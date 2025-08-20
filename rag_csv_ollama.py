#!/usr/bin/env python3
# Usage:
#   python rag_csv_ollama.py "What is the close on 09/03/2024?" data.csv

import os, re, sys, json, requests, pandas as pd
from typing import List, Tuple, Optional, Generator

# --- Configure your local Ollama model tag here ---
# OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
# OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:270m")
# OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b ")


OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434")

MAX_CTX_ROWS   = 25
MAX_ROW_CHARS  = 400
COLUMNS_AS_TEXT = None  # set to list of columns to restrict matching scope, else None

SYS_INSTRUCT_REGEX = (
    "You are a world class assistant that writes correct, specific Regular Expressions which get applied to files worth 10billion$.\n"
    "Given a user's question about a CSV, produce ONLY a single regex pattern that will match relevant rows or fields.\n"
    "The regex must be RE2/PCRE-compatible and specific.\n"
    "IMPORTANT: Look at the actual CSV structure in the sample data to understand the format.\n"
    "Analyze the column names and sample data to determine the correct pattern.\n"
    "Reasoning: low\n"
    "Respond with this JSON format only: {\"regex\": \"...\"}\n"
)


SYS_INSTRUCT_ANSWER = (
    "You are a data assistant. You will receive a user question and a small set of matched CSV rows/ matched cells .\n"
     "If a (?P<value>) capture is present in any match, that capture is the requested field value.\n"
    "Return those value(s) directly and concisely. Do NOT claim missing information when a capture exists."
    "Use ONLY the provided <matched row content> as ground truth. Read the naswer in the Matched rows and if it is not empty that is your answer."
    "If the answer is not present, say you cannot find it."

)
# --- drop-in replacement for your helpers in rag_csv_mlx.py ---

import json, requests

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:14b"

def ollama_chat_stream(system_msg: str,
                       user_msg: str,
                       *,
                       model: str = DEFAULT_MODEL,
                       temperature: float = 0.1,
                       max_tokens: int = 200,
                       stream: bool = True):
    """Yield assistant chunks from Ollama /api/chat (Qwen)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": str(system_msg)},
            {"role": "user",   "content": str(user_msg)}
        ],
        "stream": stream,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens)
        }
    }

    # DEBUG: see the exact body we send
    # print("PAYLOAD:", json.dumps(payload)[:1000])

    r = requests.post(OLLAMA_URL, json=payload, stream=stream)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        # Surface Ollama's error text (why 400?)
        print("STATUS:", r.status_code)
        print("BODY:", r.text)
        raise

    if stream:
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            obj = json.loads(line)
            if "message" in obj and "content" in obj["message"]:
                yield obj["message"]["content"]
            if obj.get("done"):
                break
    else:
        obj = r.json()
        yield obj["message"]["content"]

def ollama_chat(system_msg: str,
                user_msg: str,
                *,
                model: str = DEFAULT_MODEL,
                temperature: float = 0.1,
                max_tokens: int = 200) -> str:
    """Return a single consolidated reply. Uses named-only args to avoid
    positional mixups."""
    return ''.join(
        ollama_chat_stream(system_msg,
                           user_msg,
                           model=model,
                           temperature=temperature,
                           max_tokens=max_tokens,
                           stream=True)
    )
# --- end replacement ---

def extract_regex(text: str) -> Optional[str]:
    try:
        # keep only JSON chunk up to [end]
        # end_tag = text.find("[end]")
        # if end_tag != -1:
        #     text = text[:end_tag]
        # find JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        obj = json.loads(text[start:end+1])
        rgx = obj.get("regex")
        if not isinstance(rgx, str) or not rgx.strip():
            return None
        re.compile(rgx)  # validate
        return rgx
    except Exception:
        return None

def df_rows_as_strings(df: pd.DataFrame, columns) -> List[str]:
    rows = []
    if columns is None:
        for _, row in df.iterrows():
            s = ", ".join(f"{c}={row[c]}" for c in df.columns)
            rows.append(s)
    else:
        for _, row in df.iterrows():
            s = ", ".join(f"{c}={row[c]}" for c in columns if c in df.columns)
            rows.append(s)
    return rows
def apply_regex_capture(filename: str, pattern: str) -> List[Tuple[int, str]]:
    prog = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    results = []
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            m = prog.search(line)
            if m:
                val = m.groupdict().get('value', None)
                results.append((i, val if val else line[:MAX_ROW_CHARS]))
                if len(results) >= MAX_CTX_ROWS:
                    break
    return results

def build_answer_prompt(question: str, matches):
    parts = []
    for tup in matches:
        if len(tup) == 3:
            i, full, val = tup
            if val is not None:
                parts.append(f"- Row#{i} (capture): {val}\n  Full line: {full}")
            else:
                parts.append(f"- Row#{i}: {full}")
        else:
            # backward-compat: (i, snippet_or_value)
            i, s = tup
            parts.append(f"- Row#{i}: {s}")
    ctx = "\n".join(parts) if parts else "(no matches)"
    return (
        "You will receive a question and regex matches from a CSV.\n"
        "Each match may include the full line and (if present) the (?P<value>) capture.\n"
        "If a capture exists, that capture IS the requested value. Return it directly.\n\n"
        f"Question:\n{question}\n\nMatches:\n{ctx}\n\nReturn just the value(s) if present."
    )


def rag_csv_answer_stream(question: str, csv_path: str) -> Generator[str, None, None]:
    """Streaming version of RAG CSV answer"""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    column_context = "Available columns: " + ", ".join([f'"{col}"' for col in df.columns]) + "\n"
    csv_context = open(csv_path, 'r', encoding='utf-8').readlines()[:3]
    # Ask model to make a regex (non-streaming)
    user_regex = f"""
You will output exactly one JSON object and nothing else.

Goal:
Given a natural-language question about a CSV and minimal schema context, produce a SINGLE Python 're' compatible regex pattern that:
1) Matches the CSV row(s) that answer the question, and
2) Captures the requested field value in a single named group called (?P<value>...).

Context:
- Question: {question}
- Column names: {column_context}
- CSV sample (header + two rows):
  {csv_context}

Assumptions & constraints:
- Engine: Python 're' (no PCRE-only syntax). Do NOT use '\\K', atomic groups, or variable-length lookbehind.
- CSV fields may be quoted (e.g., "1,289.62") or unquoted without internal commas.
- Prefer literal tokens present in the question (e.g., an exact MM/DD/YYYY date, exact column name).
- Avoid over-anchoring unless necessary; if a specific date/key is in the question, anchor the line start with that key (e.g., ^KEY,).
- Use a SINGLE pattern.
- Use exactly one named capture group: (?P<value>...), capturing ONLY the requested cell value (or the remainder of the row when appropriate).
- Do NOT include flags; the consumer code sets flags.
- Output format MUST be:
  {{"regex": "YOUR_PATTERN_HERE"}}

Guidelines for constructing the pattern:
- Use this tolerant CSV cell atom to match (but not capture) a single cell: (?:"[^"]*"|[^,]*)
- When you need to CAPTURE a single CSV cell (quoted OR unquoted), use: (?P<value>"[^"]*"|[^,]*)
- If the question asks for a value by date and column (e.g., "What is the Close on 07/08/2024?"):
  - Start-of-line anchor the date: ^07/08/2024,
  - Then skip columns using the tolerant cell atom with commas,
  - Capture the target column's cell with (?P<value>"[^"]*"|[^,]*)
  - Example shape (for Close = 5th column):
    ^07/08/2024,(?:"[^"]*"|[^,]*),(?:"[^"]*"|[^,]*),(?:"[^"]*"|[^,]*),(?P<value>"[^"]*"|[^,]*)
- If the question names a column but not a unique row key, build a pattern using literals from the question to select the intended rows and capture that column’s cell.
- If multiple rows may match (e.g., a quarter/range), the same pattern should match each relevant row and capture the requested column’s value from each.
- If the question requests an entire row (e.g., "Q3 2022 data"), after anchoring the row key(s), capture the remainder of the row:
  ^KEYS,(?P<value>.*)$

Examples (illustrative):
Q: "What is the Close on 07/08/2024?"
RETURNS:
{{"regex":"^07/08/2024,(?:\\\"[^\\\"]*\\\"|[^,]*),(?:\\\"[^\\\"]*\\\"|[^,]*),(?:\\\"[^\\\"]*\\\"|[^,]*),(?P<value>\\\"[^\\\"]*\\\"|[^,]*)"}}

Q: "Give me the Volume on 09/03/2024"
RETURNS:
{{"regex":"^09/03/2024,(?:\\\"[^\\\"]*\\\"|[^,]*),(?:\\\"[^\\\"]*\\\"|[^,]*),(?:\\\"[^\\\"]*\\\"|[^,]*),(?:\\\"[^\\\"]*\\\"|[^,]*),(?P<value>\\\"[^\\\"]*\\\"|[^,]*)"}}

Now produce ONLY the JSON object with the single key "regex".
"""


#     user_regex = f"""
#     You will output exactly one JSON object and nothing else.

# Goal:
# Given a natural-language question about a CSV, produce a SINGLE Python 're' compatible regex pattern that:
# 1) Matches the CSV row(s) containing the data needed to answer the question, and
# 2) Captures the specific value(s) needed to answer the question in named group(s): (?P<value>...).

# Context:
# - Question: {question}
# - Full CSV Content:
# {open(csv_path, 'r').read()}

# Assumptions & constraints:
# - Engine: Python 're' only. Do NOT use \K, atomic groups, or variable-length lookbehind.
# - CSV fields may be quoted (e.g., "1,289.62") or unquoted without internal commas.
# - Prefer literal tokens present in the question (e.g., an exact MM/DD/YYYY date or explicit key/ID).
# - Anchor the start of line when a precise row key is present (e.g., ^KEY,), otherwise avoid over-anchoring.
# - Use ONE pattern only.
# - Use exactly ONE named capture group (?P<value>...) that captures ONLY the requested cell value.
# - Do NOT include flags; the consumer sets flags externally.
# - In the JSON you return, escape backslashes as required by JSON.
# - Output format MUST be:
#   {{"regex":"YOUR_PATTERN_HERE"}}

# Guidelines for constructing the pattern:
# - To skip columns reliably, use a tolerant CSV cell atom:
#   (?:\"[^\"]*\"|[^,]*)
# - When a specific row key is known from the question (date/id/year+quarter/etc.), begin with ^KEY, then repeat the tolerant cell atom with trailing commas to reach the target column, and capture it with optional quotes:
#   \"?(?P<value>[^\",]+)\"?
# - If multiple rows may match (e.g., a range or a quarter), the same pattern should match each relevant row and capture the requested column’s value from each.
# - If the question specifies a column name but not a unique row key, construct a pattern that selects the intended rows (based on literals from the question) and captures that column’s cell.

# Now produce ONLY the JSON object with the single key "regex".
# """
    print(user_regex)
    print(f"Question: {question}", file=sys.stderr)
    raw = ollama_chat(SYS_INSTRUCT_REGEX , user_regex, max_tokens=200, temperature=0.1)
    print(f"Model response: {raw}", file=sys.stderr)
    rgx = extract_regex(raw)
    print(f"Extracted regex: {rgx}", file=sys.stderr)

    # Fallback regex (keyword ORs) if model fails
    # if not rgx:
    #     quoted = re.findall(r'"([^"]+)"', question)
    #     tokens = quoted if quoted else [t for t in re.findall(r"[A-Za-z0-9_./-]+", question) if len(t) >= 3]
    #     rgx = "|".join(map(re.escape, tokens)) if tokens else r".*"

    # Retrieval
    rows = df_rows_as_strings(df, COLUMNS_AS_TEXT)
    matches = apply_regex_capture(csv_path, rgx)
    print(f"Matched {len(matches)} rows with regex: {rgx}", file=sys.stderr)
    print(f"Sample matches: {matches[:3]}", file=sys.stderr)
    # Answer based on matched rows (streaming)
    prompt = build_answer_prompt(question, matches)
    print(f"Answer prompt: {prompt}", file=sys.stderr)
    # yield f"[regex used: {rgx}] [matches: {len(matches)}]\n\n"
    response =ollama_chat(SYS_INSTRUCT_ANSWER,
                prompt, max_tokens=400, temperature=0.2)
    print(f"Final response: {response}", file=sys.stderr)
    return response
def main():
    if len(sys.argv) != 3:
        print("Usage: python rag_csv_ollama.py \"your question\" path/to/file.csv")
        sys.exit(1)

    a, b = sys.argv[1], sys.argv[2]
    if os.path.exists(a) and not os.path.exists(b):
        csv_path, question = a, b
    elif os.path.exists(b) and not os.path.exists(a):
        csv_path, question = b, a
    elif os.path.exists(a) and os.path.exists(b):
        print("Please provide exactly one CSV path and one question.", file=sys.stderr)
        sys.exit(1)
    else:
        print("Could not find a CSV file in the two arguments.", file=sys.stderr)
        sys.exit(1)

    for chunk in rag_csv_answer_stream(question, csv_path):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    main()