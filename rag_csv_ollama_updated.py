#!/usr/bin/env python3
# Usage:
#   OPENAI_API_KEY=... python rag_csv_openai.py "What is the Q3 2022 data?" Apple_DB.csv

import os, re, sys, json, csv
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, RateLimitError, OpenAIError

# ---------- Env / Config ----------
load_dotenv()  # loads OPENAI_API_KEY / OPENAI_MODEL if present in .env

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # override if you like
MAX_CTX_ROWS   = 25
MAX_ROW_CHARS  = 400

# Create OpenAI client (picks up OPENAI_API_KEY from env/.env)
client = OpenAI()

# ---------- Prompts ----------
SYS_INSTRUCT_REGEX = """You are a world-class assistant that writes correct, specific Regular Expressions for CSV retrieval.

You must output exactly one JSON object and nothing else.

Goal:
Given a natural-language question about a CSV and minimal schema context, produce ONE Python 're' compatible regex that:
1) Matches the CSV row(s) that contain the data needed to answer the question, and
2) Captures the specific value(s) needed in a single named group: (?P<value>...).

Context:
- Question: {question}
- Column names: {column_context}
- CSV sample (header + first 2 rows, raw text):
{csv_context}

Key requirements:
- Use (?P<value>...) to capture only the value(s) needed to answer the question.
- Handle CSV properly (quoted/unquoted, commas). Use a tolerant CSV cell atom: (?:"[^"]*"|[^,]*)
- For quoted-or-not capture, use: "?(?P<value>[^",]+)"?
- Prefer literal tokens from the question (e.g., exact date/ID). If a precise row key exists, anchor with ^KEY at line start; otherwise avoid over-anchoring.
- One pattern only. No flags in-pattern (caller sets flags).
- Engine: Python 're' only. Do NOT use \\K, atomic groups, or variable-length lookbehind.
- Escape backslashes as required by JSON.

Return ONLY:
{{"regex":"YOUR_PATTERN_HERE"}}"""

SYS_INSTRUCT_ANSWER = """You are a data assistant. You will receive a user question and a small set of matched CSV rows/cells.
Use ONLY the provided rows/cells as ground truth. If the answer is not present, say you cannot find it.
"""

# ---------- Helpers ----------
def first_n_raw_lines(path: str, n: int = 3) -> List[str]:
    lines = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            lines.append(line.rstrip("\n"))
            if i + 1 >= n:
                break
    return lines

def header_columns(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        try:
            return next(r)
        except StopIteration:
            return []

def extract_regex_from_text(text: str) -> Optional[str]:
    try:
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e == -1 or e <= s:
            return None
        obj = json.loads(text[s:e+1])
        rgx = obj.get("regex")
        if not isinstance(rgx, str) or not rgx.strip():
            return None
        re.compile(rgx)  # validate it compiles in Python 're'
        return rgx
    except Exception:
        return None

def apply_regex_capture(csv_path: str, pattern: str) -> List[Tuple[int, str]]:
    """Return [(row_index, captured_value_or_line_prefix)]."""
    prog = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    results: List[Tuple[int, str]] = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            m = prog.search(line)
            if m:
                val = m.groupdict().get("value")
                results.append((i, (val if val is not None else line[:MAX_ROW_CHARS]).rstrip()))
                if len(results) >= MAX_CTX_ROWS:
                    break
    return results

def build_answer_prompt(question: str, matches: List[Tuple[int, str]]) -> str:
    ctx = "\n".join(f"- Row#{i}: {s}" for i, s in matches) if matches else "(no matches)"
    return f"User question:\n{question}\n\nMatched rows (capped):\n{ctx}\n\nAnswer the question using ONLY the matched rows."

# ---------- OpenAI wrappers with graceful fallbacks ----------
def chat_json_schema(system: str, user: str, *, max_completion_tokens: int = 300) -> str:
    """Try strict JSON Schema (newer models)."""
    schema = {
        "name": "regex_schema",
        "schema": {
            "type": "object",
            "properties": {"regex": {"type": "string"}},
            "required": ["regex"],
            "additionalProperties": False,
        },
        "strict": True,
    }
    return client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        response_format={"type": "json_schema", "json_schema": schema},
        max_completion_tokens=max_completion_tokens,
    ).choices[0].message.content or ""

def chat_json_object(system: str, user: str, *, max_completion_tokens: int = 300) -> str:
    """Fallback: force a JSON object (no schema)."""
    return client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        response_format={"type": "json_object"},
        max_completion_tokens=max_completion_tokens,
    ).choices[0].message.content or ""

def chat_plain(system: str, user: str, *, max_completion_tokens: int = 300) -> str:
    """Last fallback: no response_format; we will parse manually."""
    return client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        max_completion_tokens=max_completion_tokens,
    ).choices[0].message.content or ""

def openai_chat_answer(system: str, user: str, *, max_completion_tokens: int = 400) -> str:
    """Plain answer step (some models reject custom temperature; omit it)."""
    return client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        max_completion_tokens=max_completion_tokens,
    ).choices[0].message.content or ""

# ---------- Pipeline: 1) regex -> 2) matches -> 3) answer ----------
def generate_regex(question: str, csv_path: str) -> str:
    cols = header_columns(csv_path)
    head = first_n_raw_lines(csv_path, 3)

    column_context = ", ".join([f'"{c}"' for c in cols]) if cols else "(unknown)"
    csv_context = "\n".join(head) if head else "(empty file)"

    user_msg = SYS_INSTRUCT_REGEX.format(
        question=question,
        column_context=column_context,
        csv_context=csv_context
    )

    # Try json_schema → json_object → plain
    print(user_msg)
    try:
        raw = chat_json_schema("Return exactly one JSON with a single key 'regex'.", user_msg,
                               max_completion_tokens=200)
    except OpenAIError:
        try:
            raw = chat_json_object("Return exactly one JSON with a single key 'regex'.", user_msg,
                                   max_completion_tokens=200)
        except OpenAIError:
            # Ask plainly and hope it's valid JSON; we still parse defensively.
            raw = chat_plain(
                "Return exactly one JSON object like {\"regex\":\"...\"} and nothing else.",
                user_msg,
                max_completion_tokens=200
            )
    print(f"Raw response: {raw}")
    rgx = extract_regex_from_text(raw)
    # if not rgx:
    #     # very mild fallback: OR of literal tokens from the question
    #     tokens = [t for t in re.findall(r"[A-Za-z0-9_./:-]+", question) if len(t) >= 3]
    #     rgx = "|".join(map(re.escape, tokens)) if tokens else r".*"
    # return rgx
    if not rgx:
        raise ValueError("Failed to extract a valid regex from the model response.")


def answer_from_regex(question: str, csv_path: str, regex: str) -> dict:
    matches = apply_regex_capture(csv_path, regex)
    prompt = build_answer_prompt(question, matches)
    try:
        final = openai_chat_answer(SYS_INSTRUCT_ANSWER, prompt, max_completion_tokens=400)
    except (APIConnectionError, RateLimitError, OpenAIError) as e:
        final = f"[OpenAI error answering] {e}"
    return {
        "regex_used": regex,
        "matches_preview": matches[:5],
        "answer": final
    }

# ---------- CLI ----------
def main():
    if len(sys.argv) != 3:
        print('Usage: OPENAI_API_KEY=... python rag_csv_openai.py "your question" path/to/file.csv', file=sys.stderr)
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

    try:
        regex = generate_regex(question, csv_path)                 # 1) make regex
        print(f"Generated regex: {regex}")
        result = answer_from_regex(question, csv_path, regex)      # 2) answer from matches only
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (APIConnectionError, RateLimitError, OpenAIError) as e:
        print(f"[OpenAI error] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
