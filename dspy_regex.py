# optimize_regex_prompt.py
# pip install -U dspy

import os, re, sys, json, csv
import dspy
from typing import List, Optional
# from dspy.teleprompt import MIPROv2
from dspy.teleprompt import GEPA

# --- LM config (defaults to local Ollama chat; override via env) ---
LM_ID   = os.environ.get("DSPY_LM_ID",   "ollama_chat/qwen2.5:14b")
LM_BASE = os.environ.get("DSPY_LM_BASE", "http://localhost:11434")
lm = dspy.LM(LM_ID, api_base=LM_BASE, api_key="", temperature=0.2, max_tokens=384)
dspy.configure(lm=lm)

# --- Signature for the optimizer's student program ---
class RegexFromCSV(dspy.Signature):
    """Return exactly {"regex":"..."} (Python 're' compatible)."""
    question: str       = dspy.InputField()
    columns:  list[str] = dspy.InputField()
    sample:   list[str] = dspy.InputField()
    regex_json: str     = dspy.OutputField()

# --- Base instructions (optimizer will refine/augment these) ---
BASE_INSTRUCTIONS = """
Output exactly one JSON object and nothing else.

Goal:
Given a natural-language question about a CSV and only minimal context (column names + the first three raw lines),
produce a SINGLE Python 're' compatible regex that:
1) Matches the relevant CSV line(s), and
2) Captures the requested cell value in ONE named group (?P<value>...).

Constraints:
- Engine: Python 're' ONLY (no \\K, no atomic groups, no variable-length lookbehind).
- Prefer literal tokens present in the question (e.g., exact keys like dates or ids).
- If a specific row key is present (e.g., a date), anchor with ^KEY, at line start.
- Cells may be quoted or unquoted; a tolerant cell atom is allowed: (?:\\\"[^\\\"]*\\\"|[^,]*)
- For quoted-or-not values, capture with optional quotes: \\\"?(?P<value>[^\\\",]+)\\\"?
- Return JSON in this exact form: {"regex":"YOUR_PATTERN"}
""".strip()

# --- Helpers (no printing, no examples) ---
def first_n_raw_lines(path: str, n: int = 3) -> List[str]:
    lines, cnt = [], 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            lines.append(line.rstrip("\n"))
            cnt += 1
            if cnt >= n: break
    return lines

def header_columns(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        try:
            return next(r)
        except StopIteration:
            return []

def extract_regex_json(text: str) -> Optional[str]:
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e <= s: return None
    try:
        obj = json.loads(text[s:e+1])
        rgx = obj.get("regex")
        return rgx if isinstance(rgx, str) and rgx.strip() else None
    except Exception:
        return None

def apply_regex_capture(path: str, pattern: str, max_preview: int = 3) -> int:
    """Return number of matched lines (cap preview internally)."""
    prog = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if prog.search(line):
                n += 1
                if n >= max_preview:  # early stop for metric speed
                    break
    return n

def tokens_from_question(question: str, columns: List[str]) -> List[str]:
    q = question.lower()
    base = re.findall(r"[a-z0-9_:/.\-]{3,}", q)
    cols = [c.strip().lower() for c in columns if isinstance(c, str) and c.strip()]
    seen, out = set(), []
    for t in base + cols:
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out

# --- Weakly supervised metric (no labels) for MiPRO ---
def make_metric(csv_path: str, question: str, columns: List[str]):
    toks = tokens_from_question(question, columns)
    def metric(example, pred, trace=None, *args, **kwargs) -> float:
        rgx = extract_regex_json(getattr(pred, "regex_json", ""))
        if not rgx:
            return 0.0
        try:
            re.compile(rgx)
        except re.error:
            return 0.0

        # Reward presence of the named capture group
        has_group = 1.0 if "(?P<value>" in rgx else 0.0

        # Prefer a small (non-zero) number of matches on the file
        n = apply_regex_capture(csv_path, rgx, max_preview=5)
        if   n == 0: count_score = 0.0
        elif n <= 5: count_score = 1.0
        elif n <=20: count_score = 0.7
        else:        count_score = 0.4

        # Encourage token alignment: the regex should reference question/column cues
        align = 0.0
        rgx_l = rgx.lower()
        if toks:
            hits = sum(1 for t in toks if t in rgx_l)
            align = hits / max(3, len(toks))  # soft normalization

        # Combine (weights sum to 1.0)
        score = 0.35 * count_score + 0.15 * has_group + 0.5 * align
        return float(max(0.0, min(1.0, score)))
    return metric

# --- Render a single, paste-ready SYSTEM PROMPT string ---
def render_system_prompt(compiled) -> str:
    instr = (getattr(compiled.signature, "instructions", "") or "").strip()
    demos = getattr(compiled, "demos", [])
    # Keep only instructions; optionally append demos as “Guidance” section.
    parts = [instr]
    if demos:
        lines = ["", "### Few-shot guidance (optional)"]
        for i, d in enumerate(demos, 1):
            lines.append(f"- Demo {i}: inputs={d.inputs()} outputs={d.outputs()}")
        parts.append("\n".join(lines))
    return "\n".join(parts).strip()

def main():
    if len(sys.argv) != 3:
        print("Usage: python optimize_regex_prompt.py \"<question>\" <file.csv>", file=sys.stderr)
        sys.exit(1)

    question, csv_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    cols   = header_columns(csv_path)
    sample = first_n_raw_lines(csv_path, 3)

    # Build student with base instructions (optimizer will refine them)
    student = dspy.Predict(RegexFromCSV)
    student.signature.instructions = BASE_INSTRUCTIONS

    # Single unlabeled example: inputs only
    ex = dspy.Example(
        question=question,
        columns=cols,
        sample=sample
    ).with_inputs("question", "columns", "sample")

    # Optimize the prompt ONLY (GEPA)
    # tele = MIPROv2(metric=make_metric(csv_path, question, cols), auto="heavy",log_dir="logs",verbose=True)
    # compiled = tele.compile(student=student, trainset=[ex], valset=[ex],provide_traceback=True)
    reflection_lm = dspy.LM(model='ollama_chat/qwen2.5:14b', api_base='http://localhost:11434', temperature=1.0, max_tokens=32000)
    tele = GEPA(metric=make_metric(csv_path, question, cols), max_full_evals=50, reflection_lm=reflection_lm)
    compiled = tele.compile(student=student, trainset=[ex], valset=[ex])
    compiled.save("optimized_regex_gepa.json")
    # Print ONLY the optimized, paste-ready system prompt
    print(render_system_prompt(compiled))

if __name__ == "__main__":
    main()
