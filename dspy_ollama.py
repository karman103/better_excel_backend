# dspy_ollama.py
# pip install -U dspy

import dspy
from dspy.teleprompt import MIPROv2

# 1) Configure LM (Ollama chat endpoint)
lm = dspy.LM(
    'ollama_chat/qwen2.5:14b',
    api_base='http://localhost:11434',
    api_key='',
    temperature=0.2,
    max_tokens=256,
)
dspy.configure(lm=lm)

# 2) Signature: declare fields with InputField/OutputField
class RewriteActive(dspy.Signature):
    # """Rewrite the sentence in active voice, preserving meaning and names."""
    sentence: str = dspy.InputField()
    answer:   str = dspy.OutputField()

# 3) Base module to optimize
student = dspy.Predict(RewriteActive)

# 4) Single training example
train = [
    dspy.Example(
        sentence="Say hello world",
        answer="print('Hello, world!')"
    ).with_inputs("sentence")
]

# 5) Metric
def exact_match(ex, pred, trace=None):
    gold = ex.answer.strip().lower()
    got  = getattr(pred, "answer", "").strip().lower()
    return float(gold == got)

# 6) Optimize with MiPROv2 (light search for tiny data)
tele = MIPROv2(metric=exact_match, auto="light")
rewriter = tele.compile(
    student=student,
    trainset=train,
    valset=train
)

# 7) Use optimized program
print(rewriter(sentence="The window was broken by the wind.").answer)

# # (Optional) Inspect the final rendered prompt/messages
# dspy.inspect_history(n=1)
rewriter.save("optimized_prpmt.json")