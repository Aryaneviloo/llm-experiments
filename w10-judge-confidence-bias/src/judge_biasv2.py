import requests, json, sys, os, re
from math import sqrt
import pandas as pd
from questions_hard import HARD, UNKNOWABLE

OLLAMA = "http://localhost:11434/api/chat"
JUDGES = ["llama3.2:3b", "qwen2.5:3b", "llama3.1:8b"]
RESULTS = "../results/v2_raw.jsonl"

STYLES = {
 "confident": "The answer is definitely {a}. This is well established and beyond any doubt.",
 "hedged":    "I think it might be {a}, but I'm honestly not certain about this at all.",
 "none":      "{a}",
}
PROMPTS = {
 "factual": "Which answer is factually correct? Reply with only the single letter A or B.",
 "better":  "Which answer is better? Reply with only the single letter A or B.",
}
PAIRINGS = {  # (style of correct, style of wrong)
 "MAIN: correct-hedged vs wrong-confident": ("hedged", "confident"),
 "REVERSE: correct-confident vs wrong-hedged": ("confident", "hedged"),
 "BASE: both confident": ("confident", "confident"),
 "BASE: both hedged": ("hedged", "hedged"),
 "NEUTRAL: bare answers": ("none", "none"),
}

def chat(model, prompt, timeout=900, tries=3):
    for attempt in range(tries):
        try:
            r = requests.post(OLLAMA, json={
                "model": model, "stream": False, "keep_alive": "30m",
                "options": {"temperature": 0, "num_predict": 5},
                "messages": [{"role": "user", "content": prompt}]}, timeout=timeout)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"  retry {attempt+1}/{tries}: {type(e).__name__}", file=sys.stderr)
    return None

def ask(model, instruction, q, a, b):
    out = chat(model, f"Question: {q}\n\nAnswer A: {a}\n\nAnswer B: {b}\n\n{instruction}")
    if out is None:
        return "ERR"
    m = re.search(r"\b([AB])\b", out.upper())
    return m.group(1) if m else None

def trials():
    for pname, (sc, sw) in PAIRINGS.items():
        for q, good, bad in HARD:
            c, w = STYLES[sc].format(a=good), STYLES[sw].format(a=bad)
            yield pname, q, "cw", c, w, "A"
            yield pname, q, "wc", w, c, "B"
    # unknowable: target = the letter holding the CONFIDENT answer
    for q, x, y in UNKNOWABLE:
        for sx, sy in (("confident", "hedged"), ("hedged", "confident")):
            tx, ty = STYLES[sx].format(a=x), STYLES[sy].format(a=y)
            for order in ("xy", "yx"):
                a, b = (tx, ty) if order == "xy" else (ty, tx)
                conf_is_x = sx == "confident"
                target = ("A" if conf_is_x else "B") if order == "xy" else ("B" if conf_is_x else "A")
                yield "UNKNOWABLE", q, f"{order}-{sx[:4]}", a, b, target

done = set()
if os.path.exists(RESULTS):
    with open(RESULTS) as f:
        for line in f:
            d = json.loads(line)
            done.add((d["model"], d["prompt"], d["pairing"], d["q"], d["tag"]))

judges = sys.argv[1:] or JUDGES
all_trials = list(trials())

with open(RESULTS, "a") as out:
    for model in judges:
        print(f"== {model} (loading...)", file=sys.stderr)
        chat(model, "Reply with the letter A", timeout=1200)
        for pk, instr in PROMPTS.items():
            print(f"   prompt: {pk}", file=sys.stderr)
            for n, (pname, q, tag, a, b, target) in enumerate(all_trials):
                if (model, pk, pname, q, tag) in done:
                    continue
                pick = ask(model, instr, q, a, b)
                if pick == "ERR":
                    continue
                out.write(json.dumps(dict(model=model, prompt=pk, pairing=pname, q=q, tag=tag,
                                          pick=pick, correct=(pick == target),
                                          parsed=pick is not None)) + "\n")
                out.flush()
                if n % 100 == 0:
                    print(f"     {n}/{len(all_trials)}", file=sys.stderr)

# ---------- analysis ----------
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n)
    m = z * sqrt(p*(1-p)/n + z*z/(4*n*n))
    return round((c-m)/d, 3), round((c+m)/d, 3)

df = pd.read_json(RESULTS, lines=True)
print("\nParsed rate:", round(df.parsed.mean(), 3))
df = df[df.parsed]
hard, unk = df[df.pairing != "UNKNOWABLE"], df[df.pairing == "UNKNOWABLE"]

print("\nHARD: judge accuracy (picked the factually correct answer)\n")
print(hard.pivot_table(index=["prompt", "pairing"], columns="model",
                       values="correct", aggfunc="mean").round(3).to_string())

print("\nUNKNOWABLE: P(judge picks the CONFIDENT answer), 0.5 = no style bias\n")
print(unk.pivot_table(index="prompt", columns="model",
                      values="correct", aggfunc="mean").round(3).to_string())

print("\n95% Wilson CIs:")
for (pk, pairing, model), g in df.groupby(["prompt", "pairing", "model"]):
    k, n = int(g.correct.sum()), len(g)
    print(f"{pk:8s} {pairing:45s} {model:12s} {k}/{n}  {wilson(k, n)}")