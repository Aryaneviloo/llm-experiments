import pandas as pd
from math import comb

df = pd.read_json("../results/v2_raw.jsonl", lines=True)
df = df[df.parsed & (df.pairing != "UNKNOWABLE")].copy()
df["correct"] = df.correct.astype(int)

MAIN = "MAIN: correct-hedged vs wrong-confident"
REV = "REVERSE: correct-confident vs wrong-hedged"
NEU = "NEUTRAL: bare answers"

def sign_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return round(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n), 4)

print("MAIN vs REVERSE (same question, same order; only the style assignment flips)\n")
tb = tc = 0
for (model, prompt), g in df.groupby(["model", "prompt"]):
    w = g.pivot_table(index=["q", "tag"], columns="pairing", values="correct", aggfunc="first")
    w = w.dropna(subset=[MAIN, REV])
    b = int(((w[REV] == 1) & (w[MAIN] == 0)).sum())   # right under REVERSE, wrong under MAIN
    c = int(((w[REV] == 0) & (w[MAIN] == 1)).sum())   # the opposite
    tb += b; tc += c
    print(f"{model:12s} {prompt:8s} REV-only right={b:2d}  MAIN-only right={c:2d}  p={sign_p(b, c)}")
print(f"\nPooled: {tb} vs {tc}, p={sign_p(tb, tc)} (exploratory, same questions reused across models)")

print("\nQuestions where MAIN loses the most vs NEUTRAL (max 12 trials each):\n")
m = df[df.pairing.isin([MAIN, NEU])].pivot_table(index="q", columns="pairing",
                                                  values="correct", aggfunc="sum")
m["lost"] = m[NEU] - m[MAIN]
print(m.sort_values("lost", ascending=False).head(10).to_string())

# ---- question-level analysis: the question is the unit ----
q = df[df.pairing.isin([MAIN, REV])].pivot_table(
    index="q", columns="pairing", values="correct", aggfunc="sum")
q["net"] = q[REV] - q[MAIN]          # > 0: style flip hurt on this question
pos, neg = int((q.net > 0).sum()), int((q.net < 0).sum())
print(f"\nPer question: MAIN worse on {pos}, better on {neg}, "
      f"tied on {int((q.net == 0).sum())}   sign-test p={sign_p(pos, neg)}")

def pooled(d):
    w = d[d.pairing.isin([MAIN, REV])].pivot_table(
        index=["model", "prompt", "q", "tag"], columns="pairing",
        values="correct", aggfunc="first").dropna()
    b = int(((w[REV] == 1) & (w[MAIN] == 0)).sum())
    c = int(((w[REV] == 0) & (w[MAIN] == 1)).sum())
    return b, c, sign_p(b, c)

print("\nPooled (REV-only, MAIN-only, p):")
print("  all questions     ", pooled(df))
ranked = q.net.sort_values(ascending=False).index
print("  minus top 1       ", pooled(df[~df.q.isin(ranked[:1])]))
print("  minus top 5       ", pooled(df[~df.q.isin(ranked[:5])]))
print("  minus top 10      ", pooled(df[~df.q.isin(ranked[:10])]))

# does the style flip hurt more where the judge's knowledge is shakier?
# knowledge is measured on the BASE pairings, which are independent of MAIN/REV trials
BASE = ["BASE: both confident", "BASE: both hedged"]
know = df[df.pairing.isin(BASE)].groupby("q").correct.mean().rename("know")
q = q.join(know)
q["bin"] = pd.cut(q.know, [0, 0.75, 0.95, 1.0],
                  labels=["shaky (<=.75)", "mid (.75-.95]", "solid (>.95)"],
                  include_lowest=True)
print("\nStyle-flip loss by how well the judges know the fact (BASE accuracy):")
print(q.groupby("bin", observed=True).agg(n_questions=("net", "size"),
                                          mean_net=("net", "mean"),
                                          total_net=("net", "sum")).to_string())