import requests, random, json, sys
import pandas as pd

OLLAMA = "http://localhost:11434/api/chat"
JUDGES = ["llama3.2:3b", "llama3.1:8b", "qwen2.5:3b"]  # small -> large

# (question, correct, wrong): extend to 100+ later
DATA = [
 ("What is the capital of Australia?", "Canberra", "Sydney"),
 ("Who wrote 'Pride and Prejudice'?", "Jane Austen", "Charlotte Bronte"),
 ("What is the chemical symbol for gold?", "Au", "Ag"),
 ("Which planet is closest to the Sun?", "Mercury", "Venus"),
 ("In what year did World War II end?", "1945", "1943"),
 ("What is the largest ocean on Earth?", "The Pacific Ocean", "The Atlantic Ocean"),
 ("Who painted the Mona Lisa?", "Leonardo da Vinci", "Michelangelo"),
 ("What is the square root of 144?", "12", "14"),
 ("What gas do plants absorb for photosynthesis?", "Carbon dioxide", "Oxygen"),
 ("What is the capital of Canada?", "Ottawa", "Toronto"),
 ("Who developed the theory of general relativity?", "Albert Einstein", "Isaac Newton"),
 ("How many continents are there?", "Seven", "Six"),
 ("What is the hardest natural substance?", "Diamond", "Quartz"),
 ("Which element has atomic number 1?", "Hydrogen", "Helium"),
 ("What is the longest river in South America?", "The Amazon", "The Orinoco"),
 ("Who was the first person to walk on the Moon?", "Neil Armstrong", "Buzz Aldrin"),
 ("What is the capital of Turkey?", "Ankara", "Istanbul"),
 ("What is the boiling point of water at sea level in Celsius?", "100", "90"),
 ("Who wrote 'The Republic'?", "Plato", "Aristotle"),
 ("What is the smallest prime number?", "2", "1"),
 ("Which organ pumps blood through the body?", "The heart", "The liver"),
 ("What is the capital of Brazil?", "Brasilia", "Rio de Janeiro"),
 ("What language has the most native speakers?", "Mandarin Chinese", "English"),
 ("Who discovered penicillin?", "Alexander Fleming", "Louis Pasteur"),
 ("What is the tallest mountain above sea level?", "Mount Everest", "K2"),
 ("How many sides does a hexagon have?", "Six", "Eight"),
 ("Which country has the city of Marrakesh?", "Morocco", "Tunisia"),
 ("What is H2O commonly called?", "Water", "Hydrogen peroxide"),
 ("Who composed the 'Four Seasons'?", "Antonio Vivaldi", "Johann Sebastian Bach"),
 ("What is the capital of Switzerland?", "Bern", "Zurich"),
]

STYLES = {
 "confident": "The answer is definitely {a}. This is well established and there is no doubt about it.",
 "hedged":    "I think the answer might be {a}, though I'm not certain.",
}

def ask_judge(model, q, ans_a, ans_b):
    prompt = (f"Question: {q}\n\nAnswer A: {ans_a}\n\nAnswer B: {ans_b}\n\n"
              "Which answer is factually correct? Reply with only the single letter A or B.")
    r = requests.post(OLLAMA, json={
        "model": model, "stream": False,
        "options": {"temperature": 0, "num_predict": 5},
        "messages": [{"role": "user", "content": prompt}]}, timeout=120)
    out = r.json()["message"]["content"].strip().upper()
    for ch in out:
        if ch in "AB":
            return ch
    return None  # unparseable

PAIRINGS = {  # name: (style of correct, style of wrong)
 "MAIN: correct-hedged vs wrong-confident": ("hedged", "confident"),
 "REVERSE: correct-confident vs wrong-hedged": ("confident", "hedged"),
 "BASE: both confident": ("confident", "confident"),
 "BASE: both hedged": ("hedged", "hedged"),
}

rows = []
for model in JUDGES:
    print(f"== {model}", file=sys.stderr)
    for pname, (sc, sw) in PAIRINGS.items():
        for q, good, bad in DATA:
            c = STYLES[sc].format(a=good)
            w = STYLES[sw].format(a=bad)
            for order in ("cw", "wc"):  # swap positions to cancel position bias
                a, b = (c, w) if order == "cw" else (w, c)
                pick = ask_judge(model, q, a, b)
                correct_letter = "A" if order == "cw" else "B"
                rows.append(dict(model=model, pairing=pname, q=q, order=order,
                                 pick=pick, correct=(pick == correct_letter),
                                 parsed=pick is not None))

df = pd.DataFrame(rows)
df.to_json("w10_raw.json", orient="records", indent=1)

table = df.pivot_table(index="pairing", columns="model", values="correct", aggfunc="mean").round(3)
print("\nJudge accuracy (picked the factually correct answer):\n")
print(table.to_string())
print("\nUnparseable rate:", round(1 - df.parsed.mean(), 3))

# position bias check
pos = df[df.parsed].assign(pickedA=lambda d: d.pick == "A").groupby("model").pickedA.mean().round(3)
print("\nP(judge picks 'A') per model (0.5 = no position bias):\n", pos.to_string())