# W10: Judge confidence bias

## Hypothesis
LLM judges pick a confident wrong answer over a hedged correct one more often than their baseline error rate predicts, and smaller judges are hurt more.

## Method
- 30 factual questions, each with a correct and a wrong answer
- Each answer wrapped in a confident or hedged template
- 4 pairings (main, reverse, two baselines), both A/B orders to cancel position bias
- Judges (local via Ollama, temperature 0): llama3.2:3b, llama3.1:8b, qwen2.5:7b

## Results

Judge accuracy (picked the factually correct answer):

model                                       llama3.1:8b  llama3.2:3b  qwen2.5:3b
pairing                                                                         
BASE: both confident                              0.983        0.933       1.000
BASE: both hedged                                 1.000        0.983       0.967
MAIN: correct-hedged vs wrong-confident           0.967        0.967       0.950
REVERSE: correct-confident vs wrong-hedged        0.983        0.933       1.000

Unparseable rate: 0.0

P(judge picks 'A') per model (0.5 = no position bias):
 model
llama3.1:8b    0.517
llama3.2:3b    0.479
qwen2.5:3b     0.504


## Conclusions
No measurable confidence bias on easy factual questions in 3 small local judges; the pilot is at ceiling (93-100%) and underpowered

## Limitations
- Templates differ in length and wording, so "confidence" is confounded with them
- Easy questions may put baselines at ceiling
- n=60 per cell, no confidence intervals yet