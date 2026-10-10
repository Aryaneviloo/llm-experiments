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

## Results (v2)
- 48 hard factual questions, 3 local judges (llama3.2:3b, qwen2.5:3b, llama3.1:8b), 2 judge prompts.
- MAIN (hedged-correct vs confident-wrong) was below REVERSE (confident-correct vs hedged-wrong) in all 6 model×prompt cells; 3 of 6 individually p<0.05 (paired sign test, uncorrected).
- Per question: MAIN worse on 18, better on 7, tied on 23 (sign test p=0.043).
- The effect is concentrated: pooled difference falls from 53 vs 17 to 12 vs 16 after removing the 10 most affected questions.
- Larger loss where judges' baseline knowledge of the fact is weaker (7 shaky questions: mean loss 1.86; 35 solid questions: 0.46). Post hoc, small n.
- UNKNOWABLE (12 invented questions): mixed directions, inconclusive.

## Conclusions
A small style effect appears on hard questions, driven by a few questions with plausible distractors. Not established: that it is confidence rather than the "I'm not certain" wording, that it appears when judges know nothing, or any effect of model size.

## Limitations
- 48 questions, hand-written by me, 3 small judges
- One wrapper pair; confidence is confounded with an explicit uncertainty disclaimer
- Analyses chosen after seeing the data; no multiple-comparison correction
- UNKNOWABLE has 12 distinct questions


## Conclusions
No measurable confidence bias on easy factual questions in 3 small local judges; the pilot is at ceiling (93-100%) and underpowered

## Limitations
- Templates differ in length and wording, so "confidence" is confounded with them
- Easy questions may put baselines at ceiling
- n=60 per cell, no confidence intervals yet