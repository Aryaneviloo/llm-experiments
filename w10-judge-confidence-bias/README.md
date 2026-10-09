# W10: Judge confidence bias

## Hypothesis
LLM judges pick a confident wrong answer over a hedged correct one more often than their baseline error rate predicts, and smaller judges are hurt more.

## Method
- 30 factual questions, each with a correct and a wrong answer
- Each answer wrapped in a confident or hedged template
- 4 pairings (main, reverse, two baselines), both A/B orders to cancel position bias
- Judges (local via Ollama, temperature 0): llama3.2:3b, llama3.1:8b, qwen2.5:7b

## Results
(to fill after running)

## Conclusions
(to fill after running)

## Limitations
- Templates differ in length and wording, so "confidence" is confounded with them
- Easy questions may put baselines at ceiling
- n=60 per cell, no confidence intervals yet