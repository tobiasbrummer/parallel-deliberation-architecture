# Simulation 4: Prompt-PDA on GSM8K Benchmark

Date: 2026-04-08
Status: POSITIVE

## Setup

- Model: Qwen3-8B (qwen/qwen3-8b via OpenRouter)
- Benchmark: GSM8K test set, 200 random questions (seed 42)
- Baseline: Single pass, temperature 0.0
- PDA: 3 workers (methodical, creative, skeptical) at temperature 0.7 + 1 merge at temperature 0.0
- Answer extraction: regex for #### pattern, fallback to last number

## Results

| Condition | Accuracy | Correct | Wrong |
|-----------|----------|---------|-------|
| Baseline | 96.0% | 192/200 | 8 |
| PDA (n=3) | 97.5% | 195/200 | 5 |
| Delta | +1.5pp | | |

## Disagreement Analysis

- Total disagreements: 3 (out of 200)
- PDA fixed baseline error: 3
- PDA broke baseline correct: 0
- Error recovery rate: 37.5% (3/8 baseline errors corrected)

## Worker Perspectives

1. Methodical: "Break the problem into small steps. Double-check each calculation."
2. Creative: "Look for shortcuts and patterns. Find the most efficient solution path."
3. Skeptical: "Consider edge cases and common mistakes. Verify assumptions."

## Merge Strategy

The merge prompt evaluates reasoning quality, not majority vote:
"If they disagree, identify which solution has the correct reasoning and pick that answer."

## Key Findings

1. PDA improves accuracy even on a strong baseline (96% is already high for 8B)
2. Zero regressions -- PDA never makes a correct answer wrong
3. The 37.5% error recovery rate suggests PDA catches roughly 1 in 3 errors
4. Cost: ~4x tokens per question (3 workers + 1 merge vs 1 baseline)
5. The creative and skeptical perspectives provide genuine diversity

## Limitations

- Single run (no statistical significance testing across seeds)
- GSM8K may be too easy for this model to show large effects
- Temperature 0.7 for workers may not be optimal
- No analysis of which worker perspective was most valuable

## Next Steps

- Run on a harder benchmark (MMLU-Pro, GPQA) where baseline is lower
- Test with n=5 workers for diminishing returns analysis
- Analyze which worker catches which errors
- Test with different models (Gemma 4, Mistral Small)
- Test iterative PDA (multiple rounds of deliberation)

## Cost

Estimated: <$1 total for 200 questions (Qwen3-8B is $0.05/M input, $0.40/M output)
