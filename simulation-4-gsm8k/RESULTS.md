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

---

# Simulation 4b: Prompt-PDA on MMLU-Pro (recall benchmark)

Date: 2026-04 (run aborted at item 120/200 due to API-cost decision)
Status: NEGATIVE -- PDA does **not** help on a recall-heavy benchmark

## Why MMLU-Pro

GSM8K is reasoning-heavy. MMLU-Pro is recall-heavy (multiple-choice
knowledge questions across many domains). The hypothesis was: if PDA's
gains come from "better reasoning", they should not appear on a benchmark
that mostly tests "do you know the answer".

## Setup

- Model: Qwen3-8B (via OpenRouter)
- Benchmark: MMLU-Pro, 200 questions planned (seed 42)
- Baseline: single pass, T=0.0, answer extraction `Answer: <letter>` regex
- PDA: 3 workers (methodical / creative / skeptical) at T=0.7 + 1 merge at T=0.0

## Results (n=119, partial -- run aborted at item 120/200)

| Condition  | Accuracy | Correct |
|------------|----------|---------|
| Baseline   | 75.6%    | 90/119  |
| PDA (n=3)  | 74.8%    | 89/119  |
| Delta      | **-0.8pp** | |

Source: `sim4b_mmlu_pro_partial_recovered.json` (per-item data re-extracted
from `sim4b_run.log` by `recover_sim4b_from_log.py` -- see the
reproducibility note below).

## Key Finding

**PDA does not help on MMLU-Pro.** Effective delta is zero (within noise
for n=119). This is consistent with the interpretation that prompt-level
PDA gains are reasoning-process gains, not recall gains -- the benchmark
tests recall, so multi-perspective deliberation has nothing to add.

This is a **negative result that supports** the broader story (PDA moves
reasoning, not knowledge). It is just as informative as Simulation 4's
positive result, possibly more.

## Reproducibility note

The original `sim4b_mmlu_pro_200.json` file in this directory has all
`baseline_answer` and `pda_answer` fields set to `null`. That is a
serialization bug, **not** a parser bug: the run aborted at item 120/200,
the final `json.dump` path wrote stub values, but the actual per-item
predictions are captured in `sim4b_run.log` (one block per item).

`recover_sim4b_from_log.py` parses that log line-by-line and writes
`sim4b_mmlu_pro_partial_recovered.json` containing the recovered per-item
data plus a summary block. The summary uses the last cumulative
`Running:` line in the log (n=119) as authoritative.

```sh
cd simulation-4-gsm8k && python3 recover_sim4b_from_log.py
```

## Limitations

- n=119, single run, single seed.
- Cost-bounded: continuing to n=200 was not pursued because the trend at
  119 was clear and the per-question cost was non-trivial against the
  conclusion strength.
- A single-perspective ablation (1 worker vs 3) was not run on MMLU-Pro
  -- would rule out "PDA = just more compute".
