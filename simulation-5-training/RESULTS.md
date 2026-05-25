# PDA Distillation Results -- Sim 5

Teacher: Qwen3-8B. Student: Qwen3-1.7B. Method: QLoRA r=16, 3 epochs.
All evaluations on 200 held-out test examples per benchmark.

## Sim 5a: Single-Benchmark Distillation (GSM8K)

Proof of concept on a single reasoning benchmark.

**Setup:** 474 correct PDA traces from GSM8K (500 generated, 94.8% correct).
Training on Colab Free T4.

| Model | GSM8K |
|---|---|
| Qwen3-1.7B Base | 24.5% |
| Qwen3-1.7B PDA-Distilled | **50.0%** |

**Finding:** PDA distillation doubles accuracy on the trained domain.
Data generation cost: ~$2.

## Sim 5b: Multi-Benchmark Generalization

Does the reasoning methodology transfer across domains?

**Setup:** 834 PDA traces total -- 474 GSM8K + 174 MATH + 186 ARC-Challenge.
16-bit training on Colab Pro A100.

| Benchmark | Base | Distilled | Delta |
|---|---|---|---|
| GSM8K | 15.5% | **60.0%** | **+44.5pp** |
| MATH | 0.0% | 4.5% | +4.5pp |
| ARC-Challenge | 0.5% | **72.5%** | **+72.0pp** |
| Average | 5.3% | 45.7% | +40.3pp |

**Findings:**

1. PDA distillation generalizes across domains (math, logic, science).
2. Largest gain on ARC-Challenge (+72pp) -- a different domain from the bulk of training data.
3. MATH stays low -- 1.7B is too small for competition-level math regardless of distillation.
4. Inference cost is identical to the base model (same speed, same memory).
5. The student appears to learn the *reasoning methodology*, not domain-specific answers.

Note on GSM8K base accuracy difference vs. Sim 5a (24.5% vs. 15.5%):
the Sim 5b eval used a different prompt template and test split sample,
which lowered the base score. The relative delta is the comparable signal.

Data generation cost: ~$1.63 (additional 400 examples for MATH + ARC).

## Sim 5c: PDA vs. CoT Distillation (head-to-head)

Apples-to-apples comparison -- same teacher, same student, same training recipe,
comparable example counts. Does the multi-perspective structure matter,
or is the gain just from any structured reasoning trace?

**Setup:** 834 PDA traces vs. 830 CoT traces (single-pass chain-of-thought
from the same Qwen3-8B teacher). Identical training: QLoRA r=16, 3 epochs.

Training data correctness: PDA 94.8% / CoT 95.8% on GSM8K -- effectively equal.

| Benchmark | Base | CoT-Distilled | PDA-Distilled | PDA vs. CoT |
|---|---|---|---|---|
| GSM8K | 15.5% | 42.5% | **56.5%** | **+14.0pp** |
| MATH | 0.0% | 8.0% | 7.0% | -1.0pp |
| ARC-Challenge | 0.5% | 71.0% | 74.0% | +3.0pp |

**Findings:**

1. PDA beats CoT clearly on GSM8K (+14pp), the most reasoning-heavy benchmark.
2. PDA slightly ahead on ARC-Challenge (+3pp), roughly tied on MATH (-1pp).
3. Most of the distillation gain comes from *any* structured reasoning trace --
   CoT alone lifts ARC from 0.5% to 71% and MATH from 0% to 8%.
4. The PDA bonus is real but localised: it shows up where multi-perspective
   deliberation actually changes the outcome (long arithmetic + multi-step word problems).
5. Same inference cost as CoT distillation, same inference cost as the base model.

## Summary

| Result | Number |
|---|---|
| Best distillation gain (single benchmark, Sim 5b) | +72pp (ARC-Challenge) |
| Best distillation gain (target benchmark, Sim 5b) | +44.5pp (GSM8K) |
| PDA advantage over CoT distillation (Sim 5c, GSM8K) | +14.0pp |
| Total data generation cost (Sim 5a + 5b) | ~$3.63 |
| Total training cost | ~$0.50 (Colab Pro) |
| Inference cost overhead vs. base model | 0 |

**Takeaway:** PDA distillation works, generalizes across domains, and the
multi-perspective structure provides a measurable bonus on reasoning-heavy
tasks beyond what plain CoT distillation achieves. The student inherits the
reasoning quality of a much larger teacher at zero inference-time cost.

## Files

| File | Content |
|---|---|
| `generate_training_data.py` | GSM8K PDA data generation (Sim 5a) |
| `generate_math_arc.py` | MATH + ARC PDA data generation (Sim 5b) |
| `generate_cot_baseline.py` | CoT baseline data generation (Sim 5c) |
| `pda_training_data.jsonl` | 500 GSM8K PDA examples (474 correct) |
| `pda_math_training.jsonl` | 200 MATH PDA examples (174 correct) |
| `pda_arc_training.jsonl` | 200 ARC PDA examples (186 correct) |
| `cot_gsm8k_training.jsonl` | 500 GSM8K CoT examples (479 correct) |
| `cot_math_training.jsonl` | 200 MATH CoT examples (163 correct) |
| `cot_arc_training.jsonl` | 200 ARC CoT examples (188 correct) |
| `pda_distillation_colab.ipynb` | Sim 5a training notebook |
| `pda_generalization_colab.ipynb` | Sim 5b training notebook |
| `sim5c_pda_vs_cot_colab.ipynb` | Sim 5c training notebook |
| `generalization_results.json` | Sim 5b raw eval results |
| `sim5c_results.json` | Sim 5c raw eval results |
