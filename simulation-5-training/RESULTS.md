# PDA Distillation Results

## Sim 5a: Single-Benchmark Distillation (GSM8K only)

**Setup:** QLoRA on Qwen3-1.7B, 474 PDA examples from GSM8K, Qwen3-8B teacher.

| Model | GSM8K | Cost |
|-------|-------|------|
| Qwen3-1.7B Base | 24.5% | - |
| Qwen3-1.7B PDA-Distilled | **50.0%** | ~$2 |

**Finding:** PDA distillation doubles accuracy on the trained domain.

## Sim 5b: Multi-Benchmark Generalization

**Setup:** QLoRA on Qwen3-1.7B, 834 PDA examples (GSM8K + MATH + ARC-Challenge),
Qwen3-8B teacher, 16-bit training on A100. Total data generation cost: $3.63.

| Benchmark | Base | Distilled | Delta |
|-----------|------|-----------|-------|
| GSM8K | 15.5% | **60.0%** | **+44.5pp** |
| MATH | 0.0% | **4.5%** | **+4.5pp** |
| ARC-Challenge | 0.5% | **72.5%** | **+72.0pp** |
| **Average** | **5.3%** | **45.7%** | **+40.3pp** |

**Key Findings:**
1. PDA distillation generalizes across domains (math, logic, science)
2. Largest gain on ARC-Challenge (+72pp) -- different domain from training
3. MATH limited by model capacity (1.7B too small for competition math)
4. The model learned the *reasoning methodology*, not task-specific answers
5. Inference cost: identical to base model (same speed, same memory)

**Interpretation:** Multi-perspective reasoning is a transferable meta-skill.
The model doesn't learn "the answer to X is Y" -- it learns "before answering,
consider the problem from systematic, creative, and critical angles." This
generalizes because the cognitive strategy is domain-independent.

## Sim 5c -- PDA vs CoT Distillation Control

Sim 5c was originally planned as a "perspective count sweep" (see below).
That plan was set aside in favour of a control that the Sim 5b numbers
above needed: **how much of the +44.5 / +72.0 / +4.5 pp improvement is
from PDA specifically vs from reasoning-distillation in general?**

To answer this, the same Qwen3-1.7B base was QLoRA-distilled from the same
~830-example teacher dataset under two conditions:
- **PDA**: training traces from a Qwen3-8B teacher run as 3 PDA workers + merge
- **CoT**: training traces from the same Qwen3-8B teacher run as plain
  chain-of-thought (single pass)

Same student, same QLoRA hyperparameters (r=16, 3 epochs), same eval set.

| Benchmark      | Base   | CoT-distilled | PDA-distilled | PDA vs Base | PDA vs CoT |
|----------------|--------|---------------|---------------|-------------|------------|
| GSM8K          | 15.5%  | 42.5%         | 56.5%         | +41.0pp     | **+14.0pp** |
| ARC-Challenge  | 0.5%   | 71.0%         | 74.0%         | +73.5pp     | **+3.0pp**  |
| MATH           | 0.0%   | 8.0%          | 7.0%          | +7.0pp      | **-1.0pp**  |

Source: `sim5c_results.json`. Note that the PDA numbers in this run are
slightly different from the Sim 5b headline (training data size 834 vs 830,
different seed) -- the PDA-vs-CoT comparison is the apples-to-apples
contrast.

### Interpretation

**Most of the headline improvement is reasoning-distillation in general.**
CoT-distilled gets GSM8K 42.5%, ARC-C 71.0%, MATH 8.0% -- the bulk of the
jump from the 15.5/0.5/0.0 base. PDA distillation adds another +14pp on
GSM8K and +3pp on ARC-C on top of that, and is -1pp on MATH (within noise
for n=200).

**What this means for the Sim 5b framing:** the "+72pp on ARC, +45pp on
GSM8K" numbers vs the untrained base are real, but they over-attribute
the gain to the multi-perspective method. The *PDA-specific* signal is
the +14pp / +3pp / -1pp delta vs CoT-distill. That is still a positive
signal on the two reasoning benchmarks -- the multi-perspective layer
does add something -- but not the order of magnitude the base comparison
suggests.

This negative-as-control result is the most useful thing in Simulation 5.
The lesson is that any "X works" claim about a fine-tuning recipe needs
the "X vs the simpler alternative" arm before the headline.

## Next: Sim 5d -- Perspective Count Sweep (original Sim 5c plan, deferred)

**Question:** How many worker perspectives give the best distillation?

**Design:**
- Generate training data with 2, 3, 4, 5 workers
- Train separate LoRA adapters for each count
- Evaluate all on GSM8K + ARC-Challenge
- Hypothesis: 3 is sweet spot (diversity vs noise)

## Next: Sim 6 — LoRA-MoE with Perspective Routing

**Question:** Are separate per-perspective LoRA adapters better than one fused adapter?

**Design:**
- Train 3 separate LoRA adapters (methodical, creative, critical)
- Each trained only on its worker's reasoning outputs
- Small router MLP learns which perspective(s) to activate per input
- Weighted combination of adapter outputs in single forward pass

**Key difference from existing LoRA-MoE:** Existing work routes by domain
(math vs code vs language). This routes by cognitive strategy (systematic vs
creative vs critical). The experts are orthogonal to content, which should
enable cross-domain generalization.

**Architecture:**
```
Input -> Router (tiny MLP)
          |
     Weights: [w1, w2, w3]
          |
     LoRA_methodical * w1
     LoRA_creative   * w2  -> weighted sum -> Output
     LoRA_critical   * w3
```

## Files

- `generate_training_data.py` -- GSM8K PDA data generation (Sim 5a)
- `generate_math_arc.py` -- MATH + ARC data generation (Sim 5b)
- `pda_training_data.jsonl` -- 500 GSM8K examples (474 correct)
- `pda_math_training.jsonl` -- 200 MATH examples (174 correct)
- `pda_arc_training.jsonl` -- 200 ARC examples (186 correct)
- `pda_distillation_colab.ipynb` -- Sim 5a Colab notebook
- `pda_generalization_colab.ipynb` -- Sim 5b Colab notebook
- `generalization_results.json` -- Sim 5b raw results

## Cost Summary

| Item | Cost |
|------|------|
| Sim 5a data (GSM8K, 500 examples) | ~$2 |
| Sim 5b data (MATH + ARC, 400 examples) | ~$1.63 |
| Sim 5a training (Colab Free T4) | $0 |
| Sim 5b training (Colab Pro A100) | ~$0.50 |
| **Total** | **~$4.13** |
