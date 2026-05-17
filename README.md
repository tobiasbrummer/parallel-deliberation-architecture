# Parallel Deliberation Architecture (PDA)

An exploration of whether language-model reasoning improves when multiple
"workers" deliberate over the same input from different perspectives and
converge to a consensus — instead of a single forward pass.

This repository is a **research log**: hypotheses, a literature survey, a
staged simulation roadmap, and the experiments that tested it. Results are
exploratory (mostly single-run) and reported as such.

## The idea

A normal transformer answers in one sequential pass. PDA asks: what if
reasoning were a *parallel deliberation* instead — several workers process
the same problem through different lenses (e.g. methodical, creative,
skeptical), exchange intermediate results, and iterate until they converge?

Two tracks are explored:

- **PDA (pragmatic)** — apply the idea to existing models, first at the
  prompt level, then by distilling the behaviour into model weights.
  See [docs/concept-pda.md](docs/concept-pda.md).
- **n-PDA (native)** — a from-scratch architecture sketch: orthogonal
  perspective subspaces, cross-attention as the deliberation channel,
  iteration to a fixed point. A thought experiment about the mathematical
  properties parallel deliberation would need.
  See [docs/concept-n-pda.md](docs/concept-n-pda.md).

The work was staged deliberately: validate the maths in a controlled
sandbox before operating on a trained model. See
[docs/simulations-roadmap.md](docs/simulations-roadmap.md).

## Results so far

### Prompt-level PDA improves reasoning accuracy (Simulation 4)

Qwen3-8B on 200 GSM8K questions. PDA = 3 workers (methodical / creative /
skeptical) plus one reasoning-quality merge.

| Condition  | Accuracy | Correct |
|------------|----------|---------|
| Baseline   | 96.0%    | 192/200 |
| PDA (n=3)  | 97.5%    | 195/200 |

PDA corrected 3 of 8 baseline errors (37.5% recovery) and produced **zero
regressions** — it never turned a correct answer wrong. Cost: ~4x tokens.
→ [simulation-4-gsm8k/RESULTS.md](simulation-4-gsm8k/RESULTS.md)

### The reasoning strategy distils into a small model (Simulation 5)

The more interesting result: the multi-perspective behaviour can be
distilled into a small model's weights, so it costs nothing extra at
inference. QLoRA on Qwen3-1.7B, trained on ~830 PDA traces from a
Qwen3-8B teacher (GSM8K + MATH + ARC-Challenge). Total data + training
cost: **~$4**.

| Benchmark      | Base   | PDA-distilled | Delta     |
|----------------|--------|---------------|-----------|
| GSM8K          | 15.5%  | 60.0%         | +44.5pp   |
| ARC-Challenge  | 0.5%   | 72.5%         | +72.0pp   |
| MATH           | 0.0%   | 4.5%          | +4.5pp    |
| **Average**    | **5.3%** | **45.7%**   | **+40.3pp** |

The largest gain is on ARC-Challenge — a *different domain* from most of
the training data — which suggests the model learned a transferable
reasoning methodology rather than task-specific answers.
→ [simulation-5-training/RESULTS.md](simulation-5-training/RESULTS.md)

### Honest limitations

- Results are single-run; no multi-seed significance testing yet.
- Sim 4's baseline (96%) is already high — GSM8K is near-saturated for
  this model, so the headroom is small.
- MATH stays low: a 1.7B model lacks the capacity for competition maths.
- The n-PDA architecture is a concept sketch, not an implementation.

## What may be new here

The literature survey (75+ papers, [docs/research-synthesis.md](docs/research-synthesis.md))
found every individual ingredient of PDA validated somewhere, but not the
combination. The one element absent from all surveyed work: using
**signal-processing metrics** — SNR, phase coherence, crest factor — as
convergence diagnostics for the deliberation. That framing comes from an
audio-engineering background and is the part most worth pursuing.

## Repository map

| Path | What |
|------|------|
| `docs/concept-pda.md` | The architecture concept (pragmatic track) |
| `docs/concept-n-pda.md` | The native from-scratch architecture sketch |
| `docs/simulations-roadmap.md` | Staged validation plan, with decision criteria |
| `docs/research-synthesis.md` | Literature survey, 75+ papers |
| `docs/experiments-spec.md` | Experimental design |
| `docs/research-notes/` | Raw research-process artifacts (kept as-is) |
| `simulation-1/` | Mathematical convergence (no model) |
| `simulation-2/` | Probing existing-model activations |
| `simulation-3/`, `simulation-3b/` | Toy model + prompt-PDA |
| `simulation-4-gsm8k/` | Prompt-PDA benchmark (GSM8K, MMLU-Pro) |
| `simulation-5-training/` | Distillation into a small model |
| `simulation-6-sweep/` | Perspective-count sweep |

Each `simulation-*` directory contains its plan, code, and a results file.

## Status

Active exploratory research. The prompt-level and distillation results are
encouraging enough to justify the next steps: multi-seed runs for
significance, harder benchmarks, and a per-perspective LoRA-MoE router
(Simulation 6).
