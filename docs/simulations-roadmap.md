# Simulations Roadmap: Validating Parallel Deliberation

Status: Planning
Date: 2026-03-27, revised 2026-03-29 (post-survey)
Authors: Toby, Claude

## Motivation

The main risk of running PDA experiments on existing models: you are
operating on the open heart of a trained transformer. Middle layers expect
exactly the input the previous layers produced. If parallel passes are
thrown together, the probability is high that, at first, only garbage comes
out — and then it is unclear whether the IDEA is bad or only the
IMPLEMENTATION.

The simulation route tests the maths in a controlled environment. No
"hopefully nothing breaks" — the system is built so that parallel processing
is the basic assumption, not a hack.

If the maths does not converge or the toy model does not learn, you know
that within days instead of after weeks of debugging on a Llama model.

## Sequence

```
Simulation 1: Mathematical convergence    (days,   no model needed)
     |
     v
Simulation 2: Probing existing models     (days,   existing model,
                                                    no training)
     |
     v
Simulation 3: Toy model from scratch      (weeks,  small training,
                                                    consumer GPU)
     |
     v
[Decision: do PDA experiments on existing  (only here, if Sim 1-3
 models still make sense?]                  positive)
```

## Simulation 1: Mathematical convergence

### What is tested
Purely mathematical, without any neural network: does iterative parallel
processing in orthogonal subspaces converge to a stable result?

Two convergence paradigms compared (revised after the survey):
- **Fixed-point iteration** (original PDA design): repeat until delta < epsilon
- **Energy minimisation** (EBT-inspired): a shared energy function, gradient
  descent to the minimum. Theoretically better scaling (Gladstone et al.,
  EBTs; Du et al., IRED).

Huginn (Geiping et al.) shows: simple iteration scales, root-finding does
not. So we test both and let the data decide.

### Setup
- Random AND semantically loaded vectors in n orthogonal subspaces (numpy/torch)
- Semantically loaded: vectors from pretrained embeddings (GloVe/word2vec),
  projected into orthogonal subspaces. Tests whether convergence also works
  with realistic structure, not only with random vectors.
- Two iteration mechanisms:
  (a) Fixed point: cross-attention-like projections, weighted sums
  (b) Energy: E(x) = sum of pairwise worker distances + regularisation,
      minimised via gradient descent on worker states
- Various merge strategies (average, phase alignment, frequency-selective, sidechain)
- Diversity enforcement: a repulsive force between workers (contrastive
  loss) that prevents all workers from collapsing immediately (Coda-Forno et al.)

### Metrics

Basic:
- Does the system converge? (delta x < epsilon, or energy gradient < epsilon)
- How many iterations/steps until convergence?
- Is the result stable? (small perturbation -> returns?)
- Does convergence depend on the number of subspaces and dimensionality?

Signal-processing diagnostics (the distinctive contribution -- in none of the 75+ papers):
- SNR curve over iterations: monotonically rising? plateau? decline?
- Phase coherence per component (SVD): where do workers agree, where not?
- Crest factor of the divergence: a concentrated point of contention vs. diffuse uncertainty?
- Correlation: which metric tracks convergence quality best?

Meta-comparison:
- Fixed point vs. energy minimisation: which paradigm converges faster, more
  stably, with less sensitivity to hyperparameters?

### Variables
- Number of subspaces: 2, 3, 5, 10, 20
- Dimensionality: 64, 256, 1024, 4096
- Merge strategy: all from the PDA experiment plan
- Convergence mechanism: fixed-point iteration vs. energy minimisation
- Degree of orthogonality: exactly orthogonal, near-orthogonal (angle
  variation), non-orthogonal (control condition)
- Vector type: random (Gaussian) vs. semantically loaded (embedding-based)
- Diversity enforcement: without vs. with contrastive repulsion (InfoNCE-like)

### Expected results
- Exactly orthogonal + average: trivial convergence
- Non-orthogonal + average: destructive interference
- Near-orthogonal: at which angle does it break?
- Signal merges: more robust than averaging under non-orthogonality?
- **NEW**: energy minimisation probably more robust than fixed-point
  iteration, especially at high subspace counts (EBT prediction)
- **NEW**: without diversity enforcement, workers collapse (PLR prediction,
  diversity decays exponentially with iteration depth)
- **NEW**: semantically loaded vectors behave differently from random ones
  (more realistic test condition)

### Effort
One Jupyter notebook. 2-3 days for a basic version + systematic sweeps.

### Decision criterion
Positive: convergence in at least near-orthogonal spaces, with at least one
merge strategy + convergence mechanism, in <20 iterations.
Negative: no stable convergence even with exact orthogonality.
-> If negative: a fundamental problem, further simulations questionable.
Differentiated: energy minimisation works, fixed point does not
-> drop the DEQ route, prefer an EBT-based architecture.

## Simulation 2: Probing existing models

### What is tested
Are the activations of existing transformers actually spread over separable
subspaces? Or is the information so entangled that orthogonal decomposition
destroys it?

### Setup
- An existing open-weights model (e.g. Qwen 2.5 1.5B)
- Extract activations from middle layers (TransformerLens / HuggingFace hooks)
- Project post-hoc into orthogonal subspaces (PCA, SVD, ICA)
- Reconstruct the answer from the individual components

### Metrics
- Explained variance per component: how much information sits in the
  dominant subspaces vs. the rest?
- Reconstruction error: if you keep only k of n components, how much does
  the output suffer?
- Semantic separability: do different components encode recognisably
  different aspects? (e.g. one component for factual knowledge, one for
  tonality, one for logical structure)
- Phase coherence per component across different inputs: do the subspaces
  stay stable or shift per input?

### Variables
- Layer depth: early, middle, late layers
- Decomposition method: PCA, SVD, ICA, sparse dictionary learning
- Number of kept components: 2, 5, 10, 50% of dimensions
- Task type: facts, reasoning, creative

### Expected results
- Hypothesis: middle layers have the most separable representations.
- Hypothesis: reasoning tasks are more strongly distributed across
  components than factual queries.
- If activations are strongly entangled and orthogonal decomposition
  systematically destroys information: the basic assumption of n-PDA is
  questionable. Then one would have to work with non-orthogonal approaches.

### Effort
TransformerLens setup + extraction: 1-2 days.
Analysis: 2-3 days.
Needs a GPU for model inference, but no training.

### Decision criterion
Positive: activations decompose into a few components that are each
semantically interpretable, with an acceptable reconstruction error (<10%).
Negative: decomposition systematically destroys information, no
reconstruction path.

## Simulation 3: Toy model from scratch

### What is tested
Can a small model with built-in parallel deliberation actually learn? Do the
gradients flow? Does orthogonality stay stable under training?

### Setup
A tiny transformer, from scratch:
- 2-4 layers, embedding dimension 128-256
- 1-5M parameters
- 2-4 workers with orthogonal subspaces
- Cross-attention between workers
- Iteration (fixed round count, simple loop, or EBT energy minimisation --
  depending on the result of Simulation 1)
- Trained on simple tasks

### Training tasks (ascending)
1. Copy: reproduce the input unchanged (tests: does the architecture break nothing?)
2. Sort: sort a list of numbers (tests: can the system learn systematic
   transformations?)
3. Simple arithmetic: addition, multiplication (tests: multi-step reasoning)
4. Simple logic: if A and B then C (tests: combining perspectives)

### Metrics
- Training loss curve: does training converge at all?
- Gradient norms: do they explode or vanish?
- Orthogonality over training: do the subspaces stay orthogonal or drift together?
- Comparison with baseline: same parameter count, same task, but a standard
  transformer without deliberation. Does PDA learn faster/better?
- Activation patterns: do different workers actually use different
  representations, or collapse to the same?

### Architecture variants to compare
- n-PDA full: orthogonal spaces + cross-attention + iteration
- Without orthogonality: shared space, but multiple workers + cross-attention
- Without cross-attention: orthogonal spaces, but only merge at the end
- Without iteration: orthogonal + cross-attention, but only one pass
- Standard transformer: same size, no deliberation

This isolates which component contributes how much.

### Loss function (multi-objective, revised after the survey)
- L_task: standard cross-entropy on the task
- L_ortho: PEGO-style dual regularisation:
  L_preserve (preserve base knowledge) + L_diversify (differentiate workers)
  (Hu et al., ECCV 2024 -- designed directly for multi-perspective training)
- L_diversity: contrastive loss (InfoNCE) between worker activations.
  MANDATORY: without explicit enforcement, workers collapse into redundant
  subspaces (Coda-Forno et al., 2025; PLR shows exponential diversity decay)
- L_convergence: penalty for slow convergence (iterations/energy to result)
- L_total = L_task + alpha * L_ortho + beta * L_diversity + gamma * L_convergence
- Curriculum: first L_ortho + L_diversity (build stable subspaces),
  then L_task (learn the task), then L_convergence (optimise efficiency)

### Effort
Implementation: 1-2 weeks (PyTorch, by hand)
Training: hours to days on a consumer GPU (RTX 3060/4060 is enough)
Analysis: 1 week

### Decision criteria
Strongly positive: the toy model learns, orthogonality stays stable, workers
differentiate, the deliberation variant beats the standard baseline.
Weakly positive: the model learns, but no clear advantage over the baseline.
-> Then it may be the task size, not the principle.
Negative: training unstable, orthogonality collapses, workers degenerate.
-> A fundamental architectural problem, not just a scaling question.

## Decision matrix after Simulations 1-3

```
Sim 1 (maths)   Sim 2 (probing)   Sim 3 (toy)     -> Next step
---------------------------------------------------------------------------
Positive        Positive          Positive         -> Train a larger model
                                                       (n-PDA research project)

Positive        Positive          Negative         -> Adjust architecture details
                                                       (loss, regularisation)
                                                       and repeat Sim 3

Positive        Negative          (skipped)        -> Prefer the PDA route
                                                       (existing models, since
                                                       orthogonal decomposition
                                                       does not match reality)

Negative        (skipped)         (skipped)        -> Fundamental problem.
                                                       Parallel deliberation
                                                       does not converge.
                                                       Rethink the approach.
```

## Connection to the LoRA-ensemble approach

If Simulation 2 shows that existing activations are NOT cleanly orthogonally
decomposable, but Simulation 1 shows that the maths WITH orthogonality works,
then the LoRA-ensemble approach is an interesting middle path:

LoRA adapters can be trained with an orthogonality regulariser, so that
different LoRAs operate in different subspaces of the model. That would be
"n-PDA properties retrofitted" onto an existing base model — easier than
from scratch, more robust than raw steering vectors.

## Revisions after the literature survey (2026-03-29)

Based on 75+ papers (three independent agent surveys, consolidated in
[research-synthesis.md](research-synthesis.md)):

### Architecture revision
- **DEQ → EBT/iteration**: root-finding (Broyden/Anderson) does not scale
  beyond 250M parameters (Huginn result). Use simple iteration
  (Huginn-style) or energy minimisation (EBT-style, Gladstone et al.)
  instead. Simulation 1 compares both paradigms directly.
- **Backbone recommendation**: Huginn-style (prelude → recurrent block →
  coda) with MoEUT routing in the recurrent block.

### Orthogonality and diversity
- **Orthogonality necessary, not sufficient**: structural orthogonality
  alone produces no meaningful semantic diversity (Zhang et al., Sep 2025).
- **Diversity enforcement is mandatory**: without contrastive losses
  (InfoNCE or similar), latent vectors collapse into redundant subspaces
  (Coda-Forno et al., Oct 2025). PLR (Tang et al., Jan 2026) confirms:
  diversity decays exponentially with depth.
- **Recommendation**: PEGO-style dual regularisation + PLR-style contrastive
  losses. Evaluate Conceptors (Abreu et al., NeurIPS 2025) for perspective
  assignment.

### The distinctive contribution
The signal-processing metrics (SNR, phase coherence, crest factor) as
convergence diagnostics appear in NONE of the 75+ papers. "Language Through
a Prism" (Tamkin et al.) comes closest (DCT on activations), but uses
spectral analysis only analytically, not as active control. This is the
genuinely new contribution the simulations focus on.

## Related documents

- [concept-pda.md](concept-pda.md) — PDA: pragmatic approach with existing models
- [concept-n-pda.md](concept-n-pda.md) — n-PDA: theoretical greenfield concept
- [research-synthesis.md](research-synthesis.md) — research synthesis
