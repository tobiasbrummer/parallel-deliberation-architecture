# Native Parallel Deliberation Architecture (n-PDA)

Status: Theoretical greenfield concept (foundational research)
Date: 2026-03-27
Authors: Toby, Claude, Gemini

## Core idea

Instead of forcing existing transformer models into a parallel deliberation
mode through inference-time interventions (which often causes representation
breakage), the architecture is designed from the ground up for parallel
vector-space reasoning. The system is natively trained to build perspectives
in isolation and iteratively merge them into a consensus.

It grew out of the question: if you started from zero, how would the
architecture have to look for parallel deliberation to work cleanly in a
mathematical sense?

## Relationship to pragmatic PDA

n-PDA is not a replacement for PDA, but a thought experiment that shows which
mathematical properties parallel deliberation needs:

- Orthogonality (no destructive interference)
- Differentiable exchange (not a rigid merge)
- Endogenous convergence (no external halting mechanism)

Insights from n-PDA feed back into the PDA experiments — e.g. orthogonal
projection as merge preprocessing. Conversely, PDA results can show which
n-PDA assumptions hold up empirically.

## The three architectural pillars

### 1. Orthogonal perspective subspaces (the foundation)

Workers do not share the same dense latent space. Different perspectives are
forced into mathematically orthogonal (right-angled) subspaces through
regularisation during training.

Advantage: no destructive interference when merging. Workers cannot
accidentally overwrite or cancel each other out.

Caveat: orthogonality in high-dimensional spaces is cheap — in a
4096-dimensional space, thousands of near-orthogonal vectors fit. The
question is not whether the subspaces *can* be orthogonal, but whether
orthogonal subspaces also encode semantically meaningful perspectives. That
is an empirical question, testable via probing experiments (see the
simulations roadmap).

### 2. Native cross-attention (the deliberation space)

The exchange is not a rigid mean-merge at the end of a layer. Instead,
workers use continuous cross-attention between their orthogonal spaces.

Worker A "reads" differentiably into the representations of worker B and
pulls exactly the features that productively complement its own perspective.

### 3. Deep Equilibrium motor (the iteration)

Iteration and the halting criterion are not controlled externally. The
network is formulated as a Deep Equilibrium Model (DEQ).

The forward pass is an endogenous loop that iterates until the system
reaches a mathematical fixed point (delta x < epsilon). Consensus is thereby
intrinsically built into the forward pass.

Advantage: no external halting, no arbitrary thresholds.
Disadvantage: DEQs are notoriously hard to train (unstable gradients).

## Architecture sketch

```
Input (text)
    |
[Embedding & orthogonal projection into n subspaces]
    |
    +---> [ DEQ loop start ] <----------------------+
    |                                               |
    |   [Worker 1] <--- cross-attention ---> [Worker n]
    |   (subspace 1)                         (subspace n)
    |                                               |
    +---> [ Convergence check: divergence < e ? ] --+
                  |                      (No: next iteration)
                (Yes)
                  |
[ Consolidated vector space ]
                  |
[ Unembedding layer ]
                  |
Output (text)
```

## Open research questions

### Training stability
DEQs are hard to train. Combining them with orthogonal regularisation for
the subspaces compounds the challenge. Possible mitigation: Jacobian-free
backpropagation, phantom gradients.

### Loss function
How do you formulate a training signal that simultaneously rewards:
- task accuracy (language understanding/generation)
- orthogonality of the perspective spaces
- fast DEQ convergence?

Approach: a weighted multi-objective loss, or curriculum training
(orthogonality first, then task, then convergence speed).

### Scaling
Can this architecture be scaled to the size of today's LLMs? DEQ models have
so far only been successful in relatively small variants.

## Possible third pillar: LoRA ensemble as learned perspectives

Between PDA (steering vectors, hand-crafted) and n-PDA (orthogonal spaces,
trained from scratch) there is a middle path:

Train several LoRA adapters on the same base model, with different
objectives or data. Each LoRA *is* a learned perspective — not hand-built,
but also not as expensive as a whole model from scratch.

Advantages:
- Immediately feasible with existing tools (PEFT, HuggingFace)
- Perspectives are learned rather than hand-crafted
- Existing research on LoRA merging (TIES, DARE, SLERP) directly applicable
- Two merge levels testable: weight merge (combine LoRAs, then forward)
  vs. activation merge (multiple forward passes with different LoRAs)

This could serve as a pragmatic entry point into learned perspectives,
before n-PDA training is even feasible.

## Related documents

- [concept-pda.md](concept-pda.md) — Parallel Deliberation Architecture: the pragmatic approach with existing models
- [simulations-roadmap.md](simulations-roadmap.md) — staged plan for validating the mathematical foundations
