# Research Synthesis: Parallel Deliberation

Consolidated from three independent research agents, 75+ papers.
Date: 2026-03-29

## Headline result

PDA's combination of all five elements (parallel workers, same input,
different perspectives in vector space, iterative deliberation, convergence
to consensus) exists in no single paper. But each element on its own is
validated, and several papers implement 3-4 of the 5.

## The 10 most important papers (by relevance to PDA)

### Tier 1: Direct architectural precedents

1. **Mixture of Thoughts (MoT)** (Fein-Ashley et al., Sep 2025)
   - Multi-expert cross-attention in a shared latent space
   - Closest to PDA. Missing: iterative convergence.
   - https://arxiv.org/abs/2509.21164

2. **Chain-of-Experts (CoE)** (2025)
   - Iterative expert communication with step-specific routing
   - PDA = "parallel CoE with convergence"
   - https://arxiv.org/html/2506.18945v1

3. **Parallel Latent Reasoning (PLR)** (Tang et al., Jan 2026)
   - Parallel latent streams via trigger tokens, contrastive diversity
   - Proof: diversity decays exponentially with depth → width scaling
   - https://arxiv.org/abs/2601.03153

4. **LatentMAS** (Zou et al., Nov 2025)
   - Multi-agent latent collaboration via KV-cache concatenation
   - 14.6% improvement, 4x faster than text-based
   - https://arxiv.org/abs/2511.20639

### Tier 2: Convergence mechanisms

5. **Energy-Based Transformers (EBTs)** (Gladstone et al., Jul 2025)
   - Generalise DEQs with better scaling behaviour (35% faster)
   - Consensus = shared energy minimum
   - ICLR 2026 submission: https://arxiv.org/abs/2507.02092

6. **IRED: Learning Iterative Reasoning through Energy Diffusion** (Du et al., ICML 2024)
   - Reasoning as energy minimisation with annealed landscapes
   - Adaptive computation, generalises to harder instances
   - https://arxiv.org/abs/2406.11179

7. **Huginn** (Geiping et al., Feb 2025, NeurIPS Spotlight)
   - 3.5B recurrent-depth transformer, scales to 132 effective layers
   - Proves: simple iteration scales, root-finding does not
   - https://arxiv.org/abs/2502.05171

### Tier 3: Perspective mechanisms

8. **Conceptors for Compositional Steering** (Abreu et al., NeurIPS 2025)
   - Boolean algebra (AND/OR/NOT) over steering matrices
   - Better than linear vector combination for perspective composition
   - https://openreview.net/forum?id=0Yu0eNdHyV

9. **PEGO: Orthogonal LoRA Groups** (Hu et al., ECCV 2024)
   - Dual regularisation: L_preserve (base) + L_diversify (workers)
   - Directly applicable to PDA's multi-perspective training
   - https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06773.pdf

### Tier 4: Critical negative results

10. **Orthogonality != semantics** (Zhang et al., Sep 2025)
    + **Diversity collapse** (Coda-Forno et al., Oct 2025)
    - Structural orthogonality alone produces no meaningful diversity
    - Without explicit enforcement, latent vectors collapse
    - PDA NEEDS contrastive losses + semantic anchoring

## Further relevant papers (by category)

### Latent reasoning
- **Coconut** (Meta, ICLR 2025): continuous thought, latent BFS
- **Latent-SFT**: 3-4 parallel paths already encoded in latent states
- **LatentSeek**: test-time policy gradient in activation space
- **Token Assorted**: hybrid latent/text tokens
- **Diffusion of Thoughts (DoT)**: diffusion as iterative reasoning refinement
- **LaDiR**: latent diffusion for text reasoning (+11.7 acc from 5→10 steps)

### DEQ and fixed points
- **RevDEQ**: exact gradients via a reversible solver
- **pcDEQ**: guaranteed convergence via Perron-Frobenius
- **Ouro-LoopLM**: a 1.4B looped LM matches 12B standard LLMs
- **Fixed-Point RNNs**: expressive models as fixed points of simple parallel models

### Orthogonal representations
- **NDM**: unsupervised subspace decomposition via orthogonal rotation
- **Transformer normalisation**: pre-norm requires orthogonal subspaces
- **Linear Representation Hypothesis**: causal inner product as the right metric
- **LLMs Encode Semantics in Low-D Subspaces**: empirical confirmation

### LoRA
- **OSRM**: orthogonal subspace projection for interference-free merging
- **SMoRA**: each rank as an expert, simplifies PDA implementation
- **LoRA-Ensemble**: implicit deep ensembles via multiple LoRAs
- **OPLoRA, CLoRA, LoraHub, MoLE, S-LoRA**: further merge/routing methods

### Signal processing in ML
- **Language Through a Prism**: DCT spectral filter on activations (Tamkin et al.)
- **SEA**: SVD-based spectral editing for alignment
- **Fourier Features**: LLMs use a Fourier representation for arithmetic
- **CKA**: Centered Kernel Alignment as a convergence metric

### SSM/Mamba
- **Mamba-3 MIMO**: multi-stream SSMs, no decode-latency overhead
- **MH-SSM / Stateformer**: parallel SSM heads with different dynamics
- **MvSSM**: multi-view SSM with cross-view interaction

### Hypernetworks
- **Attention as Hypernetwork**: multi-head attention IS already a hypernetwork
- **Hyper-CL**: perspective-specific subspace projection via hypernetwork
- **Text-to-LoRA / Zhyper**: context-aware LoRA generation
- **HyperMoE**: cross-expert knowledge transfer via hypernetwork

### Other
- **PDT / Dynamic Notes Bus**: shared latent workspace for parallel streams
- **Consensus Game**: game-theoretic equilibrium as convergence
- **DIFFormer**: attention IS energy minimisation (theoretically)
- **RoE**: existing models already contain latent dynamic experts

## Key takeaways for PDA design

### What is validated
- Latent reasoning works and beats token-level (Coconut, Huginn)
- LLMs already perform implicit parallel reasoning
- Semantics is encoded linearly in orthogonal subspaces
- Multi-agent latent collaboration brings measurable benefits
- Width scaling (parallel streams) complements depth scaling

### What needs revising
- **DEQs → EBTs**: root-finding does not scale. Use simple iteration or
  energy minimisation instead of Broyden/Anderson.
- **Orthogonality is necessary but not sufficient**: also needs contrastive
  losses and semantic anchoring.
- **Diversity must be explicitly enforced**: without InfoNCE or similar,
  workers collapse into redundant subspaces.

### The distinctive contribution here
The signal-processing metrics (SNR, phase coherence, crest factor) as
convergence diagnostics appear in none of the 75+ papers. "Language Through
a Prism" (DCT on activations) comes closest, but uses spectral analysis only
analytically, not as active control. PDA's combination of audio-DSP concepts
as a merge strategy and halting criterion is genuinely new.

## Recommended architecture (based on the survey)

```
Backbone: Huginn-style (prelude → recurrent block → coda)
          with MoEUT routing in the recurrent block

Perspectives: PEGO-style dual orthogonal regularisation
              + Conceptor-based perspective assignment
              + PLR-style contrastive losses (InfoNCE)

Deliberation space: Dynamic Notes Bus (shared latent workspace)
                    workers read/write via cross-attention

Convergence: EBT energy minimisation instead of DEQ root-finding
             Diagnostics: SNR, phase coherence, crest factor

Diversity: contrastive losses + repulsive ensemble forces
```

## Open research directions

### PDA for training (idea, 2026-04-08)

Hypothesis: PDA principles could apply to training, not just inference.
Instead of a single forward/backward pass: several parallel workers with
different "learning styles" (robustness, generalisation, efficiency)
deliberate over the best gradient update.

Related approaches:
- Population-Based Training (DeepMind 2017): parallel runs, different hyperparams
- Evolutionary Strategies (OpenAI 2017): perturbed parameters instead of gradients
- Lookahead Optimizer: simulate multiple steps ahead

PDA-specific twist: workers have not random perturbations but semantically
different perspectives on the training data. Deliberation over the consensus
gradient.

Open questions:
- Compute overhead: N parallel passes × cost per pass. Worth it if
  convergence is faster?
- Testability: could this be tested with a small toy model on a toy task?
  E.g. 3 workers training a 1M-param model on a simple task.
- Connection to Sim 1: energy minimisation converges at n=3-5. Does that
  hold for training dynamics too?

Motivation: compute sovereignty. More efficient training methods as a
counterweight to compute concentration at frontier labs. If PDA training
converges faster, the same result needs fewer GPU-hours.

Status: idea, no experiment. Earliest Sim 5+.

## Related documents

- [concept-pda.md](concept-pda.md) — PDA v2 (pragmatic approach)
- [concept-n-pda.md](concept-n-pda.md) — n-PDA (native architecture)
- [simulations-roadmap.md](simulations-roadmap.md) — simulations roadmap (revised 2026-03-29)
- [research-notes/research-agent-1.md](research-notes/research-agent-1.md) — research agent 1
- [research-notes/research-agent-2.md](research-notes/research-agent-2.md) — research agent 2
- [research-notes/research-agent-3.md](research-notes/research-agent-3.md) — research agent 3
