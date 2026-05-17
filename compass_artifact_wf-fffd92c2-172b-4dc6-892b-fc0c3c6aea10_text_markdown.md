# Systematic Literature Review: Parallel Deliberation Architecture (PDA)

**PDA's core concept — multiple workers processing the same input in parallel from different perspectives in representation space, iterating toward convergence — has no single direct predecessor, but a rapidly converging body of work validates every major component.** The most critical finding of this review is that 2024–2026 has seen an explosion of latent-space reasoning and multi-agent latent collaboration research that collectively builds most of PDA's foundation. Mixture of Thoughts (MoT), LatentMAS, Parallel Latent Reasoning (PLR), and Chain-of-Experts (CoE) each implement substantial subsets of PDA's vision. However, **no existing work combines all five elements**: (a) multiple workers, (b) same input, (c) different perspectives in vector space, (d) iterative deliberation, and (e) convergence to consensus. This gap represents PDA's novel contribution. The review covers 75+ papers across three search directions, identifying 10 high-priority works and concrete architectural recommendations.

---

## Search Direction 1: Directly related work

### Mixture of Thoughts (MoT): Learning to Aggregate What Experts Think, Not Just What They Say (Fein-Ashley et al., Sep 2025)
- **Source**: https://arxiv.org/abs/2509.21164
- **Core idea**: MoT enables latent-level collaboration among heterogeneous LLM experts. A lightweight router selects top-K experts; interaction layers project hidden states into a shared latent space where the primary expert performs cross-attention over its active peers. Pre-trained experts stay frozen; only the router and interaction layers are trained.
- **Relevance for PDA**: Architecture + Merge — the closest existing work to PDA's core concept. Multiple models process the same input from different perspectives and merge hidden states in a shared latent space via cross-attention, not at the token level.
- **What PDA can adopt**: Cross-attention merge mechanism in shared latent space; router-based expert selection; joint training objective for selection and collaboration.
- **Limitations / Differences**: Uses heterogeneous pre-trained models rather than workers from a single model. **No iterative deliberation/convergence** — single-pass only. PDA's iterative refinement is entirely missing.
- **Follow up**: yes — most critical prior work; must be thoroughly analyzed and differentiated from PDA.

### LatentMAS: Latent Collaboration in Multi-Agent Systems (Zou et al., Nov 2025)
- **Source**: https://arxiv.org/abs/2511.20639
- **Core idea**: Training-free framework enabling LLM agents to reason and communicate entirely in continuous latent space via shared latent working memory. Each agent performs autoregressive latent thought generation through last-layer hidden embeddings. KV-caches are concatenated to preserve and transfer each agent's representations losslessly. Achieves **14.6% average improvement** over single-model baselines and **4× faster inference** vs. text-based multi-agent systems.
- **Relevance for PDA**: Architecture + Merge + Convergence — demonstrates multiple agents collaborating in latent space with lossless information exchange.
- **What PDA can adopt**: Latent working memory concept; KV-cache concatenation as merge mechanism; input-output distribution alignment (realignment matrix W_a); training-free latent-space reasoning.
- **Limitations / Differences**: Sequential/hierarchical agent interaction, not truly parallel simultaneous processing. Agents take turns; no iterative convergence loop.
- **Follow up**: yes — directly relevant architecture with code available on GitHub.

### Interlat: Enabling Agents to Communicate Entirely in Latent Space (Du et al., Nov 2025)
- **Source**: https://arxiv.org/abs/2511.09149
- **Core idea**: LLM agents transmit last hidden states directly as latent communication, bypassing the discrete token bottleneck. Includes a learned compression process. Information bandwidth of hidden states (~40k bits/hidden-state) vastly exceeds tokens (~15 bits/token). Works across heterogeneous models. Achieves **24× inference speedup** via compression while maintaining performance.
- **Relevance for PDA**: Architecture + Merge — directly demonstrates inter-agent communication in activation space.
- **What PDA can adopt**: Communication adapter design; compression of latent messages; the "Off-Manifold" problem identification and solutions for cross-model latent communication.
- **Limitations / Differences**: Two-agent sequential setup, not parallel multi-worker deliberation. No convergence mechanism.
- **Follow up**: yes — critical for understanding latent-space communication mechanics.

### Parallel Latent Reasoning (PLR) for Sequential Recommendation (Tang et al., Jan 2026)
- **Source**: https://arxiv.org/html/2601.03153
- **Core idea**: Pioneers **width-level computational scaling** by exploring multiple diverse reasoning trajectories simultaneously. PLR constructs parallel reasoning streams through learnable trigger tokens in continuous latent space, preserves diversity via global reasoning regularization (contrastive/InfoNCE losses), and adaptively synthesizes multi-stream outputs through gating networks. Includes theoretical proof that **diversity decays exponentially with depth**, motivating width-level scaling.
- **Relevance for PDA**: Architecture + Merge + Convergence — architecturally very close to PDA. Multiple parallel streams in latent space with diversity enforcement and adaptive aggregation.
- **What PDA can adopt**: Learnable trigger tokens for spawning streams; contrastive losses for diversity; gating networks for adaptive aggregation; the theoretical framework for width vs. depth scaling.
- **Limitations / Differences**: Applied to recommendation domain, not general language modeling. No iterative convergence — streams are synthesized in one pass. No inter-stream communication during reasoning.
- **Follow up**: yes — width-level scaling theory directly supports PDA's approach.

### Parallel Decoder Transformer (PDT): Planner-Seeded Latent Coordination (Dec 2025)
- **Source**: https://arxiv.org/html/2512.10054
- **Core idea**: Augments a frozen decoder with a planner-seeded latent workspace ("Dynamic Notes Bus") and synchronized multi-stream output protocol. Parallel streams decode against this workspace through Speculative Note Conditioning, exchanging latent summaries at synchronized block boundaries.
- **Relevance for PDA**: Architecture + Convergence — very close to PDA's concept of a shared deliberation space. Multiple parallel streams read/write to a shared latent workspace, with agreement logic.
- **What PDA can adopt**: Dynamic Notes Bus (shared latent workspace); Speculative Note Conditioning; block-boundary synchronization and agreement logic; planner-seeded initialization.
- **Limitations / Differences**: Focused on parallel generation (output coordination), not reasoning. Single-pass with synchronized blocks, not iterative.
- **Follow up**: yes — closest architectural precedent for PDA's shared workspace concept.

### Parallel Test-Time Scaling for Latent Reasoning Models (You et al., Oct 2025)
- **Source**: https://arxiv.org/abs/2510.07745
- **Core idea**: Enables parallel test-time scaling for latent reasoning models via two sampling strategies (MC Dropout for epistemic uncertainty, Additive Gaussian Noise for aleatoric uncertainty) to create diverse latent trajectories. A Latent Reward Model (LatentRM) trained with step-wise contrastive objectives evaluates trajectory quality.
- **Relevance for PDA**: Convergence + Alternative — directly addresses creating and aggregating parallel reasoning paths in continuous latent space.
- **What PDA can adopt**: MC Dropout and Gaussian noise for diversity; LatentRM for trajectory evaluation; step-wise contrastive training for reward models operating on latent states.
- **Limitations / Differences**: Post-hoc aggregation (best-of-N selection), not iterative deliberation. Paths are independent — no cross-path communication.
- **Follow up**: yes — key for understanding aggregation mechanisms in latent space.

### Deliberation in Latent Space via Differentiable Cache Augmentation (Liu et al., Dec 2024)
- **Source**: https://arxiv.org/abs/2412.17747 (ICML 2025)
- **Core idea**: Augments a frozen LLM with a coprocessor that operates on the model's KV-cache, injecting learned latent embeddings. The coprocessor "deliberates" in latent space and distills additional computation into the cache. Achieves 10% improvement on GSM8K.
- **Relevance for PDA**: Architecture + Enabling — demonstrates a modular deliberation module communicating with a base LLM through latent embeddings. The coprocessor concept is analogous to a PDA worker.
- **What PDA can adopt**: KV-cache augmentation as communication channel; offline/asynchronous coprocessor design; enriching context through latent embeddings.
- **Limitations / Differences**: Single coprocessor, not multiple parallel ones. No multi-perspective reasoning.
- **Follow up**: yes — core architectural concept for PDA-style communication.

### Exploring System 1 and 2 Communication for Latent Reasoning in LLMs (Coda-Forno et al., Oct 2025)
- **Source**: https://arxiv.org/pdf/2510.00494
- **Core idea**: Revisits the KV-cache coprocessor design. **Critical negative finding: without explicit diversity constraints, latents fail to decorrelate or specialize.** Cross-capture analysis shows latent vectors occupy redundant subspaces.
- **Relevance for PDA**: Architecture + Convergence — directly motivates PDA's multi-perspective design by showing that diversity must be explicitly enforced.
- **What PDA can adopt**: Diagnostic tools (cross-capture heatmaps, PCA analysis); the critical insight that **diversity must be explicitly enforced**; co-finetuning strategies.
- **Limitations / Differences**: Dual-model (System 1 + 2), not multi-worker. Limited gains, suggesting the bottleneck is in latent communication design.
- **Follow up**: yes — essential negative results informing PDA design.

### Coconut: Chain of Continuous Thought (Hao et al., Meta, Dec 2024)
- **Source**: https://arxiv.org/abs/2412.06769 (ICLR 2025)
- **Core idea**: Uses the LLM's last hidden state as a "continuous thought" fed back as input embedding. Enables BFS-like parallel exploration where continuous thoughts encode multiple alternative next steps simultaneously. Outperforms CoT on logical tasks.
- **Relevance for PDA**: Enabling — foundational work proving latent-space reasoning works and can encode multiple paths.
- **What PDA can adopt**: Continuous thought representation; multi-stage curriculum training; the insight that latent representations naturally encode multiple reasoning paths.
- **Limitations / Differences**: Sequential (not parallel workers); single model; no explicit multi-perspective decomposition.
- **Follow up**: yes — foundational reference for PDA.

### Latent-SFT: Latent Reasoning as Vocabulary-Space Superposition (Deng et al., Oct 2025)
- **Source**: https://openreview.net/forum?id=ciiKoeM206
- **Core idea**: Restricts latent space to the column space of the LLM vocabulary, treating latent reasoning as a superposition over vocabulary probabilities. Each latent step carries a distribution over tokens representing simultaneous support for multiple reasoning chains. The solution "collapses" to an explicit sequence at the end. Defines metrics: **Effective Compression Rate** and **Effective Global Parallelism**.
- **Relevance for PDA**: Architecture + Convergence — the "superposition → collapse" metaphor is directly analogous to PDA's convergence mechanism.
- **What PDA can adopt**: Superposition-as-reasoning framework; collapse/convergence metaphor; ECR and EGP metrics for measuring latent parallelism.
- **Limitations / Differences**: Single model with implicit parallelism, not explicit multi-worker architecture. No inter-worker communication.
- **Follow up**: yes — provides formal metrics for latent parallelism.

### Distributional Reasoning in LLMs: Parallel Reasoning Processes in Multi-hop Reasoning (Shalev et al., Jun 2024)
- **Source**: https://arxiv.org/abs/2406.13858
- **Core idea**: Discovers that LLMs **naturally encode multiple parallel reasoning paths** in their hidden representations during multi-hop reasoning. Middle layers generate embeddings representing a set of potential intermediate answers simultaneously.
- **Relevance for PDA**: Enabling (Empirical Evidence) — provides empirical evidence that transformers already perform implicit parallel reasoning in latent space. Validates PDA's core hypothesis.
- **What PDA can adopt**: Empirical methodology for analyzing parallel paths; the finding that parallel reasoning emerges naturally.
- **Limitations / Differences**: Observational, not architectural.
- **Follow up**: yes — foundational empirical evidence.

### Diffusion of Thoughts (DoT) (Ye et al., Feb 2024, NeurIPS 2024)
- **Source**: https://arxiv.org/abs/2402.07754
- **Core idea**: Integrates diffusion models with Chain-of-Thought reasoning. All reasoning positions are refined simultaneously through the diffusion process, enabling bidirectional non-linear reasoning and self-correction.
- **Relevance for PDA**: Architecture + Alternative — provides a fundamentally different approach to iterative parallel refinement.
- **What PDA can adopt**: Iterative parallel refinement of all reasoning steps simultaneously; self-correction via global refinement.
- **Limitations / Differences**: Operates at token/embedding level; requires a diffusion language model backbone; limited to small-scale models.
- **Follow up**: yes — important alternative paradigm.

### Huginn: Scaling Test-Time Compute with Latent Reasoning (Geiping et al., Feb 2025)
- **Source**: https://arxiv.org/abs/2502.05171 (NeurIPS 2025 Spotlight)
- **Core idea**: **3.5B-parameter depth-recurrent transformer** with Prelude → Recurrent Block (looped N times) → Coda architecture. Trained on 800B tokens. Can unroll to effective depth of 132 layers at test time. Hidden states exhibit convergence patterns (fixed points for easy tokens, orbits for complex reasoning).
- **Relevance for PDA**: Architecture + Enabling — proves recurrent-depth transformers work as language models at scale. PDA could use Huginn-like architecture for individual workers.
- **What PDA can adopt**: Prelude-Recurrent-Coda architecture; stochastic depth training; per-token adaptive compute; KV-cache sharing.
- **Limitations / Differences**: Single sequential stream, no parallelism across perspectives.
- **Follow up**: yes — essential scaling reference.

### Latent Thinking Optimization (LTO) (Du et al., Sep 2025)
- **Source**: https://arxiv.org/abs/2509.26314
- **Core idea**: Shows latent thoughts leading to correct vs. incorrect answers exhibit highly distinguishable patterns. Proposes a Latent Reward Model (LRM) for evaluating latent reasoning quality and optimizing latent thinking processes.
- **Relevance for PDA**: Convergence — provides a mechanism for evaluating and guiding latent reasoning trajectories. The LRM could serve as PDA's convergence criterion.
- **What PDA can adopt**: Latent reward models for scoring intermediate states; optimization of latent thinking.
- **Limitations / Differences**: Single-model, sequential. No parallel workers.
- **Follow up**: yes — convergence/quality mechanisms for PDA.

### Additional relevant works in this direction

**Instilling Parallel Reasoning into Language Models** (Macfarlane et al., Microsoft/ICML 2025) distills parallel reasoning traces from a teacher into a student LLM. Token-level but demonstrates value of multi-perspective decomposition. **Adaptive Parallel Reasoning (APR)** (Pan et al., Apr 2025) enables spawn/join control trained via RL. **Parallel-R1** (Tencent, NeurIPS 2025) is the first RL framework for instilling parallel thinking. Both are token-level. Two comprehensive surveys — **A Survey on Parallel Reasoning** (2025) and **Reasoning Beyond Language: A Comprehensive Survey on Latent Chain-of-Thought** (2025) — provide useful taxonomies.

---

## Search Direction 2: Enabling technologies

### 2a: Deep Equilibrium Models and fixed-point convergence

### Energy-Based Transformers are Scalable Learners and Thinkers (Gladstone et al., Jul 2025)
- **Source**: https://arxiv.org/abs/2507.02092 (ICLR 2026 submission)
- **Core idea**: EBTs assign an energy scalar to every (input, candidate-prediction) pair and refine predictions via gradient-descent energy minimization until convergence. Generalize DEQs: "EBMs are a generalization of DEQ." Scale up to **35% faster** than Transformer++ across data, parameters, FLOPs, and depth.
- **Relevance for PDA**: Architecture + Convergence + Enabling — the most promising convergence mechanism alternative to DEQs. Each PDA worker could minimize an energy landscape; consensus emerges at the joint energy minimum.
- **What PDA can adopt**: Energy-based convergence instead of root-finding; energy landscape regularization; best-of-N sampling via energy comparison; scalable training with implicit gradients.
- **Limitations / Differences**: 3.3–6.6× more training FLOPs. Operates on single prediction, not multiple parallel perspectives. Not yet scaled to full foundation model size.
- **Follow up**: yes — most promising alternative to DEQs for PDA's convergence.

### Scaling Latent Reasoning via Looped Language Models / Ouro-LoopLM (Zhu et al., Oct 2025)
- **Source**: https://arxiv.org/abs/2510.25741
- **Core idea**: Pre-trained Looped Language Models (1.4B, 2.6B params) using parameter-shared recurrent transformer blocks with **entropy-regularized adaptive depth** allocation. Trained on 7.7T tokens. Ouro-1.4B matches 12B standard LLMs. Advantage comes from superior knowledge *manipulation* over storage.
- **Relevance for PDA**: Architecture + Convergence — weight-tied iterative computation in latent space is essentially PDA's deliberation loop.
- **What PDA can adopt**: Entropy-regularized depth allocation for adaptive iteration count; sandwich normalization; multi-stage training pipeline.
- **Limitations / Differences**: Single looped block, not multiple parallel workers. Limited to ~4 recurrent steps in practice.
- **Follow up**: yes — directly demonstrates fixed-point-like convergence at LLM scale.

### Fixed-Point RNNs: Interpolating from Diagonal to Dense (Movahedi et al., Mar 2025)
- **Source**: https://arxiv.org/abs/2503.10799 (NeurIPS 2025)
- **Core idea**: Parameterizes dense linear RNNs as fixed-points of parallelizable diagonal linear RNNs. Alternates channel mixing and sequence mixing until convergence. Uses implicit differentiation at fixed points, avoiding backpropagation through iterations. SOTA on state-tracking benchmarks.
- **Relevance for PDA**: Architecture + Convergence — framework of computing an expressive model as the fixed-point of simpler, parallelizable models is directly analogous to PDA workers converging.
- **What PDA can adopt**: Fixed-point parameterization trading expressivity for efficiency; workaround avoiding Jacobian inversion; guaranteed contraction regime.
- **Limitations / Differences**: Focused on sequence mixing (RNN-like). Evaluated only on toy tasks.
- **Follow up**: yes — novel mathematical framework highly relevant to convergence.

### Reversible Deep Equilibrium Models / RevDEQs (McCallum et al., Sep 2025)
- **Source**: https://arxiv.org/abs/2509.12917
- **Core idea**: Uses an algebraically reversible fixed-point solver enabling *exact* gradient calculation, eliminating the need for Jacobian regularization. SOTA on language modeling (WikiText-103) and image classification vs. comparable implicit and explicit models.
- **Relevance for PDA**: Convergence + Enabling — solves the primary training instability problem of DEQs.
- **What PDA can adopt**: Reversible fixed-point solver for exact gradients; damping parameter β for convergence speed/accuracy tradeoff; O(1) memory maintained.
- **Limitations / Differences**: Still tested at relatively small scale. Runtime can exceed explicit models.
- **Follow up**: yes — major improvement for DEQ training.

### Positive Concave Deep Equilibrium Models / pcDEQ (Gabor et al., Feb 2024)
- **Source**: https://arxiv.org/abs/2402.04029
- **Core idea**: Enforces nonnegative weights and concave activations, guaranteeing fixed-point existence/uniqueness via nonlinear Perron-Frobenius theory. Standard fixed-point iteration converges geometrically.
- **Relevance for PDA**: Convergence — provides *guaranteed* convergence without complex root-finding.
- **What PDA can adopt**: Perron-Frobenius convergence guarantees; simplified training without Broyden's method.
- **Limitations / Differences**: Constraining weights to nonnegative and activations to concave limits representational capacity.
- **Follow up**: yes — useful for guaranteed convergence in simpler PDA components.

**Key finding for DEQs**: Pure DEQs have **not** been scaled beyond WikiText-103 (~250M params). Barriers include root-finding overhead, training instability, and architectural brittleness. However, the closely related **Looped/Recurrent-Depth Transformers** have scaled to 3.5B parameters (Huginn) and 2.6B parameters (Ouro) successfully. **Energy-Based Transformers** offer a generalization of DEQs with superior scaling properties. The practical consensus is: **simple iteration with truncated BPTT scales; root-finding does not.**

### 2b: Orthogonal representations

### Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning / NDM (Huang et al., Aug 2025)
- **Source**: https://arxiv.org/abs/2508.01916
- **Core idea**: Proposes Neighbor Distance Minimization (NDM) to learn non-basis-aligned subspaces of transformer representation space in an unsupervised manner. An orthogonal matrix is learned that rotates and reflects the space before partitioning, capturing the distributed nature of neural representations. Subspaces are interpretable and correspond to model "variables."
- **Relevance for PDA**: Enabling — directly addresses how to decompose representation space into orthogonal subspaces for PDA workers.
- **What PDA can adopt**: The learned orthogonal rotation matrix approach for partitioning representation space into worker-specific subspaces. NDM provides principled unsupervised discovery of "natural" independent subspaces.
- **Limitations / Differences**: Designed for interpretability/analysis, not parallel inference.
- **Follow up**: yes — perhaps the most directly relevant paper for PDA's subspace decomposition.

### Transformer Normalisation Layers and the Independence of Semantic Subspaces (Jun 2024)
- **Source**: https://arxiv.org/abs/2406.17837
- **Core idea**: Pre-Norm transformers require semantic subspaces to be orthogonal spheres to avoid interference. Different normalisation layers impose different geometric constraints. Introduces "circuit collapse" phenomenon at ~10% norm perturbation.
- **Relevance for PDA**: Architecture — theoretical foundation for why orthogonal subspaces are necessary.
- **What PDA can adopt**: QKV-Norm architecture relaxes constraints to require only linear independence, enabling more flexible worker subspaces.
- **Limitations / Differences**: Analysis of existing norms, not construction of parallel workers.
- **Follow up**: yes — critical theoretical grounding.

### Large Language Models Encode Semantics in Low-Dimensional Linear Subspaces (Saglam et al., Jul 2025)
- **Source**: https://arxiv.org/html/2507.09709v1
- **Core idea**: Large-scale empirical study confirming semantic representations compress into compact, linearly separable clusters. Separability strengthens in deeper layers and under reasoning prompts (e.g., CoT).
- **Relevance for PDA**: Enabling — validates the linear representation hypothesis PDA relies on.
- **What PDA can adopt**: Evidence that LLMs naturally organize knowledge in low-dimensional linear subspaces that can be independently manipulated.
- **Limitations / Differences**: Observational; doesn't provide methods for enforcing separation.
- **Follow up**: yes — strong empirical validation.

### The Linear Representation Hypothesis and the Geometry of Large Language Models (Park et al., ICML 2024)
- **Source**: https://arxiv.org/abs/2311.03658
- **Core idea**: Formalizes that concepts varying independently are represented as orthogonal vectors under a non-Euclidean **"causal inner product"** estimated from the LLM unembedding matrix. Unifies probing and steering into a single framework.
- **Relevance for PDA**: Enabling — provides the mathematical framework for concept orthogonality in LLMs.
- **What PDA can adopt**: The causal inner product as the correct metric for ensuring PDA worker perspectives are truly independent.
- **Limitations / Differences**: Theoretical; doesn't address multi-agent processing.
- **Follow up**: yes — critical for defining PDA's deliberation geometry.

### PEGO: Parameter-Efficient Group with Orthogonal Regularization (Hu et al., ECCV 2024)
- **Source**: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06773.pdf
- **Core idea**: Injects a group of LoRA modules with dual orthogonal losses: **L_preserve** (preserve base model generalization) and **L_diversify** (encourage diverse knowledge learning across modules).
- **Relevance for PDA**: Architecture + Enabling — directly demonstrates orthogonal LoRA groups as diverse "perspectives."
- **What PDA can adopt**: Dual orthogonal regularization — L_preserve keeps workers grounded in base knowledge, L_diversify ensures workers learn different aspects. This is essentially PDA's multi-perspective training.
- **Limitations / Differences**: Vision domain; LoRA groups used during training, not parallel inference-time workers.
- **Follow up**: yes — very high priority; closest to PDA's orthogonal multi-perspective mechanism.

### 2c: LoRA merging and Multi-LoRA

### Rethinking Inter-LoRA Orthogonality in Adapter Merging (Zhang et al., Sep 2025)
- **Source**: https://arxiv.org/abs/2510.03262
- **Core idea**: **CRITICAL NEGATIVE FINDING: Orthogonality alone does NOT lead to semantic disentanglement or compositionality.** Proposes Orthogonal Monte Carlo Dropout for efficient enforcement, but demonstrates structural orthogonality ≠ semantic compositionality.
- **Relevance for PDA**: Enabling — essential cautionary result for PDA's design.
- **What PDA can adopt**: The lesson that **PDA needs more than just orthogonality** — it needs semantic grounding for its perspectives, not just geometric separation.
- **Limitations / Differences**: Tested primarily in diffusion model context, not LLMs for text.
- **Follow up**: yes — essential; PDA must address why its orthogonal perspectives would yield meaningful diversity.

### OSRM: Orthogonal Subspaces for Robust Model Merging (ACL 2025)
- **Source**: https://aclanthology.org/2025.acl-long.1284.pdf
- **Core idea**: Characterizes task-specific latent feature subspaces using samples, then projects LoRA updates into orthogonal subspaces to minimize interference during merging. Provides analytical solution for optimal projection.
- **Relevance for PDA**: Merge — directly addresses how to merge multiple LoRA perspectives without interference.
- **What PDA can adopt**: Sample-based subspace characterization; analytical orthogonal projection method.
- **Limitations / Differences**: Post-hoc merging, not designed for iterative deliberation.
- **Follow up**: yes — OSRM's projection could serve as PDA's convergence mechanism.

### SMoRA: Each Rank Could be an Expert (Zhao et al., Jan 2025)
- **Source**: https://arxiv.org/abs/2501.15103
- **Core idea**: Establishes equivalence between multi-LoRA MoE routing and rank partitioning within a single LoRA. Each rank dimension acts as an independent expert with dynamic activation.
- **Relevance for PDA**: Architecture — PDA workers could operate as different rank subsets within a shared LoRA.
- **What PDA can adopt**: The rank-as-expert concept could dramatically simplify PDA's implementation.
- **Limitations / Differences**: Routing for task selection, not deliberative multi-perspective processing.
- **Follow up**: yes — implementation simplification opportunity.

### LoRA-Ensemble: Efficient Uncertainty Modelling (2024)
- **Source**: https://arxiv.org/html/2405.14438v3
- **Core idea**: Creates implicit deep ensembles using multiple LoRA adaptations of a single pretrained model. Members converge across a broader loss landscape area than explicit ensembles, indicating greater diversity. 3× inference speedup over full ensembles.
- **Relevance for PDA**: Architecture — closest existing work to PDA's concept of parallel LoRA workers.
- **What PDA can adopt**: Multiple LoRA modules sharing frozen backbone for diverse parallel processing; evidence that LoRA naturally induces weight-space diversity.
- **Limitations / Differences**: Used for uncertainty quantification, not deliberative convergence.
- **Follow up**: yes — PDA is essentially "LoRA-Ensemble + deliberation/convergence."

**Additional LoRA works**: **OPLoRA** (AAAI 2025) constrains LoRA updates to be orthogonal to top-k singular directions of pretrained weights. **CLoRA** (ACL 2025) imposes null-space constraints. **LoraHub** (COLM 2024) composes multiple LoRAs via gradient-free optimization. **Instance-Level Dynamic LoRA Composition** (EMNLP 2024) extends to per-input routing. **MoLE** (2024) uses hierarchical gating for LoRA fusion. **S-LoRA** (2023) demonstrates serving thousands of concurrent adapters efficiently. All support the feasibility of PDA's multi-LoRA approach.

### 2d: Signal processing in ML

### Language Through a Prism: A Spectral Approach for Multiscale Language Representations (Tamkin, Jurafsky, Goodman, 2020)
- **Source**: https://arxiv.org/abs/2011.04823
- **Core idea**: Applies DCT (Discrete Cosine Transform) spectral filters to neuron activations in BERT. Low-pass filters isolate document-level (topic) info; high-pass filters isolate word-level (POS) info; band-pass captures utterance-level structure. Introduces a "prism layer" constraining different neurons to model different frequency scales.
- **Relevance for PDA**: Architecture + Convergence — **highly relevant.** PDA workers could operate at different frequency bands, and convergence could be measured by coherence across filtered outputs.
- **What PDA can adopt**: DCT-based spectral decomposition as perspective diversity framework; prism layer for enforcing scale specialization; spectral filtering as convergence metric.
- **Limitations / Differences**: Applied to BERT encoder, not decoder LLMs. Framework is analytical.
- **Follow up**: yes — essential foundational work for spectral perspective.

### Spectral Editing of Activations for LLM Alignment / SEA (Qiu et al., NeurIPS 2024)
- **Source**: https://arxiv.org/abs/2405.09719
- **Core idea**: Uses SVD-based spectral decomposition on LLM activations to edit internal representations. Projects activations into directions with maximal covariance with positive demonstrations.
- **Relevance for PDA**: Convergence + Merge — SVD of activation covariance can identify alignment directions. PDA could project worker outputs into spectral bases and measure convergence as alignment along desired spectral directions.
- **What PDA can adopt**: SVD cross-covariance framework; spectral spectrum visualization for merge strategy.
- **Limitations / Differences**: Single-model editing, not multi-worker convergence.
- **Follow up**: yes — bridges spectral analysis and representation engineering.

### Pre-trained LLMs Use Fourier Features to Compute Addition (Zhou et al., NeurIPS 2024)
- **Source**: https://arxiv.org/abs/2406.03445
- **Core idea**: LLMs represent numbers using Fourier features — hidden state dimensions encode numbers via frequency-sparse features. MLP layers use low-frequency features for magnitude; attention uses high-frequency for modular arithmetic.
- **Relevance for PDA**: Enabling — LLM hidden states naturally organize into frequency-decomposable components. Different workers could process different frequency components.
- **What PDA can adopt**: Evidence for natural Fourier structure; methodology for identifying frequency components.
- **Limitations / Differences**: Focused on arithmetic tasks, not general semantic reasoning.
- **Follow up**: yes — check if similar structure exists for semantic representations.

### Similarity of Neural Network Representations Revisited / CKA (Kornblith et al., ICML 2019)
- **Source**: http://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf
- **Core idea**: Introduces **Centered Kernel Alignment** as a robust metric for comparing neural network representations. Shows networks trained from different initializations learn similar representations.
- **Relevance for PDA**: Convergence — CKA provides a principled metric for measuring whether workers' representations are converging.
- **What PDA can adopt**: CKA as a convergence metric between worker representations.
- **Limitations / Differences**: Computationally expensive for real-time convergence monitoring.
- **Follow up**: yes — could serve as basis for PDA's convergence criterion.

### 2e: Steering vectors and representation engineering

### From Steering Vectors to Conceptors: Compositional Affine Activation Steering (Abreu et al., NeurIPS 2025 submission)
- **Source**: https://openreview.net/forum?id=0Yu0eNdHyV
- **Core idea**: Combines conceptor theory with activation steering. Conceptors are soft projection matrices (ellipsoids in activation space). **Critically, introduces Boolean operations over conceptors — AND, OR, NOT — for compositional steering toward multiple objectives**, outperforming traditional vector combination.
- **Relevance for PDA**: Architecture + Merge — **highly relevant.** Boolean algebra over conceptors provides a principled framework for combining or separating different perspectives.
- **What PDA can adopt**: Conceptor-based steering as worker perspective mechanism (matrices, not just vectors); Boolean composition (AND/OR/NOT) for combining perspectives during merge; optimal affine steering.
- **Limitations / Differences**: More complex than simple vectors; computational cost higher. Under review.
- **Follow up**: yes — **top priority** for PDA's composition needs.

### Steer2Adapt: Dynamically Composing Steering Vectors (Feb 2025)
- **Source**: https://arxiv.org/abs/2602.07276
- **Core idea**: Tasks share underlying concept dimensions. Constructs a reusable, low-dimensional **"semantic prior subspace"** from domain-relevant concepts. Uses Bayesian optimization with stability-aware objective to find optimal linear combination coefficients from few examples.
- **Relevance for PDA**: Architecture + Enabling — validates that different perspectives can be created by linear combination of basis steering vectors spanning meaningful subspaces.
- **What PDA can adopt**: Semantic prior subspace as the deliberation space; Bayesian optimization for finding optimal combinations; stability-aware objective as convergence criterion.
- **Limitations / Differences**: Static per-task composition, not iteratively refined.
- **Follow up**: yes — high priority; operational framework for PDA's perspective space.

### Representation Engineering: A Top-Down Approach to AI Transparency (Zou et al., 2023)
- **Source**: https://arxiv.org/abs/2310.01405
- **Core idea**: Foundational paper. Uses population-level representations to monitor and manipulate high-level cognitive phenomena. Introduces Linear Artificial Tomography (LAT). Demonstrates control over honesty, power-seeking, harmlessness.
- **Relevance for PDA**: Enabling — establishes that high-level concepts are linearly encoded and can be manipulated via activation-space interventions.
- **What PDA can adopt**: Concept-direction extraction methodology; LAT visualization for monitoring worker states.
- **Limitations / Differences**: Focuses on safety concepts; doesn't address composition or iteration.
- **Follow up**: yes — essential reference.

### Activation Addition / ActAdd (Turner et al., 2023/2024)
- **Source**: https://arxiv.org/abs/2308.10248
- **Core idea**: Lightweight steering by computing activation differences between contrastive prompt pairs and adding the resulting vector to the residual stream at inference time. Shows evidence for **compositional representations** — composing forward passes works sensibly.
- **Relevance for PDA**: Architecture + Enabling — PDA workers could be created by adding different steering vectors at inference time. **Middle layers (~layer 6 for GPT-2, ~layer 14 for 7-9B models) are optimal for intervention.**
- **What PDA can adopt**: ActAdd as mechanism for creating diverse perspectives; optimal layer selection; compositional evidence.
- **Limitations / Differences**: Single-vector steering is limited; composition not deeply explored; one-shot.
- **Follow up**: yes — foundation method for worker differentiation.

### DISCO: Disentangled Communication Steering (NeurIPS 2025)
- **Source**: https://openreview.net/forum?id=c8AjdgdHnD
- **Core idea**: Injects steering vectors into query and value representation spaces within attention heads. Q/V spaces exhibit **higher linear discriminability** of concepts than attention head outputs. Up to **19.1% higher steering efficacy** on LLaMA 3.1 8B.
- **Relevance for PDA**: Architecture — PDA workers could be differentiated in Q/V spaces, which are more concept-discriminable.
- **What PDA can adopt**: Q/V space injection as a more precise steering mechanism; disentangled control principle.
- **Limitations / Differences**: Single-concept, not multi-perspective composition.
- **Follow up**: yes — could refine PDA's steering point.

### Linear Representations of Political Perspective (ICLR 2025)
- **Source**: https://openreview.net/forum?id=rwqShzb9li
- **Core idea**: LLMs possess linear representations of complex political perspectives (not just simple traits) in attention head activation space. Steering can shift outputs between liberal and conservative stances.
- **Relevance for PDA**: Enabling — validates that complex, high-level *perspectives* are linearly represented and steerable, directly supporting PDA's core assumption.
- **What PDA can adopt**: Evidence that perspective-level concepts are linearly encoded.
- **Limitations / Differences**: One axis; PDA needs multi-dimensional perspective diversity.
- **Follow up**: yes — validates PDA's perspective-steering assumption.

**Additional works**: **Repulsive Deep Ensembles** (NeurIPS 2021) provides a principled attraction/repulsion balance for maintaining diversity during convergence — directly relevant to PDA's deliberation dynamics. **Parallel Orthogonal DNNs** (2021) demonstrates enforced diversity via Gram-Schmidt orthogonalization across parallel networks. **Can SAEs Decompose Steering Vectors?** (NeurIPS 2024 Workshop) is an important cautionary result: steering vectors fall outside SAE input distribution and are not simple sums of monosemantic features.

---

## Search Direction 3: Alternative architectures

### 3a: State Space Models with parallel streams

### Mamba-3 MIMO (Multi-Input, Multi-Output SSMs) (Gu, Dao et al., 2025)
- **Source**: https://openreview.net/forum?id=HwCvaJOiCj (ICLR 2026)
- **Core idea**: Models **multiple SSMs in parallel** instead of SISO. Exploits memory bandwidth rather than compute; adds no decode latency. Includes complex-valued state updates for richer state tracking.
- **Relevance for PDA**: Architecture + Enabling — MIMO SSMs are a direct analog to PDA's parallel workers. Multiple SSM instances process the same input from different state-space perspectives and merge via an input-dependent projection matrix.
- **What PDA can adopt**: MIMO formulation for hardware-efficient parallel perspectives with zero decode latency penalty.
- **Limitations / Differences**: Parallel streams share identical architecture; lack explicit deliberation/convergence mechanisms.
- **Follow up**: yes — one of the most promising enabling technologies for PDA-style parallel processing.

### Multi-Head State Space Model / MH-SSM (Fathullah et al., 2023)
- **Source**: https://arxiv.org/abs/2305.12498
- **Core idea**: Parallel SSM heads with specialized gating, each learning **different temporal dynamics** (local vs. global). The Stateformer variant combines MH-SSMs with attention.
- **Relevance for PDA**: Architecture — directly implements multiple workers processing the same input from different perspectives via independent SSM channels.
- **What PDA can adopt**: Parallel head structure with heterogeneous initialization encouraging specialization.
- **Limitations / Differences**: No inter-head communication. "Perspectives" are temporal, not semantic.
- **Follow up**: yes — strong architectural analog.

### MvSSM: Multi-view State-Space Model (2025)
- **Source**: https://www.sciencedirect.com/science/article/abs/pii/S0893608025009682
- **Core idea**: Multi-view representation learning as a continuous-time dynamical system. View-specific features serve as external inputs; a shared latent representation evolves as the internal state, driven by learnable dynamics with cross-view interaction.
- **Relevance for PDA**: Architecture — directly models multi-view learning through SSMs with cross-view interaction.
- **What PDA can adopt**: Framework for treating perspectives as external inputs driving a shared latent state through SSM dynamics; cross-view interaction mechanism.
- **Limitations / Differences**: Multi-modal inputs, not generating multiple perspectives from a single input.
- **Follow up**: yes — strongest conceptual alignment with PDA in the SSM space.

### 3b: Hypernetworks

### Attention as a Hypernetwork (Schug et al., ICLR 2025)
- **Source**: https://arxiv.org/abs/2406.05816
- **Core idea**: **Multi-head attention is mathematically equivalent to a hypernetwork**, where attention scores form a compact latent code configuring key-query specific operations in a linear value network. Nonlinear value networks improve compositional generalization. Functionally structured latent spaces emerge.
- **Relevance for PDA**: Architecture + Enabling — shows attention heads already implement "multiple perspectives" via hypernetwork composition. PDA could explicitly design workers as hypernetwork-configured processors.
- **What PDA can adopt**: Explicit hypernetwork formulation where a "perspective code" configures each worker's computation. Compositionality — different codes can combine to form novel operations.
- **Limitations / Differences**: Shows what attention *already does implicitly*; PDA must make perspectives explicit and deliberately iterate.
- **Follow up**: yes — **critical** theoretical foundation for hypernetwork-based perspectives.

### Hyper-CL: Conditioning Sentence Representations with Hypernetworks (Yoo et al., ACL 2024)
- **Source**: https://arxiv.org/html/2403.09490v1
- **Core idea**: A hypernetwork transforms condition embeddings into **projection layers creating different perspective-specific subspaces**. The same input is projected differently based on different conditions/perspectives.
- **Relevance for PDA**: Architecture — **almost exactly PDA's concept** applied to sentence embeddings: same input projected into different subspaces based on perspective conditions.
- **What PDA can adopt**: Hypernetwork taking "perspective embedding" and generating projection matrices for perspective-specific representations.
- **Limitations / Differences**: Sentence-level, not token-level generation. No iterative deliberation.
- **Follow up**: yes — **critical**; most direct implementation of perspective-specific weight generation via hypernetworks.

### Text-to-LoRA Hypernetworks / Zhyper (2025)
- **Source**: https://arxiv.org/html/2510.19733v1
- **Core idea**: Factorized hypernetwork framework generating **context-aware LoRA adapters from textual descriptions** in a single forward pass. Achieves competitive performance with up to 26× fewer parameters.
- **Relevance for PDA**: Enabling — perspective-specific LoRA adapters can be efficiently generated by hypernetworks conditioned on "perspective descriptors."
- **What PDA can adopt**: LoRA-based perspective modification via hypernetwork; parameter efficiency at LLM scale.
- **Limitations / Differences**: Designed for task adaptation, not parallel multi-perspective reasoning.
- **Follow up**: yes — practical implementation path.

### 3c: Modular networks and communicating MoE

### Chain-of-Experts (CoE): Unlocking Communication Power of MoE Models (2025)
- **Source**: https://arxiv.org/html/2506.18945v1
- **Core idea**: Introduces **iterative expert processing within each MoE layer** — tokens pass through experts sequentially across multiple communication steps with separate gating at each step. Expert outputs from iteration 1 influence routing in iteration 2. Residual connections between iterations and iteration-independent routers enable stable training. Expert transitions show increasing specialization (convergence) in deeper layers.
- **Relevance for PDA**: Architecture — **highest architectural relevance** among alternatives. CoE explicitly enables expert communication, multi-pass refinement, and convergence — three core PDA elements.
- **What PDA can adopt**: Iterative expert chaining with C communication steps; iteration-specific routing allowing different perspectives per step; residual connections enabling stable convergence.
- **Limitations / Differences**: CoE processes tokens *sequentially* through experts (chain), whereas PDA envisions *parallel* processing that converges. Increases latency linearly with iterations.
- **Follow up**: yes — **critical**; PDA could be viewed as "parallel CoE with convergence."

### HyperMoE: Knowledge Transfer Among Experts via Hypernetworks (Zhao et al., ACL 2024)
- **Source**: https://arxiv.org/abs/2402.12656
- **Core idea**: A shared hypernetwork across all experts/layers generates "HyperExperts" conditioned on **information from unselected experts**. Knowledge from inactive experts influences active expert computation without breaking sparsity.
- **Relevance for PDA**: Architecture + Enabling — bridges hypernetworks and MoE, enabling expert communication through the hypernetwork.
- **What PDA can adopt**: Shared cross-layer hypernetwork for information flow across experts/layers; "selection embedding" concept.
- **Limitations / Differences**: One-directional communication (inactive → active); no iterative convergence.
- **Follow up**: yes — combines hypernetworks and MoE for cross-expert communication.

### MoEUT: Mixture-of-Experts Universal Transformers (NeurIPS 2024)
- **Source**: https://arxiv.org/abs/2405.16039
- **Core idea**: **First Universal Transformer competitive with standard Transformers on language modeling.** Solves the parameter-compute ratio problem via MoE in both feedforward and attention layers. Introduces "peri-layernorm." Outperforms dense baselines from 44M to 1B parameters.
- **Relevance for PDA**: Architecture — solves how to maintain parameter efficiency with weight-shared iterative layers. Different iterations can route to different experts despite sharing weights.
- **What PDA can adopt**: MoE in iterative blocks; peri-layernorm for stable shared-layer training; layer grouping.
- **Limitations / Differences**: Depth-wise recurrence, not parallel multi-perspective. Not tested beyond 1B.
- **Follow up**: yes — MoEUT's approach could enable PDA's multi-perspective routing.

### How Many Heads Make an SSM? (Ghodsi et al., Dec 2025)
- **Source**: https://arxiv.org/abs/2512.15115
- **Core idea**: Proves the **Head-Count Theorem**: representing a linear SSM whose lag operators span a k-dimensional subspace requires exactly H=k heads. Establishes fundamental trade-offs between expressivity and long-range gradient propagation.
- **Relevance for PDA**: Enabling (Theoretical) — directly informs how many parallel workers/perspectives are needed for a given complexity.
- **What PDA can adopt**: Theoretical framework for determining optimal worker count.
- **Limitations / Differences**: Purely theoretical.
- **Follow up**: yes — important for PDA's design space.

### 3d: Neural ODEs and continuous-depth networks

### Neural ODE Transformers / DiffEqFormer (Tong et al., ICLR 2025)
- **Source**: https://arxiv.org/abs/2503.01329
- **Core idea**: Models transformer layers as non-autonomous neural ODEs; all weights (Q, K, V, FF) parameterized as continuous functions of layer index via hypernetworks. Spectral analysis reveals increasing eigenvalue magnitudes across depth.
- **Relevance for PDA**: Architecture + Convergence — continuous-depth formulation models smooth evolution for PDA's convergence. Spectral analysis provides convergence diagnostics.
- **What PDA can adopt**: Continuous parameterization for smoother deliberation dynamics; Lyapunov exponents as convergence monitoring tools.
- **Limitations / Differences**: Single trajectory; inherently sequential; eigenvalue growth may destabilize multi-perspective convergence.
- **Follow up**: yes — spectral analysis framework adaptable for monitoring PDA convergence.

### Continuous-Depth Transformers with Learned Control Dynamics (Jan 2025)
- **Source**: https://arxiv.org/abs/2601.10007
- **Core idea**: Replaces discrete middle layers with a Neural ODE block. A low-dimensional **learned control signal u** steers the trajectory, enabling inference-time attribute control (98% sentiment accuracy). Achieves 0.068% trajectory divergence between fixed and adaptive solvers.
- **Relevance for PDA**: Architecture + Alternative — **each PDA worker could use a different control signal u** to explore different regions of representation space while sharing continuous dynamics.
- **What PDA can adopt**: Control signal mechanism as "perspective injection" for workers; gradient flow stability techniques; solver-invariance testing.
- **Limitations / Differences**: Current focus on attribute steering, not reasoning convergence. ODE solving overhead compounds with multiple workers.
- **Follow up**: yes — control-signal-as-perspective is highly promising.

### 3e: Energy-Based Models for consensus

### Learning Iterative Reasoning through Energy Diffusion / IRED (Du, Mao, Tenenbaum, ICML 2024)
- **Source**: https://arxiv.org/abs/2406.11179
- **Core idea**: Formulates reasoning as energy minimization over learned energy landscapes. Learns a sequence of **annealed energy functions** with progressively increasing difficulty. Adapts optimization steps to problem difficulty. Outperforms baselines on Sudoku, matrix completion, graph pathfinding, generalizing to unseen harder instances.
- **Relevance for PDA**: Convergence + Alternative — **the most directly relevant for PDA's convergence mechanism**. Instead of iterative fixed-point convergence, defines answers as energy minima. Multiple workers could optimize different energy landscapes or from different initializations.
- **What PDA can adopt**: Annealed energy landscapes as convergence curriculum; adaptive computation; combined score function + energy landscape supervision.
- **Limitations / Differences**: Task-specific energy functions, not general language reasoning. Requires continuous gradient optimization. Computationally expensive.
- **Follow up**: yes — critical; annealed energy landscapes could replace/augment PDA's convergence.

### The Consensus Game: Language Model Generation via Equilibrium Search (Jacob et al., ICLR 2024)
- **Source**: https://arxiv.org/abs/2310.09139
- **Core idea**: Casts LM decoding as a regularized imperfect-information signaling game between Generator and Discriminator. Finds equilibrium via no-regret learning. Training-free. **LLaMA-7B with equilibrium-ranking outperforms LLaMA-65B** on multiple benchmarks.
- **Relevance for PDA**: Convergence + Alternative — directly addresses consensus between multiple "perspectives" of the same model using game-theoretic equilibrium.
- **What PDA can adopt**: Regularized equilibrium as convergence target; no-regret learning for iterative strategy updates; decomposition of a model into perspectives.
- **Limitations / Differences**: Operates over discrete candidate answers, not continuous representation space. Limited to two perspectives.
- **Follow up**: yes — game-theoretic equilibrium is a strong convergence alternative.

### DIFFormer: Diffusion-inspired Transformers (JMLR 2025)
- **Source**: http://www.jmlr.org/papers/volume26/23-1672/23-1672.pdf
- **Core idea**: Derives transformer attention as a diffusion process on latent complete graphs where attention weights minimize an energy function. Proves convergence speed and addresses over-smoothing risk.
- **Relevance for PDA**: Convergence — proves **attention IS energy minimization**, providing theoretical grounding that PDA's attention-based convergence can be analyzed through energy frameworks.
- **What PDA can adopt**: Energy-constrained diffusion framework; techniques to avoid over-smoothing (degenerate consensus).
- **Limitations / Differences**: Applied to GNNs, not language models.
- **Follow up**: yes — connection between attention and energy supports PDA.

### 3f: Diffusion in representation space

### LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning (Kang et al., 2025)
- **Source**: https://arxiv.org/abs/2510.04573
- **Core idea**: Constructs a structured latent reasoning space using a VAE. A latent diffusion model iteratively denoises thought-token blocks with bidirectional attention. Shows **+11.7 accuracy points from 5→10 denoising steps**, demonstrating scalable test-time compute.
- **Relevance for PDA**: Architecture + Alternative — **directly demonstrates diffusion as an iterative deliberation mechanism in latent space.** The blockwise denoising process is analogous to PDA's iterative refinement.
- **What PDA can adopt**: VAE-constructed latent reasoning space; blockwise bidirectional attention; adaptive test-time compute via variable denoising steps; flow matching loss to prevent collapse.
- **Limitations / Differences**: Single diffusion trajectory, not multiple parallel perspectives. Requires separate VAE and diffusion model.
- **Follow up**: yes — strong alternative deliberation mechanism.

### Efficient Parallel Samplers for Recurrent-Depth Models (2025)
- **Source**: https://arxiv.org/html/2510.14961v1
- **Core idea**: Develops diffusion forcing sampler for recurrent-depth models (like Huginn). Establishes that **recurrent-depth models can be interpreted as latent-space diffusion models.** Formal equivalence between recurrent depth and latent diffusion.
- **Relevance for PDA**: Architecture — unifies iterative recurrence and diffusion, meaning PDA can leverage both theoretical toolkits.
- **What PDA can adopt**: Parallel sampling algorithms from diffusion literature; formal equivalence for theoretical grounding.
- **Limitations / Differences**: Focused on efficient sampling, not multi-perspective reasoning.
- **Follow up**: yes — parallel samplers could accelerate PDA inference.

### 3g: Recurrent and Universal Transformers

### Looped Transformers for Length Generalization (Fan et al., 2024)
- **Source**: https://arxiv.org/html/2409.15647v5
- **Core idea**: Decoder-only looped transformer with **input injection** (original input added at each step) and step-dependent supervision. Achieves universal length generalization on RASP-L tasks. Shows input injection prevents information loss.
- **Relevance for PDA**: Architecture + Convergence — input injection ensures workers maintain connection to original problem across iterations.
- **What PDA can adopt**: Input injection at each iteration; step-dependent supervision for training intermediate states.
- **Limitations / Differences**: Synthetic tasks; single loop trajectory.
- **Follow up**: yes — directly applicable to PDA.

### Routing Experts (RoE): Dynamic Expert Routing in Existing MLLMs (ICLR 2025)
- **Source**: https://openreview.net/forum?id=vtT09dYPGI
- **Core idea**: Standard non-MoE models **already contain latent dynamic experts** — can be turned into MoE-like systems through learned routing without architectural changes. **3.3% performance gain** while being **1.61× faster**.
- **Relevance for PDA**: Enabling — PDA-like multi-perspective processing might be achievable by routing through existing subnetworks in pre-trained models.
- **What PDA can adopt**: Principle that pre-trained models have implicit perspective-specific pathways.
- **Limitations / Differences**: Routing for efficiency, not multi-perspective reasoning.
- **Follow up**: moderate — interesting for practical deployment.

---

## Synthesis: What this means for PDA

### The 10 most promising works and why

1. **Mixture of Thoughts (MoT)** — Closest architecture: latent-level multi-expert collaboration via cross-attention in shared latent space. PDA adds what MoT lacks: iterative convergence.

2. **Chain-of-Experts (CoE)** — Closest to PDA's iterative deliberation: experts communicate across multiple steps with iteration-specific routing and residual connections. PDA should be viewed as "parallel CoE with convergence."

3. **LatentMAS** — Validates multi-agent latent-space collaboration with 14.6% accuracy gains and 4× speedup over text-based systems. Demonstrates that latent collaboration practically works.

4. **Parallel Latent Reasoning (PLR)** — Provides theoretical proof that diversity decays exponentially with depth, motivating width-level scaling. Contrastive diversity enforcement and gating aggregation directly applicable.

5. **Energy-Based Transformers (EBTs)** — The strongest convergence mechanism alternative. Generalizes DEQs with better scaling (35% faster) and natural energy-landscape convergence. Each PDA worker could minimize a shared energy function.

6. **IRED (Energy Diffusion for Reasoning)** — Demonstrates reasoning as energy minimization with annealed landscapes and adaptive compute. The annealing curriculum could replace fixed-point convergence.

7. **Conceptors for Compositional Steering** — Boolean algebra (AND/OR/NOT) over steering objectives provides the most principled framework for composing and separating perspectives.

8. **Hyper-CL** — Most direct implementation of perspective-specific weight generation via hypernetworks: same input projected into different subspaces based on perspective conditions.

9. **PEGO** — Dual orthogonal regularization (L_preserve + L_diversify) for LoRA groups is essentially PDA's multi-perspective training mechanism.

10. **System 1 and 2 Communication** — Critical negative finding that explicit diversity enforcement is mandatory. Without it, latent vectors collapse into redundant subspaces.

### Which PDA assumptions are strengthened

**Strongly validated**: (a) Latent-space reasoning works and can outperform token-level reasoning (Coconut, Huginn, Latent-SFT). (b) LLMs naturally perform implicit parallel reasoning in hidden representations (Distributional Reasoning). (c) Semantic concepts are linearly encoded in orthogonal subspaces (Park et al., Saglam et al.). (d) Multiple steering vectors can be composed to create meaningful perspectives (Steer2Adapt, Conceptors). (e) Multi-agent latent collaboration outperforms text-based approaches (LatentMAS, Interlat). (f) Width-level scaling (parallel streams) provides complementary benefits to depth scaling (PLR).

**Moderately validated**: (a) Fixed-point convergence mechanisms work at scale, but only via simple iteration (Ouro, Huginn), not root-finding (DEQs stall at 250M). (b) LoRA-based perspectives are feasible (LoRA-Ensemble, PEGO, SMoRA).

### Which PDA assumptions are weakened or challenged

**Orthogonality ≠ semantic disentanglement**: Zhang et al. (2025) demonstrate that structural orthogonality alone does not produce meaningful perspective diversity. PDA must include semantic grounding beyond geometric separation.

**Diversity collapses without explicit enforcement**: Coda-Forno et al. (2025) show that simply providing a latent communication channel is insufficient. PLR proves diversity decays exponentially with depth. PDA needs strong contrastive or repulsive forces (PLR's InfoNCE losses, Repulsive Deep Ensembles).

**Root-finding convergence does not scale**: Pure DEQ-style Broyden/Anderson methods have not been scaled beyond 250M parameters. PDA should adopt simple iteration with truncated BPTT (Huginn/Ouro style) or energy minimization (EBTs/IRED) instead.

**Interpretability of latent reasoning is limited**: Lu et al. (Jul 2025) found "limited evidence of interpretable latent CoT" in Huginn, and "only marginal gains" beyond a point. PDA's deliberation process may be difficult to inspect or debug.

### Concrete architecture adjustments based on findings

1. **Replace DEQ root-finding with energy-based convergence.** Use EBT-style energy landscapes where each worker minimizes a shared energy function. The energy formulation naturally defines consensus (joint energy minimum) and provides a measurable convergence criterion.

2. **Adopt the Prelude-Recurrent-Coda backbone** from Huginn for each worker, with MoEUT-style mixture-of-experts in the recurrent block for parameter efficiency. Use entropy-regularized adaptive depth from Ouro/LoopLM.

3. **Implement perspective diversity via three complementary mechanisms**: (a) PEGO-style dual orthogonal regularization (preserve base + diversify workers) during training; (b) Conceptor-based or steering-vector-based perspective assignment at inference; (c) PLR-style contrastive losses to prevent diversity collapse.

4. **Use the Dynamic Notes Bus concept** from PDT as PDA's shared deliberation space. Workers read/write latent summaries to a shared workspace at synchronized boundaries, combining CoE's iterative communication with MoT's cross-attention merge mechanism.

5. **Consider hypernetwork-generated perspective weights** (Hyper-CL/Text-to-LoRA pattern) as an alternative or complement to LoRA-based perspectives. A single hypernetwork conditioned on "perspective embeddings" generates lightweight worker-specific modifications.

6. **Monitor convergence using spectral coherence** (from Language Through a Prism and SEA): when low-frequency spectral components align across workers, deliberation is converging on semantics, while high-frequency diversity is maintained for detail refinement.

7. **For SSM-based variants**, adopt Mamba-3 MIMO's multi-stream architecture with MvSSM-style cross-view interaction as a potentially more efficient and less fragile alternative to transformer-based PDA.

### Blind spots: what we did not search for but should

- **Neuroscience of parallel deliberation**: How do biological neural ensembles achieve consensus? Predictive coding, neural oscillation synchronization, and neural population coding literature may inform PDA's design.

- **Distributed consensus algorithms**: Classical computer science consensus (Paxos, Raft, Byzantine fault tolerance) could inform convergence protocol design, especially for robustness guarantees.

- **Quantum computing metaphors**: Superposition, measurement/collapse, and entanglement have direct analogs in PDA's design. The quantum cognition literature may offer useful formalisms.

- **Information bottleneck theory**: How much information should each worker retain vs. compress? The Information Bottleneck principle could guide the design of worker communication bandwidth.

- **Curriculum learning for iterative architectures**: How to train PDA's iterative convergence? Coconut's multi-stage curriculum is a start, but systematic work on curricula for convergent multi-agent systems is sparse.

- **Adversarial robustness of latent merging**: What happens when one worker's perspective is adversarially corrupted? Robustness literature for ensemble methods should be consulted.

- **Memory and context**: How does PDA's deliberation interact with long-context processing? The intersection of latent reasoning and retrieval-augmented generation is under-explored.

- **Hardware-aware architecture search**: What parallel deliberation structures map most efficiently to modern GPU/TPU architectures? The systems perspective (beyond S-LoRA and Occult) deserves deeper investigation.

- **Sparse autoencoders (SAEs) for deliberation monitoring**: While SAEs struggle to decompose steering vectors, they could be valuable for monitoring what each PDA worker is "thinking about" during deliberation, enabling interpretable convergence tracking.