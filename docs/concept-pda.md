# Parallel Deliberation Architecture

Status: Early idea stage, not yet tested.
Date: 2026-03-26, extended 2026-03-27
Authors: Toby, Claude

## Core idea

A model architecture in which reasoning does not happen sequentially
token-by-token, but as a parallel deliberation of multiple workers in vector
space.

The overall concept consists of three independent hypotheses with
increasing speculation:

### Hypothesis 1: Parallel reasoning (the core)
Multiple workers process the same input in parallel from different
perspectives. The results are merged and flow into the next step. A
transformer layer need not be a single forward pass — it can be several
parallel passes whose results are merged before things continue. This
hypothesis stands on its own and is testable with existing tools.

### Hypothesis 2: Dynamic parameter allocation (the extension)
If workers exist anyway, parameters can be distributed dynamically instead
of using fixed experts (an MoE upgrade). Not "you ARE the maths expert", but
"here is a maths question, take the matching weights". Requires Hypothesis 1
but is not a prerequisite for it.

### Hypothesis 3: Holistic output (speculation)
The output emerges as a whole rather than token by token (diffusion-like).
Currently the furthest from anything testable. Related to research on
diffusion-based language models (MDLM, SEDD), which struggle with similar
problems (the sequentiality of language vs. parallel generation).

## Library metaphor

A library (the parameter space). In it:
- 3 entrance staff: text -> word translation -> internal representation (embedding layer)
- n runners (workers): sent off with sub-questions, can access any shelf
- 1 side table (deliberation space): where runners merge results, discuss, iterate
- 1 control mechanism: emerges from the interaction (convergence), no separate module
- 3 exit staff: internal representation -> word translation -> text (unembedding layer)
- special runners: can "run out" (web search, tool calls in vector space)

## Differences from existing architectures

### vs. dense transformer
- Dense: 1 sequential forward pass through all layers, token by token
- PDA: parallel processing, dynamic, iterative, a result instead of a token

### vs. MoE (Mixture of Experts)
- MoE: a router picks fixed experts (fixed weight subsets), 1 per token, no communication
- PDA: workers dynamically get the parameters they need. Not "you ARE the
  maths expert" but "here is a maths question, take the matching weights".
  Workers are generic. Workers communicate with each other (deliberation).

### vs. Chain of Thought / Chain of Continuous Thought
- CoT: sequential reasoning in language (token output as intermediate steps)
- CCoT: reasoning in continuous vectors, but sequential
- PDA: reasoning in vectors, PARALLEL, with iteration

### vs. diffusion models
- Diffusion: iterative refinement of an overall result
- PDA: similar on the output side (a result instead of a token sequence),
  but with structured deliberation instead of undirected denoising steps

### More closely related work

The comparisons above cover the large architecture classes. But there are
works that sit closer to PDA, from which the concept must distinguish itself
more sharply:

**Parallel decoding (Medusa, EAGLE, Lookahead Decoding)**
- Generate multiple token candidates in parallel, then verify against the base model.
- Goal: inference speedup at the same output. No different reasoning.
- PDA: parallelism not for speed, but for perspective diversity. Different
  workers should deliver different results, not the same result faster.

**Multi-Head Latent Attention (MLA, e.g. DeepSeek-V2)**
- Compresses the KV cache via low-rank projection, multiple attention heads
  work on latent representations.
- Parallelism within a layer, but all heads see the same input and are
  trained jointly. No deliberation, no consensus.
- PDA: workers are more strongly decoupled (different steering vectors), and
  the merge is explicitly iterative rather than a learned layer.

**Tree of Thoughts (ToT) / Graph of Thoughts (GoT)**
- Multiple reasoning paths in parallel, with evaluation and backtracking.
- Operate at the token/text level, not in vector space.
- PDA: similar structure (parallel, then evaluate), but the deliberation
  happens in the activations, not in generated text.

**Best-of-N / majority voting**
- Generate multiple completions, pick the best or decide by majority.
- Full model runs, no interaction between the paths.
- PDA: workers share parameters and communicate via the deliberation space.
  Not "generate 10 answers and pick", but "think from 10 perspectives and
  converge to one".

## Architecture sketch

### Core architecture (Hypothesis 1: perspective-based)

```
Input (text)
    |
[Embedding layer: 3 layers, text -> internal representation]
    |
    +------------------+------------------+
    |                  |                  |
[Worker 1]       [Worker 2]    ...  [Worker n]
Steering vec A   Steering vec B      Steering vec N
    |                  |                  |
    +------------------+------------------+
                       |
                [Deliberation space]        <-- cross-attention between worker outputs
                       |
                [Measure divergence]        <-- cosine distance or similar
                       |
              Divergence < threshold?
              /                    \
            yes                    no
             |                      |
    [Consolidated               [Results + divergence info
     result]                     as new input -> next round]
             |
[Unembedding layer: 3 layers, internal representation -> text]
    |
Output
```

### Extended architecture (with Hypothesis 2: dynamic parameters)

```
Input (text)
    |
[Embedding layer]
    |
[Topic classifier]  -->  [Steering vector pool]
    |                              |
    +--- selects matching vectors -+
    |
[Workers 1..n with dynamically assigned steering vectors + parameter subsets]
    |
[Deliberation + iteration as above]
    |
Output
```

## Hardware vision

- 1 parameter server (GPU with lots of VRAM): holds all weights (e.g. 100B)
- n worker GPUs (small, e.g. 8GB): generic compute units
- coordinator (CPU or light GPU): orchestrates parameter distribution

Per iteration each worker receives only the parameters it needs (~8B subset).
Scaling by width (more small GPUs) instead of height (bigger GPUs).

Bottleneck: bandwidth from parameter server to workers.
8B fp16 = ~16GB. NVLink: ~18ms, PCIe 5.0: ~250ms.
Mitigation: streaming (compute during transfer), locality (reuse parameters).

## Open questions

### Critical (Hypothesis 1)
1. **Compatibility**: middle layers of existing models expect a specific
   input. Parallel processing + merging will probably break the
   representations. -> Retraining needed, at least for the deliberation mechanism.

2. **Training signal**: what do you optimise the deliberation on?
   Next-token prediction does not fit. Outcome-based RL? "Was this
   deliberation result better than without deliberation?"
   Partially addressed by consensus as an internal signal (see below).

3. **Steering vector suitability**: at what level of abstraction can
   meaningful perspective vectors be extracted? Tonality/truthfulness work.
   Semantic perspectives ("critical", "analytical") are still unclear.

### Important (Hypothesis 1)
4. **Convergence**: how to ensure the deliberation terminates?
   Halting based on a divergence threshold + rate of change + timeout.

5. **Merge mechanism**: how do you optimally combine worker outputs?
   Weighted average, concatenation + projection, cross-attention?
   Probably task-dependent — itself an open question.

6. **Evaluation**: on which tasks do you measure whether it is better?
   Reasoning benchmarks? Creative tasks? Open problems?

### Important (Hypothesis 2)
7. **Dynamic parameter allocation**: how do you decide which parameters a
   worker needs? Learned routing? Topic-based heuristic? Or: is it enough if
   all workers access the same full parameter set and the differentiation
   runs only through steering vectors?

8. **Tool integration**: tools currently return text. For vector-space
   integration, each tool needs an encoder into the internal representation space.

### Worth exploring
9. **Scaling behaviour**: does it improve linearly with more workers?
   Sublinearly? Is there a sweet spot?
10. **Emergent specialisation**: do generic workers develop specialisations
    through training?

## Consensus as an internal quality signal

Multiple workers can process the same sub-question in parallel. The degree
of agreement is an internal quality signal:

- **High consensus** (8/10 same conclusion): the solution is robust, pass
  the result on.
- **Low consensus** (5:5): either two equivalent answers or an
  underdetermined question. Both are valuable information. -> More
  iteration, or report uncertainty to the output.
- **No consensus**: the question may be ill-posed. -> Back to the
  coordinator for reformulation.

Difference from classic ensemble methods: ensemble = different models,
measures model variance. PDA = same parameters, different starting
points/questions, measures the stability of the solution space.

This partially solves the training-signal problem: consensus is an internal
signal, needs no external evaluator. It can serve as a component of the
halting mechanism (high consensus -> stop) and as a confidence signal in the
output.

### Limitation: shared blindness
Consensus is a stability and halting signal, but not a proof of quality. If
all workers use the same parameters and differ only through steering
vectors, they share the same learned biases. High consensus can mean "all
agree" OR "all are wrong in the same way".

Consequence: for evaluating the architecture, an external task benchmark is
indispensable (reasoning benchmarks, ground-truth comparisons). Consensus is
useful as a runtime heuristic (when to stop, how confident the result is),
but not as the sole quality criterion during development.

## Perspectives instead of task decomposition

Central design decision: workers do not get different sub-tasks, but
different perspectives on the same task.

### Why no task decomposition
An active coordinator that intelligently decomposes tasks must already
understand the task to do so — a reasoning problem BEFORE the actual
reasoning (chicken-and-egg). Also, a decomposed task has to be reassembled
afterwards, and the reassembly is often as hard as the problem itself.

### The perspective approach
Instead: all workers start with the same input, but with different steering
vectors. Not "you handle sub-question A", but "you look at the problem
through lens X". The diversity comes from the starting conditions, not from
an explicit task split.

Advantages:
- No learned coordinator needed
- Diversity comes from the initialisation
- Convergence comes from the deliberation space
- Reassembly is a convergence problem (mathematically tractable), not a
  recombination problem

### Fixed perspective vectors (stage 1)
The simplest approach: a fixed set of steering vectors, fixed assignment.
Each worker has its focus — e.g. critical, realistic, analytical, creative.
Similar to experts in MoE, but more lightweight: the worker is the same,
only the lens differs.

Open question: at what level of abstraction do steering vectors work
reliably? Existing research shows good results for tonality, language,
truthfulness. Whether high-level semantic perspectives ("critical",
"economic") can be cleanly separated in activation space is unclear. First
step: work with demonstrably extractable dimensions and check whether the
diversity in the output is still productive.

### Dynamic perspective assignment (stage 2)
A light classifier recognises the topic and selects matching steering
vectors from a predefined pool. Not a full coordinator, but a routing based
on simple topic recognition.

### Fully dynamic (stage 3)
Steering vectors are generated at inference time. Requires stages 1 and 2 to
deliver positive results.

## Iterative staged model

The deliberation runs in rounds:

```
Round 1: n workers process the input in parallel with different perspectives
         -> n result vectors
         -> measure divergence

         Divergence low?  -> pass the result on (consensus reached)
         Divergence high? -> results as new input in round 2

Round 2: n workers process the enriched input
         (original question + results + divergence info from round 1)
         -> measure divergence
         -> continue or stop

...

Round k: timeout (e.g. k=5 or k=10) -> use the result,
         communicate uncertainty in the output
```

Each round has more context than the previous one: the workers see not only
the original question, but also where the previous round disagreed. This is
iterative refinement with built-in information gain.

Halting criteria:
- Primary: divergence below threshold (consensus)
- Secondary: rate of change between rounds (convergence)
- Fallback: maximum round count (timeout)

## Signal-processing-based convergence metrics

Cosine similarity between worker outputs is one-dimensional: "similar or
not". Signal-processing concepts deliver a much richer picture for
evaluating the deliberation and the halting decision.

### SNR (signal-to-noise ratio)
From audio engineering: the ratio of useful signal to noise.

Transfer: the consensus direction (average of all worker outputs) is the
"signal", each worker's individual deviation from it is "noise".
SNR = ||mean(worker_outputs)||^2 / mean(||worker_i - mean||^2)

Track SNR over rounds:
- SNR rising -> the result is getting clearer, deliberation is productive
- SNR stagnating -> convergence reached, further rounds bring nothing
- SNR falling -> deliberation destabilising, stop immediately

Advantage over simple divergence: SNR measures not just WHETHER it
converges, but HOW FAST, and detects destabilisation early.

### Phase coherence
From multi-microphone setups: measures how strongly multiple signals
correlate, to separate signal from diffuse room sound.

Transfer: decompose worker outputs into their dominant components (SVD/PCA).
Per component, measure the coherence between workers.

- High coherence in a component -> workers agree on this aspect, that is
  stable "signal"
- Low coherence -> either noise (irrelevant) OR a productive perspective
  difference (the most interesting information)

Distinguishing "noise" from "productive difference": if the incoherent
components stay stable over rounds (the same workers always deviate in the
same direction), it is structured disagreement. If they fluctuate randomly
over rounds, it is noise.

### Crest factor
From audio analysis: the ratio of peak to RMS (average). Measures whether a
signal has concentrated peaks or is evenly distributed.

Transfer: applied to the divergence vector between workers.
Crest factor = max(|divergence|) / rms(divergence)

- High crest factor -> disagreement is concentrated in a few dimensions.
  Meaning: there is a specific point of contention. The workers agree
  overall, but disagree at one concrete spot.
  -> Investigate those dimensions specifically, or focus an additional
     iteration only on the contested aspect.

- Low crest factor -> disagreement is broadly spread.
  Meaning: general uncertainty, the workers have no clear consensus in any
  direction.
  -> More iteration, or communicate uncertainty in the output.

### Combination as a halting decision

The three metrics together give a multi-dimensional picture:

```
SNR high + coherence high + any crest factor
-> Strong consensus, stop, the result is robust.

SNR rising + coherence rising
-> Not done yet, but on a good path. Keep iterating.

SNR high + coherence low in individual components + crest factor high
-> Consensus overall, but a specific point of contention.
   Use the result, but communicate uncertainty specifically.

SNR low + coherence low + crest factor low
-> General uncertainty. Either keep iterating or output the result
   with a high uncertainty marker.

SNR falling (in any combination)
-> Deliberation destabilising. Stop immediately, use the last stable result.
```

## Experiment plan

### Basic principles

Each run is compared against a clear baseline: a normal single forward pass
of the same model on the same data. All variables (worker count, merge
strategy, round count) are varied one at a time, not simultaneously —
otherwise it is unclear what caused the difference.

A negative result for one configuration does not mean the core hypothesis is
refuted. There are several knobs (merge, vectors, count, layer selection),
and it only gets serious once no configuration shows an effect.

### Baseline

For each data point:
- A single forward pass through the full model, without modification
- Same model, same input, same decoding parameters
- Repeated multiple times (e.g. 10x per question) to measure base model variance

The baseline variance is decisive: if the base model already swings between
right and wrong over 10 runs, the PDA effect must clearly exceed that swing.

### Independent variables (what we change)

**A: Worker count**
- 2, 3, 5, 10 workers vs. baseline (1)
- Question: from which count does an effect appear? Is there a sweet spot
  beyond which more workers add no value or even hurt?

**B: Merge strategy**

Standard ML merges:
- Weighted average (simplest, baseline merge)
- Concatenation + linear projection back to original dimension
- Element-wise maximum (the strongest activation wins)
- Attention-based merge (worker outputs as keys/values, query from the average)

Signal-processing-based merges (from the audio-engineering analogy):
- Phase alignment: before the merge, measure cosine similarity between
  worker activations. In-phase parts (same direction in vector space)
  reinforce each other — constructive interference. Counter-phase parts mark
  spots where workers disagree — the interesting spots, which can be handled
  separately or used as an uncertainty signal.
- Frequency-selective merge: decompose activations into components via SVD
  or PCA. "Low frequencies" (dominant components, coarse semantics) simply
  averaged, "high frequencies" (fine differences, details) weighted more
  strongly or handled separately. Analogous to multiband processing in audio.
- Sidechain merge: an asymmetric merge in which one worker output modulates
  the other instead of an equal-weight combination. Element-wise
  multiplication or a learned gating function. One worker delivers the
  "what", the other controls the "how strongly". From the sidechain
  compression analogy (SWE).

All of these operations are differentiable (cosine similarity, SVD,
element-wise multiplication), so in principle integrable into a training
process.

Combinations: signal-processing merges can also be combined with standard
merges. E.g. phase alignment as preprocessing before an attention merge, or
frequency decomposition followed by a weighted average per band.

Question: which mechanism preserves the most information with the least
representation breakage? Do signal-processing-based approaches deliver more
robust results than naive averaging?

**C: Round count (iteration)**
- 1 round (pure merge, no feedback)
- 2, 3, 5, 10 rounds (merge result as new input)
- Question: does the divergence converge between rounds? From which round
  are the changes only minimal? Does task quality improve with more rounds
  or stagnate/degrade?

**D: Steering vectors**
- Without steering vectors (only different random seeds / dropout masks as diversity)
- Known, validated dimensions (e.g. a truthfulness vector from existing research)
- Semantic perspectives (critical, analytical, creative — if extractable)
- Question: do you need steering vectors at all for the effect, or is pure
  perturbation enough? And if so, what kind of vectors helps?

**E: Layer selection**
- Only middle layers (e.g. layers 8-16 in a 24-layer model)
- Early layers (1-8)
- Late layers (16-24)
- Question: in which layers is the merge most productive?
  Hypothesis: middle layers, because the most abstract representations sit there.

### Dependent variables (what we measure)

**Functionality (does it run at all)**
- Perplexity of the output (measured by a reference model): is the output
  linguistically coherent or degenerate?
- Share of degenerate outputs (repetition, nonsense, empty outputs)

**Task quality (is it better)**
- Accuracy on benchmarks with ground truth:
  - GSM8K (multi-step maths reasoning)
  - ARC-Challenge (scientific reasoning)
  - TruthfulQA (resistance to common misconceptions)
  - HellaSwag (commonsense completion)
- Accuracy difference from the single pass (delta), not just absolute value

**Robustness (is it more stable)**
- Variance of accuracy over multiple runs of the same question
- Share of cases where PDA is right and the single pass wrong (and vice versa)
- Consistency: how often does the same setup give the same result on repetition?

**PDA-specific (deliberation dynamics)**

Basic metrics:
- Divergence between worker outputs (cosine distance in activation space)
- Correlation between consensus and task accuracy (is high consensus
  actually a good predictor of correct answers, or a case of shared
  blindness?)

Signal-processing metrics (see section above):
- SNR over rounds: rising (productive convergence), stagnating (done), or
  falling (destabilisation)?
- Phase coherence per component: in which dimensions do workers agree, in
  which not? Is incoherence stable (structural disagreement) or random (noise)?
- Crest factor of the divergence: is disagreement concentrated (a specific
  point of contention) or diffuse (general uncertainty)?
- Combination: which of the three metrics correlates most strongly with task
  accuracy? Which is best suited as a halting criterion?

### Order of execution

```
Phase 1: Feasibility
- 1 model (e.g. Qwen 2.5 1.5B)
- 2 workers, 1 round
- Only weighted average as the merge
- Only middle layers
- Without steering vectors (random perturbation)
- 50-100 questions from GSM8K
- Question: does coherent output come out at all?

Phase 2a: Standard ML merges (if Phase 1 positive)
- Same setup, but: compare average, concatenation, max, attention
- Identify the best standard strategy

Phase 2b: Signal-processing merges
- Test phase alignment, frequency-selective, sidechain
- Also combinations (e.g. phase alignment + attention)
- Compare against the best standard strategy from 2a

Important: if Phase 1 fails with averaging (output degenerates), still run
Phase 2b. The naive average is the most questionable merge mechanism; phase
alignment or a frequency-selective merge could solve exactly the problems
that occur with averaging (destructive interference, blurring of representations).

Phase 3: Worker count scaling
- Best merge strategy from Phase 2
- Workers: 2, 3, 5, 10
- Identify the sweet spot

Phase 4: Steering vectors
- Best setup from Phase 2+3
- Compare different steering vector types
- Question: does targeted perspective diversity bring more than random perturbation?

Phase 5: Iteration + convergence metrics
- Best setup from Phase 2+3+4
- 1, 2, 3, 5, 10 rounds
- Measure SNR, phase coherence, crest factor per round
- Convergence curve: identify the point of diminishing returns
- Comparison: which metric correlates best with task accuracy?
- Derive a halting rule: e.g. "stop when SNR stagnates AND crest factor
  is below threshold"

Phase 6: Broad evaluation
- Best overall setup on all benchmarks
- Compare different task types (where does PDA help, where not?)
- Optional: a second model for comparison
```

### Result scenarios

**Scenario A: Phase 1 fails (output degenerates)**
-> Move directly to Phase 2b (signal-processing merges). The weighted
   average is the most naive merge and the most prone to destructive
   interference. Phase alignment or a frequency-selective merge address
   exactly this problem by steering the merge instead of blindly averaging.
   Also vary layer selection.

**Scenario B: Phase 1 works, but no quality gain**
-> Still run Phase 4 (steering vectors). Without targeted perspective
   diversity, a pure random merge may be only noise. The effect could appear
   only with meaningful vectors.

**Scenario C: Quality gain, but no robustness improvement**
-> Still interesting. Means: a parallel merge can beat the baseline, but not
   reliably. The question then becomes: can iteration (Phase 5) increase the
   reliability?

**Scenario D: Positive on reasoning, neutral on facts**
-> The expected result. Would confirm that PDA helps on tasks that benefit
   from perspective diversity, not on pure recall.

**Scenario E: All negative across all configurations**
-> The core hypothesis is then genuinely weakened. But even that would be a
   result: "parallel perspectives in the activation space of existing models
   bring no measurable advantage" is a usable statement.

## Connection to other projects

- Experiment "thought-process framing" (2026-03-25): showed that processing
  = language. PDA would address exactly this limit.
- KV-injection research (Toby): the technical basis for vector-space manipulation.
- Self-referential processing has measurable activation patterns. Could
  serve as a reference point for "what happens in the middle layers".
- Semantic Wave Encoding (SWE, Toby): audio-engineering analogies for
  vector-space operations. Concepts like phase alignment, frequency
  decomposition, and sidechain modulation as a thinking frame for merge
  strategies. The SWE experience showed: not all sub-ideas worked, but
  individual concepts (e.g. phase as a negation marker) were usable. Same
  expectation here: even if PDA as a whole architecture does not work out,
  individual merge mechanisms or the consensus signal can be useful on their own.

## Related documents

- [concept-n-pda.md](concept-n-pda.md) — n-PDA (Native PDA): a greenfield
  architecture that builds parallel deliberation in natively instead of
  layering it onto existing models. Orthogonal perspective spaces,
  cross-attention, a Deep Equilibrium motor. A thought experiment: which
  mathematical properties does parallel deliberation need to work cleanly?
- [simulations-roadmap.md](simulations-roadmap.md) — staged plan to validate
  the mathematical foundations BEFORE experimenting on existing models.

### Recommended order
The PDA experiments on existing models (experiment plan above) are possibly
not the best first step. The simulation route tests the core maths in a
controlled environment and delivers solid results faster. Recommendation:

1. Work through the simulations roadmap (weeks, not months)
2. Decide on the basis of the results: the PDA route or the n-PDA route
3. Run the PDA experiment plan (above) only if the simulations are positive
   AND probing shows that existing representations are decomposable
