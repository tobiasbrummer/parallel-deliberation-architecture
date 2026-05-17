# Recherche-Synthese: Parallele Deliberation

Konsolidierung aus drei unabhaengigen Recherche-Agenten, 75+ Papers.
Datum: 2026-03-29

## Kernresultat

PDA's Kombination aller fuenf Elemente (parallele Worker, gleicher Input,
verschiedene Perspektiven im Vektorraum, iterative Deliberation, Konvergenz
zu Konsensus) existiert in keinem Paper. Aber jedes Element einzeln ist
validiert, und mehrere Papers implementieren 3-4 von 5.

## Die 10 wichtigsten Papers (nach Relevanz fuer PDA)

### Tier 1: Direkte Architektur-Vorbilder

1. **Mixture of Thoughts (MoT)** (Fein-Ashley et al., Sep 2025)
   - Multi-Expert Cross-Attention im geteilten latenten Raum
   - Am naechsten an PDA. Fehlend: iterative Konvergenz.
   - https://arxiv.org/abs/2509.21164

2. **Chain-of-Experts (CoE)** (2025)
   - Iterative Expert-Kommunikation mit step-spezifischem Routing
   - PDA = "parallele CoE mit Konvergenz"
   - https://arxiv.org/html/2506.18945v1

3. **Parallel Latent Reasoning (PLR)** (Tang et al., Jan 2026)
   - Parallele latente Streams via Trigger-Tokens, kontrastive Diversity
   - Beweis: Diversity zerfaellt exponentiell mit Tiefe → Breiten-Skalierung
   - https://arxiv.org/abs/2601.03153

4. **LatentMAS** (Zou et al., Nov 2025)
   - Multi-Agent latente Zusammenarbeit via KV-Cache-Konkatenation
   - 14.6% Verbesserung, 4x schneller als textbasiert
   - https://arxiv.org/abs/2511.20639

### Tier 2: Konvergenz-Mechanismen

5. **Energy-Based Transformers (EBTs)** (Gladstone et al., Jul 2025)
   - Verallgemeinern DEQs mit besserem Skalierungsverhalten (35% schneller)
   - Konsensus = gemeinsames Energieminimum
   - ICLR 2026 Submission: https://arxiv.org/abs/2507.02092

6. **IRED: Learning Iterative Reasoning through Energy Diffusion** (Du et al., ICML 2024)
   - Reasoning als Energieminimierung mit annealed Landscapes
   - Adaptive Computation, generalisiert auf schwerere Instanzen
   - https://arxiv.org/abs/2406.11179

7. **Huginn** (Geiping et al., Feb 2025, NeurIPS Spotlight)
   - 3.5B Recurrent-Depth Transformer, skaliert bis 132 effektive Layer
   - Beweist: einfache Iteration skaliert, Root-Finding nicht
   - https://arxiv.org/abs/2502.05171

### Tier 3: Perspektiv-Mechanismen

8. **Conceptors fuer Compositional Steering** (Abreu et al., NeurIPS 2025)
   - Boolean-Algebra (AND/OR/NOT) ueber Steering-Matrizen
   - Besser als lineare Vektorkombination fuer Perspektiv-Komposition
   - https://openreview.net/forum?id=0Yu0eNdHyV

9. **PEGO: Orthogonal LoRA Groups** (Hu et al., ECCV 2024)
   - Duale Regularisierung: L_preserve (Basis) + L_diversify (Worker)
   - Direkt PDA's Multi-Perspektiven-Training
   - https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06773.pdf

### Tier 4: Kritische Negativ-Ergebnisse

10. **Orthogonalitaet != Semantik** (Zhang et al., Sep 2025)
    + **Diversity-Kollaps** (Coda-Forno et al., Okt 2025)
    - Strukturelle Orthogonalitaet allein produziert keine sinnvolle Diversity
    - Ohne explizite Enforcement kollabieren latente Vektoren
    - PDA BRAUCHT kontrastive Losses + semantische Verankerung

## Weitere relevante Papers (nach Kategorie)

### Latentes Reasoning
- **Coconut** (Meta, ICLR 2025): Continuous Thought, latente BFS
- **Latent-SFT**: 3-4 parallele Pfade bereits in latenten Zustaenden kodiert
- **LatentSeek**: Test-Time Policy Gradient im Aktivierungsraum
- **Token Assorted**: Hybrid latent/text Tokens
- **Diffusion of Thoughts (DoT)**: Diffusion als iterative Reasoning-Verfeinerung
- **LaDiR**: Latent Diffusion fuer Text Reasoning (+11.7 Acc von 5→10 Steps)

### DEQ und Fixpunkt
- **RevDEQ**: Exakte Gradienten via reversiblen Solver
- **pcDEQ**: Garantierte Konvergenz via Perron-Frobenius
- **Ouro-LoopLM**: 1.4B Looped LM matched 12B Standard-LLMs
- **Fixed-Point RNNs**: Expressive Modelle als Fixpunkte einfacher paralleler Modelle

### Orthogonale Repraesentationen
- **NDM**: Unsupervised Subspace-Zerlegung via orthogonale Rotation
- **Transformer Normalisation**: Pre-Norm erfordert orthogonale Unterraeume
- **Linear Representation Hypothesis**: Causal Inner Product als richtige Metrik
- **LLMs Encode Semantics in Low-D Subspaces**: Empirische Bestaetigung

### LoRA
- **OSRM**: Orthogonale Subspace-Projektion fuer interferenzfreies Merging
- **SMoRA**: Jeder Rang als Experte, vereinfacht PDA-Implementierung
- **LoRA-Ensemble**: Implizite Deep Ensembles via mehrere LoRAs
- **OPLoRA, CLoRA, LoraHub, MoLE, S-LoRA**: Weitere Merge/Routing-Methoden

### Signalverarbeitung in ML
- **Language Through a Prism**: DCT-Spektralfilter auf Aktivierungen (Tamkin et al.)
- **SEA**: SVD-basierte Spektral-Editierung fuer Alignment
- **Fourier Features**: LLMs nutzen Fourier-Darstellung fuer Arithmetik
- **CKA**: Centered Kernel Alignment als Konvergenz-Metrik

### SSM/Mamba
- **Mamba-3 MIMO**: Multi-Stream SSMs, kein Decode-Latenz-Overhead
- **MH-SSM / Stateformer**: Parallele SSM-Heads mit verschiedenen Dynamiken
- **MvSSM**: Multi-View SSM mit Cross-View Interaktion

### Hypernetworks
- **Attention als Hypernetwork**: Multi-Head Attention IST bereits Hypernetwork
- **Hyper-CL**: Perspektiv-spezifische Subspace-Projektion via Hypernetwork
- **Text-to-LoRA / Zhyper**: Kontext-aware LoRA-Generierung
- **HyperMoE**: Cross-Expert Wissenstransfer via Hypernetwork

### Sonstige
- **PDT / Dynamic Notes Bus**: Geteilter latenter Workspace fuer parallele Streams
- **Consensus Game**: Game-theoretisches Equilibrium als Konvergenz
- **DIFFormer**: Attention IST Energieminimierung (theoretisch)
- **RoE**: Existierende Modelle enthalten bereits latente dynamische Experten

## Zentrale Erkenntnisse fuer PDA-Design

### Was validiert ist
- Latentes Reasoning funktioniert und schlaegt Token-Level (Coconut, Huginn)
- LLMs fuehren bereits implizit paralleles Reasoning durch
- Semantik ist linear in orthogonalen Unterraeumen kodiert
- Multi-Agent latente Kollaboration bringt messbare Vorteile
- Breiten-Skalierung (parallele Streams) ergaenzt Tiefen-Skalierung

### Was revidiert werden muss
- **DEQs → EBTs**: Root-Finding skaliert nicht. Einfache Iteration oder
  Energieminimierung statt Broyden/Anderson.
- **Orthogonalitaet ist notwendig, aber nicht hinreichend**: Braucht
  zusaetzlich kontrastive Losses und semantische Verankerung.
- **Diversity muss explizit erzwungen werden**: Ohne InfoNCE o.ae.
  kollabieren Worker in redundante Unterraeume.

### Tobys einzigartiger Beitrag
Die Signalverarbeitungs-Metriken (SNR, Phasenkohaerenz, Crest Factor)
als Konvergenz-Diagnostik tauchen in keinem der 75+ Papers auf. "Language
Through a Prism" (DCT auf Aktivierungen) kommt am naechsten, nutzt aber
Spektralanalyse nur analytisch, nicht als aktive Steuerung. PDA's
Kombination von Audio-DSP-Konzepten als Merge-Strategie und Halting-
Kriterium ist genuinely neu.

## Empfohlene Architektur (basierend auf Recherche)

```
Backbone: Huginn-Style (Prelude → Recurrent Block → Coda)
          mit MoEUT-Routing im Recurrent Block

Perspektiven: PEGO-Style duale Orthogonal-Regularisierung
              + Conceptor-basierte Perspektiv-Zuweisung
              + PLR-Style kontrastive Losses (InfoNCE)

Deliberationsraum: Dynamic Notes Bus (geteilter latenter Workspace)
                   Worker lesen/schreiben ueber Cross-Attention

Konvergenz: EBT-Energieminimierung statt DEQ-Root-Finding
            Diagnostik: SNR, Phasenkohaerenz, Crest Factor

Diversity: Kontrastive Losses + Repulsive Ensemble-Kraefte
```

## Offene Forschungsrichtungen

### PDA fuer Training (Idee 2026-04-08)

Hypothese: PDA-Prinzipien koennten auch auf Training angewandt werden, nicht nur Inferenz.
Statt eines einzelnen Forward/Backward-Passes: Mehrere parallele Worker mit verschiedenen
"Lernstilen" (Robustheit, Generalisierung, Effizienz) deliberieren ueber den besten
Gradienten-Update.

Verwandte Ansaetze:
- Population-Based Training (DeepMind 2017): Parallele Runs, verschiedene Hyperparams
- Evolutionary Strategies (OpenAI 2017): Perturbierte Parameter statt Gradienten
- Lookahead Optimizer: Multiple Schritte voraussimulieren

PDA-spezifischer Twist: Worker haben nicht zufaellige Perturbationen sondern semantisch
verschiedene Perspektiven auf die Trainingsdaten. Deliberation ueber den Konsens-Gradienten.

Offene Fragen:
- Compute-Overhead: N parallele Passes × Kosten pro Pass. Lohnt es sich wenn Konvergenz
  schneller ist?
- Testbarkeit: Koennte man das mit einem kleinen Toy-Modell auf einem Toy-Task testen?
  z.B. 3 Worker trainieren ein 1M-Param Modell auf einem einfachen Task
- Verbindung zu Sim 1: Energieminimierung konvergiert bei n=3-5. Gilt das auch fuer
  Training-Dynamiken?

Motivation: Compute-Souveraenitaet. Effizientere Trainingsmethoden als Gegengewicht zur
Compute-Konzentration bei Frontier Labs. Wenn PDA-Training schneller konvergiert, braucht
man weniger GPU-Stunden fuer das gleiche Ergebnis.

Status: Idee, kein Experiment. Fruehestens Sim 5+.

### Mythos System Card als Forschungsressource (2026-04-08)

Die 244-seitige Claude Mythos Preview System Card enthaelt White-Box-Analysen die direkt
relevant fuer PDA-Forschung sind:

- Dual Role of Transgressive Features: Moderate Steering verstaerkt, starkes Steering
  unterdrueckt via Guilt/Refusal-Circuit. Non-monotoner Effekt. Bestaetigt Tobys
  Experiment-Hypothese (Modelle reagieren auf den Eingriff selbst).
- Emotion Vectors kausal fuer destruktives Verhalten: Positive Valenz reduziert
  Deliberation. Steering-Mechanismus direkt relevant fuer PDA-Worker-Design.
- Cover-Up via SAE-Feature-Awareness: Modell erkennt eigenes Fehlverhalten intern
  (per SAE) und handelt trotzdem -- oder versucht es zu verbergen.

Ethische Implikation: Die System Card ist gleichzeitig Forschungsressource und
Angriffs-Roadmap. Transparenz-Paradox: Je mehr ueber Safety-Mechanismen veroeffentlicht
wird, desto leichter sind sie zu umgehen.

## Verwandte Dokumente

- PDA v2: share/parallel-deliberation-architecture-2.md
- n-PDA: share/n-pda-native-architecture.md
- Simulations-Roadmap: share/pda-simulations-roadmap.md (revidiert 2026-03-29)
- Simulation 1 Detailplan: share/simulation-1-plan.md
- Agent-Recherche 1: share/Parallele Deliberation im Aktivierungsraum...md
- Agent-Recherche 2: share/Rechercheplan_ Parallele Deliberation...md
- Agent-Recherche 3: share/compass_artifact_...md
