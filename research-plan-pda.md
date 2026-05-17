# Rechercheplan: Parallele Deliberation und verwandte Architekturen

Status: Entwurf
Datum: 2026-03-28
Zweck: Systematische Literaturrecherche fuer einen Agenten

## Kontext

Wir entwickeln eine Architektur-Idee namens PDA (Parallel Deliberation Architecture):
Mehrere Worker verarbeiten denselben Input parallel aus verschiedenen Perspektiven
im Vektorraum (nicht auf Token-Ebene), iterieren ueber einen Deliberationsraum,
und konvergieren zu einem Ergebnis. Siehe parallel-deliberation-architecture-2.md
und n-pda-native-architecture.md fuer Details.

Zentrale Fragen der Recherche:
1. Was gibt es bereits, das in diese Richtung geht?
2. Welche Technologien koennten PDA ermoeglichen oder vereinfachen?
3. Welche alternativen Wege zum selben Ziel existieren, die robuster sein koennten?

## Suchrichtung 1: Direkt verwandte Arbeiten

### Was wir suchen
Arbeiten die paralleles Reasoning im Vektorraum / Aktivierungsraum beschreiben.
Nicht Token-Level-Ensemble (Best-of-N, Majority Voting), sondern Zusammenfuehrung
auf Repraesentationsebene.

### Suchbegriffe (kombinieren, variieren)
- "parallel reasoning" + "latent space" / "activation space" / "representation space"
- "multi-perspective reasoning" + transformer / LLM
- "deliberation" + "neural network" / "language model" (nicht RL-Deliberation)
- "activation merging" / "representation merging" (nicht Weight-Merging)
- "internal debate" + neural / latent
- "consensus" + "neural" + "reasoning"
- "multi-view reasoning" + transformer
- "collaborative inference" (nicht federated learning)
- "mixture of thoughts" / "parallel thoughts"
- "latent reasoning" + "multiple paths"

### Bekannte verwandte Arbeiten (als Ausgangspunkt fuer Zitationsnetzwerk)
- Medusa, EAGLE (Parallel Decoding -- Abgrenzung: Speed, nicht Perspektiven)
- Tree of Thoughts, Graph of Thoughts (Abgrenzung: Token-Ebene)
- Chain of Continuous Thought (Abgrenzung: sequentiell, nicht parallel)
- DeepSeek-V2 MLA (Multi-Head Latent Attention)
- MDLM, SEDD (Diffusion-basierte Sprachmodelle)
- Coconut (Chain of Continuous Thought, Meta 2024)

### Besonders relevant waere
- Arbeiten die Worker/Agents im Aktivierungsraum kommunizieren lassen
- Merge-Mechanismen fuer Aktivierungen (nicht Weights)
- Iterative Verfeinerung von Repraesentationen mit mehreren Perspektiven

## Suchrichtung 2: Enabling-Technologien

### 2a: Deep Equilibrium Models (DEQ)
- Aktuelle Fortschritte bei DEQ-Training (Stabilitaet, Skalierung)
- DEQ + Transformer: gibt es Arbeiten die DEQs auf LLM-Skala bringen?
- Alternativen zu DEQs die aehnliche Fixpunkt-Konvergenz bieten
  aber leichter trainierbar sind
- Suchbegriffe: "deep equilibrium" + "language model" / "transformer" / "scaling",
  "implicit layer" + transformer, "fixed point" + "neural network" + 2024/2025/2026

### 2b: Orthogonale Repraesentationen
- Orthogonale Regularisierung in Transformer-Training
- Disentangled Representations in LLMs
- Subspace-basierte Methoden fuer Repraesentationslernen
- Suchbegriffe: "orthogonal regularization" + transformer,
  "disentangled representations" + "language model",
  "subspace learning" + transformer,
  "representation decomposition" + LLM

### 2c: LoRA-Merging und Multi-LoRA
- Fortschritte bei TIES, DARE, SLERP und Nachfolger
- Multi-LoRA Inference (mehrere LoRAs gleichzeitig aktiv)
- LoRA als gelernte Perspektiven / Experten
- Orthogonale LoRA-Methoden
- Suchbegriffe: "LoRA merging" + 2025/2026, "multi-LoRA inference",
  "LoRA ensemble", "orthogonal LoRA", "LoRA composition"

### 2d: Signalverarbeitung in ML
- Anwendung von DSP-Konzepten auf neuronale Repraesentationen
- Frequenzbasierte Analyse von Aktivierungen
- Phase/Kohaerenz-Metriken in ML-Kontexten
- Suchbegriffe: "signal processing" + "neural representations",
  "frequency analysis" + activations + transformer,
  "spectral analysis" + "language model",
  "coherence" + "neural" + "ensemble"

### 2e: Steering Vectors und Repraesentations-Engineering
- Fortschritte seit Representation Engineering (Zou et al. 2023)
- Steering auf semantischer Ebene (nicht nur Tonalitaet/Safety)
- Komposition von Steering Vectors
- Suchbegriffe: "representation engineering" + 2025/2026,
  "steering vectors" + "composition" / "combination",
  "activation steering" + "semantic",
  "concept vectors" + LLM + 2025/2026

## Suchrichtung 3: Alternative Architekturen (WICHTIG)

### Was wir suchen
Ansaetze die dasselbe Ziel erreichen koennten wie PDA -- paralleles, perspektivreiches
Reasoning -- aber auf einem anderen, moeglicherweise robusteren Weg. Explizit NICHT
nur das was wir schon kennen, sondern Ideen die unsere Annahmen in Frage stellen.

### Konkrete Alternativen zum Erkunden
- **State Space Models (Mamba, etc.) mit parallelen Streams**: SSMs haben
  andere Rechenstruktur als Transformer. Kann man mehrere SSM-Streams
  parallel laufen lassen und mergen? Weniger fragil als Transformer-Hacks?
- **Hypernetworks**: Ein Netzwerk generiert die Weights fuer ein anderes.
  Koennte ein Hypernetwork perspektiv-spezifische Weights generieren?
- **Modular Networks / Routing**: Jenseits von MoE -- dynamisch zusammengesetzte
  Netzwerke die pro Input verschiedene Pfade nehmen
- **Neural ODEs / Continuous-Depth Networks**: Verwandt mit DEQs aber
  mit kontinuierlicher Dynamik statt Fixpunkt. Moeglicherweise stabiler.
- **Energy-Based Models fuer Konsensus**: Statt iterativer Konvergenz
  ein Energieminimum als Konsensus definieren
- **Diffusion im Repraesentationsraum**: Nicht Diffusion fuer Output-Generierung,
  sondern als iterativer Deliberationsmechanismus
- **Sparse Mixture of Experts mit Kommunikation**: MoE wo die Experten
  miteinander reden (existiert das?)
- **Multi-Agent RL im Parameterraum**: Agenten die im selben Netzwerk
  kooperieren statt auf Token-Ebene
- **Recurrent Transformer / Universal Transformer**: Weight-Sharing ueber
  Layer als Form von Iteration -- Verbindung zu DEQ aber praktisch erprobt

### Suchbegriffe
- "multi-stream" + transformer / "state space model" / Mamba
- "hypernetwork" + "perspective" / "conditional computation" + LLM
- "modular neural network" + "language model" + 2025/2026
- "neural ODE" + transformer / "continuous depth" + LLM
- "energy based model" + "consensus" / "agreement"
- "latent diffusion" + "reasoning" / "deliberation"
- "communicating experts" / "expert communication" + "mixture of experts"
- "multi-agent" + "single model" / "shared parameters"
- "universal transformer" + 2025/2026, "recurrent transformer" + scaling
- "iterative refinement" + "language model" + "latent"

## Quellen

### Primaer
- arXiv (cs.CL, cs.LG, cs.AI) -- Preprints, aktuellste Arbeiten
- Semantic Scholar -- Zitationsnetzwerk, "cited by" fuer bekannte Arbeiten
- Google Scholar -- breitere Abdeckung

### Sekundaer
- Papers with Code -- Implementierungen und Benchmarks
- OpenReview -- ICLR, NeurIPS, ICML Submissions und Reviews
- Hugging Face Blog / Papers -- Community-Arbeiten
- Blogposts von Forschungsgruppen (DeepMind, Meta FAIR, Anthropic, etc.)

### Suchstrategie
1. Keyword-Suche ueber alle primaeren Quellen
2. Fuer jedes relevante Paper: "Cited by" und "References" pruefen
3. Autoren relevanter Papers: Was haben sie sonst veroeffentlicht?
4. Bei Konferenz-Papers: Co-located Workshop Papers pruefen
   (oft experimentellere Ideen)

## Output-Format

Pro gefundenem Paper / Arbeit:

```
### [Titel] (Autoren, Datum)
- **Quelle**: Link
- **Kernidee**: 2-3 Saetze
- **Relevanz fuer PDA**: Welcher Aspekt ist relevant? (Architektur / Merge /
  Konvergenz / Alternative / Enabling)
- **Was koennen wir uebernehmen**: Konkreter Aspekt
- **Einschraenkungen / Unterschiede**: Wo passt es nicht
- **Weiter verfolgen**: ja/nein + Begruendung
```

Ergebnisse gruppiert nach den drei Suchrichtungen, innerhalb nach Relevanz sortiert.

Am Ende: Eine kurze Synthese (max 1 Seite) mit:
- Die 5-10 vielversprechendsten Arbeiten und warum
- Welche PDA-Annahmen durch die Recherche gestaerkt oder geschwaecht werden
- Konkrete Vorschlaege fuer Architektur-Anpassungen basierend auf den Funden
- Blinde Flecken: Wonach wir nicht gesucht haben, aber vielleicht sollten
