# Parallele Deliberation im Aktivierungsraum und verwandte Architekturen

## Executive Summary

Dieser Bericht untersucht aktuelle Forschungsarbeiten zu parallelem Reasoning im latenten/aktivierungsbasierten Raum von Sprachmodellen, zu Enabling-Technologien (DEQs, orthogonale Repräsentationen, LoRA-Merging, Signalverarbeitung, Representation Engineering) sowie zu alternativen Architekturen (SSMs/Mamba, Hypernetworks, modulare Netze, Diffusion, Energy-Based Models etc.).
Es zeigt sich, dass es inzwischen mehrere konkrete Ansätze zu **latenter bzw. kontinuierlicher Reasoning-Dynamik** gibt (Coconut/Chain of Continuous Thought, Latent-SFT, LatentSeek, Token Assorted, Latent TTS), erste explizite Frameworks für **Parallel Latent Reasoning (PLR)** existieren und dass Repräsentations-Steuerung und Aktivierungs-Merging in mehreren Linien (RepE, Activation-Informed Merging, Deep Model Merging) aktiv erforscht werden.[^1][^2][^3][^4]
Parallel dazu entwickeln sich Deep Equilibrium Networks und State-Space-Modelle (Mamba-2, Mamba-3) zu ernsthaften Transformeralternativen mit intrinsischer Fixpunkt- bzw. rekurrenter Dynamik, die für eine PDA-ähnliche Architektur als Backbones interessant sind.[^5][^6][^7][^8]

Kernresultat: Es gibt **kein** Papier, das genau die im PDA-Plan skizzierte Architektur (mehrere Worker, die denselben Input im Aktivierungsraum parallel aus unterschiedlichen Perspektiven verarbeiten und ihre Aktivierungen iterativ mergen) implementiert; aber es existieren mehrere Bausteine und eng verwandte Richtungen, auf denen eine PDA-Implementierung aufsetzen kann.[^3][^4][^9]

***

## 1. Direkt verwandte Arbeiten zu latentem und parallelem Reasoning

### 1.1 Coconut – Chain of Continuous Thought

**Training Large Language Models to Reason in a Continuous Latent Space (Coconut)** führt das Konzept der "continuous thought"-Vektoren ein: Statt textuelle Chain-of-Thought-Sequenzen zu generieren, wird der letzte Hidden State als latenter Reasoning-Status genommen und mehrfach in den eigenen Input zurückgeführt.[^10][^11][^4]
Dadurch entsteht eine iterative Reasoning-Dynamik **im latenten Raum**, die in Experimenten auf logischen Benchmarks bei kürzeren Sequenzen bessere Genauigkeit als klassische CoT-Ansätze erreicht und eine Art latente Breitensuche über mehrere mögliche Reasoning-Pfade erlaubt.[^12][^10]

**Relevanz für PDA:**
- Coconut zeigt, dass sich Reasoning vollständig im kontinuierlichen Aktivierungsraum abwickeln lässt, ohne Token-level CoT.
- Die iterative Rückkopplung des latenten Zustands ist konzeptionell nahe an einem "Deliberationsraum" mit Fixpunktkonvergenz.

**Was übernehmbar ist:**
- Design-Pattern "letzter Hidden State als Thought-Vektor", der wieder in den Input-Embedding-Stream eingespeist wird.
- Training mit latentem Reasoning-Head, der Belohnungs-/Rewardähnliche Signale im latenten Raum nutzt.

**Einschränkungen:**
- Coconut realisiert typischerweise **einen** latenten Reasoning-Pfad, der in sich latent parallel ist, aber nicht explizit mehrere Worker-Agenten mit separaten Perspektiven modelliert.[^10]

**Weiter verfolgen:** Ja – Coconut ist eine der Hauptreferenzen für kontinuierliches Reasoning und liefert direkt nutzbare Mechanismen für eine PDA-Iteration.

### 1.2 Latent-SFT – Latent Reasoning als Superposition von Pfaden

**Latent-SFT: Latent Reasoning in LLMs as a Vocabulary-Space Superposition** interpretiert latentes Reasoning als Superposition mehrerer Reasoning-Chains im Spaltenraum der Vokabular-Matrix („Hilbertraum“-Analogie).[^2]
Die Arbeit definiert metrische Größen wie "effective global parallelism" und zeigt empirisch, dass latente Reasoning-Zustände Informationen zu **3–4 parallelen Reasoningpfaden** tragen, statt nur einen Pfad zu repräsentieren.[^2]

**Relevanz für PDA:**
- Stützt die Annahme, dass latente Zustände natürlicherweise **parallelisierte** Reasoning-Inhalte kodieren.
- Liefert eine quantitative Sicht auf Parallelität (Parallelismus-Metriken), die als Diagnostik für eine PDA-Implementierung dienen kann.

**Was übernehmbar ist:**
- Metriken für Parallelismus und Kompression latenter Reasoning-Pfade.
- Idee, dass Parallelität bereits implizit in der latenten Repräsentation existiert und explizite Worker evtl. nur diese Struktur entpacken/ausnutzen müssen.

**Einschränkungen:**
- Die Arbeit zeigt Parallelität, nutzt sie aber primär analytisch; ein expliziter Multi-Worker-Mechanismus wird nicht implementiert.[^2]

**Weiter verfolgen:** Ja – insbesondere als theoretischer Unterbau für Parallelitätsmetriken im Aktivierungsraum.

### 1.3 LatentSeek – Policy-Gradient-Update im Latent Space

**Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space (LatentSeek)** optimiert latente Repräsentationen testzeitlich per Policy Gradient.[^1]
Der Ansatz aktualisiert die eingebetteten Zustände iterativ anhand eines selbstgenerierten Rewards, ohne die Modellgewichte zu verändern, und erreicht auf GSM8K, MATH-500 und AIME24 bessere Ergebnisse als klassische CoT- und Fine-Tuning-Methoden.[^1]

**Relevanz für PDA:**
- Zeigt, dass **testzeitliche Optimierung im Aktivierungsraum** praktikabel ist und signifikante Reasoning-Gewinne bringt.
- Der Policy-Gradient-Schritt kann als eine Art "Deliberationsupdate" interpretiert werden, ähnlich den PDA-Ideen von iterativer Verfeinerung.

**Was übernehmbar ist:**
- Mechanik: Latent-Update-Schritt mit Gradienten aus einem lokalen Reward.
- Struktur: Trennung von Basis-LLM und zusätzlicher Optimierungsdynamik im latenten Raum.

**Einschränkungen:**
- Kein explizites Multi-Worker- oder Multi-Perspektiven-Setup; es wird ein einzelner latenter Zustand optimiert.

**Weiter verfolgen:** Ja – als Blaupause für Optimierungs-basierte Deliberation (insbesondere für "hard"-PDA-Varianten, die explizit im Aktivierungsraum optimieren).

### 1.4 Token Assorted – Hybrid aus latenten und Text-Tokens

**Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning** führt latente diskrete Tokens (VQ-VAE) ein, die frühe Reasoning-Schritte komprimieren, während spätere Schritte in Textform verbleiben.[^13][^14]
Im Training werden latente und Text-Tokens gemischt, was zu kürzeren Reasoning-Traces bei verbesserter Leistung auf Maze- und logischen/mathematischen Benchmarks führt.[^14][^13]

**Relevanz für PDA:**
- Demonstriert, dass Reasoning-Informationen **teilweise in komprimierter latenter Form** gehalten und später wieder entfaltet werden können.
- Könnte als Mechanismus dienen, um Worker-Perspektiven in kompakten Codes zu repräsentieren und später zu mergen.

**Was übernehmbar ist:**
- VQ-VAE-basierte latente Tokens als Repräsentationsform für Zwischenzustände.
- Trainingsrezepte für Mischungen aus latenten und expliziten Reasoning-Schritten.

**Einschränkungen:**
- Kein explizites Parallel-Setup; mehrere Pfade werden primär sequentiell, nicht simultan, betrachtet.

**Weiter verfolgen:** Eher als Enabler für effiziente PDA-Implementierungen mit begrenzten Kontextlängen.

### 1.5 Parallel Latent Reasoning (PLR) für Recommendation

**Parallel Latent Reasoning for Sequential Recommendation** stellt ein generisches PLR-Framework vor, das mehrere latente Reasoning-Streams über "Trigger Tokens" erzeugt und diese über eine Mixture-of-Reasoning-Streams-Aggregation wieder zusammenführt.[^15][^9][^16]
Durch globale Regularisierung wird Diversität zwischen Streams erhalten, und parallelisierte Breiten-Skalierung in der latenten Breite (statt Tiefe) verbessert Accuracy und Effizienz in Empfehlungsszenarien deutlich.[^9]

**Relevanz für PDA:**
- Dies ist **das direkteste bekannte Pendant** zu PDA: mehrere parallele latente Reasoning-Trajektorien, Diversity-Enforcement und ein expliziter Aggregationsschritt.
- Arbeitet bereits mit Trigger-Tokens als Steuerungseinheiten für unterschiedliche Perspektiven.

**Was übernehmbar ist:**
- Architekturpattern "Trigger-Tokens → parallele Streams → Aggregation".
- Regularisierungsmethoden zur Sicherstellung unterscheidbarer Perspektiven.
- Mixture-of-Streams-Aggregation als Soft-Merge-Mechanik.

**Einschränkungen:**
- Fokus auf sequentielle Empfehlung, nicht auf generelles Language Reasoning.
- Aggregation erfolgt meist in einem statischen Head, nicht als mehrstufige Deliberationsschleife.

**Weiter verfolgen:** Unbedingt – PLR ist das beste existierende Vorbild für PDA-ähnliche multi-stream latente Reasoning-Systeme.

### 1.6 Parallel Test-Time Scaling für latente Reasoning-Modelle

**Parallel Test-Time Scaling for Latent Reasoning Models** generalisiert Test-Time-Scaling (TTS) auf latente Reasoning-Modelle, indem mehrere latente Trajektorien über Dropout und Rauschen gesampelt und durch ein Latent Reward Model (LatentRM) bewertet und aggregiert werden.[^17]
Experimente zeigen, dass paralleles Sampling im latenten Raum ähnlich wie CoT-Ensembles skaliert, aber effizienter ist.[^17]

**Relevanz für PDA:**
- Belegt, dass **parallele latente Reasoning-Trajektorien** mit expliziter Bewertungs- und Aggregationsfunktion praktisch funktionieren.
- Der LatentRM-Ansatz liefert eine Vorlage für PDA-interne "Critic"-Module.

**Was übernehmbar ist:**
- Sampling von latenten Zuständen via Unsicherheits-induzierte Stochastik.
- Latent Reward Models zur Trajektorien-Selektion.

**Einschränkungen:**
- Parallele Pfade werden nicht iterativ interagierend weiterentwickelt, sondern primär für Auswahl/Ensembling genutzt.

**Weiter verfolgen:** Ja – insbesondere zur Gestaltung des PDA-Konsensmechanismus.

### 1.7 Distributional / parallele Reasoning-Analysen

Arbeiten zur "distributional reasoning" zeigen, dass LLMs im Inneren mehrere parallele Reasoning-Pfade verfolgen, die sich in Aktivierungsmustern für Multi-Hop-Aufgaben niederschlagen.[^18]
Dies liefert weitere Evidenz, dass Parallelität im latenten Raum bereits vorhanden ist und nur noch explizit zugänglich gemacht werden muss.

**Relevanz für PDA:**
- Stützt die Grundannahme, dass sich mehrere Pfade im Aktivierungsraum lesen oder gezielt verstärken lassen.

**Weiter verfolgen:** Eher als interpretatives Werkzeug denn als unmittelbare Architekturvorlage.

***

## 2. Enabling-Technologien

### 2.1 Deep Equilibrium Networks und RevDEQ

DEQs modellieren das Netz als implizite unendliche Tiefe und berechnen Ausgaben als Fixpunkt einer nichtlinearen Transformation; sie nutzen Root-Finding (z.B. Broyden) und implizite Differentiation für Gradienten.[^6][^19]
Neuere Arbeiten wie **Reversible Deep Equilibrium Models (RevDEQ)** zeigen, dass mit reversiblen Fixpunkt-Solvern exakte Gradienten mit deutlich weniger Funktionsauswertungen möglich sind und dass solche Modelle Transformer-XL bei ähnlicher Parameterzahl in Sprachmodellierung schlagen.[^7][^5]

**Relevanz für PDA:**
- Bietet eine mathematisch saubere Grundlage für **Fixpunkt-basierte Deliberation** im Aktivierungsraum.
- Die Idee von unendlicher Tiefe mit Weight-Sharing ist eng verwandt mit iterativer Repräsentationsverfeinerung.

**Was übernehmbar ist:**
- Nutzung eines DEQ- oder RevDEQ-Blocks als Deliberationskern, der eine PDA-Fixpunktiteration implementiert.
- Interpretationsrahmen für Konvergenz und Stabilität (z.B. Regularisierung des Fixpunkts, Kontrolle der Lipschitz-Konstanten).

**Einschränkungen:**
- Bestehende Arbeiten adressieren primär Standard-Sequenzmodellierung, nicht explizit multi-perspektivische Worker.

**Weiter verfolgen:** Ja – vor allem RevDEQ als Kandidat-Backbone für eine PDA-Schicht mit kontrollierbarer Konvergenzdynamik.

### 2.2 Representation Engineering und Steering Vectors

**Representation Engineering: A Top-Down Approach to AI Transparency** sowie nachfolgende Arbeiten und Blogposts (Alignment Forum, Theia Vogel, neuere RepE-Zusammenfassungen) systematisieren das Extrahieren und Anwenden von Konzeptvektoren zur Steuerung von LLM-Aktivierungen während der Inferenz.[^20][^21][^22][^23]
Steering-Vektoren können z.B. Wahrheitsliebe, Sicherheit, Humor, "Happiness" oder andere abstrakte Konzepte verstärken oder abschwächen und werden meist als additive Verschiebungen in ausgewählten Layern eingesetzt.[^24][^21]

**Relevanz für PDA:**
- Bietet einen Mechanismus, um **unterschiedliche Perspektiven im Aktivierungsraum zu kodieren** (z.B. "formale Strenge", "heuristisch", "sicherheitsorientiert").
- Kann genutzt werden, um Worker zu definieren: jeweils ein gemeinsamer Grundzustand plus spezifische Steering-Vektoren.

**Was übernehmbar ist:**
- Methoden zur Gewinnung hochpräziser Konzeptvektoren.
- Verfahren zur Kombination mehrerer Steering-Vektoren (z.B. lineare Kombinationen, orthogonalisierte Subspaces).

**Einschränkungen:**
- Bisher meist auf monotone Steuerung (z.B. mehr/weniger von einem Konzept) fokussiert, nicht auf komplexe, sich entwickelnde Deliberationspfade.

**Weiter verfolgen:** Ja – als primärer Mechanismus zur **Definition semantisch unterschiedlicher Worker** innerhalb desselben Modells.

### 2.3 Activation- und Representation-Merging

Der Bereich des Model Merging im Aktivierungs-/Repräsentationsraum ist fragmentiert, aber liefert wichtige Bausteine:
- **Deep Model Merging** und verwandte Arbeiten analysieren Aktivierungsräume, um Gewichte so zu aggregieren, dass Aktivierungsverteilungen möglichst konsistent bleiben.[^25]
- **Activation-Informed Merging (AIM)** integriert Aktivierungsinformationen explizit in den Parameter-Merging-Prozess, um robuste kombinierte Modelle zu erhalten.[^26]
- Biologische/Genomik-Arbeiten zu Reverse-Complement-Netzen diskutieren *Representation Merging* von Vorwärts- und Rückwärts-Strang als Durchschnitt/Ensemble im Aktivierungsraum.[^27][^28]

**Relevanz für PDA:**
- PDA benötigt Mechanismen zum **Mergen mehrerer latenter Pfade/Worker-Aktivierungen** ohne Destruktion wichtiger Information.
- Aktivierungsinformiertes Merging könnte auf Worker-Outputs angewendet werden, um einen stabilen Konsensvektor zu bilden.

**Was übernehmbar ist:**
- Konzepte wie Korrelation-basierte Neuronen/Head-Selektion vor dem Merge.
- Idee der "Activation Renormalization" zur Vermeidung von Skalenverschiebungen beim Mergen.[^29]

**Einschränkungen:**
- Fokus liegt meist auf Weight-Space-Merging, nicht auf per-Instance-Aktivierungs-Merging im laufenden Inferenzprozess.

**Weiter verfolgen:** Ja – als Inspirationsquelle für konsensstiftende Merge-Operatoren im Aktivierungsraum.

### 2.4 LoRA-Merging und Multi-LoRA

Neuere Arbeiten zu LoRA-Merging gehen deutlich über einfache gewichtete Mittel hinaus:
- **IterIS** behandelt LoRA-Merging als iteratives Inferenz-Alignment-Problem, um eine gemeinsame Adapterstruktur über mehrere Tasks/Domänen zu finden.[^30]
- **LoRA-LEGO** zerlegt LoRA-Ränge in Minimal Semantic Units (MSUs) und nutzt Rank-wise Clustering zur flexiblen Rekombination von Fähigkeiten.[^31]
- **LoRA Soups** und **Adaptive LoRA Merge** kombinieren mehrere Adapter mit lernbaren Gewichten und zusätzlichem Pruning, um kompakte Multi-Skill-Modelle zu erzeugen.[^32][^33]
- **Multi-Linguistic LoRA Merging (MLM)** trennt Task- und Sprachadapter und kombiniert sie über einfache Interpolation plus Post-Merge-Alignment.[^34]

**Relevanz für PDA:**
- Zeigt, dass **Perspektiven in Form von LoRAs** modular kombiniert werden können – z.B. ein Worker = Basis-LLM + spezifische LoRA.[^31]
- Legt nahe, dass Worker-spezifische Fähigkeiten ohne vollständige Modellkopien realisierbar sind.

**Was übernehmbar ist:**
- Rank-wise Zerlegung von Adaptern als feinkörnige Perspektivbausteine (MSUs).
- Training-freie oder -arme Merge-Strategien für Worker-Ensembles.

**Einschränkungen:**
- Merging findet im Parameterraum statt; PDA zielt primär auf Aktivierungsraum-Merging während der Inferenz.

**Weiter verfolgen:** Ja – vor allem für praktische Implementationen, bei denen Worker durch unterschiedliche Adapter realisiert werden.

### 2.5 Signalverarbeitungskonzepte in neuronalen Repräsentationen

Mehrere Übersichtsartikel und spezialisierte Arbeiten wenden spektrale und frequenzbasierte Analysen auf neuronale Aktivierungen an, etwa zur Untersuchung von Hoch-/Niederfrequenzkomponenten und zur Herleitung von Regularisierungsstrategien.[^35][^36]
DEQ-Analysen legen nahe, dass DEQs Vorteile beim Lernen von Hochfrequenzkomponenten haben, was für detaillierte Reasoning-Strukturen relevant sein kann.[^36][^19]

**Relevanz für PDA:**
- Ermöglicht **Kohaerenzmetriken** im Aktivierungsraum: z.B. ob Worker-Pfade komplementäre Frequenzbänder abdecken.
- Könnte als Diagnoseinstrument dienen, um redundante Worker-Perspektiven zu erkennen.

**Was übernehmbar ist:**
- Spektralanalysen zur Bewertung von Diversität und Redundanz.

**Einschränkungen:**
- Bisher primär analytisch, weniger als aktiver Steuerungsmechanismus eingesetzt.

**Weiter verfolgen:** Mittel – vor allem für spätere Optimierungen und theoretische Analysen von PDA.

***

## 3. Alternative Architekturen mit Potenzial für PDA-ähnliches Parallel-Reasoning

### 3.1 State-Space-Modelle (Mamba, Mamba-2, Mamba-3, Multi-Stream-SSMs)

State-Space-Modelle wie Mamba und seine Nachfolger implementieren sequenzielle Verarbeitung mit linearem (oder besserem) Zeitverhalten und explizitem verstecktem Zustand, der über die Sequenz propagiert wird.[^37][^38][^8]
Neuere Varianten wie **Mamba-3** führen komplexwertige Zustandsupdates und MIMO-SSM-Formulierungen ein, die mehrere Inputs/Outputs pro Schritt verarbeiten und dadurch die Effizienz und Modellkapazität erhöhen.[^8]

Zusätzlich existieren domänenspezifische Multi-Stream-SSM-Ansätze, bei denen verschiedene zeitliche oder räumliche Skalen in parallelen Streams verarbeitet und später fusioniert werden.[^39][^40][^41]

**Relevanz für PDA:**
- SSMs mit Multi-Stream- oder MIMO-Struktur sind natürliche Kandidaten, um **mehrere parallele Reasoning-Streams** mit geteiltem Zustand abzubilden.
- Linearere Dynamik und konstante Speicherskalierung sind attraktiv für viele Worker.

**Was übernehmbar ist:**
- Nutzung eines Mamba-ähnlichen SSM-Kerns, der mehrere Streams (z.B. verschiedene Perspektiven) in einem gemeinsamen State speichert.
- Explizite Multi-Stream-Fusion als PDA-Merge-Mechanismus.

**Einschränkungen:**
- Bisher primär für Vision/Audio/Time-Series; großskalige textuelle Reasoning-Modelle auf Mamba-Backbone sind noch weniger etabliert.

**Weiter verfolgen:** Ja – insbesondere Mamba-3 MIMO-SSMs und Multi-Stream-Varianten als mögliche PDA-Backbones jenseits von Transformern.

### 3.2 Hypernetworks und modulare Netze

Hypernetworks, die Gewichte eines Zielnetzes konditional generieren, werden zunehmend für konditionale Anpassung und modulare Fähigkeiten eingesetzt; aktuelle Literatur nutzt sie u.a. für domänenspezifische Anpassung und dynamische Pfadwahl.[^42][^43]
Gleichzeitig zeigt die Literatur zu modularem Routing, MoE und dynamisch zusammengesetzten Netzen, dass Modelle effektiv zwischen verschiedenen Expertenpfaden wählen und diese kombinieren können.[^43][^44]

**Relevanz für PDA:**
- Ein Hypernetwork könnte **perspektiv-spezifische Parameter** für Worker generieren, ohne separate vollständige Netze zu halten.
- Modulbasierte Architekturen mit dynamischem Routing können Worker-Pfade realisieren, die auf Instanzebene gewählt und gemischt werden.

**Was übernehmbar ist:**
- Mechanismus zur On-the-fly-Generierung von Worker-Köpfen oder Layer-Varianten.
- Interpretabler Routing-Score als Proxy für Worker-"Zustimmung".

**Einschränkungen:**
- bisher selten auf latenten Reasoning-Space fokussiert; meist Token- oder Feature-level.

**Weiter verfolgen:** Ja, aber eher mittel- bis langfristig – hoher Implementierungsaufwand.

### 3.3 Diffusions- und kontinuierliche Generationsmodelle für Reasoning

Projekte wie **MDLM** und neuere Discrete Diffusion Language Models fokussieren auf parallele, iterative Textgenerierung über Diffusionsschritte, während neuere Arbeiten wie **Planner and Executor: Collaboration between Discrete Diffusion And Autoregressive Models** einen Diffusionsplaner mit einem ARM-Executor koppeln, auch mit latenten Kommunikationspfaden.[^45][^42]
Außerdem schlägt eine aktuelle Survey zu Alternativen zu Next-Token-Prediction explizit Kategorien wie "Latent Reasoning" und "Continuous Generation" mit Diffusion/Flow Matching/Energy-based Methods vor.[^42]

**Relevanz für PDA:**
- Diffusionsbasierte Reasoning-Modelle sind intrinsisch iterativ und parallel in der Sample-Dynamik.
- Ein Diffusionsprozess im Aktivierungsraum könnte als Deliberationsprozess mit stochastischen, aber konvergierenden Updates dienen.

**Was übernehmbar ist:**
- Iterative Refinement-Pattern; insbesondere das gleichzeitige Aktualisieren vieler Positionen in jedem Schritt.
- Kombinierte Latent–Token-Architekturen, bei denen ein latenter Reasoner emittierte Textketten eines ARM steuert.[^45]

**Einschränkungen:**
- Training komplexer und teurer; Nutzen für rein textuelle Reasoning-Aufgaben noch nicht vollständig etabliert.

**Weiter verfolgen:** Selektiv – insbesondere hybride Planner/Executor-Ansätze mit latenter Kommunikation.

### 3.4 Energy-Based Models und Konsensus über Energieminima

Während es bislang nur vereinzelte Arbeiten gibt, die Energy-Based Models (EBMs) im NLP-Kontext nutzen, liegen aus anderen Domänen Beispiele vor, wie Energieminimierung als Konsensmechanismus über verschiedene Hypothesenräume dienen kann.[^42]
Diese Konzepte könnten auf latente Reasoning-Zustände übertragen werden, bei denen PDA-Worker gemeinsam einen Energiescore minimieren.

**Relevanz für PDA:**
- Bietet eine alternative Sicht auf Konsens: statt Weighted Averaging könnte ein EBM über Worker-Zustände definiert werden, dessen Minimum als "Deliberationsresultat" gilt.

**Weiter verfolgen:** Konzepte interessant, derzeit aber wenig konkrete LLM-Arbeiten.

***

## 4. Synthese: Implikationen für PDA

### 4.1 Bestätigung und Erweiterung zentraler PDA-Annahmen

Die Literatur zu Coconut, Latent-SFT, LatentSeek, Token Assorted und PLR bestätigt, dass **Reasoning im latenten Raum** praktikabel, effizient und leistungsfähig ist, insbesondere im Vergleich zu rein textbasiertem CoT.[^11][^13][^9][^10][^2]
Außerdem zeigen PLR und parallele Test-Time-Scaling-Ansätze, dass **parallele latente Trajektorien mit expliziter Aggregation** die Performance steigern und sich gut skalieren lassen.[^16][^9][^17]

Damit werden folgende PDA-Annahmen gestützt:
- Es ist sinnvoll, mehrere Reasoningpfade explizit im Aktivierungsraum zu repräsentieren.
- Ein separater Konsens-/Merge-Mechanismus kann Mehrwert gegenüber einfachem Best-of-N auf Tokenebene liefern.
- Deliberation kann zum Teil als Optimierung/Test-Time-Adaption im Aktivierungsraum formuliert werden.

### 4.2 Konkrete Architektur-Bausteine für PDA

Aus der Literatur lassen sich mehrere konkrete Bausteine destillieren:
- **Worker-Perspektiven:**
  - Realisierung via Steering/Concept Vectors (RepE), unterschiedlichen LoRA-Adaptern oder PLR-ähnlichen Trigger-Tokens.[^21][^23][^9][^31]
- **Deliberationskern:**
  - Coconut-/LatentSeek-artige latente Iteration (Feed-back des Thought-Vektors; Policy-Gradient-Updates im Aktivierungsraum) oder ein DEQ/RevDEQ-Fixpunktlayer.[^4][^7][^10][^1]
- **Parallelisierung:**
  - PLR-Style parallele Streams mit globaler Diversitätsregularisierung, ggf. auf einem Mamba/Mamba-3 MIMO-SSM-Backbone.[^8][^9]
- **Merge/Konsens:**
  - Latent Reward Models (LatentRM), Activation-Informed Merging (AIM), Korrelation-/Redundanz-basierte Fusion und ggf. ein EBM-artiges Energieminimum.[^25][^26][^17]

### 4.3 Vorschlag für eine PDA-Variante inspiriert von der Literatur

Basierend auf den gefundenen Arbeiten bietet sich als "native" PDA-Variante in etwa folgendes Muster an:
- Ausgangsbasis: Transformer- oder Mamba-Backbone mit einem speziellen **Deliberation Head**, der einen Thought-Vektor erzeugt (Coconut).[^37][^10]
- Trigger-Mechanismus: Ein Set von **Trigger-/Steering-Vektoren**, die aus RepE/Steering-Vektor-Arbeiten oder LoRA-Adaptern stammen und verschiedene Perspektiven implementieren.[^23][^21][^31]
- Multi-Stream-Setup: Der Thought-Vektor wird dupliziert und mit unterschiedlichen Triggern modifiziert; anschließend laufen **M parallele Deliberationsschritte** (PLR), entweder in einem geteilten Backbone mit MIMO-SSM oder in M separaten Forward-Pässen mit Weight-Sharing.[^9][^8]
- Konsenslayer: Ein LatentRM-ähnlicher Critic bewertet die Zwischenergebnisse, ein AIM-inspirierter Merge-Operator kombiniert hochbewertete Pfade zu einem neuen zentralen Thought-Vektor.[^26][^17]
- Iteration: Der gemergte Thought-Vektor wird erneut als Input eingespeist (Coconut/DEQ-Style) und der Zyklus wiederholt, bis Konvergenzkriterien erfüllt sind (z.B. geringe Änderung im Aktivierungsraum oder Energiescore).[^6][^10]

### 4.4 Blinde Flecken und offene Fragen

Die Recherche zeigt zugleich mehrere offene Punkte:
- **Theorie der Aktivierungs-Merge-Operatoren:** Es gibt viele heuristische Merge-Strategien, aber kaum systematische Untersuchungen zur Informationskonservierung beim Mergen mehrerer latenter Pfade.[^25][^26]
- **Stabilität paralleler Fixpunkte:** DEQ/RevDEQ behandeln Fixpunktstabilität, aber nicht explizit die Interaktion mehrerer konkurrierender/kooperierender Pfade im selben Raum.[^19][^7]
- **Safety/Alignment bei latenter Deliberation:** Representation Engineering wird für Safety-Steering genutzt, aber die Kombination mit parallel latentem Reasoning (z.B. mehrere konkurrierende Safety-/Utility-Pfade) ist noch kaum untersucht.[^21][^12]
- **Skalierung auf sehr große Modelle:** Viele der demonstrierten Systeme arbeiten auf kleineren bis mittleren Modellgrößen; wie sich mehrstufige PDA-Deliberation bei LLMs mit hunderten Milliarden Parametern verhält, ist offen.[^43][^42]

Zusätzlich wurden einige Richtungen aus Zeitgründen nur gestreift:
- Explizite Multi-Agent-RL-Ansätze im Parameterraum.
- Energie-basierte Reasoning-Modelle für textuelle Aufgaben.
- Genaue Verbindung von DEQ-Fixpunktdynamik und CoT-ähnlichen Reasoning-Mustern.

***

## 5. Die 5–10 vielversprechendsten Arbeiten (kurz)

1. **Coconut – Training LLMs to Reason in a Continuous Latent Space**: Kernarbeit zu kontinuierlichem Thought-Vektor-Reasoning, direkte Vorlage für Deliberationsloops.[^4][^10]
2. **Latent-SFT – Latent Reasoning in LLMs as a Vocabulary-Space Superposition**: Quantitative Evidenz für Parallelität im latenten Raum; liefert Parallelismusmetriken.[^2]
3. **LatentSeek – Test-Time Policy Gradient in Latent Space**: Demonstriert testzeitliche Optimierung im Aktivierungsraum als leistungsfähige Alternative zu CoT/RLHF.[^1]
4. **Token Assorted – Mixing Latent and Text Tokens**: Praktische Hybridisierung von latenten und textuellen Reasoning-Schritten, relevant für Kontextökonomie.[^13][^14]
5. **Parallel Latent Reasoning (PLR)**: Explizites Multi-Stream-Latent-Reasoning mit Trigger-Tokens und Mixture-of-Streams-Aggregation; direktes Analog zu PDA.[^16][^9]
6. **Parallel Test-Time Scaling for Latent Reasoning Models**: Paralleles Sampling im latenten Raum plus Latent Reward Model, geeignet als Konsensmodul.[^17]
7. **Representation Engineering / Activation Steering (Zou et al. + Folgeliteratur)**: Systematische Methoden für Konzeptvektoren und Steering; Grundlage für Perspektiven-Definition.[^22][^23][^21]
8. **Activation-Informed Merging (AIM)**: Nutzt Aktivierungen explizit zur Verbesserung von Model-Merging; Ideengeber für PDA-Merge-Operatoren.[^26]
9. **Reversible Deep Equilibrium Models (RevDEQ)**: Fixpunkt-Architektur mit exakten Gradienten und hoher Effizienz; aussichtsreicher Deliberationskern.[^5][^7]
10. **Mamba-3 / Multi-Stream-SSMs**: State-Space-Backbones mit Multi-Input-/Multi-Stream-Fähigkeiten; natürliche Basis für many-stream PDA.[^39][^8]

***

## 6. Konkrete nächste Schritte für die PDA-Architektur

Basierend auf der Literatur ergeben sich mehrere konkrete Empfehlungen:
- **Kurzfristig (Prototyping):**
  - Coconut-artige Thought-Vektor-Schleifen in ein bestehendes LLM integrieren und mit 2–4 parallelen PLR-inspirierten Streams kombinieren, gesteuert durch einfache Steering-Vektoren.[^10][^9]
  - Ein LatentRM-ähnliches Modul als Kritiker trainieren, das die Qualität von Worker-Thought-Vektoren bewertet.
- **Mittelfristig (Architekturdesign):**
  - Evaluierung eines RevDEQ- oder DEQ-Blocks als Deliberationskern und Vergleich mit expliziten Mehrschicht-Backbones.[^7][^6]
  - Untersuchung verschiedener Merge-Operatoren (Mittelung, gewichtete Summe, AIM-inspirierte aktivierungsbewusste Fusion) hinsichtlich Stabilität und Informationsverlust.[^25][^26]
- **Langfristig (Robustheit & Theorie):**
  - Entwicklung theoretischer Rahmenbedingungen für Konvergenz und Stabilität mehrerer interagierender latenter Reasoning-Pfade.
  - Systematische Analyse von Safety- und Alignment-Fragen bei latenter Deliberation mit Representation Engineering.

Diese Schritte sollten es ermöglichen, die im Rechercheplan beschriebene Parallel Deliberation Architecture iterativ auf einer gut fundierten Basis aus dem aktuellen Forschungsstand zu konzipieren und experimentell zu evaluieren.

---

## References

1. [Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space](https://arxiv.org/abs/2505.13308) - Reasoning ability, a core component of human intelligence, continues to pose a significant challenge...

2. [Latent Reasoning in LLMs as a Vocabulary-Space Superposition](https://arxiv.org/html/2510.15522v1) - To assess the parallel reasoning capability of latent models, we ... Training large language models ...

3. [Parallel Latent Reasoning (PLR) - Emergent Mind](https://www.emergentmind.com/topics/parallel-latent-reasoning-plr) - PLR refers to the simultaneous generation and processing of multiple distinct reasoning paths inside...

4. [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/html/2412.06769v3)

5. [[PDF] Reversible Deep Equilibrium Models - arXiv](https://arxiv.org/pdf/2509.12917.pdf) - Here, we introduce Reversible Deep Equilibrium Models (RevDEQs) that compute exact gradients via an ...

6. [Deep Equilibrium Networks (DEQs) - Emergent Mind](https://www.emergentmind.com/topics/deep-equilibrium-networks-deqs) - Deep Equilibrium Networks (DEQs) are implicit neural architectures that compute outputs as the fixed...

7. [Reversible Deep Equilibrium Models - arXiv](https://arxiv.org/html/2509.12917v2) - Here, we introduce Reversible Deep Equilibrium Models (RevDEQs) that compute exact gradients via an ...

8. [Mamba-3: Improved Sequence Modeling using State Space Principles](https://www.emergentmind.com/papers/2603.15569) - ... multi-stream decoding (varying batch/beam sizes) with MIMO states ... This paper introduces Mamb...

9. [[PDF] Parallel Latent Reasoning for Sequential Recommendation - arXiv.org](https://arxiv.org/pdf/2601.03153.pdf)

10. [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769) - Large language models (LLMs) are typically constrained to reason in the language space, where they e...

11. [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/html/2412.06769) - Large language models (LLMs) are restricted to reason in the "language
space", where they typically ...

12. [Worries about latent reasoning in LLMs - LessWrong](https://www.lesswrong.com/posts/D2Aa25eaEhdBNeEEy/worries-about-latent-reasoning-in-llms) - Why not use a continuous latent space for reasoning? ... The authors didn't mention doing the parall...

13. [Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning](https://arxiv.org/abs/2502.03275) - Large Language Models (LLMs) excel at reasoning and planning when trained on chainof-thought (CoT) d...

14. [Token Assorted: Mixing Latent and Text Tokens for Improved Language
  Model Reasoning](http://arxiv.org/pdf/2502.03275.pdf) - Large Language Models (LLMs) excel at reasoning and planning when trained on
chainof-thought (CoT) d...

15. [Parallel Latent Reasoning for Sequential Recommendation - Takara ...](https://tldr.takara.ai/p/2601.03153) - To address this limitation, we propose Parallel Latent Reasoning (PLR), a novel framework that pione...

16. [Parallel Latent Reasoning for Sequential Recommendation - arXiv](https://arxiv.org/abs/2601.03153) - To address this limitation, we propose \textbf{Parallel Latent Reasoning (PLR)}, a novel framework t...

17. [Parallel Test-Time Scaling for Latent Reasoning Models](https://arxiv.org/abs/2510.07745) - Parallel test-time scaling (TTS) is a pivotal approach for enhancing large language models (LLMs), t...

18. [Distributional reasoning in LLMs: Parallel reasoning processes in
  multi-hop reasoning](https://arxiv.org/html/2406.13858v1) - Large language models (LLMs) have shown an impressive ability to perform
tasks believed to require t...

19. [[PDF] Separation and Bias of Deep Equilibrium Models on Expressivity ...](https://zhouchenlin.github.io/Publications/2024-NeurIPS-DEQ.pdf) - The deep equilibrium model (DEQ) generalizes the conventional feedforward neural network by fixing t...

20. [[PDF] arXiv:2502.19649v3 [cs.LG] 12 Mar 2025 - Jan Wehner](https://janwehner.com/files/representation_engineering.pdf) - Representation Engineering (RepE) is a novel paradigm for controlling the behavior of LLMs. Unlike t...

21. [An Introduction to Representation Engineering - AI Alignment Forum](https://www.alignmentforum.org/posts/3ghj8EuKzwD3MQR5G/an-introduction-to-representation-engineering-an-activation) - Representation Engineering (aka Activation Steering/Engineering) is a new paradigm for understanding...

22. [Representation Engineering Mistral-7B an Acid Trip - Theia Vogel](https://vgel.me/posts/representation-engineering/) - Playing around with the Representation Engineering paper, I made some interesting control vectors, a...

23. [Representation Engineering: A Top-Down Approach to AI ... - arXiv](https://arxiv.org/abs/2310.01405) - In this paper, we identify and characterize the emerging area of representation engineering (RepE), ...

24. [A Language Model's Guide Through Latent Space](https://arxiv.org/pdf/2402.14433.pdf) - Concept guidance has emerged as a cheap and simple way to control the
behavior of language models by...

25. [Deep Model Merging: The Sister of Neural Network ...](https://arxiv.org/html/2410.12927v2)

26. [Activation-Informed Merging of Large Language Models](https://arxiv.org/pdf/2502.02421.pdf) - ...merging, a method that combines the parameters and embeddings of
multiple fine-tuned large langua...

27. [Towards a Better Understanding of Reverse-Complement ...](https://www.biorxiv.org/content/10.1101/2020.11.04.368803.full) - Note that when the form of representation merging is “averaging”, a post-hoc conjoined model is esse...

28. [DeePaC: predicting pathogenic potential of novel DNA with reverse ...](https://academic.oup.com/bioinformatics/article/36/1/81/5531656) - ... neural architecture. We also considered three methods of strain representation merging. Although...

29. [$C^2M^3$: Cycle-Consistent Multi-Model Merging](https://arxiv.org/pdf/2405.17897.pdf) - In this paper, we present a novel data-free method for merging neural
networks in weight space. Diff...

30. [[PDF] IterIS: Iterative Inference-Solving Alignment for LoRA Merging](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_IterIS_Iterative_Inference-Solving_Alignment_for_LoRA_Merging_CVPR_2025_paper.pdf) - In response, LoRA merging presents an effec- tive solution by combining multiple LoRAs into a unifie...

31. [ICLR Poster Merging LoRAs like Playing LEGO](https://iclr.cc/virtual/2025/poster/28655) - ... LoRA. Experiments across various benchmarks demonstrate that our method outperforms existing app...

32. [[PDF] Adaptive LoRA Merge with Parameter Pruning for Low-Resource ...](https://aclanthology.org/2025.findings-acl.990.pdf) - In addition, we review studies on LLM layer analysis that inspired us to conduct parameter pruning d...

33. [[PDF] LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks](https://aclanthology.org/2025.coling-industry.55.pdf) - – We introduce Learnable Concatenation (CAT), a LoRA merging technique that involves a sim- ple weig...

34. [MLM: Multi-linguistic LoRA Merging - NeurIPS](https://neurips.cc/virtual/2025/126638) - In this work, we propose Multi-Linguistic LoRA Merging (MLM), a modular fine-tuning framework that d...

35. [Emergence of a High-Dimensional Abstraction Phase in Language
  Transformers](https://arxiv.org/pdf/2405.15471.pdf) - A language model (LM) is a mapping from a linguistic context to an output
token. However, much remai...

36. [NeurIPS Poster Separation and Bias of Deep Equilibrium Models on ...](https://nips.cc/virtual/2024/poster/94976) - The deep equilibrium model (DEQ) generalizes the conventional feedforward neural network by fixing t...

37. [State Space Duality](https://goombalab.github.io/blog/2024/mamba2-part1-model/) - Homepage of the Goomba AI Lab @ CMU MLD.

38. [[PDF] Enhancing Multimodal Mamba with Local and Global Cross-modal ...](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_AlignMamba_Enhancing_Multimodal_Mamba_with_Local_and_Global_Cross-modal_Alignment_CVPR_2025_paper.pdf) - These approaches can be categorized into two main types: multi-stream and single-stream methods. Mul...

39. [TSkel-Mamba: Temporal Dynamic Modeling via State Space Model ...](https://arxiv.org/html/2512.11503v1) - IV-C Multi-stream Strategy · IV-D Comparison with ... arXiv:2512.11503v1 [cs.CV] 12 Dec 2025. TSkel ...

40. [TSSMamba: A temporal–spectral–spatial state space model for multi ...](https://www.sciencedirect.com/science/article/pii/S1569843226000476) - For example, MSC-GAN adopts a multi-stream complementary architecture to facilitate cross-temporal i...

41. [M²S²L: Mamba-based Multi-Scale Spatial-temporal Learning for ...](https://arxiv.org/html/2511.05564v1) - Recently, state space models, particularly Mamba [gu2023mamba] ... Multi-stream approaches demonstra...

42. [Alternatives To Next Token Prediction In Text Generation - A Survey](https://arxiv.org/abs/2509.24435) - The paradigm of Next Token Prediction (NTP) has driven the unprecedented success of Large Language M...

43. [Survey of Different Large Language Model Architectures: Trends, Benchmarks, and Challenges](https://ieeexplore.ieee.org/document/10720163/) - Large Language Models (LLMs) represent a class of deep learning models adept at understanding natura...

44. [Attend or Perish: Benchmarking Attention in Algorithmic Reasoning](https://arxiv.org/pdf/2503.01909.pdf) - Can transformers learn to perform algorithmic tasks reliably across
previously unseen input/output d...

45. [Planner and Executor: Collaboration between Discrete Diffusion And Autoregressive Models in Reasoning](https://arxiv.org/abs/2510.15244) - Current autoregressive language models (ARMs) achieve high accuracy but require long token sequences...

