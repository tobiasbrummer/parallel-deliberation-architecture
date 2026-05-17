# **Rechercheplan: Parallele Deliberation und verwandte Architekturen**

## **Kontext und architektonische Grundlagen**

Die Entwicklung der Parallel Deliberation Architecture (PDA) markiert einen signifikanten Paradigmenwechsel in der Konstruktion und Skalierung von Sprachmodellen (Large Language Models, LLMs). Historisch bedingt konzentrieren sich Ansätze zur Erhöhung der Inferenzkapazität (Test-Time Compute) primär auf die sequenzielle Generierung von diskreten Sprach-Token, wie es bei traditionellen Chain-of-Thought (CoT) Verfahren der Fall ist. Diese sequenziellen Architekturen leiden jedoch unter kumulativen Fehlern, mangelnder Fehlertoleranz und einer fundamentalen Ineffizienz bei der Repräsentation multipler, unentschiedener kognitiver Zustände.  
PDA adressiert diese Limitierungen durch die Verlegung des Deliberationsprozesses in den kontinuierlichen Vektorraum (Aktivierungsraum). Indem mehrere "Worker" (isolierte Inferenzpfade oder Aufmerksamkeitsköpfe) denselben Input parallel aus verschiedenen semantischen Perspektiven verarbeiten, entsteht ein hochdimensionaler Deliberationsraum. Das System iteriert über diese Vektorrepräsentationen, ohne sie vorzeitig in diskrete Token zu zwingen, und konvergiert durch spezifische Verschmelzungsmechanismen (Merging) zu einem fundierten Konsens.  
Die nachfolgende systematische Analyse evaluiert den aktuellen Stand der Forschung (Fokus 2025/2026), um die theoretischen Annahmen der PDA zu validieren, technologische Befähiger (Enablers) zu identifizieren und robuste alternative Architekturen zu prüfen.

## ---

**Suchrichtung 1: Direkt verwandte Arbeiten**

Der Bereich des "Latent Space Reasoning" hat in der jüngsten Forschung eine enorme Beschleunigung erfahren. Die analysierten Arbeiten demonstrieren, dass Modelle, die im kontinuierlichen Aktivierungsraum rechnen, nicht nur effizienter sind, sondern auch komplexe Suchalgorithmen (wie die Breitensuche) nativ abbilden können.  
Die folgende Tabelle fasst die konzeptionellen Unterschiede zwischen klassischen Token-basierten und den hier untersuchten Vektor-basierten Ansätzen zusammen:

| Dimension | Traditionelles CoT (Token-Level) | Parallel Latent Reasoning (Activation-Level) |
| :---- | :---- | :---- |
| **Kommunikationsbandbreite** | Niedrig ($\\sim$ 15 Bits pro Token) | Hoch ($\\sim$ 40.000 Bits pro Hidden State) 1 |
| **Informationszustand** | Diskret, kollabiert auf ein Wort | Kontinuierlich, Superposition multipler Hypothesen 2 |
| **Skalierungsachse** | Tiefenskalierung (längere Texte) | Breitenskalierung (simultane Vektor-Ströme) 3 |
| **Fehlerkorrektur** | Erfordert Neu-Generierung des Pfades | Iteratives Gradienten-Update / Denoisierung 4 |

Die nachfolgenden Publikationen wurden als hochrelevant für die Konstruktion des PDA-Deliberationsraums identifiziert.

### **(Tang et al., Jan 2026\)**

* **Quelle**: 3  
* **Kernidee**: Die Forschung etabliert das Framework "Parallel Latent Reasoning" (PLR), welches das Konzept der reinen Tiefenskalierung (Depth-Level Scaling) verwirft und stattdessen eine Breitenskalierung (Width-Level Computational Scaling) einführt. Das System generiert über spezifische, lernbare "Trigger-Token" simultan mehrere divergierende Inferenzströme im kontinuierlichen Vektorraum, deren semantische Diversität durch globale Regularisierungsmetriken erzwungen wird, bevor ein adaptives Netzwerk die Ströme zu einem finalen Resultat aggregiert.3  
* **Relevanz für PDA**: Die Publikation validiert die Kernhypothese der PDA-Architektur auf empirischer und theoretischer Ebene. Sie beweist, dass das parallele Nachdenken in multiplen Vektorströmen fundamentale Generalisierungsvorteile bietet und die abnehmenden Erträge (Diminishing Returns) klassischer sequenzieller Ansätze überwindet.6  
* **Was können wir übernehmen**: Der "Mixture-of-Reasoning-Streams" (MoRS) Aggregationsmechanismus ist für den PDA-Merge-Knoten essenziell. Die Autoren belegen, dass eine naive Mittelwertbildung von Vektoren (Mean-Pooling) zu katastrophaler Interferenz führt, da hochpräzise Perspektiven durch fehlerhafte Ströme verwässert werden.5 MoRS nutzt ein leichtgewichtiges Gating-Netzwerk, das basierend auf der initialen "Fast-Thinking"-Repräsentation ($h\_0$) adaptive Gewichte ($g$) berechnet: $g \= \\text{softmax}(W\_g h\_0 \+ b\_g)$. Die finale Repräsentation ($z\_{rea}$) ergibt sich aus der gewichteten Summe der individuellen Ströme ($z\_m$): $z\_{rea} \= \\sum\_{m=1}^M g\_m z\_m$.5  
* **Einschränkungen / Unterschiede**: PLR ist spezifisch auf sequentielle Empfehlungssysteme und das Modellieren von Nutzerinteraktionen ausgelegt, weshalb die Metriken stark auf Verhaltenssequenzen optimiert sind und nicht unmittelbar auf offene NLP-Generierungsaufgaben übertragen werden können.5  
* **Weiter verfolgen**: Ja. Die mathematische Formulierung des MoRS-Gatings und das "Reasoning Contrastive Learning" (RCL) zur Erzwingung von Inter-Stream-Diversität sind direkte Blaupausen für PDA.5

### **(Du et al., Nov 2025\)**

* **Quelle**: 1  
* **Kernidee**: Die Interlat-Architektur befreit KI-Agenten von der Limitierung der natürlichen Sprache, indem sie eine direkte Kommunikation über die "Last-Layer Hidden States" des Transformators etabliert. Durch einen leichtgewichtigen neuronalen Adapter werden diese hochdimensionalen Repräsentationen, die das interne "Wissen" des Agenten kodieren, direkt in den Eingabestrom eines kollaborierenden Agenten injiziert.1  
* **Relevanz für PDA**: Diese Arbeit liefert den empirischen Beweis, dass der Aktivierungsraum (Latent Space) ein weit überlegenes Medium für den Austausch komplexer kognitiver Zustände ist. Die Informationsbandbreite steigt drastisch an, was PDA-Workern ermöglichen würde, nuancierte Unsicherheiten und parallele Hypothesen verlustfrei zu synchronisieren.1  
* **Was können wir übernehmen**: Die Erkenntnisse zur Informationskompression. Die Studie belegt, dass latente Nachrichtenströme auf lediglich 8 Token komprimiert werden können, was die Latenz der Kommunikation um den Faktor 24 reduziert, ohne die Aufgabenerfolgsrate zu beeinträchtigen.9 Ein derartiger Kompressionsmechanismus garantiert, dass der iterative PDA-Deliberationsraum auch bei vielen Workern recheneffizient bleibt.  
* **Einschränkungen / Unterschiede**: Interlat operiert primär im Kontext von Multi-Agenten-Systemen (Sender-Empfänger-Paradigma) bei Reinforcement Learning Aufgaben, wohingegen PDA einen homogenen Deliberationsraum innerhalb eines einzelnen, monolithischen Modells anstrebt.1  
* **Weiter verfolgen**: Ja. Das Phänomen, dass latent kommunizierende Modelle ein emergentes, tiefgründigeres exploratives Verhalten aufweisen und nicht nur oberflächliches Pattern-Matching betreiben, bestätigt den fundamentalen Wert der PDA-Vision.11

### **(Zhu et al., 2025\) / \[Coconut\] (Hao et al., 2025\)**

* **Quelle**: 2  
* **Kernidee**: Diese konzeptionell verwandten Arbeiten ersetzen den traditionellen autoregressiven Dekodierungsschritt durch ein Denken in "kontinuierlichen Token". Anstatt Wahrscheinlichkeiten über ein Vokabular zu berechnen und auf ein Wort zu kollabieren, speist das Modell den kontinuierlichen Hidden-State-Vektor direkt als Input-Embedding in den nächsten Rechenschritt ein.2  
* **Relevanz für PDA**: Die Methodik erlaubt es dem Modell, eine Superposition mehrerer potenzieller Argumentationspfade im Vektorraum aufrechtzuerhalten, was mathematisch einer latenten Breitensuche (Breadth-First Search, BFS) entspricht.2 Dies ist das theoretische Fundament für die Koexistenz verschiedener PDA-Perspektiven.  
* **Was können wir übernehmen**: Die "Multi-Token Sampling" (MTS) Strategie. Hierbei werden in jedem Vorwärtsdurchlauf $K$ diskrete Token als Wahrscheinlichkeits-Simplex beprobt und zu einem kontinuierlichen Token komponiert, um das exakte Niveau der Parallelität präzise zu steuern.12  
* **Einschränkungen / Unterschiede**: CoT2 und Coconut zielen weiterhin auf eine sequentielle Verlängerung des Inferenzprozesses ab (kontinuierliche Iterationstiefe), während PDA den Raum lateral aufspaltet.  
* **Weiter verfolgen**: Ja. Die Beweise, dass die Kapazität für paralleles Reasoning direkt von der Dimensionalität des Embedding-Raums abhängt, sind essenziell für das Sizing der PDA-Architektur.12

### **(Yang et al., Feb 2026\)**

* **Quelle**: 16  
* **Kernidee**: ManCAR adressiert das kritische Problem des "Latent Drift", bei dem unstrukturierte Vektor-Reasoning-Prozesse in unplausible oder degenerative Repräsentationsräume abdriften. Die Architektur definiert eine "kollaborative Mannigfaltigkeit" (collaborative manifold) basierend auf einer Graphen-Topologie und zwingt den Reasoner durch eine Variations-Zielfunktion, diese Mannigfaltigkeit während der iterativen Verfeinerung nicht zu verlassen.16  
* **Relevanz für PDA**: Iterative Vektor-Deliberation ist hochgradig anfällig für Instabilitäten. ManCAR bietet den mathematischen Rahmen, um den Deliberationsraum der PDA-Worker zu stabilisieren und auf logisch kohärente Regionen zu beschränken.  
* **Was können wir übernehmen**:  
  1. Die Regularisierungsfunktion via Kullback-Leibler-Divergenz ($L^{(t')}*{reg} \= D*{KL}(q |

| p^{(t')}\\theta)$), die die Vektoren an einen plausiblen "Teacher Prior" bindet.16 2\. Den Mechanismus für "Adaptive Test-Time Stopping": Das System stoppt die Iteration dynamisch, sobald die Divergenz zwischen aufeinanderfolgenden Vorhersageverteilungen unter einen Schwellenwert $\\epsilon$ fällt ($D{KL}(p^{(t'-1)} |  
| p^{(t')}) \< \\epsilon$), wodurch nutzloses "Over-Thinking" eliminiert wird.16

* **Einschränkungen / Unterschiede**: Die Mannigfaltigkeit wird hier durch Interaktionsgraphen in Empfehlungssystemen definiert. Für PDA in NLP muss die Mannigfaltigkeit stattdessen durch semantische Sprachmodelle (z. B. durch Pre-Training) definiert werden.  
* **Weiter verfolgen**: Zwingend. Ohne geometrische Restriktionen wird der PDA-Konvergenzprozess im Rauschen kollabieren.

### **(Chen et al., Dez 2025\)**

* **Quelle**: 19  
* **Kernidee**: Die Arbeit formalisiert den internen Inferenzprozess als implizite "latente Debatte". Interne Signale (Hidden States, Attention) werden als Pro- oder Contra-Argumente interpretiert, und ein Aggregationsmodul löst diese Konflikte mathematisch auf, um zu einer Entscheidung zu gelangen.19  
* **Relevanz für PDA**: Bietet einen strukturierten Rahmen, um den Konsensusprozess (Merge) zwischen divergierenden Vektoren als Argumentationsprozess zu modellieren.  
* **Was können wir übernehmen**: Das "Quantitative Bipolar Argumentation Framework" (QBAF). Durch die Anwendung "gradueller Semantik" (gradual semantics) propagiert das System Konflikte und Verstärkungen, um das Ungleichgewicht (Imbalance) zwischen Vektoren zu berechnen, anstatt diese lediglich linear zu interpolieren.22  
* **Einschränkungen / Unterschiede**: Das Framework fungiert primär als Surrogat zur Interpretierbarkeit und Halluzinationserkennung, nicht als Generierungsarchitektur für neues Wissen.22  
* **Weiter verfolgen**: Ja. Insbesondere die Erkenntnis, dass eine extrem hohe Vektor-Debatten-Varianz in den mittleren Transformerschichten ein zuverlässiger Prädiktor für Halluzinationen ist, liefert PDA eine Metrik zur Erkennung von unlösbaren Konflikten.21

### **(Hu et al., Jan 2026\)**

* **Quelle**: 23  
* **Kernidee**: PaCoRe ersetzt tiefes sequenzielles Nachdenken durch massiv parallele Exploration. In mehreren Runden werden parallele Inferenz-Trajektorien gestartet, zu kompakten Nachrichten zusammengefasst (Compaction) und über eine Message-Passing-Architektur synchronisiert, um die nächste Runde zu leiten.24  
* **Relevanz für PDA**: Zeigt die praktische Umsetzbarkeit und enorme Leistungsfähigkeit koordinierter Parallelität, die GPT-5-ähnliche Leistung (94.5% auf HMMT 2025\) mit Modellen der 8B-Parameter-Klasse erreicht.24  
* **Was können wir übernehmen**: Die "Functional Region Partitioning". Das Modell lernt, den Repräsentationsraum in Argumentations- und Explorationsregionen zu unterteilen und nutzt Unsicherheitsmetriken (wie semantische Entropie), um optimale Verzweigungspunkte (Branching) zu identifizieren.24  
* **Einschränkungen / Unterschiede**: PaCoRe operiert durch Kompaktierung von Sprach-Trajektorien in den Kontext (Message Passing auf Token-Ebene), während PDA den Prozess rein im Vektorraum (Continuous Space) abbilden will.  
* **Weiter verfolgen**: Ja, als Referenzmodell für die Effizienz von Parallelität gegenüber reiner Sequenzialität.

## ---

**Suchrichtung 2: Enabling-Technologien**

Um die Interaktion, Isolierung und Verschmelzung der Vektoren im PDA-System technisch zu realisieren, liefert die aktuelle Forschung im Bereich der Netzwerkmechanik entscheidende algorithmische Werkzeuge.

### **2a: Deep Equilibrium Models (DEQ) und Rekurrente Tiefe**

Die Konstruktion eines iterativen Deliberationsraums, der ohne feste Schichtanzahl zu einem Konsens konvergiert, findet ihr mathematisches Äquivalent in Fixpunkt-Iterationen.

* **Recurrent Depth Approach (Geiping et al., NeurIPS 2025\)** 25: Die Autoren skalieren Test-Time Compute durch die Integration rekurrenter Blöcke, die den Vektorraum in einer iterativen Schleife unendlich ausrollen können, ohne den Parameter-Footprint zu erhöhen.25 Die Architektur demonstriert "Zero-shot per-token adaptive compute" 25, wodurch komplexe Logik-Token automatisch tiefer deliberiert werden.  
* **Fixpunkt-Stabilität in Hierarchical Reasoning Models (HRM)** 26: HRMs implementieren hierarchische Rekurrenz, vermeiden das ressourcenintensive Backpropagation-Through-Time (BPTT) und erzielen überragende Ergebnisse bei 27M Parametern.26 Es zeigt sich jedoch, dass die Fixpunkt-Annahme in der Praxis häufig durch "Over-Thinking" verletzt wird. Dies kann laut aktuellen Studien durch gezielte Datenaugmentations-Mischungen stabilisiert werden.29  
* **Implikation für PDA**: Der PDA-Deliberationsraum muss als Deep Equilibrium Model strukturiert werden. Die Iteration der Worker-Outputs wird durch implizite Differenzierung trainiert. Die Fixpunkt-Metrik (relative Änderung des Konsens-Vektors nahe Null) dient als nativer Stopp-Mechanismus, der externe Token-Prädiktoren überflüssig macht.

### **2b: Signalverarbeitung (DSP) für neuronale Repräsentationen**

Die Separierung von Worker-Perspektiven aus einem überlagerten Vektorraum kann durch Techniken der digitalen Signalverarbeitung (DSP) im Aufmerksamkeitsprozess (Attention) gelöst werden.

* **Spectral Attention for Transformers (Huang et al., März 2026\)** 30: Diese revolutionäre Architektur filtert die Attention-Score-Matrix direkt im Frequenzraum.30 Die Matrix wird mittels Fast Fourier Transform (FFT) konvertiert, durch lernbare Masken modifiziert und per IFFT in die Zeitdomäne zurückgeführt.30 Das "Adaptive Frequency Attention" Modul generiert dynamische, inhaltsabhängige Modulationsgewichte ($W \= \\tanh(\\text{Linear}(C))$), um Frequenzen zu filtern.31 Die Ergebnisse belegen eine Reduktion der Perplexität um bis zu 15.3%, wobei niedrige Frequenzen globale Abhängigkeiten und hohe Frequenzen lokale Logikmuster isolieren.30  
* **Phase Coherence Tracing (Kerridge et al., Dez 2025\)** 34: Diese Methode misst die interne Koordination eines LLMs durch die Berechnung von Phasen-Ausrichtungen (angular displacement) der Attention-Flüsse.34 Sie offenbart, dass selbst-referenzielle Interferenzen akkumulieren und messbare Kohärenz-Plateaus bilden.34  
* **Implikation für PDA**: PDA kann die Frequenzanalyse nativ für das "Perspektiven-Routing" nutzen. Anstatt Worker durch Prompting in verschiedene Rollen zu zwingen, weist das System Worker A einen Low-Pass-Filter zu (Erzwingung einer global-semantischen, strategischen Perspektive) und Worker B einen High-Pass-Filter (Erzwingung der Analyse logischer Randbedingungen). Die anschließende Vektorfusion wird dadurch kollisionsfrei, da die Worker in unterschiedlichen physikalischen Frequenzbändern operieren.

### **2c: Repräsentations-Engineering und Affines Steering**

Der Merge-Knoten der PDA erfordert fortgeschrittene kompositorische Operationen, die über die naive Addition von Vektoren hinausgehen.

* **From Steering Vectors to Conceptors (Abreu et al., NeurIPS 2025\)** 35: Diese Arbeit revolutioniert die Vektor-Addition durch die Einführung von "Conceptors". Diese agieren als weiche Projektionsmatrizen, die Sätze von Aktivierungsvektoren komprimieren.35 Der entscheidende technologische Sprung besteht darin, dass Conceptors Boolesche Operationen (wie AND, OR, NOT) direkt im Repräsentationsraum ermöglichen.35  
* **Identifizierbarkeit von Vektoren** 36: Die Analyse zeigt, dass simple Steering-Vektoren extrem anfällig für Überlagerungen und nicht-identifizierbar sind.  
* **Implikation für PDA**: Wenn Worker ihre Vektor-Repräsentationen zusammenführen, löst die Boolesche Komposition über Conceptors das Problem des Widerspruchs. Eine logische AND-Operation im Vektorraum extrahiert den unbestreitbaren Konsens der Worker, während eine OR-Operation ein erweitertes Array möglicher Lösungsräume zur weiteren Iteration aufspannt.

### **2d: LoRA-Routing und Gruppierte Ausführung**

Techniken aus dem Parameter-Raum (Weight Merging) inspirieren die Architektur der parallelen Worker-Inferenz.

* **HiLoRA (2026)** 37: Nutzt orthogonale Low-Rank-Zerlegung ($\\Delta W\_i \= B\_r A\_r \+ B\_{c,j} A\_{c,j} \+ B\_{\\ell,i} A\_{\\ell,i}$) und adaptive Clusterbildung zur Isolierung von Wissen.  
* **tLoRA / Grouped Execution** 38: Durch adaptives Nano-Batching können mehrere LoRAs ohne den typischen Kernel-Launch-Overhead auf derselben GPU simultan ausgeführt werden.38  
* **Implikation für PDA**: Die Hardware-Beschleunigungs-Mechanismen von tLoRA können genutzt werden, um die parallelen Worker der PDA (die als orthogonale Aufmerksamkeitsköpfe oder Aktivierungspfade implementiert sind) latenzfrei und hardware-optimiert abzuarbeiten.

## ---

**Suchrichtung 3: Alternative Architekturen zur PDA-Vision**

Um die Resilienz der PDA zu evaluieren, wurden alternative Architekturen untersucht, die eine multiperspektivische, latente Konvergenz über radikal abweichende mathematische Paradigmen anstreben.

### **(Kang et al., 2025/2026)**

* **Quelle**: 4  
* **Mechanismus**: LaDiR verwirft die sequenzielle Autoregression vollständig. Über einen Variational Autoencoder (VAE) werden logische Textschritte in Blöcke latenter "Gedanken-Token" komprimiert.4 Der Reasoning-Prozess wird als Entrauschungs-Aufgabe (Denoising) formuliert: Ein Diffusionsmodell nutzt Flow Matching, um ein Vektorfeld ($u\_\\theta(z\_t, t)$) von purem Rauschen ($t=1$) rückwärts durch die Zeit in eine kohärente logische Repräsentation ($t=0$) zu transformieren.4  
* **Relevanz & Alternative**: Anstatt PDA-Worker durch wiederholte Feed-Forward-Berechnungen iterieren zu lassen, modelliert LaDiR den Deliberationsprozess als Diffusion. Ein bemerkenswertes Feature ist die explizite "Diversity Guidance", die abstoßende Kräfte (repulsive forces) zwischen latenten Trajektorien nutzt, um eine Kollabierung der Lösungen zu verhindern.4  
* **Fazit**: Die Modellierung von fehlerhaftem Reasoning als "lexikalisches Rauschen", das durch Diffusions-Iterationen aus dem Vektor herausgefiltert wird, bietet eine robuste, hochgradig parallele Alternative zur klassischen Transformer-Deliberation.

### **(Logical Intelligence, Jan 2026\)**

* **Quelle**: 43  
* **Mechanismus**: EBMs bewerten nicht das nächste Token, sondern die Validität eines vollständigen Lösungspfades oder Systemzustandes über eine globale Energielandschaft. Das "Kona 1.0" Modell generiert inferenzzeitliche Pläne simultan (non-autoregressiv).43 Ein Skalarwert (die Energie) zeigt die Güte des Zustands an: Niedrige Energie entspricht der Einhaltung logischer Randbedingungen, hohe Energie indiziert Konflikte, was eine präzise Lokalisierung exakter Fehlerquellen in partiellen Trajektorien erlaubt.43  
* **Relevanz & Alternative**: EBMs stellen das mächtigste Paradigma für den PDA-Merge-Knoten dar. Der Konsensprozess benötigt keine manuell definierten Pooling-Gewichte. Der optimale Konsensvektor ist per Definition derjenige Punkt im Deliberationsraum, an dem das Energiefunktional des Systems sein globales Minimum erreicht.  
* **Fazit**: Während EBMs in offenen, kreativen NLP-Aufgaben noch schwächeln, sind sie für die Lösung harter, deterministischer Nebenbedingungen (Constraint Satisfaction) in der PDA-Architektur essenziell.

### **\[Optimizing Multi-Agent Captioning via Consensus-Aware Gradient Fusion\] (Liu, März 2026\)**

* **Quelle**: 47  
* **Mechanismus**: Das WeatherTGD-Framework nutzt ein "Text Gradient Descent" (TGD) Verfahren, um Outputs heterogener Agenten zu verschmelzen.47 Die Kerninnovation ist der "Consensus-Aware Gradient Fusion" (CAGF) Mechanismus, der über Einbettungs-Ähnlichkeiten operiert: Informationsfragmente mit hoher Ähnlichkeit ($sim(g\_i, g\_j) \\ge 0.8$) werden zu einem Basis-Konsens verschmolzen, während unike, domänenspezifische Erkenntnisse (Ähnlichkeit $\< 0.6$) explizit erhalten bleiben.47  
* **Relevanz & Alternative**: Dieses zweistufige Verfahren liefert den exakten Algorithmus für die Vektorfusion in PDA. Es extrahiert das geteilte Signal und bewahrt gleichzeitig das tiefgreifende Fachwissen der spezialisierten Worker, anstatt dieses herauszumitteln.  
* **Fazit**: Die algorithmischen Schwellenwerte dieses Fusionsprozesses können direkt in die Kontrolllogik der PDA-Architektur überführt werden.

### **\[Heterogeneous Agent Collaborative Policy Optimization (HACPO)\] (Zhang et al., März 2026\)**

* **Quelle**: 50  
* **Mechanismus**: HACPO adressiert die Kollaboration von Agenten mit fundamental unterschiedlichen Architekturen und Parameter-Dimensionen.50 Das System nutzt eine "Agent-Capability-Aware Advantage Estimation", bei der das Lernen durch den relativen Leistungsunterschied moduliert wird.50 Ein Diskrepanz-Koeffizient ($\\omega$) kalibriert die Baseline und verstärkt Gradienten von performanten Agenten, während fehlerhaftes Rauschen schwacher Agenten unterdrückt wird.50  
* **Relevanz & Alternative**: Falls PDA mit Workern unterschiedlicher Kapazität (z.B. einem massiven Strategie-Kopf und mehreren kleinen Faktenprüfer-Köpfen) operiert, garantiert dieser Mechanismus, dass schwache Worker die Konvergenz nicht sabotieren, sondern lediglich explorative Randimpulse liefern.  
* **Fazit**: Ein überlegenes Design-Muster für das Reinforcement Learning und Pre-Training (RLVR) der PDA-Gewichte.

### **(Tian, NeurIPS 2025\)**

* **Quelle**: 53  
* **Mechanismus**: Diese fundierte theoretische Analyse beweist, dass der Gewichtsraum in Netzwerken, die auf Logikaufgaben trainiert wurden (z.B. modulare Addition), algebraische Semi-Ring-Strukturen aufweist.53 Die Verlustfunktionen agieren als Ring-Homomorphismen. Dies ermöglicht es, aus mehreren nicht-optimalen Teillösungen auf rein algebraischem Weg (durch Ring-Addition und Ring-Multiplikation) eine globale Optimallösung zu konstruieren.53  
* **Relevanz & Alternative**: Es liefert den mathematischen Beweis, dass neuronale Logik-Repräsentationen streng kompositionell sind. Die Vermengung partieller Perspektiven in PDA muss nicht auf Black-Box-Interpolationen hoffen, sondern kann deterministischen algebraischen Gesetzen folgen, sofern das Modell entsprechend regularisiert ist (z.B. durch L2-Verlust und Weight Decay).53  
* **Fazit**: Stützt das theoretische Fundament der PDA-Zusammenführung massiv.

## ---

**Synthese und strategische Empfehlungen**

Die umfassende Literaturauswertung der Jahrgänge 2025 und 2026 zeichnet die Konturen einer neuen Epoche in der Architektur von Foundation Models. Die Industrie vollzieht eine rapide Abkehr vom textbasierten "Chain-of-Thought" (wie in o1/o3-Modellen) hin zu massiv parallelem **Latent Space Reasoning**.  
Die Ergebnisse dieser Recherche validieren die Kernprämissen der *Parallel Deliberation Architecture (PDA)* vollumfänglich. Sie offerieren jedoch gleichzeitig hochentwickelte Werkzeuge, um die bisherigen, von Instabilität geprägten Annahmen des Frameworks durch beweisbar korrekte mathematische Konstrukte zu ersetzen.

### **Top-Architektur- und Konzept-Empfehlungen**

1. Parallel Latent Reasoning (PLR) & MoRS-Aggregator 3: Belegt empirisch die Überlegenheit von Width-Scaling im Vektorraum. Der "Mixture-of-Reasoning-Streams" Algorithmus muss das naive Mean-Pooling in der PDA ersetzen.  
2. ManCAR (Manifold-Constraints) 16: Identifiziert und löst den katastrophalen "Latent Drift". Die PDA-Architektur muss zwingend KL-Divergenz-Regularisierungen implementieren, die die iterierenden Vektoren an eine gültige semantische Mannigfaltigkeit binden.  
3. Spectral Attention (FFT im Transformer) 30: Die Isolation von Frequenzbändern liefert die physikalische Grundlage für das PDA-Perspektiven-Routing. Anstatt auf unscharfes Prompting zu vertrauen, operieren die PDA-Worker in sauberen, orthogonalen Frequenzbereichen.  
4. Conceptors & Boolean Affine Steering 35: Ermöglicht logische (AND/OR) Vektorverschmelzungen, was die PDA-Synthese von Rauschen befreit.  
5. Energy-Based Models (EBM) 43: Definiert das Ziel der PDA-Deliberation als Energieminimum, was eine globale Kohärenz der Lösungspfade erzwingt, ohne autoregressive Flaschenhälse zu durchlaufen.

### **Status der PDA-Annahmen**

**Signifikant gestärkte Annahmen:** Die Überlegenheit des **Aktivierungsraums (Latent Space)** über die Token-Ebene ist durch Arbeiten wie *Interlat* 9 und *CoT2* 12 unbestreitbar belegt. Der Informationsverlust bei der Diskretisierung wird eliminiert, und die Übertragung von Superpositions-Zuständen sowie kontextueller Unsicherheit wird möglich. Zudem wurde das **Width-Scaling** (die parallele Erkundung) als weitaus ressourceneffizienter validiert als das reine Verlängern sequenzieller Denkpfade.5  
**Geschwächte Annahmen und identifizierte Gefahren:**

1. **Divergenz der Iteration:** Die naive Annahme, dass freies iteratives Rechnen zu einem schärferen Konzept konvergiert, wird durch Studien zu *Deep Equilibrium Models* und *ManCAR* 16 widerlegt. Ohne explizite Begrenzung des Lösungsraums führen wiederholte Forward-Passes zu degenerierten Vektoren (Over-Thinking).  
2. **Interferenz bei der Fusion:** Die direkte Aggregation heterogener Perspektiven ist destruktiv. Wie in *WeatherTGD* 47 und *PLR* 5 bewiesen, verschlechtert eine statische Vektor-Addition die Performance im Vergleich zum besten Einzel-Worker. Eine intelligente, "Consensus-Aware" Synthese ist unerlässlich.

### **Konkrete Vorschläge für das Design der PDA**

1. **Injektion von Trigger-Token:** Das Modell initiiert die PDA-Verzweigung nicht blind, sondern durch das Generieren lernbarer Vektor-Trigger. Diese aktivieren den "Parallel Reasoning Mode", woraufhin der Hidden State in orthogonale Pfade geklont wird.  
2. **Spektrale Isolation der Worker:** Den parallelen Workern werden mittels *Fast Fourier Transform (FFT)* innerhalb der Attention-Matrix spezifische Frequenzbänder zugewiesen.30 Worker A (Low-Pass) evaluiert den globalen strategischen Kontext; Worker B (High-Pass) validiert feingranulare logische Constraints.  
3. **Zweistufiger Conceptor-Merge:** Der Konsensus-Knoten nutzt keine Durchschnittsbildung. Er wendet Boolesche *Conceptor*\-Operatoren 35 an: Im ersten Schritt (Konsens) wird eine AND-Schnittmenge der Vektoren gebildet, die nur übereinstimmende Kernfaktoren passieren lässt (analog zur $0.8$-Ähnlichkeitsgrenze in WeatherTGD 47). Im zweiten Schritt werden stark divergierende, domänenspezifische Features (OR-Verknüpfung) hinzugefügt, um spezialisiertes Wissen zu erhalten.  
4. **Energie-gestoppte DEQ-Iteration:** Der Deliberationsprozess wird als rekurrentes Deep Equilibrium Model formuliert. Die Iteration der Vektorfusionen läuft so lange, bis die Veränderung der Systemenergie (EBM-Metrik) oder die KL-Divergenz zwischen zwei Zeitschritten einen Schwellenwert $\\epsilon$ unterschreitet (Adaptive Test-Time Computation 16).

### **Blinde Flecken der aktuellen Recherche**

Die Literaturrecherche fokussierte stark auf die mathematische und konzeptionelle Formulierung latenter Deliberation. In zukünftigen Phasen muss dringend das **Hardware/I-O Profiling** adressiert werden. Die massiv parallele Vektor-Synchronisation zwischen GPUs erfordert gigantische Memory-Bandbreiten (All-Reduce Operations), was die Vorteile des Latent Space Reasoning durch Latenzen auf Chip-Ebene zunichtemachen könnte. Ebenso verlangt die **Trainingsdynamik** bei Modellen mit zyklischen Gradienten (BPTT vs. Implicit Differentiation) eine gesonderte Untersuchung, um Vanishing-Gradient-Effekte im Deliberationsraum auszuschließen.

#### **Works cited**

1. Enabling Agents to Communicate Entirely in Latent Space \- OpenReview, accessed on March 28, 2026, [https://openreview.net/forum?id=rmYbgsehTd](https://openreview.net/forum?id=rmYbgsehTd)  
2. Training Large Language Model to Reason in a Continuous Latent Space | OpenReview, accessed on March 28, 2026, [https://openreview.net/forum?id=tG4SgayTtk](https://openreview.net/forum?id=tG4SgayTtk)  
3. Sequential User-based Recurrent Neural Network ... \- ResearchGate, accessed on March 28, 2026, [https://www.researchgate.net/publication/319285113\_Sequential\_User-based\_Recurrent\_Neural\_Network\_Recommendations](https://www.researchgate.net/publication/319285113_Sequential_User-based_Recurrent_Neural_Network_Recommendations)  
4. \[2510.04573\] LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning \- arXiv, accessed on March 28, 2026, [https://arxiv.org/abs/2510.04573](https://arxiv.org/abs/2510.04573)  
5. \[2601.03153\] Parallel Latent Reasoning for Sequential Recommendation \- arXiv, accessed on March 28, 2026, [https://arxiv.org/abs/2601.03153](https://arxiv.org/abs/2601.03153)  
6. Parallel Latent Reasoning for Sequential Recommendation \- arXiv, accessed on March 28, 2026, [https://arxiv.org/pdf/2601.03153](https://arxiv.org/pdf/2601.03153)  
7. Parallel Latent Reasoning for Sequential Recommendation \- ResearchGate, accessed on March 28, 2026, [https://www.researchgate.net/publication/399522624\_Parallel\_Latent\_Reasoning\_for\_Sequential\_Recommendation](https://www.researchgate.net/publication/399522624_Parallel_Latent_Reasoning_for_Sequential_Recommendation)  
8. \[2511.09149\] Enabling Agents to Communicate Entirely in Latent Space \- arXiv, accessed on March 28, 2026, [https://arxiv.org/abs/2511.09149](https://arxiv.org/abs/2511.09149)  
9. Enabling Agents to Communicate Entirely in Latent Space | Request PDF \- ResearchGate, accessed on March 28, 2026, [https://www.researchgate.net/publication/397556064\_Enabling\_Agents\_to\_Communicate\_Entirely\_in\_Latent\_Space](https://www.researchgate.net/publication/397556064_Enabling_Agents_to_Communicate_Entirely_in_Latent_Space)  
10. Enabling Agents to Communicate Entirely in Latent Space \- arXiv, accessed on March 28, 2026, [https://arxiv.org/html/2511.09149v2](https://arxiv.org/html/2511.09149v2)  
11. Enabling Agents to Communicate Entirely in Latent Space \- arXiv, accessed on March 28, 2026, [https://arxiv.org/html/2511.09149v1](https://arxiv.org/html/2511.09149v1)  
12. Continuous Chain of Thought Enables Parallel Exploration and Reasoning \- OpenReview, accessed on March 28, 2026, [https://openreview.net/forum?id=sTPKDKn5ig](https://openreview.net/forum?id=sTPKDKn5ig)  
13. Training Large Language Models to Reason in a Continuous Latent Space \- arXiv, accessed on March 28, 2026, [https://arxiv.org/html/2412.06769v3](https://arxiv.org/html/2412.06769v3)  
14. Continuous Chain of Thought Enables Parallel Exploration and Reasoning \- arXiv, accessed on March 28, 2026, [https://arxiv.org/pdf/2505.23648](https://arxiv.org/pdf/2505.23648)  
15. Continuous Chain of Thought Enables Parallel Exploration and Reasoning \- arXiv.org, accessed on March 28, 2026, [https://arxiv.org/html/2505.23648v1](https://arxiv.org/html/2505.23648v1)  
16. ManCAR: Manifold-Constrained Latent Reasoning with ... \- arXiv, accessed on March 28, 2026, [https://arxiv.org/abs/2602.20093](https://arxiv.org/abs/2602.20093)  
17. ManCAR: Manifold-Constrained Latent Reasoning with Adaptive Test-Time Computation for Sequential Recommendation \- arXiv.org, accessed on March 28, 2026, [https://arxiv.org/html/2602.20093v1](https://arxiv.org/html/2602.20093v1)  
18. ManCAR: Manifold-Constrained Latent Reasoning with Adaptive Test-Time Computation for Sequential Recommendation \- ResearchGate, accessed on March 28, 2026, [https://www.researchgate.net/publication/401132801\_ManCAR\_Manifold-Constrained\_Latent\_Reasoning\_with\_Adaptive\_Test-Time\_Computation\_for\_Sequential\_Recommendation](https://www.researchgate.net/publication/401132801_ManCAR_Manifold-Constrained_Latent_Reasoning_with_Adaptive_Test-Time_Computation_for_Sequential_Recommendation)  
19. Latent Debate: A Surrogate Framework for Interpreting LLM Thinking \- arXiv, accessed on March 28, 2026, [https://arxiv.org/html/2512.01909v1](https://arxiv.org/html/2512.01909v1)  
20. Latent Debate: A Surrogate Framework for Interpreting LLM Thinking \- arXiv.org, accessed on March 28, 2026, [https://arxiv.org/html/2512.01909v2](https://arxiv.org/html/2512.01909v2)  
21. Latent Debate: A Surrogate Framework for Interpreting LLM Thinking \- arXiv, accessed on March 28, 2026, [https://www.arxiv.org/pdf/2512.01909](https://www.arxiv.org/pdf/2512.01909)  
22. Latent Debate: A Surrogate Framework for Interpreting LLM ... \- arXiv, accessed on March 28, 2026, [https://arxiv.org/abs/2512.01909](https://arxiv.org/abs/2512.01909)  
23. \[2601.05593\] PaCoRe: Learning to Scale Test-Time Compute with Parallel Coordinated Reasoning \- arXiv.org, accessed on March 28, 2026, [https://arxiv.org/abs/2601.05593](https://arxiv.org/abs/2601.05593)  
24. Parallel Coordinated Reasoning (PaCoRe) \- Emergent Mind, accessed on March 28, 2026, [https://www.emergentmind.com/topics/parallel-coordinated-reasoning-pacore](https://www.emergentmind.com/topics/parallel-coordinated-reasoning-pacore)  
25. Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach, accessed on March 28, 2026, [https://neurips.cc/virtual/2025/poster/117966](https://neurips.cc/virtual/2025/poster/117966)  
26. What is a hierarchical reasoning model (HRM)? \- IBM, accessed on March 28, 2026, [https://www.ibm.com/think/topics/hierarchical-reasoning-model](https://www.ibm.com/think/topics/hierarchical-reasoning-model)  
27. Hierarchical Reasoning Models: Thinking in Layers | Apolo AI Launchpad, accessed on March 28, 2026, [https://www.apolo.us/blog-posts/hierarchical-reasoning-models-thinking-in-layers](https://www.apolo.us/blog-posts/hierarchical-reasoning-models-thinking-in-layers)  
28. Hierarchical Reasoning Models: When 27M Parameters Outperform Chain-of-Thought, accessed on March 28, 2026, [https://towardsai.net/p/machine-learning/hierarchical-reasoning-models-when-27m-parameters-outperform-chain-of-thought](https://towardsai.net/p/machine-learning/hierarchical-reasoning-models-when-27m-parameters-outperform-chain-of-thought)  
29. Are Your Reasoning Models Reasoning or Guessing? A Mechanistic Analysis of Hierarchical Reasoning Models \- arXiv, accessed on March 28, 2026, [https://arxiv.org/html/2601.10679](https://arxiv.org/html/2601.10679)  
30. Spectral attention for transformers: frequency-domain filtering of attention maps \- ResearchGate, accessed on March 28, 2026, [https://www.researchgate.net/publication/403094419\_Spectral\_attention\_for\_transformers\_frequency-domain\_filtering\_of\_attention\_maps](https://www.researchgate.net/publication/403094419_Spectral_attention_for_transformers_frequency-domain_filtering_of_attention_maps)  
31. (PDF) Spectral attention for transformers: frequency-domain filtering of attention maps \- ResearchGate, accessed on March 28, 2026, [https://www.researchgate.net/publication/403094419\_Spectral\_attention\_for\_transformers\_frequency-domain\_filtering\_of\_attention\_maps/download](https://www.researchgate.net/publication/403094419_Spectral_attention_for_transformers_frequency-domain_filtering_of_attention_maps/download)  
32. Learned frequency masks from adaptive spectral attention showing... | Download Scientific Diagram \- ResearchGate, accessed on March 28, 2026, [https://www.researchgate.net/figure/Learned-frequency-masks-from-adaptive-spectral-attention-showing-head-specific-frequency\_fig2\_403094419](https://www.researchgate.net/figure/Learned-frequency-masks-from-adaptive-spectral-attention-showing-head-specific-frequency_fig2_403094419)  
33. Training efficiency trade-off analysis showing the relationship between... \- ResearchGate, accessed on March 28, 2026, [https://www.researchgate.net/figure/Training-efficiency-trade-off-analysis-showing-the-relationship-between-computational\_fig1\_403094419](https://www.researchgate.net/figure/Training-efficiency-trade-off-analysis-showing-the-relationship-between-computational_fig1_403094419)  
34. (PDF) Phase Coherence Tracing in Transformer Based Large ..., accessed on March 28, 2026, [https://www.researchgate.net/publication/398891007\_Phase\_Coherence\_Tracing\_in\_Transformer\_Based\_Large\_Language\_Models\_Through\_Self\_Referential\_Token\_Interference](https://www.researchgate.net/publication/398891007_Phase_Coherence_Tracing_in_Transformer_Based_Large_Language_Models_Through_Self_Referential_Token_Interference)  
35. From Steering Vectors to Conceptors: Compositional Affine Activation Steering for LLMs, accessed on March 28, 2026, [https://openreview.net/forum?id=0Yu0eNdHyV](https://openreview.net/forum?id=0Yu0eNdHyV)  
36. On the Non-Identifiability of Steering Vectors in Large Language Models \- arXiv.org, accessed on March 28, 2026, [https://arxiv.org/html/2602.06801v3](https://arxiv.org/html/2602.06801v3)  
37. HiLoRA: Hierarchical Low-Rank Adaptation for Personalized Federated Learning \- arXiv.org, accessed on March 28, 2026, [https://arxiv.org/html/2603.02785v1](https://arxiv.org/html/2603.02785v1)  
38. tLoRA: Efficient Multi-LoRA Training with Elastic Shared Super-Models \- arXiv, accessed on March 28, 2026, [https://arxiv.org/html/2602.07263v1](https://arxiv.org/html/2602.07263v1)  
39. tLoRA: Efficient Multi-LoRA Training with Elastic Shared Super-Models \- arXiv, accessed on March 28, 2026, [https://arxiv.org/html/2602.07263v2](https://arxiv.org/html/2602.07263v2)  
40. Beyond Mode Elicitation: Diversity-Preserving Reinforcement Learning via Latent Diffusion Reasoner \- arXiv, accessed on March 28, 2026, [https://arxiv.org/html/2602.01705v2](https://arxiv.org/html/2602.01705v2)  
41. LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning \- arXiv, accessed on March 28, 2026, [https://arxiv.org/html/2510.04573v2](https://arxiv.org/html/2510.04573v2)  
42. LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning \- arXiv, accessed on March 28, 2026, [https://arxiv.org/html/2510.04573v5](https://arxiv.org/html/2510.04573v5)  
43. Energy-Based Models for AI Reasoning: Beyond LLM Limitations, accessed on March 28, 2026, [https://logicalintelligence.com/blog/energy-based-models-for-reasoning](https://logicalintelligence.com/blog/energy-based-models-for-reasoning)  
44. KLDrive: Fine-Grained 3D Scene Reasoning for Autonomous Driving based on Knowledge Graph \- arXiv, accessed on March 28, 2026, [https://arxiv.org/html/2603.21029v1](https://arxiv.org/html/2603.21029v1)  
45. This New AI Model Thinks Without Language (w/ Eve Bodnia of Logical Intelligence), accessed on March 28, 2026, [https://www.youtube.com/watch?v=rvwBsWDOFIE](https://www.youtube.com/watch?v=rvwBsWDOFIE)  
46. Energy-Based Transformers are Scalable Learners and Thinkers | OpenReview, accessed on March 28, 2026, [https://openreview.net/forum?id=ZBj3Qp1bYg](https://openreview.net/forum?id=ZBj3Qp1bYg)  
47. Optimizing Multi-Agent Weather Captioning via Text Gradient ... \- arXiv, accessed on March 28, 2026, [https://arxiv.org/abs/2603.21673](https://arxiv.org/abs/2603.21673)  
48. Optimizing Multi-Agent Weather Captioning via Text Gradient Descent: A Training-Free Approach with Consensus-Aware Gradient Fusion \- ResearchGate, accessed on March 28, 2026, [https://www.researchgate.net/publication/403072499\_Optimizing\_Multi-Agent\_Weather\_Captioning\_via\_Text\_Gradient\_Descent\_A\_Training-Free\_Approach\_with\_Consensus-Aware\_Gradient\_Fusion](https://www.researchgate.net/publication/403072499_Optimizing_Multi-Agent_Weather_Captioning_via_Text_Gradient_Descent_A_Training-Free_Approach_with_Consensus-Aware_Gradient_Fusion)  
49. Optimizing Multi-Agent Weather Captioning via Text Gradient Descent: A Training-Free Approach with Consensus-Aware Gradient Fusion \- arXiv, accessed on March 28, 2026, [https://arxiv.org/html/2603.21673v1](https://arxiv.org/html/2603.21673v1)  
50. Heterogeneous Agent Collaborative Reinforcement Learning \- arXiv, accessed on March 28, 2026, [https://arxiv.org/abs/2603.02604](https://arxiv.org/abs/2603.02604)  
51. Heterogeneous Agent Collaborative Reinforcement Learning \- arXiv, accessed on March 28, 2026, [https://arxiv.org/html/2603.02604v1](https://arxiv.org/html/2603.02604v1)  
52. Heterogeneous Agent Collaborative Reinforcement Learning \- arXiv, accessed on March 28, 2026, [https://arxiv.org/pdf/2603.02604](https://arxiv.org/pdf/2603.02604)  
53. Composing Global Solutions to Reasoning Tasks via Algebraic ..., accessed on March 28, 2026, [https://openreview.net/forum?id=tD7MLq0dbZ](https://openreview.net/forum?id=tD7MLq0dbZ)  
54. In-Context Algebra \- arXiv.org, accessed on March 28, 2026, [https://arxiv.org/html/2512.16902v2](https://arxiv.org/html/2512.16902v2)  
55. Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets | OpenReview, accessed on March 28, 2026, [https://openreview.net/forum?id=tD7MLq0dbZ\&referrer=%5Bthe%20profile%20of%20Yuandong%20Tian%5D(%2Fprofile%3Fid%3D\~Yuandong\_Tian1)](https://openreview.net/forum?id=tD7MLq0dbZ&referrer=%5Bthe+profile+of+Yuandong+Tian%5D\(/profile?id%3D~Yuandong_Tian1\))