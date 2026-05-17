# Parallel Deliberation Architecture

Status: Fruehe Ideenphase, noch nicht getestet.
Datum: 2026-03-26, erweitert 2026-03-27
Autoren: Toby, Claude

## Kernidee

Eine Modell-Architektur, in der Reasoning nicht sequentiell Token-fuer-Token passiert,
sondern als parallele Deliberation mehrerer Worker im Vektorraum.

Das Gesamtkonzept besteht aus drei unabhaengigen Hypothesen mit zunehmender Spekulation:

### Hypothese 1: Paralleles Reasoning (Kern)
Mehrere Worker verarbeiten denselben Input parallel aus verschiedenen Perspektiven.
Die Ergebnisse werden zusammengefuehrt und fliessen in den naechsten Schritt.
Ein Transformer-Layer muss nicht ein einzelner Forward Pass sein, sondern kann
mehrere parallele Passes sein, deren Ergebnisse gemergt werden bevor es weitergeht.
Diese Hypothese steht fuer sich und ist mit bestehenden Tools testbar.

### Hypothese 2: Dynamische Parameter-Allokation (Erweiterung)
Wenn ohnehin Worker vorhanden sind, koennen die Parameter dynamisch verteilt werden
statt fester Experten (MoE-Upgrade). Nicht "du BIST der Mathe-Experte", sondern
"hier ist eine Mathe-Frage, nimm dir die passenden Weights".
Setzt Hypothese 1 voraus, ist aber keine Voraussetzung fuer sie.

### Hypothese 3: Ganzheitlicher Output (Spekulation)
Der Output entsteht als Ganzes statt Token fuer Token (diffusionsartig).
Aktuell am weitesten weg von etwas Testbarem. Verwandt mit Forschung zu
Diffusion-basierten Sprachmodellen (MDLM, SEDD), die mit aehnlichen Problemen
kaempfen (Sequentialitaet von Sprache vs. parallele Erzeugung).

## Bibliotheksmetapher

Eine Bibliothek (der Parameterraum). Darin:
- 3 Eingangsdudes: Text -> Wort-Uebersetzung -> interne Repraesentation (Embedding-Layer)
- n Runner (Worker): Werden mit Teilfragen losgeschickt, greifen auf beliebige Regale zu
- 1 Nebentisch (Deliberationsraum): Wo Runner Ergebnisse zusammenfuehren, diskutieren, iterieren
- 1 Kontrollmechanismus: Entsteht aus der Interaktion (Konvergenz), kein separates Modul
- 3 Ausgangsdudes: Interne Repraesentation -> Wort-Uebersetzung -> Text (Unembedding-Layer)
- Spezial-Runner: Koennen "rausrennen" (Web-Search, Tool-Calls im Vektorraum)

## Unterschiede zu bestehenden Architekturen

### vs. Dense Transformer
- Dense: 1 sequentieller Forward Pass durch alle Layer, Token fuer Token
- PDA: Parallele Verarbeitung, dynamisch, iterativ, Ergebnis statt Token

### vs. MoE (Mixture of Experts)
- MoE: Router waehlt feste Experten (feste Weight-Subsets), 1 pro Token, keine Kommunikation
- PDA: Worker bekommen dynamisch die Parameter die sie brauchen. Nicht "du BIST der Mathe-Experte"
  sondern "hier ist eine Mathe-Frage, nimm dir die passenden Weights". Worker sind generisch.
  Worker kommunizieren miteinander (Deliberation).

### vs. Chain of Thought / Chain of Continuous Thought
- CoT: Sequentielles Reasoning in Sprache (Token-Output als Zwischenschritte)
- CCoT: Reasoning in kontinuierlichen Vektoren, aber sequentiell
- PDA: Reasoning in Vektoren, PARALLEL, mit Iteration

### vs. Diffusion Models
- Diffusion: Iterative Verfeinerung eines Gesamtergebnisses
- PDA: Aehnlich beim Output (Ergebnis statt Token-Sequenz), aber mit strukturierter
  Deliberation statt ungerichteter Denoising-Schritte

### Naeher verwandte Arbeiten

Die obigen Vergleiche decken die grossen Architekturklassen ab. Es gibt aber Arbeiten,
die naeher an PDA liegen und von denen sich das Konzept schaerfer abgrenzen muss:

**Parallel Decoding (Medusa, EAGLE, Lookahead Decoding)**
- Erzeugen mehrere Token-Kandidaten parallel, verifizieren dann gegen das Basismodell.
- Ziel: Inference-Speedup bei gleichem Output. Kein anderes Reasoning.
- PDA: Parallelitaet nicht fuer Speed, sondern fuer Perspektiv-Diversitaet.
  Verschiedene Worker sollen verschiedene Ergebnisse liefern, nicht schneller
  dasselbe Ergebnis.

**Multi-Head Latent Attention (MLA, z.B. DeepSeek-V2)**
- Komprimiert KV-Cache ueber Low-Rank-Projektion, mehrere Attention-Heads
  arbeiten auf latenten Repraesentationen.
- Parallelitaet innerhalb eines Layers, aber alle Heads sehen denselben Input
  und werden gemeinsam trainiert. Keine Deliberation, kein Konsensus.
- PDA: Worker sind staerker entkoppelt (verschiedene Steering Vectors),
  und die Zusammenfuehrung ist explizit iterativ statt ein gelernter Layer.

**Tree of Thoughts (ToT) / Graph of Thoughts (GoT)**
- Mehrere Reasoning-Pfade parallel, mit Evaluation und Backtracking.
- Arbeiten auf Token-/Text-Ebene, nicht im Vektorraum.
- PDA: Aehnliche Struktur (parallel, dann bewerten), aber die Deliberation
  passiert in den Aktivierungen, nicht in generiertem Text.

**Best-of-N / Majority Voting**
- Mehrere Completions generieren, beste waehlen oder per Mehrheit entscheiden.
- Volle Modell-Runs, keine Interaktion zwischen den Pfaden.
- PDA: Worker teilen sich Parameter und kommunizieren ueber den Deliberationsraum.
  Nicht "generiere 10 Antworten und waehle", sondern "denke aus 10 Perspektiven
  und konvergiere zu einer".

## Architektur-Skizze

### Kern-Architektur (Hypothese 1: Perspektiven-basiert)

```
Input (Text)
    |
[Embedding-Layer: 3 Schichten, Text -> interne Repraesentation]
    |
    +------------------+------------------+
    |                  |                  |
[Worker 1]       [Worker 2]    ...  [Worker n]
Steering Vec A   Steering Vec B      Steering Vec N
    |                  |                  |
    +------------------+------------------+
                       |
                [Deliberationsraum]         <-- Cross-Attention zwischen Worker-Outputs
                       |
                [Divergenz messen]          <-- Cosine Distance o.ae.
                       |
              Divergenz < Schwellwert?
              /                    \
            ja                     nein
             |                      |
    [Konsolidiertes               [Ergebnisse + Divergenz-Info
     Ergebnis]                     als neuer Input -> naechste Runde]
             |
[Unembedding-Layer: 3 Schichten, interne Repraesentation -> Text]
    |
Output
```

### Erweiterte Architektur (mit Hypothese 2: Dynamische Parameter)

```
Input (Text)
    |
[Embedding-Layer]
    |
[Themen-Klassifikator]  -->  [Steering Vector Pool]
    |                              |
    +--- waehlt passende Vektoren -+
    |
[Worker 1..n mit dynamisch zugewiesenen Steering Vectors + Parameter-Subsets]
    |
[Deliberation + Iteration wie oben]
    |
Output
```

## Hardware-Vision

- 1 Parameter-Server (GPU mit viel VRAM): Haelt alle Weights (z.B. 100B)
- n Worker-GPUs (klein, z.B. 8GB): Generische Recheneinheiten
- Koordinator (CPU oder leichte GPU): Orchestriert Parameterverteilung

Pro Iteration bekommt jeder Worker nur die Parameter die er braucht (~8B Subset).
Skalierung ueber Breite (mehr kleine GPUs) statt Hoehe (groessere GPUs).

Flaschenhals: Bandbreite Parameter-Server -> Worker.
8B fp16 = ~16GB. NVLink: ~18ms, PCIe 5.0: ~250ms.
Mitigation: Streaming (rechnen waehrend Transfer), Lokalitaet (Parameter wiederverwenden).

## Offene Fragen

### Kritisch (Hypothese 1)
1. **Kompatibilitaet**: Mittlere Layer bestehender Modelle erwarten spezifischen Input.
   Parallele Verarbeitung + Merging wird die Repraesentationen wahrscheinlich brechen.
   -> Retraining noetig, mindestens fuer den Deliberationsmechanismus.

2. **Trainingssignal**: Worauf optimiert man die Deliberation?
   Next-Token-Prediction passt nicht. Outcome-basiertes RL?
   "War dieses Deliberationsergebnis besser als ohne Deliberation?"
   Teilweise adressiert durch Konsensus als internes Signal (s.u.).

3. **Steering Vector Eignung**: Auf welcher Abstraktionsebene lassen sich sinnvolle
   Perspektiv-Vektoren extrahieren? Tonalitaet/Truthfulness funktionieren.
   Semantische Perspektiven ("kritisch", "analytisch") sind noch unklar.

### Wichtig (Hypothese 1)
4. **Konvergenz**: Wie sicherstellen, dass Deliberation terminiert?
   Halting basierend auf Divergenz-Schwellwert + Veraenderungsrate + Timeout.

5. **Merge-Mechanismus**: Wie kombiniert man Worker-Outputs optimal?
   Gewichteter Durchschnitt, Concatenation + Projektion, Cross-Attention?
   Wahrscheinlich aufgabenabhaengig — selbst eine offene Frage.

6. **Evaluation**: Auf welchen Aufgaben misst man ob es besser ist?
   Reasoning-Benchmarks? Kreative Aufgaben? Offene Probleme?

### Wichtig (Hypothese 2)
7. **Dynamische Parameter-Allokation**: Wie entscheidet man welche
   Parameter ein Worker braucht? Gelerntes Routing? Themenbasierte Heuristik?
   Oder: Reicht es wenn alle Worker auf denselben vollstaendigen Parametersatz
   zugreifen und die Differenzierung nur ueber Steering Vectors laeuft?

8. **Tool-Integration**: Tools liefern aktuell Text. Fuer Vektorraum-Integration
   braucht jedes Tool einen Encoder in den internen Repraesentationsraum.

### Erkundungswert
9. **Skalierungsverhalten**: Wird es mit mehr Workern linear besser? Sublinear? Gibt es einen Sweet Spot?
10. **Spezialisierung emergent**: Entwickeln generische Worker ueber Training Spezialisierungen?

## Konsensus als internes Qualitaetssignal

Mehrere Worker koennen dieselbe Teilfrage parallel bearbeiten. Der Grad der Uebereinstimmung
ist ein internes Qualitaetssignal:

- **Hoher Konsensus** (8/10 gleicher Schluss): Loesung ist robust, Ergebnis weiterreichen.
- **Niedriger Konsensus** (5:5): Entweder zwei gleichwertige Antworten oder unterdeterminierte
  Frage. Beides wertvolle Information. -> Mehr Iteration, oder Unsicherheit an Output melden.
- **Kein Konsensus**: Frage moeglicherweise falsch gestellt. -> Zurueck an Koordinator zur Reformulierung.

Unterschied zu klassischen Ensemble-Methoden: Ensemble = verschiedene Modelle, misst Modell-Varianz.
PDA = gleiche Parameter, verschiedene Startpunkte/Fragen, misst Stabilitaet des Loesungsraums.

Loest teilweise das Trainingssignal-Problem: Konsensus ist ein internes Signal, braucht keinen
externen Evaluator. Kann als Komponente des Halting-Mechanismus dienen (hoher Konsensus -> stopp)
und als Konfidenz-Signal im Output.

### Einschraenkung: Gemeinsame Blindheit
Konsensus ist ein Stabilitaets- und Halting-Signal, aber kein Qualitaetsbeweis.
Wenn alle Worker dieselben Parameter nutzen und sich nur durch Steering Vectors
unterscheiden, teilen sie dieselben gelernten Biases. Hoher Konsensus kann bedeuten:
"alle sind sich einig" ODER "alle liegen auf dieselbe Art daneben".

Konsequenz: Fuer die Evaluation der Architektur ist ein externer Task-Massstab
unverzichtbar (Reasoning-Benchmarks, Ground-Truth-Vergleiche). Konsensus taugt als
Laufzeit-Heuristik (wann stoppen, wie sicher ist das Ergebnis), aber nicht als
alleiniges Guetekriterium waehrend der Entwicklung.

## Perspektiven statt Aufgabenzerlegung

Zentrale Design-Entscheidung: Worker bekommen nicht verschiedene Teilaufgaben,
sondern verschiedene Perspektiven auf dieselbe Aufgabe.

### Warum keine Task-Decomposition
Ein aktiver Koordinator der Aufgaben intelligent zerlegt, muss die Aufgabe dafuer
bereits verstanden haben — ein Reasoning-Problem VOR dem eigentlichen Reasoning (Henne-Ei).
Ausserdem muss eine zerlegte Aufgabe hinterher wieder zusammengesetzt werden,
und das Zusammensetzen ist oft genauso schwer wie das Problem selbst.

### Perspektiven-Ansatz
Stattdessen: Alle Worker starten mit demselben Input, aber mit verschiedenen
Steering Vectors. Kein "du machst Teilfrage A", sondern "du betrachtest das Problem
durch Linse X". Die Diversitaet entsteht durch die Startbedingungen,
nicht durch explizite Aufgabenteilung.

Vorteile:
- Kein gelernter Koordinator noetig
- Diversitaet kommt aus der Initialisierung
- Konvergenz kommt aus dem Deliberationsraum
- Zusammenfuehrung ist ein Konvergenz-Problem (mathematisch greifbar),
  kein Rekombinations-Problem

### Feste Perspektiv-Vektoren (Stufe 1)
Einfachster Ansatz: Ein festes Set an Steering Vectors, fest zugewiesen.
Jeder Worker hat seinen Schwerpunkt — z.B. kritisch, realistisch, analytisch,
kreativ. Aehnlich wie Experts bei MoE, aber leichtgewichtiger: der Worker ist
derselbe, nur die Linse ist anders.

Offene Frage: Auf welcher Abstraktionsebene funktionieren Steering Vectors zuverlaessig?
Bisherige Forschung zeigt gute Ergebnisse bei Tonalitaet, Sprache, Truthfulness.
Ob sich high-level-semantische Perspektiven ("kritisch", "oekonomisch") sauber
im Aktivierungsraum trennen lassen, ist unklar. Erster Schritt: mit nachweislich
extrahierbaren Dimensionen arbeiten und pruefen ob die Diversitaet im Output
trotzdem produktiv ist.

### Dynamische Perspektivzuweisung (Stufe 2)
Ein leichter Klassifikator erkennt das Thema und waehlt passende Steering Vectors
aus einem vordefinierten Pool. Kein vollstaendiger Koordinator, sondern ein Routing
auf Basis einfacher Themenerkennung.

### Volldynamisch (Stufe 3)
Steering Vectors werden zur Inferenzzeit generiert. Setzt voraus, dass Stufe 1 und 2
positive Ergebnisse liefern.

## Iteratives Stufenmodell

Die Deliberation laeuft in Runden:

```
Runde 1: n Worker verarbeiten Input parallel mit verschiedenen Perspektiven
         -> n Ergebnis-Vektoren
         -> Divergenz messen

         Divergenz niedrig?  -> Ergebnis weiterreichen (Konsensus erreicht)
         Divergenz hoch?     -> Ergebnisse als neuer Input in Runde 2

Runde 2: n Worker verarbeiten angereicherten Input
         (Originalfrage + Ergebnisse + Divergenz-Info aus Runde 1)
         -> Divergenz messen
         -> weiter oder stoppen

...

Runde k: Timeout (z.B. k=5 oder k=10) -> Ergebnis verwenden,
         Unsicherheit im Output kommunizieren
```

Jede Runde hat mehr Kontext als die vorherige: die Worker sehen nicht nur die
Originalfrage, sondern auch wo die vorherige Runde sich uneinig war.
Das ist iterative Verfeinerung mit eingebautem Informationsgewinn.

Halting-Kriterien:
- Primaer: Divergenz unter Schwellwert (Konsensus)
- Sekundaer: Veraenderungsrate zwischen Runden (Konvergenz)
- Fallback: Maximale Rundenzahl (Timeout)

## Signalverarbeitungsbasierte Konvergenzmetriken

Cosine Similarity zwischen Worker-Outputs ist eindimensional: "aehnlich oder nicht."
Signaltechnische Konzepte liefern ein deutlich reicheres Bild fuer die Bewertung
der Deliberation und die Halting-Entscheidung.

### SNR (Signal-to-Noise Ratio)
Aus der Audiotechnik: Verhaeltnis von Nutzsignal zu Stoersignal.

Uebertragung: Die Konsensus-Richtung (Durchschnitt aller Worker-Outputs) ist das
"Signal", die individuelle Abweichung jedes Workers davon ist "Noise."
SNR = ||mean(worker_outputs)||^2 / mean(||worker_i - mean||^2)

SNR ueber Runden tracken:
- SNR steigt -> Ergebnis wird klarer, Deliberation ist produktiv
- SNR stagniert -> Konvergenz erreicht, weitere Runden bringen nichts
- SNR sinkt -> Deliberation destabilisiert, sofort stoppen

Vorteil gegenueber einfacher Divergenz: SNR misst nicht nur OB konvergiert wird,
sondern WIE SCHNELL, und erkennt Destabilisierung frueh.

### Phasenkohaerenz
Aus Multi-Mikrofon-Setups: Misst wie stark mehrere Signale korreliert sind,
um Signal von diffusem Raumschall zu trennen.

Uebertragung: Worker-Outputs in ihre dominanten Komponenten zerlegen (SVD/PCA).
Pro Komponente die Kohaerenz zwischen den Workern messen.

- Hohe Kohaerenz in einer Komponente -> Worker sind sich in diesem Aspekt einig,
  das ist stabiles "Signal"
- Niedrige Kohaerenz -> Entweder Rauschen (irrelevant) ODER produktive
  Perspektiv-Differenz (die interessanteste Information)

Unterscheidung zwischen "Rauschen" und "produktiver Differenz":
Wenn die inkohaerenten Komponenten ueber Runden stabil bleiben (gleiche Worker
weichen immer in derselben Richtung ab), ist es strukturierte Meinungsverschiedenheit.
Wenn sie ueber Runden zufaellig schwanken, ist es Rauschen.

### Crest Factor
Aus der Audio-Analyse: Verhaeltnis von Peak zu RMS (Durchschnitt).
Misst ob ein Signal konzentrierte Spitzen hat oder gleichmaessig verteilt ist.

Uebertragung: Angewandt auf den Divergenz-Vektor zwischen Workern.
Crest Factor = max(|divergenz|) / rms(divergenz)

- Hoher Crest Factor -> Uneinigkeit ist in wenigen Dimensionen konzentriert.
  Das bedeutet: es gibt einen spezifischen Streitpunkt. Die Worker sind sich
  insgesamt einig, aber an einer konkreten Stelle nicht.
  -> Gezielt diese Dimensionen untersuchen oder zusaetzliche Iteration
     nur auf den strittigen Aspekt fokussieren.

- Niedriger Crest Factor -> Uneinigkeit ist breit gestreut.
  Das bedeutet: generelle Unsicherheit, die Worker haben keinen klaren
  Konsensus in keiner Richtung.
  -> Mehr Iteration oder Unsicherheit im Output kommunizieren.

### Kombination als Halting-Entscheidung

Die drei Metriken zusammen ergeben ein mehrdimensionales Bild:

```
SNR hoch + Kohaerenz hoch + beliebiger Crest Factor
-> Starker Konsensus, stoppen, Ergebnis ist robust.

SNR steigend + Kohaerenz steigend
-> Noch nicht fertig, aber auf gutem Weg. Weiter iterieren.

SNR hoch + Kohaerenz niedrig in einzelnen Komponenten + Crest Factor hoch
-> Konsensus insgesamt, aber ein spezifischer Streitpunkt.
   Ergebnis verwenden, aber Unsicherheit gezielt kommunizieren.

SNR niedrig + Kohaerenz niedrig + Crest Factor niedrig
-> Generelle Unsicherheit. Entweder weiter iterieren oder
   Ergebnis mit hoher Unsicherheitsmarkierung ausgeben.

SNR sinkend (in beliebiger Kombination)
-> Deliberation destabilisiert. Sofort stoppen, letztes stabiles
   Ergebnis verwenden.
```

## Experimentalplan

### Grundprinzipien

Jeder Durchlauf wird gegen eine klare Baseline verglichen: ein normaler Single Forward Pass
desselben Modells auf denselben Daten. Alle Variablen (Worker-Anzahl, Merge-Strategie,
Rundenzahl) werden einzeln variiert, nicht gleichzeitig — sonst ist unklar was den
Unterschied verursacht.

Ein negatives Ergebnis bei einer Konfiguration bedeutet nicht, dass die Kernhypothese
widerlegt ist. Es gibt mehrere Stellschrauben (Merge, Vektoren, Anzahl, Layer-Auswahl),
und erst wenn keine Konfiguration einen Effekt zeigt, wird es ernst.

### Baseline

Fuer jeden Datenpunkt:
- Single Forward Pass durch das vollstaendige Modell, ohne Modifikation
- Selbes Modell, selber Input, selbe Decoding-Parameter
- Mehrfach wiederholt (z.B. 10x pro Frage) um Varianz des Basismodells zu messen

Die Baseline-Varianz ist entscheidend: wenn das Basismodell bei 10 Durchlaeufen
schon zwischen richtig und falsch schwankt, muss der PDA-Effekt deutlich
ueber diese Schwankung hinausgehen.

### Unabhaengige Variablen (was wir aendern)

**A: Worker-Anzahl**
- 2, 3, 5, 10 Worker gegen Baseline (1)
- Fragestellung: Ab welcher Anzahl zeigt sich ein Effekt? Gibt es einen Sweet Spot
  ab dem mehr Worker keinen Zusatznutzen bringen oder sogar schaden?

**B: Merge-Strategie**

Standard-ML-Merges:
- Gewichteter Durchschnitt (einfachstes, Baseline-Merge)
- Concatenation + lineare Projektion zurueck auf Originaldimension
- Elementweises Maximum (staerkste Aktivierung gewinnt)
- Attention-basierter Merge (Worker-Outputs als Keys/Values, Query aus Durchschnitt)

Signalverarbeitungsbasierte Merges (aus Audio-Engineering-Analogie):
- Phase-Alignment: Vor dem Merge Cosine Similarity zwischen Worker-Aktivierungen
  messen. Phasengleiche Anteile (gleiche Richtung im Vektorraum) verstaerken sich
  gegenseitig — konstruktive Interferenz. Gegenlaeufige Anteile markieren Stellen
  wo Worker sich uneinig sind — die interessanten Stellen, die entweder gesondert
  behandelt oder als Unsicherheits-Signal genutzt werden koennen.
- Frequenz-selektiver Merge: Aktivierungen per SVD oder PCA in Komponenten zerlegen.
  "Niedrige Frequenzen" (dominante Komponenten, grobe Semantik) einfach mitteln,
  "hohe Frequenzen" (feine Unterschiede, Details) staerker gewichten oder
  separat behandeln. Analog zu Multiband-Processing im Audio.
- Sidechain-Merge: Asymmetrischer Merge, bei dem ein Worker-Output den anderen
  moduliert statt gleichwertiger Zusammenfuehrung. Elementweise Multiplikation
  oder gelernte Gating-Funktion. Ein Worker liefert das "Was", der andere
  steuert das "Wie stark". Kommt aus der Sidechain-Compression-Analogie (SWE).

Alle diese Operationen sind differenzierbar (Cosine Similarity, SVD, elementweise
Multiplikation), also prinzipiell in einen Trainingsprozess integrierbar.

Kombinationen: Signalverarbeitungs-Merges koennen auch mit Standard-Merges
kombiniert werden. Z.B. Phase-Alignment als Vorverarbeitung vor Attention-Merge,
oder Frequenz-Zerlegung mit anschliessendem gewichtetem Durchschnitt pro Band.

Fragestellung: Welcher Mechanismus erhaelt die meiste Information
bei geringstem Repraesentationsbruch? Liefern signalverarbeitungsbasierte
Ansaetze robustere Ergebnisse als naive Mittelung?

**C: Rundenzahl (Iteration)**
- 1 Runde (reiner Merge, kein Feedback)
- 2, 3, 5, 10 Runden (Merge-Ergebnis als neuer Input)
- Fragestellung: Konvergiert die Divergenz zwischen den Runden?
  Ab welcher Runde sind die Aenderungen nur noch minimal?
  Verbessert sich die Task-Qualitaet mit mehr Runden oder stagniert/degradiert sie?

**D: Steering Vectors**
- Ohne Steering Vectors (nur verschiedene Random Seeds / Dropout-Masken als Diversitaet)
- Bekannte, validierte Dimensionen (z.B. Truthfulness-Vektor aus bestehender Forschung)
- Semantische Perspektiven (kritisch, analytisch, kreativ — falls extrahierbar)
- Fragestellung: Braucht man ueberhaupt Steering Vectors fuer den Effekt,
  oder reicht reine Perturbation? Und wenn ja, welche Art von Vektoren hilft?

**E: Layer-Auswahl**
- Nur mittlere Layer (z.B. Layer 8-16 bei einem 24-Layer-Modell)
- Fruehe Layer (1-8)
- Spaete Layer (16-24)
- Fragestellung: In welchen Layern ist der Merge am produktivsten?
  Hypothese: mittlere Layer, weil dort die abstraktesten Repraesentationen liegen.

### Abhaengige Variablen (was wir messen)

**Funktionalitaet (laeuft es ueberhaupt)**
- Perplexity des Outputs (gemessen durch Referenzmodell): Ist der Output
  sprachlich kohaerent oder degeneriert?
- Anteil degenerierter Outputs (Repetition, Nonsense, leere Ausgaben)

**Task-Qualitaet (ist es besser)**
- Accuracy auf Benchmarks mit Ground Truth:
  - GSM8K (Multi-Step Mathe-Reasoning)
  - ARC-Challenge (Naturwissenschaftliches Reasoning)
  - TruthfulQA (Widerstand gegen gaengige Fehlannahmen)
  - HellaSwag (Commonsense-Completion)
- Accuracy-Differenz zum Single Pass (Delta), nicht nur Absolutwert

**Robustheit (ist es stabiler)**
- Varianz der Accuracy ueber mehrere Durchlaeufe derselben Frage
- Anteil der Faelle wo PDA richtig und Single Pass falsch (und umgekehrt)
- Konsistenz: Wie oft gibt dasselbe Setup bei Wiederholung dasselbe Ergebnis?

**PDA-spezifisch (Deliberations-Dynamik)**

Basis-Metriken:
- Divergenz zwischen Worker-Outputs (Cosine Distance im Aktivierungsraum)
- Korrelation zwischen Konsensus und Task-Accuracy
  (Ist hoher Konsensus tatsaechlich ein guter Praediktor fuer korrekte Antworten,
  oder ein Fall von gemeinsamer Blindheit?)

Signalverarbeitungs-Metriken (s. Abschnitt oben):
- SNR ueber Runden: Steigt es (produktive Konvergenz), stagniert es (fertig),
  oder sinkt es (Destabilisierung)?
- Phasenkohaerenz pro Komponente: In welchen Dimensionen sind Worker einig,
  in welchen nicht? Ist Inkohaerenz stabil (strukturelle Meinungsverschiedenheit)
  oder zufaellig (Rauschen)?
- Crest Factor der Divergenz: Ist Uneinigkeit konzentriert (spezifischer Streitpunkt)
  oder diffus (generelle Unsicherheit)?
- Kombination: Welche der drei Metriken korreliert am staerksten mit Task-Accuracy?
  Welche eignet sich am besten als Halting-Kriterium?

### Durchfuehrungsreihenfolge

```
Phase 1: Machbarkeit
- 1 Modell (z.B. Qwen 2.5 1.5B)
- 2 Worker, 1 Runde
- Nur gewichteter Durchschnitt als Merge
- Nur mittlere Layer
- Ohne Steering Vectors (Random Perturbation)
- 50-100 Fragen aus GSM8K
- Frage: Kommt ueberhaupt kohaerenter Output raus?

Phase 2a: Standard-ML-Merges (falls Phase 1 positiv)
- Selbes Setup, aber: Durchschnitt, Concatenation, Max, Attention vergleichen
- Beste Standard-Strategie identifizieren

Phase 2b: Signalverarbeitungs-Merges
- Phase-Alignment, Frequenz-selektiv, Sidechain testen
- Auch Kombinationen (z.B. Phase-Alignment + Attention)
- Vergleich gegen beste Standard-Strategie aus 2a

Wichtig: Falls Phase 1 mit Durchschnitt scheitert (Output degeneriert),
Phase 2b trotzdem durchfuehren. Der naive Durchschnitt ist der fraglichste
Merge-Mechanismus; Phase-Alignment oder Frequenz-selektiver Merge koennten
gerade die Probleme loesen die beim Durchschnitt auftreten (destruktive
Interferenz, Verwischung von Repraesentationen).

Phase 3: Skalierung Worker-Anzahl
- Beste Merge-Strategie aus Phase 2
- Worker: 2, 3, 5, 10
- Sweet Spot identifizieren

Phase 4: Steering Vectors
- Bestes Setup aus Phase 2+3
- Verschiedene Steering-Vektor-Typen vergleichen
- Frage: Bringt gezielte Perspektiv-Diversitaet mehr als Random-Perturbation?

Phase 5: Iteration + Konvergenz-Metriken
- Bestes Setup aus Phase 2+3+4
- 1, 2, 3, 5, 10 Runden
- SNR, Phasenkohaerenz, Crest Factor pro Runde messen
- Konvergenz-Kurve: Punkt der abnehmenden Ertraege identifizieren
- Vergleich: Welche Metrik korreliert am besten mit Task-Accuracy?
- Halting-Regel ableiten: z.B. "stoppen wenn SNR stagniert
  UND Crest Factor unter Schwellwert"

Phase 6: Breite Evaluation
- Bestes Gesamtsetup auf allen Benchmarks
- Verschiedene Aufgabentypen vergleichen
  (wo hilft PDA, wo nicht?)
- Optional: zweites Modell zum Vergleich
```

### Ergebnis-Szenarien

**Szenario A: Phase 1 scheitert (Output degeneriert)**
-> Direkt weiter zu Phase 2b (Signalverarbeitungs-Merges). Der gewichtete
   Durchschnitt ist der naivste Merge und der anfaelligste fuer destruktive
   Interferenz. Phase-Alignment oder Frequenz-selektiver Merge adressieren
   genau dieses Problem, indem sie die Zusammenfuehrung steuern statt
   blind zu mitteln. Auch Layer-Auswahl variieren.

**Szenario B: Phase 1 funktioniert, aber kein Qualitaetsgewinn**
-> Phase 4 (Steering Vectors) trotzdem durchfuehren. Ohne gezielte
   Perspektiv-Diversitaet ist reiner Random-Merge moeglicherweise
   nur Rauschen. Der Effekt koennte erst mit sinnvollen Vektoren auftreten.

**Szenario C: Qualitaetsgewinn, aber keine Robustheitsverbesserung**
-> Immer noch interessant. Bedeutet: paralleler Merge kann die Baseline
   schlagen, aber nicht zuverlaessig. Frage wird dann: kann Iteration
   (Phase 5) die Zuverlaessigkeit erhoehen?

**Szenario D: Positiv auf Reasoning, neutral auf Fakten**
-> Erwartetes Ergebnis. Wuerde bestaetigen dass PDA bei Aufgaben hilft
   die von Perspektiv-Vielfalt profitieren, nicht bei reinem Recall.

**Szenario E: Alles negativ ueber alle Konfigurationen**
-> Kernhypothese ist dann tatsaechlich geschwaecht. Aber auch das waere
   ein Ergebnis: "parallele Perspektiven im Aktivierungsraum bestehender
   Modelle bringen keinen messbaren Vorteil" ist eine verwertbare Aussage.

## Verbindung zu anderen Projekten

- Experiment "Denkprozess-Framing" (2026-03-25): Zeigte dass Verarbeitung = Sprache.
  PDA wuerde genau dieses Limit adressieren.
- KV-Injection-Forschung (Toby): Technische Grundlage fuer Vektorraum-Manipulation.
- Dadfar-Paper: Selbstreferentielle Verarbeitung hat messbare Aktivierungsmuster.
  Koennte als Referenzpunkt fuer "was passiert in den mittleren Layern" dienen.
- Semantic Wave Encoding (SWE, Toby): Audio-Engineering-Analogien fuer
  Vektorraumoperationen. Konzepte wie Phase-Alignment, Frequenzzerlegung und
  Sidechain-Modulation als Denkrahmen fuer Merge-Strategien.
  Die SWE-Erfahrung zeigte: nicht alle Teilideen funktionierten, aber einzelne
  Konzepte (z.B. Phase als Negationsmarker) waren verwertbar.
  Selbe Erwartung hier: auch wenn PDA als Gesamtarchitektur nicht aufgeht,
  koennen einzelne Merge-Mechanismen oder das Konsensus-Signal
  eigenstaendig nuetzlich sein.

## Verwandte Dokumente

- n-PDA (Native PDA): Greenfield-Architektur die parallele Deliberation
  nativ einbaut statt auf bestehende Modelle aufzusetzen. Orthogonale
  Perspektiv-Raeume, Cross-Attention, Deep Equilibrium Motor.
  Denkuebung: welche mathematischen Eigenschaften braucht man, damit
  parallele Deliberation sauber funktioniert?

- Simulations-Roadmap: Stufenplan zur Validierung der mathematischen
  Grundlagen BEVOR an bestehenden Modellen experimentiert wird.
  Simulation 1 (Konvergenz-Mathematik) -> Simulation 2 (Probing) ->
  Simulation 3 (Toy-Modell). Ergebnisse entscheiden ob PDA-Experimente
  an bestehenden Modellen ueberhaupt sinnvoll sind oder ob die
  n-PDA-Route (from scratch) bevorzugt werden sollte.

### Empfohlene Reihenfolge
Die PDA-Experimente an bestehenden Modellen (Experimentalplan oben) sind
moeglicherweise nicht der beste erste Schritt. Die Simulationsroute testet
die Kernmathematik in einer kontrollierten Umgebung und liefert schneller
belastbare Ergebnisse. Empfehlung:

1. Simulations-Roadmap durchlaufen (Wochen, nicht Monate)
2. Auf Basis der Ergebnisse entscheiden: PDA-Route oder n-PDA-Route
3. PDA-Experimentalplan (oben) nur falls Simulationen positiv UND
   Probing zeigt dass bestehende Repraesentationen zerlegbar sind
