# Simulations-Roadmap: Parallele Deliberation validieren

Status: Planung
Datum: 2026-03-27, revidiert 2026-03-29 (post-Recherche)
Autoren: Toby, Claude

## Motivation

Das Hauptrisiko der PDA-Experimente an bestehenden Modellen: Man operiert am offenen
Herzen eines trainierten Transformers. Mittlere Layer erwarten exakt den Input den
die vorherigen Layer produzieren. Wenn parallele Passes zusammengeworfen werden,
ist die Wahrscheinlichkeit hoch, dass zunaechst nur Muell rauskommt — und dann ist
unklar ob die IDEE schlecht ist oder nur die UMSETZUNG.

Die Simulationsroute testet die Mathematik in einer kontrollierten Umgebung.
Kein "hoffentlich bricht nichts" — das System wird so gebaut, dass parallele
Verarbeitung die Grundannahme ist, nicht ein Hack.

Wenn die Mathematik nicht konvergiert oder das Toy-Modell nicht lernt, weiss man
das nach Tagen statt nach Wochen Debugging an einem Llama-Modell.

## Reihenfolge

```
Simulation 1: Mathematische Konvergenz     (Tage,    kein Modell noetig)
     |
     v
Simulation 2: Probing an bestehenden       (Tage,    bestehendes Modell,
               Modellen                               kein Training)
     |
     v
Simulation 3: Toy-Modell from scratch      (Wochen,  kleines Training,
                                                      Consumer-GPU)
     |
     v
[Entscheidung: PDA-Experimente an          (erst hier, falls Sim 1-3
 bestehenden Modellen noch sinnvoll?]       positiv)
```

## Simulation 1: Mathematische Konvergenz

### Was wird getestet
Rein mathematisch, ohne jedes neuronale Netz: Konvergiert iterative parallele
Verarbeitung in orthogonalen Unterraeumen zu einem stabilen Ergebnis?

Zwei Konvergenz-Paradigmen im Vergleich (revidiert nach Recherche):
- **Fixpunkt-Iteration** (urspruengliches PDA-Design): Wiederholung bis Delta < epsilon
- **Energieminimierung** (EBT-inspiriert): Gemeinsame Energiefunktion, Gradient Descent
  bis Minimum. Theoretisch besser skalierbar (Gladstone et al., EBTs; Du et al., IRED).

Huginn (Geiping et al.) zeigt: Einfache Iteration skaliert, Root-Finding nicht.
Deshalb testen wir beides und lassen die Daten entscheiden.

### Setup
- Zufaellige UND semantisch geladene Vektoren in n orthogonalen Unterraeumen (numpy/torch)
- Semantisch geladen: Vektoren aus vortrainierten Embeddings (GloVe/word2vec),
  projiziert in orthogonale Unterraeume. Testet ob Konvergenz auch mit realistischer
  Struktur funktioniert, nicht nur mit Zufallsvektoren.
- Zwei Iterationsmechanismen:
  (a) Fixpunkt: Cross-Attention-aehnliche Projektionen, gewichtete Summen
  (b) Energie: E(x) = Summe paarweiser Worker-Distanzen + Regularisierung,
      minimiert via Gradient Descent auf Worker-Zustaende
- Verschiedene Merge-Strategien (Durchschnitt, Phase-Alignment, Frequenz-selektiv, Sidechain)
- Diversity-Enforcement: Repulsive Kraft zwischen Workern (kontrastiver Loss),
  die verhindert dass alle Worker sofort kollabieren (Coda-Forno et al.)

### Metriken

Basis:
- Konvergiert das System? (Delta x < epsilon bzw. Energie-Gradient < epsilon)
- Wie viele Iterationen/Steps bis Konvergenz?
- Ist das Ergebnis stabil? (Kleine Perturbation -> zurueck?)
- Haengt Konvergenz von Unterraumanzahl und Dimensionalitaet ab?

Signalverarbeitungs-Diagnostik (Tobys Beitrag -- in keinem der 75+ Papers):
- SNR-Kurve ueber Iterationen: Monoton steigend? Plateau? Abfall?
- Phasenkohaerenz pro Komponente (SVD): Wo sind Worker einig, wo nicht?
- Crest Factor der Divergenz: Konzentrierter Streitpunkt vs. diffuse Unsicherheit?
- Korrelation: Welche Metrik trackt Konvergenzqualitaet am besten?

Meta-Vergleich:
- Fixpunkt vs. Energieminimierung: Welches Paradigma konvergiert schneller,
  stabiler, mit weniger Sensitivitaet gegenueber Hyperparametern?

### Variablen
- Anzahl Unterraeume: 2, 3, 5, 10, 20
- Dimensionalitaet: 64, 256, 1024, 4096
- Merge-Strategie: alle aus PDA-Experimentalplan
- Konvergenz-Mechanismus: Fixpunkt-Iteration vs. Energieminimierung
- Orthogonalitaetsgrad: exakt orthogonal, fast-orthogonal (Winkel-Variation),
  nicht orthogonal (Kontrollbedingung)
- Vektortyp: Zufaellig (Gaussian) vs. semantisch geladen (Embedding-basiert)
- Diversity-Enforcement: Ohne vs. mit kontrastiver Repulsion (InfoNCE-artig)

### Erwartete Ergebnisse
- Exakt orthogonal + Durchschnitt: Triviale Konvergenz
- Nicht orthogonal + Durchschnitt: Destruktive Interferenz
- Fast-orthogonal: Ab welchem Winkel bricht es?
- Signal-Merges: Robuster als Durchschnitt bei Nicht-Orthogonalitaet?
- **NEU**: Energieminimierung wahrscheinlich robuster als Fixpunkt-Iteration,
  besonders bei hoher Unterraumzahl (EBT-Vorhersage)
- **NEU**: Ohne Diversity-Enforcement kollabieren Worker (PLR-Vorhersage,
  Diversity zerfaellt exponentiell mit Iterationstiefe)
- **NEU**: Semantisch geladene Vektoren verhalten sich anders als zufaellige
  (realistischere Testbedingung)

### Aufwand
Ein Jupyter Notebook. 2-3 Tage fuer Grundversion + systematische Sweeps.

### Entscheidungskriterium
Positiv: Konvergenz bei mindestens fast-orthogonalen Raeumen, mit mindestens
einer Merge-Strategie + Konvergenz-Mechanismus, in <20 Iterationen.
Negativ: Keine stabile Konvergenz auch bei exakter Orthogonalitaet.
-> Wenn negativ: fundamentales Problem, weitere Simulationen fragwuerdig.
Differenziert: Energieminimierung funktioniert, Fixpunkt nicht
-> DEQ-Route streichen, EBT-basierte Architektur bevorzugen.

## Simulation 2: Probing an bestehenden Modellen

### Was wird getestet
Sind die Aktivierungen bestehender Transformer tatsaechlich ueber trennbare
Unterraeume verteilt? Oder ist die Information so verflochten, dass orthogonale
Zerlegung sie zerstoert?

### Setup
- Bestehendes Open-Weights-Modell (z.B. Qwen 2.5 1.5B)
- Aktivierungen aus mittleren Layern extrahieren (TransformerLens / HuggingFace hooks)
- Post-hoc in orthogonale Unterraeume projizieren (PCA, SVD, ICA)
- Aus den einzelnen Komponenten die Antwort rekonstruieren

### Metriken
- Erklaerte Varianz pro Komponente: Wie viel Information steckt in den
  dominanten Unterraeumen vs. den restlichen?
- Rekonstruktionsfehler: Wenn man nur k von n Komponenten behaelt,
  wie stark leidet der Output?
- Semantische Trennbarkeit: Kodieren verschiedene Komponenten erkennbar
  verschiedene Aspekte? (z.B. eine Komponente fuer Faktenwissen,
  eine fuer Tonalitaet, eine fuer logische Struktur)
- Phasenkohaerenz pro Komponente ueber verschiedene Inputs:
  Bleiben die Unterraeume stabil oder verschieben sie sich pro Input?

### Variablen
- Layer-Tiefe: fruehe, mittlere, spaete Layer
- Zerlegungsmethode: PCA, SVD, ICA, Sparse Dictionary Learning
- Anzahl behaltener Komponenten: 2, 5, 10, 50% der Dimensionen
- Aufgabentyp: Fakten, Reasoning, kreativ

### Erwartete Ergebnisse
- Hypothese: Mittlere Layer haben die trennbarsten Repraesentationen.
- Hypothese: Reasoning-Aufgaben sind staerker ueber Komponenten verteilt
  als Faktenabfragen.
- Falls Aktivierungen stark verflochten sind und orthogonale Zerlegung
  systematisch Information zerstoert: Grundannahme von n-PDA ist fragwuerdig.
  Dann muesste man mit nicht-orthogonalen Ansaetzen arbeiten.

### Aufwand
TransformerLens Setup + Extraktion: 1-2 Tage.
Analyse: 2-3 Tage.
Braucht GPU fuer Modell-Inference, aber kein Training.

### Entscheidungskriterium
Positiv: Aktivierungen lassen sich in wenige Komponenten zerlegen die jeweils
semantisch interpretierbar sind, mit akzeptablem Rekonstruktionsfehler (<10%).
Negativ: Zerlegung zerstoert systematisch Information, kein Rekonstruktionspfad.

## Simulation 3: Toy-Modell from scratch

### Was wird getestet
Kann ein kleines Modell mit eingebauter paralleler Deliberation tatsaechlich
lernen? Fliessen die Gradienten? Bleibt die Orthogonalitaet unter Training stabil?

### Setup
Winziger Transformer, from scratch:
- 2-4 Layer, Embedding-Dimension 128-256
- 1-5M Parameter
- 2-4 Worker mit orthogonalen Unterraeumen
- Cross-Attention zwischen Workern
- Iteration (feste Rundenzahl, einfacher Loop, oder EBT-Energieminimierung --
    je nach Ergebnis von Simulation 1)
- Trainiert auf einfachen Aufgaben

### Trainingsaufgaben (aufsteigend)
1. Kopieren: Input unveraendert reproduzieren (testet: bricht die Architektur nichts?)
2. Sortieren: Liste von Zahlen sortieren (testet: kann das System systematische
   Transformationen lernen?)
3. Einfache Arithmetik: Addition, Multiplikation (testet: Multi-Step-Reasoning)
4. Einfache Logik: Wenn A und B dann C (testet: Kombination von Perspektiven)

### Metriken
- Trainings-Loss-Kurve: Konvergiert das Training ueberhaupt?
- Gradientennormen: Explodieren oder verschwinden sie?
- Orthogonalitaet ueber Training: Bleiben die Unterraeume orthogonal
  oder driften sie zusammen?
- Vergleich mit Baseline: Selbe Parameteranzahl, selbe Aufgabe,
  aber Standard-Transformer ohne Deliberation. Lernt PDA schneller/besser?
- Aktivierungsmuster: Nutzen verschiedene Worker tatsaechlich verschiedene
  Repraesentationen, oder kollabieren sie zur selben?

### Architektur-Varianten zum Vergleich
- n-PDA full: Orthogonale Raeume + Cross-Attention + Iteration
- Ohne Orthogonalitaet: Geteilter Raum, aber mehrere Worker + Cross-Attention
- Ohne Cross-Attention: Orthogonale Raeume, aber nur Merge am Ende
- Ohne Iteration: Orthogonal + Cross-Attention, aber nur ein Durchlauf
- Standard-Transformer: Selbe Groesse, keine Deliberation

Damit kann man isolieren welche Komponente wie viel beitraegt.

### Loss-Funktion (Multi-Objective, revidiert nach Recherche)
- L_task: Standard Cross-Entropy auf die Aufgabe
- L_ortho: PEGO-Style duale Regularisierung:
  L_preserve (Basis-Wissen erhalten) + L_diversify (Worker differenzieren)
  (Hu et al., ECCV 2024 -- direkt fuer Multi-Perspektiven-Training konzipiert)
- L_diversity: Kontrastiver Loss (InfoNCE) zwischen Worker-Aktivierungen.
  PFLICHT: Ohne explizites Enforcement kollabieren Worker in redundante
  Unterraeume (Coda-Forno et al., 2025; PLR zeigt exponentiellen Diversity-Zerfall)
- L_convergence: Penalty fuer langsame Konvergenz (Iterationen/Energie bis Ergebnis)
- L_total = L_task + alpha * L_ortho + beta * L_diversity + gamma * L_convergence
- Curriculum: Erst L_ortho + L_diversity (stabile Unterraeume aufbauen),
  dann L_task (Aufgabe lernen), dann L_convergence (Effizienz optimieren)

### Aufwand
Implementierung: 1-2 Wochen (PyTorch, von Hand)
Training: Stunden bis Tage auf einer Consumer-GPU (RTX 3060/4060 reicht)
Analyse: 1 Woche

### Entscheidungskriterien
Stark positiv: Toy-Modell lernt, Orthogonalitaet bleibt stabil, Worker
differenzieren sich, Deliberation-Variante schlaegt Standard-Baseline.
Schwach positiv: Modell lernt, aber kein klarer Vorteil gegenueber Baseline.
-> Dann liegt es moeglicherweise an der Aufgabengroesse, nicht am Prinzip.
Negativ: Training instabil, Orthogonalitaet kollabiert, Worker degenerieren.
-> Fundamentales Architekturproblem, nicht nur Skalierungsfrage.

## Entscheidungsmatrix nach Simulation 1-3

```
Sim 1 (Mathe)    Sim 2 (Probing)    Sim 3 (Toy)     -> Naechster Schritt
---------------------------------------------------------------------------
Positiv          Positiv            Positiv          -> Groesseres Modell trainieren
                                                        (n-PDA Forschungsprojekt)

Positiv          Positiv            Negativ          -> Architektur-Details anpassen
                                                        (Loss, Regularisierung)
                                                        und Sim 3 wiederholen

Positiv          Negativ            (uebersprungen)  -> PDA-Route bevorzugen
                                                        (bestehende Modelle, da
                                                        orthogonale Zerlegung
                                                        nicht zur Realitaet passt)

Negativ          (uebersprungen)    (uebersprungen)  -> Fundamentales Problem.
                                                        Parallele Deliberation
                                                        konvergiert nicht.
                                                        Ansatz ueberdenken.
```

## Verbindung zu LoRA-Ensemble-Ansatz

Falls Simulation 2 zeigt, dass bestehende Aktivierungen NICHT sauber orthogonal
zerlegbar sind, aber Simulation 1 zeigt dass die Mathematik MIT Orthogonalitaet
funktioniert, dann ist der LoRA-Ensemble-Ansatz ein interessanter Mittelweg:

LoRA-Adapter koennen mit einem Orthogonalitaets-Regularizer trainiert werden,
sodass verschiedene LoRAs in verschiedenen Unterraeumen des Modells operieren.
Das waere quasi "nachtraeglich eingebaute n-PDA-Eigenschaften" auf einem
bestehenden Basismodell — leichter als from scratch, robuster als rohe
Steering Vectors.

## Revisionen nach Literaturrecherche (2026-03-29)

Basierend auf 75+ Papers (drei unabhaengige Agent-Recherchen, konsolidiert
in share/research-synthesis.md):

### Architektur-Revision
- **DEQ → EBT/Iteration**: Root-Finding (Broyden/Anderson) skaliert nicht
  ueber 250M Parameter (Huginn-Ergebnis). Einfache Iteration (Huginn-Style)
  oder Energieminimierung (EBT-Style, Gladstone et al.) stattdessen.
  Simulation 1 vergleicht beide Paradigmen direkt.
- **Backbone-Empfehlung**: Huginn-Style (Prelude → Recurrent Block → Coda)
  mit MoEUT-Routing im Recurrent Block.

### Orthogonalitaet und Diversity
- **Orthogonalitaet notwendig, nicht hinreichend**: Strukturelle Orthogonalitaet
  allein produziert keine sinnvolle semantische Diversity (Zhang et al., Sep 2025).
- **Diversity-Enforcement ist Pflicht**: Ohne kontrastive Losses (InfoNCE o.ae.)
  kollabieren latente Vektoren in redundante Unterraeume (Coda-Forno et al., Okt 2025).
  PLR (Tang et al., Jan 2026) bestaetigt: Diversity zerfaellt exponentiell mit Tiefe.
- **Empfehlung**: PEGO-Style duale Regularisierung + PLR-Style kontrastive Losses.
  Conceptors (Abreu et al., NeurIPS 2025) fuer Perspektiv-Zuweisung evaluieren.

### Tobys einzigartiger Beitrag
Die Signalverarbeitungs-Metriken (SNR, Phasenkohaerenz, Crest Factor) als
Konvergenz-Diagnostik tauchen in KEINEM der 75+ Papers auf. "Language Through
a Prism" (Tamkin et al.) kommt am naechsten (DCT auf Aktivierungen), nutzt aber
Spektralanalyse nur analytisch, nicht als aktive Steuerung. Dies ist der
genuinely neue Beitrag, auf den sich die Simulationen konzentrieren.

## Verwandte Dokumente

- PDA (Parallel Deliberation Architecture): Pragmatischer Ansatz mit bestehenden Modellen
- n-PDA (Native PDA): Theoretisches Greenfield-Konzept
- Recherche-Synthese: share/research-synthesis.md
