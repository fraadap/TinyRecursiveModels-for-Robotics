# 🗣️ Scaletta Discorso: TinyRecursiveModels per Robotica (LIBERO)

## 1. Introduzione e Obiettivo
"Professore, l'obiettivo di queste settimane è stato adattare l'architettura **TinyRecursiveModels (TRM)** — nata per il ragionamento astratto — al controllo robotico continuo sul benchmark **LIBERO**. Ho iterato su **tre approcci distinti** per arrivare a una configurazione che garantisse non solo la convergenza della loss, ma un effettivo successo nel task simulato."

### Workflow Generale
```
Dataset LIBERO → Preprocessing → Training BC (Optuna) → Valutazione → Video/Metriche
     ↓                              ↓
  Video + Testo + Azioni GT     TRM Policy (Visio-Linguistica Ricorsiva)
```

---

## 2. Evoluzione degli Approcci (Cosa ho provato)

### ❌ Approccio 1: Baseline "Component-Based" (Vecchio Notebook)
"Il primo tentativo (`notebook_old`) si basava su un'architettura complessa con teste separate:"
- **Architettura**: Usava una *Component Attention* finale per separare le feature per Posizione, Rotazione e Gripper.
- **Loss**: Utilizzavo una **Huber Loss** combinata con una **Smoothness Regularization** per evitare movimenti a scatti.
- **Risultato**: -_. Inoltre, la smoothness penalty a volte impediva i movimenti rapidi necessari per afferrare l'oggetto.

#### Schema Approccio 1
```
┌─────────────────────────────────────────────────────────────┐
│              TRM POLICY (Component-Based)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Osservazione RGB                                           │
│  (128×128×3)                                                │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────┐                                        │
│  │  VisualEncoder  │                                        │
│  └────────┬────────┘                                        │
│           │ x_cond                                          │
│           ▼                                                 │
│  ┌─────────────────────────────────────┐                    │
│  │       RECURSIVE BLOCK × N           │                    │
│  └───────────────────┬─────────────────┘                    │
│                      │ h_final                              │
│                      ▼                                      │
│           ┌──────────────────────┐                          │
│           │  Component Attention │ (Split features)         │
│           └────┬──────┬──────┬───┘                          │
│                │      │      │                              │
│      ┌─────────▼┐ ┌───▼────┐ ┌▼─────────┐                   │
│      │ Head Pos │ │Head Rot│ │Head Grip │                   │
│      └─────┬────┘ └───┬────┘ └────┬─────┘                   │
│            │          │           │                         │
│            ▼          ▼           ▼                         │
│         [x,y,z]    [r,p,y]      [grip]                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 😐 Approccio 2: Visual-Only TRM (Intermedio)
"Ho quindi semplificato l'architettura, mantenendo la struttura ricorsiva ma rimuovendo le teste separate e la component attention."
- **Configurazione**: Solo Encoder Visivo (ResNet) + Blocco Ricorsivo + Head Unico (7-DoF).
- **Varianti Encoder Testate**: In questa fase ho sperimentato tre diversi encoder visivi per capire quale estraesse le feature migliori:
    1.  **Custom CNN**: Una rete convoluzionale a 4 strati addestrata da zero.
    2.  **ResNet18 Frozen**: Pre-addestrata su ImageNet con pesi congelati (solo un adapter finale addestrabile).
    3.  **ResNet18 Fine-tuned**: Pre-addestrata ma con pesi sbloccati per adattarsi al dominio robotico.
- **Risultato**: Migliore della baseline, ma il robot falliva nei task spaziali ambigui (es. "prendi la ciotola nera" vs "ciotola rossa"). Senza input testuale, il modello cercava di mediare tra i possibili target, fallendo il task.

#### Schema Approccio 2
```
┌─────────────────────────────────────────────────────────────┐
│                 TRM POLICY (Visual-Only)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Osservazione RGB                                           │
│  (128×128×3)                                                │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────┐                                        │
│  │  VisualEncoder  │                                        │
│  └────────┬────────┘                                        │
│           │ x_cond                                          │
│           ▼                                                 │
│  ┌─────────────────────────────────────┐                    │
│  │       RECURSIVE BLOCK × N           │                    │
│  │  (Ragionamento puramente visivo)    │                    │
│  └───────────────────┬─────────────────┘                    │
│                      │ h_final                              │
│                      ▼                                      │
│               ┌─────────────┐                               │
│               │ Action Head │ (Unico MLP)                   │
│               └──────┬──────┘                               │
│                      ▼                                      │
│                Azione (7-DoF)                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 🤏 Approccio 2.5: Visual + Proprioception (Variante)
"Un'evoluzione dell'approccio puramente visivo è stata l'integrazione della **propriocezione**."
- **Modifica**: Ho concatenato all'embedding visivo anche lo stato interno del robot (configurazione giunti e stato gripper), proiettato tramite un MLP lineare.
- **Obiettivo**: Fornire al modello una "consapevolezza corporea" per migliorare la cinematica inversa implicita.
- **Risultato**: Ha migliorato la fluidità locale dei movimenti e la gestione del gripper, ma non ha risolto il problema principale: l'ambiguità semantica del task. Il robot si muoveva meglio, ma spesso verso l'oggetto sbagliato.

#### Schema Approccio 2.5
```
┌─────────────────────────────────────────────────────────────┐
│              TRM POLICY (Visual + Proprioception)           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Osservazione RGB             Stato Robot (Propriocezione)  │
│  (128×128×3)                  (Joints + Gripper)            │
│       │                              │                      │
│       ▼                              ▼                      │
│  ┌─────────────────┐          ┌──────────────────┐          │
│  │  VisualEncoder  │          │   Linear Proj    │          │
│  └────────┬────────┘          └────────┬─────────┘          │
│           │                            │                    │
│           └──────────────┬─────────────┘                    │
│                          ▼                                  │
│                  ┌──────────────┐                           │
│                  │ Concatenation│                           │
│                  └───────┬──────┘                           │
│                          │ x_cond                           │
│                          ▼                                  │
│  ┌─────────────────────────────────────┐                    │
│  │       RECURSIVE BLOCK × N           │                    │
│  └───────────────────┬─────────────────┘                    │
│                      │ h_final                              │
│                      ▼                                      │
│               ┌─────────────┐                               │
│               │ Action Head │                               │
│               └──────┬──────┘                               │
│                      ▼                                      │
│                Azione (7-DoF)                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### ✅ Approccio 3: Visio-Linguistic TRM (Attuale & Migliore)
"L'approccio attuale (`notebook.py`) è quello che ha dato i risultati migliori e un **Success Rate diverso da zero**."
- **Integrazione CLIP**: Ho integrato un Text Encoder (CLIP ViT-L/14) congelato.
- **Meccanismo**: Le feature visive e l'embedding del prompt ("pick up the black bowl") vengono fusi *prima* di entrare nel blocco ricorsivo.
- **Perché funziona**: Il blocco ricorsivo ora non deve solo "capire l'immagine", ma può usare il testo per filtrare le feature visive (attenzione selettiva). Il robot dimostra di capire *dove* sono gli oggetti e *dove* metterli.
- **Training**: Ho introdotto **Optuna** per la ricerca iperparametri e una **Loss Mista (MSE + L1)** che ha stabilizzato notevolmente il training rispetto alla Huber loss.

#### Schema Architettura Finale
```
┌─────────────────────────────────────────────────────────────┐
│                 TRM POLICY (Visio-Linguistica)              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Osservazione RGB             Prompt Testuale               │
│  (128×128×3)                  "pick up the bowl"            │
│       │                              │                      │
│       ▼                              ▼                      │
│  ┌─────────────────┐          ┌──────────────────┐          │
│  │ ResNet18 Adapter│          │ CLIP Text Encoder│          │
│  └────────┬────────┘          └────────┬─────────┘          │
│           │                            │                    │
│           └──────────────┬─────────────┘                    │
│                          ▼                                  │
│                  ┌──────────────┐                           │
│                  │ Fusion Layer │                           │
│                  └───────┬──────┘                           │
│                          │ x_cond                           │
│                          ▼                                  │
│  ┌─────────────────────────────────────┐                    │
│  │       RECURSIVE BLOCK × N           │                    │
│  │  (Self-Attention + MLP + Residual)  │                    │
│  └───────────────────┬─────────────────┘                    │
│                      │ h_final                              │
│                      ▼                                      │
│               ┌─────────────┐                               │
│               │ Action Head │                               │
│               └──────┬──────┘                               │
│                      ▼                                      │
│                Azione (7-DoF)                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Focus Tecnico: Come funziona l'Evaluation
"Un punto cruciale è che non ci limitiamo a guardare la Validation Loss, ma eseguiamo una valutazione in **Closed-Loop Simulation**. Ecco come è strutturata la pipeline di test:"

```
   ┌──────────────────────┐
   │      SIMULATORE      │◄──────┐
   │       (LIBERO)       │       │
   └──────────┬───────────┘       │
              │ Osservazione      │ Azione
              ▼                   │ (Denormalizzata)
   ┌──────────────────────┐       │
   │      TRM POLICY      │───────┘
   └──────────────────────┘
```

1.  **Setup dell'Ambiente (Parallelo)**: Istanzio l'ambiente `OffScreenRenderEnv` di LIBERO caricando il file BDDL specifico. Eseguiamo **5 tentativi (try) in parallelo** per ogni task per avere una stima robusta del Success Rate e velocizzare l'inferenza.
2.  **Inizializzazione**: Resetto il robot a uno stato iniziale noto (preso dal dataset di test) per garantire condizioni riproducibili.
3.  **Loop di Controllo (Inferenza)**:
    *   Ad ogni step, catturo l'immagine RGB (128x128).
    *   Passo immagine + Prompt al modello TRM.
    *   **Denormalizzazione**: Fondamentale. L'output del modello è normalizzato (Z-score); lo rimoltiplico per la deviazione standard e aggiungo la media calcolata sul training set. Senza questo, il robot farebbe movimenti microscopici o esplosivi.
4.  **Verifica del Successo (BDDL)**:
    *   Non uso una semplice distanza euclidea.
    *   LIBERO verifica ad ogni step dei **predicati logici** (es. `InZone(bowl, table_center)` AND `Grasped(bowl)`).
    *   L'episodio è un successo solo se tutte le condizioni logiche sono soddisfatte entro 600 step.

---

## 4. Strategie di Robustezza: Data Augmentation
"Infine, vorrei sottolineare l'importanza della **Data Augmentation** per ottenere questi risultati. Dato che il dataset per task è limitato (circa 50 dimostrazioni), il rischio di overfitting era alto. Ho implementato due tecniche applicate on-the-fly durante il training:"

1.  **Color Jitter (Luminosità/Contrasto)**:
    *   Vario casualmente la luminosità delle immagini.
    *   *Obiettivo*: Rendere il modello robusto a cambiamenti di illuminazione, costringendolo a basarsi sulla forma e semantica degli oggetti piuttosto che sui valori esatti dei pixel.

2.  **Random Crop & Resize**:
    *   Effettuo un crop casuale (zoom) e ridimensiono all'originale.
    *   *Obiettivo*: Simula piccole variazioni nella posizione della telecamera e impedisce al modello di memorizzare le coordinate pixel esatte, favorendo la generalizzazione spaziale.

"Senza queste tecniche, la Validation Loss tendeva a divergere dopo poche epoche; con l'augmentation, il training è risultato molto più stabile."

---

## 5. Sviluppi Futuri (Future Works)
"Per migliorare ulteriormente le performance e la generalizzazione, abbiamo identificato 4 direzioni principali per il futuro:"

1.  **Unificazione Propriocezione + Visio-Linguistico**:
    *   Vogliamo combinare l'Approccio 2.5 (stato del robot) con l'Approccio 3 (CLIP).
    *   *Ipotesi*: Fornire al modello sia l'intento semantico (Testo) che la consapevolezza corporea (Propriocezione) dovrebbe migliorare la precisione nel grasping fine.

2.  **Bilanciamento Dimensionale (Visual > Text)**:
    *   Attualmente proiettiamo sia l'immagine che il testo nello stesso `hidden_dim`.
    *   CLIP ViT-L/14 produce embedding da **768 dimensioni**. Riteniamo che mappare una frase semplice in un vettore così grande (o ridurlo drasticamente) sia sbilanciato rispetto alla ricchezza informativa di un'immagine RGB.
    *   *Idea*: Provare a embeddare la parte visuale in un vettore di dimensione **doppia** rispetto a quella testuale prima della fusione, per dare più "banda" ai dettagli percettivi.

3.  **Fine-tuning di CLIP**:
    *   Attualmente il Text Encoder è congelato (frozen).
    *   *Idea*: Sbloccare gli ultimi layer di CLIP per adattare gli embedding al dominio specifico di LIBERO (es. il concetto di "ciotola rossa" nel simulatore potrebbe differire leggermente da quello appreso su internet).

4.  **Ottimizzazione Avanzata**:
    *   Nonostante la mini grid-search con Optuna abbia trovato buoni parametri, vorremmo esplorare scheduler di Learning Rate più sofisticati (es. ciclici o con restart aggressivi) e ampliare lo spazio di ricerca degli iperparametri per spremere le massime performance dall'architettura ricorsiva.

---

## 6. Conclusioni
"In sintesi, l'architettura ricorsiva sembra beneficiare enormemente del condizionamento testuale. Il modello attuale riesce a completare task di pick-and-place spaziale, dimostrando che la ricorsività aiuta a raffinare la policy visuo-motoria quando guidata dalla semantica."
