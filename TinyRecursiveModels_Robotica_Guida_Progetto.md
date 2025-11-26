# Guida Completa al Progetto: TinyRecursiveModels per il Controllo Robotico

## Indice
1. [Introduzione e Panoramica del Progetto](#1-introduzione-e-panoramica-del-progetto)
2. [Spiegazione Teorica dei TinyRecursiveModels](#2-spiegazione-teorica-dei-tinyrecursivemodels)
3. [I Dataset: LIBERO e ROBOCASA](#3-i-dataset-libero-e-robocasa)
4. [Workflow del Progetto (Indicazioni del Professore)](#4-workflow-del-progetto-indicazioni-del-professore)
5. [Guida Dettagliata agli Step del Progetto](#5-guida-dettagliata-agli-step-del-progetto)
6. [Riferimenti Bibliografici](#6-riferimenti-bibliografici)

---

## 1. Introduzione e Panoramica del Progetto

### 1.1 Obiettivo Generale
Il progetto mira ad **adattare l'architettura TinyRecursiveModels (TRM)** — una delle architetture più recenti e impattanti nel campo del ragionamento ricorsivo — **a task di controllo robotico**. L'obiettivo è valutare se le capacità di ragionamento ricorsivo del modello possano migliorare:
- **Sample efficiency**: apprendere con meno dimostrazioni
- **Task generalization**: generalizzare a nuovi task non visti durante il training

### 1.2 Sfide Principali
Questo progetto si colloca all'intersezione di due aree di ricerca:
1. **Architetture ricorsive/looped** per il ragionamento
2. **Imitation learning** per la robotica

### 1.3 Risultato Atteso
Il risultato sarà un agente robotico che utilizza TRM per predire azioni, **allenato esclusivamente in Imitation Learning** (senza simulazione durante il training) utilizzando video e azioni ground truth dalla data collection. La valutazione avverrà poi in simulazione su benchmark standard (LIBERO/ROBOCASA), producendo sia **risultati quantitativi** (success rate) che **qualitativi** (video degli episodi di test).

---

## 2. Spiegazione Teorica dei TinyRecursiveModels

### 2.1 Cosa sono i TinyRecursiveModels (TRM)?

I **TinyRecursiveModels** rappresentano un paradigma innovativo nell'architettura delle reti neurali che si basa sul principio del **ragionamento iterativo ricorsivo**. A differenza delle architetture tradizionali che processano l'input in un singolo passaggio feedforward, i TRM utilizzano una rete molto piccola che viene **applicata ripetutamente** sullo stesso input, raffinando progressivamente la soluzione.

#### Caratteristiche Fondamentali:

1. **Architettura Minimalista**: 
   - Tipicamente solo **2 layer** con circa **7-27 milioni di parametri**
   - Dimensioni drasticamente inferiori rispetto ai Large Language Models (LLMs)
   - Efficienza computazionale in fase di training e inferenza

2. **Ricorsione Iterativa**:
   - La stessa rete viene eseguita **N volte** in loop
   - Ad ogni iterazione, l'output viene ri-processato come input
   - Il numero di iterazioni può essere fisso o **adattivo** (appreso dal modello)

3. **Think Tokens / Latent Thought**:
   - Il modello opera in uno **spazio latente continuo**
   - Non produce token testuali intermedi (come Chain-of-Thought)
   - Il "ragionamento" avviene attraverso trasformazioni successive nello spazio latente

### 2.2 Fondamenti Matematici

L'architettura TRM può essere formalizzata come segue:

**Notazione:**
- $x$ = input iniziale
- $h^{(0)}$ = rappresentazione iniziale (encoding di $x$)
- $f_\theta$ = funzione di trasformazione (la rete ricorsiva)
- $N$ = numero di iterazioni ricorsive

**Processo Ricorsivo:**
$$h^{(t+1)} = f_\theta(h^{(t)}, x) \quad \text{per } t = 0, 1, ..., N-1$$

**Output Finale:**
$$y = g_\phi(h^{(N)})$$

dove $g_\phi$ è un decoder che mappa la rappresentazione finale allo spazio di output desiderato.

### 2.3 Relazione con Architetture Correlate

I TRM si inseriscono in una famiglia più ampia di architetture:

| Architettura | Caratteristica Distintiva |
|--------------|---------------------------|
| **Universal Transformers** (2018) | Applicano lo stesso blocco transformer ripetutamente con halt adaptivo |
| **Looped Transformers** (2023) | Condivisione dei pesi tra layer, training per learning in-context |
| **Hierarchical Reasoning Model (HRM)** | Due reti che ricorrono a frequenze diverse |
| **TinyRecursiveModels (TRM)** (2025) | Semplificazione massima: singola rete tiny con 2 layer |

### 2.4 Vantaggi per Task di Ragionamento Complesso

I TRM hanno dimostrato risultati eccezionali su benchmark come:

- **ARC-AGI** (Abstraction and Reasoning Corpus): 45% su ARC-AGI-1, 8% su ARC-AGI-2 con soli 7M parametri
- **Sudoku-Extreme**: 86% accuracy con 423,168 puzzle di test
- **Maze solving**: Navigazione in labirinti complessi

**Perché funzionano:**
1. **Computazione adattiva**: più iterazioni per problemi più difficili
2. **Raffinamento progressivo**: ogni passo corregge errori del precedente
3. **Generalizzazione composizionale**: apprendono primitive che combinano ricorsivamente

### 2.5 Perché Applicarli alla Robotica?

Il controllo robotico presenta sfide analoghe ai puzzle di ragionamento:

| Caratteristica | Ragionamento (ARC/Sudoku) | Controllo Robotico |
|----------------|---------------------------|--------------------|
| **Sequenzialità** | Passi logici concatenati | Sequenze di azioni |
| **Vincoli** | Regole del puzzle | Vincoli fisici e di task |
| **Feedback** | Coerenza della soluzione | Stato dell'ambiente |
| **Composizionalità** | Combinare pattern | Combinare skill atomiche |

**Ipotesi di Ricerca**: La capacità di TRM di raffinare iterativamente le rappresentazioni potrebbe tradursi in:
- Migliore **accoppiamento percezione-azione** attraverso cicli di feedback interno
- **Pianificazione implicita** senza necessità di moduli separati
- **Robustezza** a perturbazioni attraverso correzione iterativa

---

## 3. I Dataset: LIBERO e ROBOCASA

### 3.1 LIBERO (Lifelong Robot Learning)

#### Descrizione Generale

**LIBERO** è un benchmark progettato specificamente per studiare il **trasferimento di conoscenza** nel contesto del **lifelong robot learning**. Pubblicato nel 2023 da Bo Liu et al. (UT Austin, Stanford), rappresenta uno standard per valutare algoritmi di apprendimento continuo in robotica.

#### Composizione del Dataset

| Task Suite | # Task | Focus |
|------------|--------|-------|
| **LIBERO-Spatial** | 10 | Trasferimento di conoscenza spaziale (posizioni, relazioni) |
| **LIBERO-Object** | 10 | Trasferimento relativo a oggetti (proprietà, affordance) |
| **LIBERO-Goal** | 10 | Trasferimento di obiettivi (cosa fare) |
| **LIBERO-90** | 90 | Pre-training multitask |
| **LIBERO-10** | 10 | Testing lifelong learning |

**Totale: 130 task di manipolazione**

#### Caratteristiche Tecniche

- **Simulatore**: MuJoCo-based (robosuite)
- **Robot**: Franka Panda (7-DoF arm + gripper)
- **Osservazioni**: Immagini RGB (128×128)
- **Azioni**: 7-DoF (delta posizione end-effector + gripper)
- **Reward**: Sparse (+1 al completamento)
- **Dimostrazioni**: 50 demo per task (teleoperate da umani)

#### Paper Correlati che Utilizzano LIBERO

1. **"LIBERO: Benchmarking Knowledge Transfer"** (Liu et al., 2023)
   - Paper originale che introduce il benchmark
   - Valuta BC-RNN, BC-Transformer, BC-ViLT

2. **"Multi-Task Interactive Robot Fleet Learning"** (Liu et al., 2024)
   - Utilizza LIBERO per visual world models
   - Approccio fleet learning con VLM

3. **"HAMLET: History-Aware Vision-Language-Action Model"** (2025)
   - Adatta VLA pre-trained per task history-dependent
   - Benchmark su LIBERO

### 3.2 ROBOCASA

#### Descrizione Generale

**ROBOCASA** è un framework di simulazione su larga scala per training di robot generalisti in ambienti domestici, con focus su scene di cucina. Pubblicato a RSS 2024 da Stanford/NVIDIA.

#### Composizione del Dataset

**Asset 3D:**
- **2,500+ oggetti** in 150+ categorie (frutta, verdura, utensili, contenitori)
- **Mobili interattivi**: armadi, cassetti, forno, microonde, lavello
- **100 texture** per pareti/pavimenti/contatori (generate da Midjourney)

**Task Suite:**

| Categoria | Esempi | # Task |
|-----------|--------|--------|
| **Skill Atomiche** | Pick-and-place, open/close doors, turn knobs | 25 |
| **Task Composite** | Brewing coffee, restocking supplies, steaming vegetables | 75 |

**Totale: 100 task**

#### Caratteristiche Tecniche

- **Simulatore**: Robosuite/MuJoCo
- **Robot supportati**: Mobile manipulator, humanoid, quadruped con braccio
- **Generazione procedurale**: Scene generate con guida LLM (GPT-4)
- **Diversità visiva**: Domain randomization via text-to-image AI
- **Dimostrazioni**: Human teleoperation + automated trajectory generation

#### Paper Correlati che Utilizzano ROBOCASA

1. **"RoboCasa: Large-Scale Simulation"** (Nasiriany et al., 2024)
   - Paper originale che introduce il framework
   - Mostra scaling trend in imitation learning

2. **"ROVER: Recursive Reasoning Over Videos"** (2025)
   - VLM per comprensione video in task robotici
   - Crea dataset di 543 video da RoboCasa

3. **"ACG: Action Coherence Guidance for Flow-based VLA"** (2025)
   - Riduzione jitter nelle policy VLA
   - Benchmark su RoboCasa

4. **"REMAC: Self-Reflective Multi-Agent Collaboration"** (2025)
   - VLM per planning long-horizon
   - Valutazione su RoboCasa

### 3.3 Confronto LIBERO vs ROBOCASA

| Aspetto | LIBERO | ROBOCASA |
|---------|--------|----------|
| **Focus** | Lifelong learning, transfer | Generalità, scalabilità |
| **Complessità scene** | Media | Alta (cucine realistiche) |
| **Diversity** | Controllata (per ablation) | Massimizzata (per generalizzazione) |
| **# Demo** | 50 per task | Variabile + synthetic augmentation |
| **Task length** | Brevi (single goal) | Lunghi (composite) |
| **Ideale per** | Studio transfer sistematico | Training policy generaliste |

**Raccomandazione per il progetto**: 
- **Iniziare con LIBERO** per la sua struttura più semplice e controllata
- **Estendere a ROBOCASA** per validare scalabilità e generalizzazione

---

## 4. Workflow del Progetto (Indicazioni del Professore)

> ⚠️ **NOTA IMPORTANTE**: Questa sezione riassume le indicazioni specifiche del professore per lo svolgimento del progetto.

### 4.1 Panoramica del Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WORKFLOW DEL PROGETTO                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FASE 1: DATA COLLECTION                                                    │
│  ─────────────────────                                                      │
│  • Utilizzare LIBERO e/o ROBOCASA per raccogliere dimostrazioni            │
│  • I benchmark forniscono video + ground truth actions                      │
│  • Selezionare un SOTTOGRUPPO di task (opzione consigliata)                │
│    OPPURE tutti i task (per confronto con state-of-the-art)                │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  FASE 2: TRAINING TRM (IMITATION LEARNING - NO SIMULAZIONE)                │
│  ───────────────────────────────────────────────────────────────           │
│  • Allenare TRM sui dati raccolti (video → azioni)                         │
│  • Modificare SOLO input/output dimensions di TRM                          │
│  • Sperimentare data augmentation per migliorare performance               │
│  • TRM già supporta task prompts in input (utile per multi-task)           │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  FASE 3: HYPERPARAMETER SEARCH VELOCE                                      │
│  ────────────────────────────────────                                       │
│  • Testare varie configurazioni con LR ALTI e POCHE EPOCHE                 │
│  • La versione che impara meglio → training completo                       │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  FASE 4: VALUTAZIONE IN SIMULAZIONE                                        │
│  ───────────────────────────────────                                        │
│  • Testare il modello allenato nei benchmark (LIBERO/ROBOCASA)             │
│  • Estrarre RISULTATI QUANTITATIVI: success rate, episode length           │
│  • Estrarre RISULTATI QUALITATIVI: video degli episodi di test             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Selezione dei Task

**Opzione 1: Sottogruppo di Task (Consigliato per iniziare)**

Selezionare task **presenti in entrambi i benchmark** per avere performance comparabili:

| Task Category | LIBERO | ROBOCASA | Comparabile |
|---------------|--------|----------|-------------|
| Pick and Place | ✅ LIBERO-Spatial | ✅ Atomic tasks | ✅ Sì |
| Open/Close Doors | ✅ LIBERO-Object | ✅ Atomic tasks | ✅ Sì |
| Object Manipulation | ✅ LIBERO-Goal | ✅ Composite tasks | ⚠️ Parziale |

**Suggerimento pratico:**
- Iniziare con **5-10 task simili** tra i due benchmark
- Esempio: pick-and-place tasks da entrambi
- Permette confronto diretto cross-benchmark

**Opzione 2: Tutti i Task (Per State-of-the-Art)**

- LIBERO: 130 task totali
- ROBOCASA: 100 task totali
- Richiede più risorse computazionali
- Permette confronto diretto con paper pubblicati

### 4.3 Adattamento di TRM per Robotica

**Cosa modificare (secondo il professore):**

```python
# Modifiche MINIME richieste a TRM:

# 1. INPUT DIMENSION
# Original TRM: input per grid/puzzle (es. 30x30 tokens)
# Robotica: features da immagini (es. 128x128 RGB → encoder → 256-dim)

# 2. OUTPUT DIMENSION  
# Original TRM: output per grid/puzzle
# Robotica: 7-DoF actions (delta pos x,y,z + delta rot + gripper)

# 3. TASK PROMPT (già supportato!)
# TRM accetta già task prompts → utile per multi-task
# Integrare descrizione testuale del task
```

**Data Augmentation da sperimentare:**

| Tecnica | Descrizione | Priorità |
|---------|-------------|----------|
| **Color Jitter** | Variare brightness, contrast, saturation | Alta |
| **Random Crop** | Crop casuale + resize | Alta |
| **Action Noise** | Aggiungere rumore gaussiano alle azioni | Media |
| **Temporal Augmentation** | Subsample/skip frames | Media |
| **Spatial Augmentation** | Flip orizzontale (con flip azioni) | Bassa |

### 4.4 Strategia di Training Veloce

**Approccio consigliato dal professore:**

```python
# FASE 1: Hyperparameter search veloce
hyperparams_to_test = [
    {'lr': 1e-3, 'epochs': 10, 'batch_size': 32},
    {'lr': 5e-4, 'epochs': 10, 'batch_size': 64},
    {'lr': 1e-4, 'epochs': 10, 'batch_size': 32},
    {'lr': 3e-4, 'epochs': 15, 'batch_size': 128},
]

# Per ogni configurazione:
for config in hyperparams_to_test:
    model = TRMPolicy(...)
    train_quick(model, **config)
    loss_curve = evaluate_on_validation(model)
    
    # Criterio: quale impara meglio/più velocemente?
    if loss_curve.is_best():
        best_config = config

# FASE 2: Training completo con la migliore configurazione
final_model = TRMPolicy(...)
train_full(final_model, **best_config, epochs=100)  # Più epoche
```

**Metriche per selezionare la configurazione migliore:**
1. Velocità di convergenza della loss
2. Loss finale su validation set
3. Stabilità del training (no spike/divergenza)

### 4.5 Output Finali Richiesti

**Risultati Quantitativi:**
- Success Rate per task
- Episode Length medio
- Confronto TRM vs baseline (se disponibile)

**Risultati Qualitativi (VIDEO):**
```python
# Script per estrarre video di test
def record_episode_video(policy, env, task_id, output_path):
    frames = []
    obs = env.reset()
    done = False
    
    while not done:
        # Salva frame
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        
        # Policy step
        action = policy(preprocess(obs))
        obs, reward, done, info = env.step(action)
    
    # Salva video
    save_video(frames, output_path, fps=30)
    return info.get('success', False)
```

---

## 5. Guida Dettagliata agli Step del Progetto

## STEP 1: Data Collection (LIBERO/ROBOCASA)

> 📌 **Obiettivo**: Costruire un dataset di video + ground truth actions utilizzabile per allenare TRM in Imitation Learning.

### 5.1.1 Setup dell'Ambiente

**Requisiti Software:**
```bash
# Creare ambiente conda
conda create -n robotic_trm python=3.8
conda activate robotic_trm

# Installare dipendenze base
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib
```

**Installare LIBERO:**
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .
```

**Installare RoboCasa:**
```bash
git clone https://github.com/robocasa/robocasa.git
cd robocasa
pip install -e .
```

### 5.1.2 Download Dataset Dimostrazioni

**LIBERO:**
```bash
cd LIBERO

# Download tutti i dataset
python benchmark_scripts/download_libero_datasets.py

# OPPURE download specifico (raccomandato per iniziare)
python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial

# Alternativa: da HuggingFace
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```

**Formato delle dimostrazioni:**
- File HDF5 contenenti:
  - `obs/agentview_rgb`: immagini RGB (50 demo × T timestep × 128 × 128 × 3)
  - `actions`: azioni 7-DoF (50 × T × 7)
  - `states`: stati del robot
  - `rewards`: reward sparse

### 5.1.3 Esplorazione e Preprocessing dei Dati

**Script di esplorazione:**
```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Caricare un file demo
demo_file = "path/to/libero_spatial/task_0/demo.hdf5"

with h5py.File(demo_file, 'r') as f:
    # Esplorare struttura
    print("Keys:", list(f.keys()))
    print("Demo keys:", list(f['data'].keys()))
    
    # Esempio prima demo
    demo_0 = f['data/demo_0']
    obs = demo_0['obs/agentview_rgb'][:]
    actions = demo_0['actions'][:]
    
    print(f"Observations shape: {obs.shape}")  # (T, 128, 128, 3)
    print(f"Actions shape: {actions.shape}")    # (T, 7)
    
    # Visualizzare frame
    plt.imshow(obs[0])
    plt.title("First frame")
    plt.show()
```

**Preprocessing raccomandato:**
1. **Normalizzazione immagini**: [0, 255] → [-1, 1] o standardizzazione
2. **Normalizzazione azioni**: z-score per ogni dimensione
3. **Data augmentation**: random crop, color jitter, rotation
4. **Chunking**: dividere traiettorie lunghe in sequenze di lunghezza fissa

### 5.1.4 Creazione DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader

class LIBERODataset(Dataset):
    def __init__(self, demo_paths, sequence_length=10, transform=None):
        self.demos = []
        self.sequence_length = sequence_length
        self.transform = transform
        
        for path in demo_paths:
            with h5py.File(path, 'r') as f:
                for demo_key in f['data'].keys():
                    obs = f[f'data/{demo_key}/obs/agentview_rgb'][:]
                    actions = f[f'data/{demo_key}/actions'][:]
                    self.demos.append({'obs': obs, 'actions': actions})
    
    def __len__(self):
        return sum(len(d['obs']) - self.sequence_length for d in self.demos)
    
    def __getitem__(self, idx):
        # Trovare demo e timestep corretto
        # ... (implementazione completa)
        
        return {
            'observations': obs_seq,  # (seq_len, 128, 128, 3)
            'actions': action_seq     # (seq_len, 7)
        }
```

---

## STEP 2: Adattamento Architettura TRM

### 5.2.1 Architettura TRM Base

**Componenti principali da implementare:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualEncoder(nn.Module):
    """Encoder CNN per osservazioni visive"""
    def __init__(self, obs_shape=(128, 128, 3), hidden_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Calcolare dim output
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            conv_out = self.conv(dummy).shape[1]
        
        self.fc = nn.Linear(conv_out, hidden_dim)
    
    def forward(self, x):
        # x: (B, H, W, C) → (B, C, H, W)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        return self.fc(self.conv(x))

class RecursiveBlock(nn.Module):
    """Blocco ricorsivo del TRM"""
    def __init__(self, hidden_dim=256, num_heads=4):
        super().__init__()
        
        # Self-attention per ragionamento
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # MLP per trasformazione
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, h, x_cond):
        """
        h: hidden state corrente (B, D)
        x_cond: conditioning dall'input (B, D)
        """
        # Combina hidden e conditioning
        combined = h + x_cond
        combined = combined.unsqueeze(1)  # (B, 1, D)
        
        # Self-attention
        attn_out, _ = self.attention(combined, combined, combined)
        h = self.norm1(combined + attn_out)
        
        # MLP
        mlp_out = self.mlp(h)
        h = self.norm2(h + mlp_out)
        
        return h.squeeze(1)

class TRMPolicy(nn.Module):
    """Policy completa basata su TRM per controllo robotico"""
    def __init__(
        self,
        obs_shape=(128, 128, 3),
        action_dim=7,
        hidden_dim=256,
        num_recursions=8,
        adaptive_halt=True
    ):
        super().__init__()
        
        self.encoder = VisualEncoder(obs_shape, hidden_dim)
        self.recursive_block = RecursiveBlock(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        self.num_recursions = num_recursions
        self.adaptive_halt = adaptive_halt
        
        if adaptive_halt:
            # Predittore per halting probability
            self.halt_predictor = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
    
    def forward(self, obs, max_recursions=None):
        """
        obs: (B, H, W, C) osservazione visiva
        Returns: azioni predette (B, action_dim)
        """
        max_rec = max_recursions or self.num_recursions
        
        # Encoding iniziale
        x_cond = self.encoder(obs)
        h = x_cond.clone()  # Inizializza hidden state
        
        if self.adaptive_halt:
            # Adaptive computation time
            halt_probs = []
            remainders = torch.ones(obs.shape[0], device=obs.device)
            n_updates = torch.zeros(obs.shape[0], device=obs.device)
            accumulated_h = torch.zeros_like(h)
            
            for t in range(max_rec):
                h = self.recursive_block(h, x_cond)
                halt_p = self.halt_predictor(h).squeeze(-1)
                halt_probs.append(halt_p)
                
                # Accumulare output pesato
                accumulated_h += remainders.unsqueeze(-1) * h
                
                # Aggiornare remainder
                still_running = remainders > 0.01
                remainders = remainders * (1 - halt_p)
                n_updates += still_running.float()
            
            h = accumulated_h
        else:
            # Fixed recursions
            for t in range(max_rec):
                h = self.recursive_block(h, x_cond)
        
        # Predizione azione
        action = self.action_head(h)
        return action
```

### 5.2.2 Adattamenti Specifici per Robotica

**1. Action Chunking (importante per smooth control):**

```python
class TRMPolicyChunked(TRMPolicy):
    """Versione con action chunking per trajectory più smooth"""
    def __init__(self, chunk_size=10, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        # Modificare action head per predire chunk
        self.action_head = nn.Linear(
            kwargs.get('hidden_dim', 256), 
            kwargs.get('action_dim', 7) * chunk_size
        )
    
    def forward(self, obs, **kwargs):
        h = super().forward_features(obs, **kwargs)  # Get hidden
        action_chunk = self.action_head(h)
        return action_chunk.view(-1, self.chunk_size, 7)
```

**2. Condizionamento su Language:**

```python
class TRMPolicyLanguage(nn.Module):
    """TRM con condizionamento su istruzione linguistica"""
    def __init__(self, hidden_dim=256, **kwargs):
        super().__init__()
        self.visual_encoder = VisualEncoder(**kwargs)
        
        # Text encoder (può essere frozen CLIP/T5)
        self.text_encoder = nn.Embedding(10000, hidden_dim)  # Semplificato
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Recursive block che integra visual + language
        self.recursive_block = RecursiveBlockMultiModal(hidden_dim)
        # ...
```

### 5.2.3 Training con Imitation Learning (Behavior Cloning)

> ⚠️ **IMPORTANTE**: Il training avviene SOLO su dati offline (video + azioni). NON si usa simulazione durante il training!

```python
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb  # Per logging

def train_bc(model, train_loader, val_loader, config):
    """
    Training con Behavior Cloning.
    
    Args:
        model: TRMPolicy
        train_loader: DataLoader con (obs, actions, task_prompt)
        val_loader: DataLoader per validation
        config: dict con lr, epochs, etc.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            obs = batch['observations'].cuda()           # (B, H, W, C)
            target_actions = batch['actions'].cuda()     # (B, 7)
            task_prompt = batch.get('task_prompt', None) # (B, seq_len) opzionale
            
            optimizer.zero_grad()
            
            # Forward pass (con task prompt se disponibile)
            if task_prompt is not None:
                pred_actions = model(obs, task_prompt=task_prompt.cuda())
            else:
                pred_actions = model(obs)
            
            # Loss MSE per continuous actions
            loss = F.mse_loss(pred_actions, target_actions)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['observations'].cuda()
                target_actions = batch['actions'].cuda()
                pred_actions = model(obs)
                val_loss += F.mse_loss(pred_actions, target_actions).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Logging
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'lr': scheduler.get_last_lr()[0]})
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
        
        scheduler.step()
    
    return model
```

### 5.2.4 Script di Training Rapido per Hyperparameter Search

> 💡 **Strategia del professore**: Testare varie versioni con LR alti e poche epoche per trovare la configurazione migliore.

```python
import itertools

def quick_hyperparameter_search(train_loader, val_loader):
    """
    Ricerca veloce dei migliori iperparametri.
    La versione che impara meglio → training completo.
    """
    
    # Griglia di iperparametri da testare
    param_grid = {
        'lr': [1e-3, 5e-4, 1e-4, 3e-4],
        'hidden_dim': [128, 256],
        'num_recursions': [4, 8],
        'batch_size': [32, 64],
    }
    
    results = []
    
    # Test rapido: poche epoche per ogni configurazione
    QUICK_EPOCHS = 10
    
    for lr in param_grid['lr']:
        for hidden_dim in param_grid['hidden_dim']:
            for num_rec in param_grid['num_recursions']:
                
                print(f"\nTesting: lr={lr}, hidden={hidden_dim}, rec={num_rec}")
                
                # Creare modello
                model = TRMPolicy(
                    hidden_dim=hidden_dim,
                    num_recursions=num_rec
                ).cuda()
                
                # Training veloce
                config = {'lr': lr, 'epochs': QUICK_EPOCHS}
                train_bc(model, train_loader, val_loader, config)
                
                # Valutare
                model.eval()
                final_val_loss = evaluate_validation(model, val_loader)
                
                results.append({
                    'lr': lr,
                    'hidden_dim': hidden_dim,
                    'num_recursions': num_rec,
                    'val_loss': final_val_loss
                })
                
                print(f"  → Val Loss: {final_val_loss:.4f}")
    
    # Trovare migliore configurazione
    best_config = min(results, key=lambda x: x['val_loss'])
    print(f"\n🏆 Best config: {best_config}")
    
    return best_config

def train_final_model(best_config, train_loader, val_loader):
    """
    Training completo con la migliore configurazione.
    """
    model = TRMPolicy(
        hidden_dim=best_config['hidden_dim'],
        num_recursions=best_config['num_recursions']
    ).cuda()
    
    # Training completo con più epoche
    config = {
        'lr': best_config['lr'],
        'epochs': 100  # Molte più epoche per training finale
    }
    
    trained_model = train_bc(model, train_loader, val_loader, config)
    
    return trained_model
```

### 5.2.5 Integrazione Task Prompts

> ✅ **Vantaggio TRM**: Già progettato per ricevere task prompts in input!

```python
class TRMPolicyWithPrompt(TRMPolicy):
    """
    TRM con supporto per task prompts (descrizioni testuali del task).
    Utile per multi-task learning.
    """
    def __init__(self, vocab_size=10000, prompt_dim=64, **kwargs):
        super().__init__(**kwargs)
        
        # Encoder per task prompt
        self.prompt_embedding = nn.Embedding(vocab_size, prompt_dim)
        self.prompt_proj = nn.Linear(prompt_dim, kwargs.get('hidden_dim', 256))
        
    def forward(self, obs, task_prompt=None):
        # Encoding visivo
        x_cond = self.encoder(obs)
        
        # Aggiungere task prompt se disponibile
        if task_prompt is not None:
            # task_prompt: (B, seq_len) token ids
            prompt_emb = self.prompt_embedding(task_prompt)  # (B, seq_len, prompt_dim)
            prompt_emb = prompt_emb.mean(dim=1)  # (B, prompt_dim) - pool over tokens
            prompt_proj = self.prompt_proj(prompt_emb)  # (B, hidden_dim)
            x_cond = x_cond + prompt_proj  # Fuse visual + language
        
        # Ricorsione standard
        h = x_cond.clone()
        for t in range(self.num_recursions):
            h = self.recursive_block(h, x_cond)
        
        return self.action_head(h)
```

---

## STEP 3: Valutazione in Simulazione

> 📌 **Obiettivo**: Testare il modello allenato (offline) nel benchmark simulato. Estrarre sia risultati **quantitativi** (success rate) che **qualitativi** (video episodi).

> ⚠️ **NOTA**: LIBERO è più intuitivo e semplice; ROBOCASA più recente e decorato. Entrambi sono "plug and play".

### 5.3.1 Setup Evaluation Environment

```python
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

def create_eval_env(task_suite_name="libero_spatial", task_id=0):
    """Creare ambiente per valutazione"""
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    
    task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), 
        task.problem_folder, 
        task.bddl_file
    )
    
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128
    }
    
    env = OffScreenRenderEnv(**env_args)
    return env, task_suite.get_task_init_states(task_id)
```

### 5.3.2 Metriche di Valutazione

```python
def evaluate_policy(policy, task_suite, num_episodes=50, max_steps=500):
    """Valutazione completa della policy"""
    results = {
        'success_rate': [],
        'episode_lengths': [],
        'returns': []
    }
    
    for task_id in range(len(task_suite)):
        env, init_states = create_eval_env(task_suite, task_id)
        
        task_successes = []
        task_lengths = []
        
        for ep in range(num_episodes):
            # Reset con initial state fisso per reproducibilità
            env.reset()
            env.set_init_state(init_states[ep % len(init_states)])
            
            obs = env.get_observation()
            done = False
            ep_reward = 0
            step = 0
            
            while not done and step < max_steps:
                # Preprocessing
                obs_tensor = preprocess_obs(obs).cuda()
                
                # Predizione azione
                with torch.no_grad():
                    action = policy(obs_tensor).cpu().numpy()
                
                # Step ambiente
                obs, reward, done, info = env.step(action)
                ep_reward += reward
                step += 1
            
            task_successes.append(info.get('success', False))
            task_lengths.append(step)
        
        results['success_rate'].append(np.mean(task_successes))
        results['episode_lengths'].append(np.mean(task_lengths))
        
        env.close()
    
    return results
```

### 5.3.3 Estrazione Video degli Episodi (Risultati Qualitativi)

> 🎬 **IMPORTANTE**: Oltre ai risultati quantitativi, è necessario estrarre video degli episodi di test!

```python
import cv2
import os
from datetime import datetime

def record_evaluation_videos(policy, task_suite, output_dir, num_episodes_per_task=5):
    """
    Registra video degli episodi di valutazione.
    
    Args:
        policy: modello TRM allenato
        task_suite: suite di task da valutare
        output_dir: directory per salvare i video
        num_episodes_per_task: quanti video per task
    
    Returns:
        dict con risultati e path dei video
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    for task_id in range(len(task_suite)):
        task = task_suite.get_task(task_id)
        task_name = task.name.replace(' ', '_')
        task_dir = os.path.join(output_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        
        env, init_states = create_eval_env(task_suite, task_id)
        
        task_results = []
        
        for ep in range(num_episodes_per_task):
            # Setup video writer
            video_path = os.path.join(task_dir, f"episode_{ep}.mp4")
            frames = []
            
            # Reset environment
            env.reset()
            env.set_init_state(init_states[ep % len(init_states)])
            obs = env.get_observation()
            done = False
            step = 0
            max_steps = 500
            
            while not done and step < max_steps:
                # Cattura frame per video
                frame = env.render(mode='rgb_array')
                frames.append(frame)
                
                # Policy inference
                obs_tensor = preprocess_obs(obs).cuda()
                with torch.no_grad():
                    action = policy(obs_tensor).cpu().numpy().squeeze()
                
                # Step
                obs, reward, done, info = env.step(action)
                step += 1
            
            # Aggiungi ultimo frame
            frames.append(env.render(mode='rgb_array'))
            
            # Salva video
            save_video_mp4(frames, video_path, fps=30)
            
            # Log risultato
            success = info.get('success', False)
            task_results.append({
                'episode': ep,
                'success': success,
                'steps': step,
                'video_path': video_path
            })
            
            print(f"Task {task_name}, Ep {ep}: {'✅ SUCCESS' if success else '❌ FAIL'} ({step} steps)")
        
        env.close()
        results[task_name] = task_results
    
    # Genera report summary
    generate_video_report(results, output_dir)
    
    return results

def save_video_mp4(frames, path, fps=30):
    """Salva lista di frame come video MP4."""
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for frame in frames:
        # OpenCV usa BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Video salvato: {path}")

def generate_video_report(results, output_dir):
    """Genera report HTML con tutti i video e statistiche."""
    html_content = """
    <html>
    <head><title>TRM Evaluation Report</title></head>
    <body>
    <h1>TRM Robot Policy Evaluation</h1>
    <p>Generated: {timestamp}</p>
    """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    for task_name, episodes in results.items():
        success_rate = sum(ep['success'] for ep in episodes) / len(episodes)
        avg_steps = sum(ep['steps'] for ep in episodes) / len(episodes)
        
        html_content += f"""
        <h2>{task_name}</h2>
        <p>Success Rate: {success_rate:.1%} | Avg Steps: {avg_steps:.1f}</p>
        <div style="display: flex; flex-wrap: wrap;">
        """
        
        for ep in episodes:
            status = "✅" if ep['success'] else "❌"
            video_rel_path = os.path.relpath(ep['video_path'], output_dir)
            html_content += f"""
            <div style="margin: 10px;">
                <p>{status} Episode {ep['episode']}</p>
                <video width="320" controls>
                    <source src="{video_rel_path}" type="video/mp4">
                </video>
            </div>
            """
        
        html_content += "</div>"
    
    html_content += "</body></html>"
    
    report_path = os.path.join(output_dir, "evaluation_report.html")
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"Report generato: {report_path}")
```

### 5.3.4 Confronto con Baseline (Opzionale)

**Baseline da confrontare (se si vuole comparare con state-of-the-art):**

| Baseline | Descrizione | Implementazione |
|----------|-------------|-----------------|
| **BC-RNN** | Behavior Cloning con LSTM | robomimic |
| **BC-Transformer** | Transformer policy | robomimic |
| **Diffusion Policy** | Diffusion model per azioni | diffusion_policy repo |

### 5.3.5 Esperimenti di Ablation

**Ablation suggerite:**

1. **Numero di ricorsioni**:
   ```python
   for n_rec in [1, 2, 4, 8, 16, 32]:
       model = TRMPolicy(num_recursions=n_rec)
       results[f"n_rec_{n_rec}"] = evaluate(model)
   ```

2. **Dimensione hidden**:
   ```python
   for hidden in [64, 128, 256, 512]:
       model = TRMPolicy(hidden_dim=hidden)
       results[f"hidden_{hidden}"] = evaluate(model)
   ```

3. **Adaptive vs Fixed halt**:
   ```python
   for adaptive in [True, False]:
       model = TRMPolicy(adaptive_halt=adaptive)
       results[f"adaptive_{adaptive}"] = evaluate(model)
   ```

4. **Transfer across task suites**:
   - Train su LIBERO-Spatial → Test su LIBERO-Object
   - Valutare generalizzazione a distribution shift

---

## 6. Riferimenti Bibliografici

### TinyRecursiveModels e Architetture Correlate
1. Jolicoeur-Martineau, A. (2025). "Less is More: Recursive Reasoning with Tiny Networks." *arXiv:2510.04871*
2. McGovern, R.K. (2025). "Test-time Adaptation of Tiny Recursive Models." *arXiv:2511.02886*
3. Qasim, K.U. & Zhang, J. (2025). "Accelerating Training Speed of Tiny Recursive Models via Curriculum Guided Adaptive Recursion." *arXiv:2511.08653*
4. Yang, L. et al. (2024). "Looped Transformers are Better at Learning Learning Algorithms." *ICLR 2024*
5. Gong, Z. et al. (2025). "What Makes Looped Transformers Perform Better Than Non-Recursive Ones." *arXiv:2510.10089*

### LIBERO Benchmark
6. Liu, B. et al. (2023). "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning." *NeurIPS 2023*. [arXiv:2306.03310]
7. Website: https://libero-project.github.io/
8. GitHub: https://github.com/Lifelong-Robot-Learning/LIBERO

### ROBOCASA Benchmark
9. Nasiriany, S. et al. (2024). "RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots." *RSS 2024*. [arXiv:2406.02523]
10. Website: https://robocasa.ai/
11. GitHub: https://github.com/robocasa/robocasa

### Robot Learning e VLA
12. Schroeder, P. et al. (2025). "ROVER: Recursive Reasoning Over Videos with VLMs for Embodied Tasks." *arXiv:2508.01943*
13. Park, M. et al. (2025). "ACG: Action Coherence Guidance for Flow-based VLA models." *arXiv:2510.22201*
14. Koo, M. et al. (2025). "HAMLET: History-Aware Vision-Language-Action Model." *arXiv:2510.00695*

### Imitation Learning
15. Chi, C. et al. (2023). "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." *RSS 2023*
16. Mandlekar, A. et al. (2022). "robomimic: A Framework for Robot Learning." *CoRL 2021*

---

## Appendice A: Timeline Suggerita

| Settimana | Attività | Deliverable |
|-----------|----------|-------------|
| 1-2 | Setup ambiente, download dataset LIBERO, esplorazione dati | Dataset pronto, DataLoader funzionante |
| 3-4 | Implementazione TRM base, modifica input/output dimensions | Modello TRM adattato |
| 5 | **Hyperparameter search veloce** (LR alti, poche epoche) | Best config identificata |
| 6-7 | Training completo con best config, data augmentation | Modello allenato |
| 8-9 | Valutazione in simulazione LIBERO, estrazione video | Risultati quantitativi + video |
| 10 | (Opzionale) Estensione a ROBOCASA | Risultati cross-benchmark |
| 11-12 | Analisi risultati, scrittura report, video showcase | Report finale |

## Appendice B: Checklist del Progetto

### ✅ Step 1: Data Collection
- [ ] Ambiente conda configurato
- [ ] LIBERO installato e funzionante
- [ ] Dataset scaricato (almeno libero_spatial)
- [ ] DataLoader implementato e testato
- [ ] (Opzionale) ROBOCASA installato

### ✅ Step 2: Training TRM
- [ ] Architettura TRM adattata (input/output dimensions)
- [ ] Script di training Imitation Learning funzionante
- [ ] Hyperparameter search completata
- [ ] Data augmentation implementata
- [ ] Modello finale allenato e salvato
- [ ] (Opzionale) Supporto task prompts

### ✅ Step 3: Valutazione
- [ ] Script di valutazione in simulazione
- [ ] **Risultati quantitativi**: success rate per task
- [ ] **Risultati qualitativi**: video degli episodi salvati
- [ ] Report HTML con video generato
- [ ] (Opzionale) Confronto con baseline

---

## Note Finali

Questo progetto rappresenta un'opportunità unica di contribuire alla ricerca di frontiera nell'intersezione tra architetture di ragionamento ricorsivo e robot learning. I TinyRecursiveModels hanno mostrato risultati straordinari su task di ragionamento astratto; applicarli alla robotica potrebbe rivelare nuove prospettive su come il "pensiero iterativo" possa migliorare il controllo sensomotorio.

### 🎯 Riassunto delle Indicazioni del Professore

1. **Data Collection**: Usare LIBERO e/o ROBOCASA per costruire dataset (video + azioni)
2. **Selezione Task**: Sottogruppo di task (consigliato) o tutti (per state-of-the-art)
3. **Training**: Solo Imitation Learning (NO simulazione durante training)
4. **Modifiche TRM**: Principalmente input/output dimensions + sperimentare data augmentation
5. **Hyperparameter Search**: LR alti + poche epoche → scegliere config migliore
6. **Valutazione**: Testare in simulazione (plug-and-play)
7. **Output**: Risultati quantitativi (success rate) + qualitativi (VIDEO episodi)

### 💡 Chiavi del Successo

1. **Iniziare semplice**: BC su singolo task LIBERO-Spatial
2. **Iterare velocemente**: Usare strategia del professore (LR alti, poche epoche)
3. **Documentare tutto**: WandB/TensorBoard per tracking esperimenti
4. **Sperimentare data augmentation**: Può fare la differenza in robotica
5. **Salvare i video**: Sono fondamentali per il deliverable finale!

### 🔗 Risorse Utili

- **TRM originale**: Studiare l'architettura e come gestisce i task prompts
- **LIBERO docs**: https://lifelong-robot-learning.github.io/LIBERO/
- **ROBOCASA docs**: https://robocasa.ai/docs/
- **robomimic**: Framework utile per baseline BC

Buon lavoro! 🤖
