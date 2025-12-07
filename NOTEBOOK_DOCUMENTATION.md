# 📘 Documentazione: TinyRecursiveModels per Controllo Robotico

Questo documento descrive in dettaglio tutte le fasi del notebook `notebook.py`, spiegando le scelte tecniche e le motivazioni dietro ogni componente.

---

## 📑 Indice

1. [Panoramica del Progetto](#1-panoramica-del-progetto)
2. [Setup e Installazione](#2-setup-e-installazione)
3. [Caricamento e Esplorazione Dataset](#3-caricamento-e-esplorazione-dataset)
4. [Dataset Class: LIBERODataset](#4-dataset-class-liberodataset)
5. [Architettura TRM](#5-architettura-trm)
   - [Visual Encoder](#51-visual-encoder)
   - [Text Encoder (CLIP)](#52-text-encoder-clip)
   - [Recursive Block](#53-recursive-block)
   - [TRM Policy](#54-trm-policy)
   - [Adaptive Halting (ACT)](#55-adaptive-halting-act)
6. [Training: Behavior Cloning](#6-training-behavior-cloning)
7. [Hyperparameter Search (Optuna)](#7-hyperparameter-search-optuna)
8. [Valutazione in Simulazione](#8-valutazione-in-simulazione)
9. [Utility e Visualizzazioni](#9-utility-e-visualizzazioni)

---

## 1. Panoramica del Progetto

### Obiettivo
Adattare l'architettura **TinyRecursiveModels (TRM)** — originariamente progettata per task di ragionamento astratto — al **controllo robotico** utilizzando il benchmark **LIBERO**. Il modello è condizionato sia dalle osservazioni visive che dalle istruzioni testuali del task.

### Workflow
```
Dataset LIBERO → Preprocessing → Training BC (Optuna) → Valutazione → Video/Metriche
     ↓                              ↓
  Video + Testo + Azioni GT     TRM Policy (Visio-Linguistica Ricorsiva)
```

### Perché TRM per Robotica?
I TRM utilizzano **ragionamento iterativo**: lo stesso blocco neurale viene applicato N volte, permettendo al modello di "pensare più a lungo" su input complessi. Questo potrebbe essere utile in robotica per:
- Pianificazione implicita delle azioni
- Gestione di situazioni ambigue
- Migliore accoppiamento percezione-azione

---

## 2. Setup e Installazione

### Patch Critica (Matplotlib)
All'inizio del notebook viene applicato un **Monkey Patch** per `matplotlib`. Questo è necessario per prevenire crash del kernel dovuti a conflitti ABI o problemi di backend grafico durante l'importazione di librerie di simulazione (come `robosuite` o `libero`) in ambienti headless.

```python
mock_mpl = MagicMock()
sys.modules["matplotlib"] = mock_mpl
# ... altri moduli matplotlib mockati ...
```

### Fasi di Setup

```python
# Step 1: Clone repository LIBERO
!git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

# Step 2: Download Dataset
!python LIBERO/benchmark_scripts/download_libero_datasets.py ...
```

### Librerie Chiave

| Libreria | Scopo |
|----------|-------|
| **einops** | Manipolazione tensori leggibile (reshape, permute) |
| **wandb** | Tracking esperimenti e visualizzazione metriche |
| **transformers** | Caricamento di CLIP per embedding testuali |
| **optuna** | Ottimizzazione iperparametri (TPE Sampler) |
| **h5py** | Gestione file dati robotici |

---

## 3. Caricamento e Esplorazione Dataset

### Struttura File HDF5 LIBERO

```
demo_file.hdf5
├── data/
│   ├── demo_0/
│   │   ├── obs/
│   │   │   ├── agentview_rgb    # (T, 128, 128, 3) immagini RGB
│   │   │   └── ...
│   │   ├── actions              # (T, 7) azioni 7-DoF
│   │   └── ...
```

### Formato Azioni (7-DoF)
```
actions = [dx, dy, dz, droll, dpitch, dyaw, gripper]
           ├─────────────┤ ├──────────────────┤    └─ -1 (apri) / +1 (chiudi)
           delta posizione   delta orientamento
```

### Funzione di Esplorazione
```python
def explore_libero_dataset(data_path):
    # 1. Trova tutti i file .hdf5
    # 2. Analizza struttura primo file
    # 3. Stampa shape di immagini e azioni
    # 4. Visualizza 4 frame equidistanti
```

**Perché esplorare?** Capire il formato dati è cruciale prima di costruire il DataLoader.

---

## 4. Dataset Class: LIBERODataset

### Architettura della Classe

```python
class LIBERODataset(Dataset):
    def __init__(self, hdf5_files, sequence_length, image_size, 
                 normalize_actions, augmentation, max_demos_per_task):
        # Carica tutti i dati in memoria
        # Calcola statistiche per normalizzazione
    
    def __getitem__(self, idx):
        # Restituisce:
        # - 'observations': (C, H, W) normalizzato [0,1]
        # - 'actions': (7,) normalizzato Z-score
        # - 'prompt': stringa descrizione task (es. "pick up the black bowl")
```

### Normalizzazione Azioni (Z-Score)

```python
# Calcolo statistiche
mean = all_actions.mean(axis=0)  # (7,)
std = all_actions.std(axis=0) + 1e-6  # (7,) + epsilon per stabilità

# Normalizzazione
actions_norm = (actions - mean) / std
```

### Data Augmentation

| Tecnica | Probabilità | Descrizione |
|---------|-------------|-------------|
| **Color Jitter** | 50% | Brightness random in [0.8, 1.2] |
| **Random Crop** | 30% | Crop 90% + resize a dimensione originale |

```python
def _augment_obs(self, obs):
    if np.random.rand() < 0.5:
        brightness = np.random.uniform(0.8, 1.2)
        obs = np.clip(obs * brightness, 0, 1)
    # ...
```

**Perché augmentation?**
- Aumenta la robustezza a variazioni di illuminazione
- Riduce overfitting su dataset piccoli
- Simula variazioni che il robot potrebbe incontrare

---

## 5. Architettura TRM

Il modello è ora **Visio-Linguistico**, combinando input visivi e istruzioni testuali.

### 5.1 Visual Encoder

Il codice supporta due modalità:
1.  **Pretrained (Default)**: `ResNet18` pre-addestrata su ImageNet.
2.  **Custom**: Una CNN a 4 livelli addestrata da zero.

### 5.2 Text Encoder (CLIP)

Utilizza **CLIP (ViT-L/14)** per generare embedding semantici delle istruzioni del task.

```python
class PromptEncoder(nn.Module):
    # Tokenizer e TextModel di CLIP
    # Output: Embedding (B, hidden_dim)
```
L'embedding testuale viene fuso con quello visivo (concatenazione + proiezione lineare) prima di entrare nel blocco ricorsivo.

### 5.3 Recursive Block

```python
class RecursiveBlock(nn.Module):
    def forward(self, h, x_cond):
        # h: hidden state corrente
        # x_cond: conditioning (Vision + Text)
        
        combined = h + x_cond           # Fusione
        combined = combined.unsqueeze(1) 
        
        # Self-attention + residual
        attn_out = self.attention(combined, combined, combined)
        combined = self.norm1(combined + attn_out)
        
        # MLP + residual
        mlp_out = self.mlp(combined)
        h_new = self.norm2(combined + mlp_out)
        
        return h_new.squeeze(1)
```

### 5.4 TRM Policy

```python
class TRMPolicy(nn.Module):
    def forward(self, obs, prompts):
        # 1. Encoding Visivo
        vis_feat = self.encoder(obs)
        
        # 2. Encoding Testuale
        text_feat = self.prompt_encoder(prompts)
        
        # 3. Fusione
        x_cond = self.fusion_adapter(cat([vis_feat, text_feat]))
        h = x_cond.clone()
        
        # 4. Ricorsione (N volte)
        for t in range(self.num_recursions):
            h = self.recursive_block(h, x_cond)
        
        # 5. Predizione azione
        actions = self.action_head(h)
        return actions
```

### 5.5 Adaptive Halting (ACT)
Opzionalmente, il modello può usare l'Adaptive Computation Time per decidere dinamicamente quante ricorsioni eseguire per ogni input, basandosi su un "halt predictor" sigmoideo.

---

## 6. Training: Behavior Cloning

### Loss Function Mista
La loss function combina MSE e L1 per robustezza:
$$\mathcal{L} = 0.7 \cdot \text{MSE} + 0.3 \cdot \text{L1}$$

### Mixed Precision (AMP)
Il trainer utilizza `torch.cuda.amp` per accelerare il training e ridurre l'uso di memoria VRAM, mantenendo la stabilità numerica tramite `GradScaler`.

### Scheduler
Supporta due modalità:
1.  **Cosine Annealing Warm Restarts**: Per cicli di learning rate.
2.  **Linear Warmup + Constant**: (Default) Riscaldamento graduale seguito da LR costante.

---

## 7. Hyperparameter Search (Optuna)

Invece di una Grid Search manuale, il notebook utilizza **Optuna** con il sampler **TPE (Tree-structured Parzen Estimator)**.

```python
def optuna_random_search(...):
    # Definisce spazio di ricerca (LR, hidden_dim, dropout, etc.)
    # Esegue N trial
    # Ottimizza la Validation Loss
```

Questo approccio è più efficiente nel trovare combinazioni ottimali di iperparametri rispetto alla ricerca casuale pura o a griglia.

---

## 8. Valutazione in Simulazione

### Funzione `evaluate_model`
Questa funzione gestisce l'interazione con l'ambiente simulato **LIBERO**:
1.  Carica il modello addestrato e le statistiche di normalizzazione.
2.  Inizializza l'ambiente `OffScreenRenderEnv` per il task specifico.
3.  Esegue il loop di controllo:
    - Ottiene osservazione.
    - Preprocessa immagine.
    - Passa immagine e **prompt testuale** alla policy.
    - Denormalizza l'azione.
    - Esegue step nell'ambiente.
4.  Salva video degli episodi e calcola il Success Rate.

---

## 📊 Riepilogo Architettura Aggiornata

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