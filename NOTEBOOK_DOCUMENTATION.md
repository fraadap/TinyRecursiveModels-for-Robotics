# 📘 Documentazione: TinyRecursiveModels per Controllo Robotico

Questo documento descrive in dettaglio tutte le fasi del notebook `kaggle_trm_robotics.py`, spiegando le scelte tecniche e le motivazioni dietro ogni componente.

---

## 📑 Indice

1. [Panoramica del Progetto](#1-panoramica-del-progetto)
2. [Setup e Installazione](#2-setup-e-installazione)
3. [Caricamento e Esplorazione Dataset](#3-caricamento-e-esplorazione-dataset)
4. [Dataset Class: LIBERODataset](#4-dataset-class-liberodataset)
5. [Architettura TRM](#5-architettura-trm)
   - [Visual Encoder](#51-visual-encoder)
   - [Recursive Block](#52-recursive-block)
   - [TRM Policy](#53-trm-policy)
   - [Adaptive Halting (ACT)](#54-adaptive-halting-act)
6. [Training: Behavior Cloning](#6-training-behavior-cloning)
7. [Hyperparameter Search](#7-hyperparameter-search)
8. [Valutazione in Simulazione](#8-valutazione-in-simulazione)
9. [Utility e Visualizzazioni](#9-utility-e-visualizzazioni)

---

## 1. Panoramica del Progetto

### Obiettivo
Adattare l'architettura **TinyRecursiveModels (TRM)** — originariamente progettata per task di ragionamento astratto — al **controllo robotico** utilizzando il benchmark **LIBERO**.

### Workflow
```
Dataset LIBERO → Preprocessing → Training BC → Valutazione → Video/Metriche
     ↓                              ↓
  Video + Azioni GT         TRM Policy (ricorsiva)
```

### Perché TRM per Robotica?
I TRM utilizzano **ragionamento iterativo**: lo stesso blocco neurale viene applicato N volte, permettendo al modello di "pensare più a lungo" su input complessi. Questo potrebbe essere utile in robotica per:
- Pianificazione implicita delle azioni
- Gestione di situazioni ambigue
- Migliore accoppiamento percezione-azione

---

## 2. Setup e Installazione

### Fasi di Setup

```python
# Step 1: Clone repository LIBERO
!git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

# Step 2-3: Installazione dipendenze
!pip install matplotlib tokenizers==0.15.2
!pip install -r <(grep -v -e "matplotlib" -e "tokenizers" requirements.txt) --no-deps

# Step 4: Copia dataset
!mkdir -p /kaggle/working/dataset && cp -r .../datasets/* /kaggle/working/dataset/

# Step 5: Librerie aggiuntive
!pip install einops wandb transformers robomimic
```

### Librerie Chiave

| Libreria | Scopo |
|----------|-------|
| **einops** | Manipolazione tensori leggibile (reshape, permute) |
| **wandb** | Tracking esperimenti e visualizzazione metriche |
| **transformers** | Embedding testuali per task prompts (futuro) |
| **robomimic** | Utility per file HDF5 robotici |

### Verifica GPU
```python
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")  # ⚠️ Molto più lento!
```

**Nota**: Su Kaggle, selezionare GPU P100 o T4 in Settings → Accelerator.

---

## 3. Caricamento e Esplorazione Dataset

### Struttura File HDF5 LIBERO

```
demo_file.hdf5
├── data/
│   ├── demo_0/
│   │   ├── obs/
│   │   │   ├── agentview_rgb    # (T, 128, 128, 3) immagini RGB
│   │   │   ├── robot0_eef_pos   # posizione end-effector
│   │   │   └── ...
│   │   ├── actions              # (T, 7) azioni 7-DoF
│   │   └── rewards              # (T,) reward per step
│   ├── demo_1/
│   └── ...
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
        # Trova demo e timestep corrispondenti
        # Applica preprocessing
        # Restituisce {observations, actions}
```

### Normalizzazione Azioni (Z-Score)

```python
# Calcolo statistiche
mean = all_actions.mean(axis=0)  # (7,)
std = all_actions.std(axis=0) + 1e-6  # (7,) + epsilon per stabilità

# Normalizzazione
actions_norm = (actions - mean) / std
```

**Perché normalizzare?**
- Le diverse dimensioni delle azioni hanno scale diverse (posizione vs rotazione)
- La normalizzazione aiuta la convergenza del training
- L'epsilon (1e-6) previene divisione per zero

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

### 5.1 Visual Encoder

```python
class VisualEncoder(nn.Module):
    # CNN ispirata a Nature DQN (Mnih et al., 2015)
    
    Conv2d(3, 32, kernel=8, stride=4)  →  ReLU
    Conv2d(32, 64, kernel=4, stride=2) →  ReLU  
    Conv2d(64, 128, kernel=3, stride=1) → ReLU → Flatten
    Linear(conv_out, hidden_dim) → LayerNorm
```

**Flusso dimensionale** (per immagine 128×128):
```
Input:  (B, 128, 128, 3)
        ↓ permute HWC→CHW
        (B, 3, 128, 128)
        ↓ Conv1
        (B, 32, 31, 31)
        ↓ Conv2
        (B, 64, 14, 14)
        ↓ Conv3
        (B, 128, 12, 12)
        ↓ Flatten
        (B, 18432)
        ↓ Linear + LayerNorm
Output: (B, 256)  # hidden_dim
```

**Perché questa architettura?**
- Progressiva riduzione spaziale cattura features a scale diverse
- LayerNorm stabilizza l'output per i blocchi successivi

### 5.2 Recursive Block

```python
class RecursiveBlock(nn.Module):
    def forward(self, h, x_cond):
        # h: hidden state corrente
        # x_cond: conditioning dall'encoder (costante)
        
        combined = h + x_cond           # Fusione
        combined = combined.unsqueeze(1) # (B, 1, D) per attention
        
        # Self-attention + residual
        attn_out = self.attention(combined, combined, combined)
        combined = self.norm1(combined + attn_out)
        
        # MLP + residual
        mlp_out = self.mlp(combined)
        h_new = self.norm2(combined + mlp_out)
        
        return h_new.squeeze(1)
```

**Componenti chiave:**

| Componente | Scopo |
|------------|-------|
| `h + x_cond` | Combina stato latente con input originale (conditioning) |
| Self-Attention | Permette al modello di "ragionare" sulle proprie rappresentazioni |
| MLP 4× | Trasformazione non-lineare (espande, trasforma, comprime) |
| LayerNorm + Residual | Stabilità del training con molte ricorsioni |

**Perché il conditioning `x_cond`?**
Il TRM applica lo stesso blocco N volte, ma l'input originale potrebbe "sbiadire". Aggiungendo `x_cond` a ogni step, manteniamo l'informazione dell'osservazione ancorata.

### 5.3 TRM Policy

```python
class TRMPolicy(nn.Module):
    def forward(self, obs):
        # 1. Encoding
        x_cond = self.encoder(obs)
        h = x_cond.clone()
        
        # 2. Ricorsione (N volte)
        for t in range(self.num_recursions):
            h = self.recursive_block(h, x_cond)
        
        # 3. Predizione azione
        actions = self.action_head(h)
        return actions
```

**Diagramma del flusso:**
```
Osservazione (128×128×3)
        ↓
   VisualEncoder
        ↓
   x_cond (256-dim)
        ↓
   ┌─────────────────────────────┐
   │  RecursiveBlock × N        │ ← Stesso blocco, pesi condivisi!
   │  h₀ → h₁ → h₂ → ... → hₙ   │
   └─────────────────────────────┘
        ↓
   ActionHead (MLP)
        ↓
   Azione (7-dim)
```

### 5.4 Adaptive Halting (ACT)

**Cos'è l'Adaptive Halting?**

Invece di applicare il blocco ricorsivo un numero **fisso** di volte (es. 8), il modello impara **quando fermarsi** in base alla complessità dell'input.

```python
def _forward_adaptive(self, h, x_cond, B):
    remainders = torch.ones(B)      # "Energia" rimanente per ogni sample
    accumulated_h = torch.zeros_like(h)  # Output accumulato
    
    for t in range(self.num_recursions):
        h = self.recursive_block(h, x_cond)
        
        # Predici probabilità di fermarsi
        halt_p = self.halt_predictor(h)  # σ(Linear(h)) ∈ [0,1]
        
        # Accumula output pesato per remainder
        accumulated_h += remainders * h
        
        # Aggiorna remainder
        remainders = remainders * (1 - halt_p)
        
        # Early stopping se tutti hanno fermato
        if remainders.max() < 0.01:
            break
    
    return accumulated_h
```

**Intuizione matematica:**

Per ogni sample $i$ al passo $t$:
$$h_{\text{final}}^{(i)} = \sum_{t=1}^{T} r_t^{(i)} \cdot h_t^{(i)}$$

dove $r_t$ è il "remainder" (quanto peso dare a questo step).

**Perché è utile in robotica?**

| Situazione | Ricorsioni | Motivo |
|------------|------------|--------|
| Pick-and-place semplice | 3-4 | Oggetto visibile, azione diretta |
| Scena occlusione | 6-8 | Serve più "ragionamento" |
| Ambiente complesso | 8+ | Pianificazione più profonda |

**Trade-off:**
- ✅ Efficienza computazionale (meno ricorsioni quando possibile)
- ✅ Adattività alla complessità del task
- ⚠️ Training più complesso (bisogna regolarizzare l'halting)

---

## 6. Training: Behavior Cloning

### BehaviorCloningTrainer

```python
class BehaviorCloningTrainer:
    def train(self):
        for epoch in range(epochs):
            # Training
            for batch in train_loader:
                pred = model(obs)
                loss = MSE(pred, target_actions)
                loss.backward()
                optimizer.step()
            
            # Validation + Early Stopping
            val_loss = validate()
            if val_loss < best:
                save_checkpoint()
```

### Loss Function: MSE

$$\mathcal{L} = \frac{1}{N \cdot 7} \sum_{i=1}^{N} \sum_{j=1}^{7} (a_{ij}^{\text{pred}} - a_{ij}^{\text{target}})^2$$

**Perché MSE?**
- Azioni sono continue (non categoriche)
- Penalizza errori grandi più di quelli piccoli
- Standard per regressione

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Perché?**
- Previene gradient explosion durante training
- Particolarmente importante con architetture ricorsive
- Stabilizza il training con learning rate alti

### Learning Rate Schedule: Cosine Annealing

```python
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
```

```
LR
 ↑
 │  ╭──────╮
 │ ╱        ╲
 │╱          ╲
 └────────────╲─────→ epochs
```

**Vantaggi:**
- Esplorazione iniziale con LR alto
- Fine-tuning finale con LR basso
- Smooth transition (no step bruschi)

---

## 7. Hyperparameter Search

### Strategia: Quick Search

Seguendo le indicazioni del professore: **LR alti + poche epoche** per identificare velocemente la configurazione migliore.

```python
param_grid = {
    'lr': [1e-3, 5e-4, 3e-4, 1e-4],
    'hidden_dim': [128, 256],
    'num_recursions': [4, 8],
}

# Per ogni config: 10 epoche veloci
for config in all_configs:
    train_quick(config, epochs=10)
    val_loss = evaluate()
    results.append((config, val_loss))

best_config = min(results, key=lambda x: x[1])
```

### Numero di Configurazioni

$4 \times 2 \times 2 = 16$ configurazioni totali

Con 10 epoche ciascuna, è gestibile in poche ore su GPU.

### Criteri di Selezione

1. **Val Loss finale** (primario)
2. **Velocità di convergenza** (secondario)
3. **Stabilità** (no spike nella loss)

---

## 8. Valutazione in Simulazione

### PolicyEvaluator

```python
class PolicyEvaluator:
    def evaluate_on_task(self, env, init_states, num_episodes=50):
        successes = []
        
        for ep in range(num_episodes):
            env.reset()
            env.set_init_state(init_states[ep])
            
            while not done:
                obs = env.get_observation()
                action = policy(preprocess(obs))
                action = denormalize(action)  # Importante!
                obs, reward, done, info = env.step(action)
            
            successes.append(info['success'])
        
        return {'success_rate': mean(successes)}
```

### Denormalizzazione Azioni

```python
# Durante training: azioni normalizzate
action_norm = (action - mean) / std

# Durante inferenza: denormalizzare!
action_real = action_norm * std + mean
```

**⚠️ Errore comune**: dimenticare la denormalizzazione causa azioni completamente sbagliate!

### Metriche

| Metrica | Descrizione | Target |
|---------|-------------|--------|
| **Success Rate** | % episodi completati | >60% |
| **Avg Episode Length** | Step medi per episodio | <200 |
| **Std Episode Length** | Variabilità | Bassa = consistenza |

### Recording Video

```python
def record_episode():
    frames = []
    while not done:
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        # ... step ...
    
    save_video(frames, path, fps=30)
```

**Output**: Video MP4 degli episodi per analisi qualitativa.

---

## 9. Utility e Visualizzazioni

### Analisi Predizioni

```python
def analyze_model_predictions(model, val_loader):
    # Per N campioni:
    # 1. Visualizza immagine osservazione
    # 2. Bar plot: target vs predicted actions
    # 3. Calcola MSE per campione
```

**Output**: Griglia che mostra quanto le predizioni si discostano dal target.

### Visualizzazione Stati Ricorsivi

```python
def visualize_recursive_states(model, obs):
    actions, states = model(obs, return_all_states=True)
    
    # states: lista di hidden states [h0, h1, ..., h8]
    
    # 1. Heatmap: evoluzione features
    # 2. Line plot: norma L2 per step
```

**Cosa cercare:**
- **Convergenza**: norma L2 che si stabilizza → il modello ha "finito di pensare"
- **Cambiamenti graduali**: transizioni smooth tra stati
- **Divergenza**: norma che esplode → problema di training

---

## 📊 Riepilogo Architettura

```
┌─────────────────────────────────────────────────────────────┐
│                    TRM POLICY                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Osservazione RGB (128×128×3)                               │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────┐                                        │
│  │  VisualEncoder  │  CNN + LayerNorm                       │
│  │  (3→32→64→128)  │                                        │
│  └────────┬────────┘                                        │
│           │ x_cond (256-dim)                                │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────┐                    │
│  │       RECURSIVE BLOCK × N           │                    │
│  │  ┌─────────────────────────────┐    │                    │
│  │  │ h + x_cond                  │    │                    │
│  │  │     ↓                       │    │                    │
│  │  │ Self-Attention (4 heads)    │    │                    │
│  │  │     ↓                       │    │                    │
│  │  │ MLP (256 → 1024 → 256)      │    │                    │
│  │  │     ↓                       │    │                    │
│  │  │ LayerNorm + Residual        │    │                    │
│  │  └─────────────────────────────┘    │                    │
│  │         ↓                           │                    │
│  │    h_new (feedback al prossimo)     │                    │
│  └─────────────────────────────────────┘                    │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │   Action Head   │  Linear(256→256→7)                     │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│     Azione (7-DoF)                                          │
│  [dx, dy, dz, droll, dpitch, dyaw, gripper]                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## ❓ FAQ Tecniche

### Q: Perché usare pesi condivisi nel Recursive Block?
**A:** È l'essenza del TRM! Con pesi condivisi:
- Meno parametri (7-27M invece di N×27M)
- Il modello impara un "algoritmo" generale, non step specifici
- Può generalizzare a più ricorsioni durante inferenza

### Q: Quante ricorsioni sono ottimali?
**A:** Dipende dal task:
- **4 ricorsioni**: task semplici, veloce
- **8 ricorsioni**: buon compromesso (default)
- **16+ ricorsioni**: solo se adaptive halting

### Q: Come gestire il gripper (discreto) vs posizione (continua)?
**A:** Nell'implementazione corrente trattiamo tutto come continuo. Alternative:
- Soglia su gripper output (>0 = chiudi, <0 = apri)
- Head separato con softmax per gripper
- Mixture of experts

### Q: Perché non usare RL dopo BC?
**A:** Come indicato dal professore, il focus è su **Imitation Learning puro**. RL può essere aggiunto successivamente ma complica significativamente il setup.

---

## 📚 Riferimenti

1. **TinyRecursiveModels**: arXiv:2510.04871, 2511.02886
2. **LIBERO Benchmark**: https://lifelong-robot-learning.github.io/LIBERO/
3. **Behavior Cloning**: Pomerleau, 1988
4. **Adaptive Computation Time**: Graves, 2016
5. **Nature DQN (CNN architecture)**: Mnih et al., 2015

---

*Documento generato per il progetto AML - TinyRecursiveModels for Robotics*
