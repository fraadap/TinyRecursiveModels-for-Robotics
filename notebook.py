# %% [markdown]
# # Libero

# %%
import sys
from unittest.mock import MagicMock

# The Patch
mock_mpl = MagicMock()
sys.modules["matplotlib"] = mock_mpl
sys.modules["matplotlib.pyplot"] = mock_mpl
sys.modules["matplotlib.cm"] = mock_mpl
sys.modules["matplotlib.colors"] = mock_mpl
sys.modules["matplotlib.transforms"] = mock_mpl
sys.modules["matplotlib.ticker"] = mock_mpl
sys.modules["matplotlib._path"] = mock_mpl

# %%
import h5py
import numpy as np
from PIL import Image
from IPython.display import display, HTML

# Path to your demo file
file_path = "dataset/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5"

print(f"Opening file: {file_path}")

try:
    with h5py.File(file_path, "r") as f:
        # Loop through all demos in the file
        for demo_name in f["data"]:
            print(f"\n=== Demo: {demo_name} ===")
            
            # Access the image data (AgentView RGB)
            # Note: Ensure this path exists in your specific HDF5 structure
            if "obs/agentview_rgb" in f[f"data/{demo_name}"]:
                dataset = f[f"data/{demo_name}/obs/agentview_rgb"]
                num_images = dataset.shape[0]
                print(f"Total frames: {num_images}")
                
                # Pick indices: every 15th frame + the last one
                indices = list(range(0, num_images, 15))
                if num_images - 1 not in indices:
                    indices.append(num_images - 1)
                
                # Display images horizontally using HTML/PIL (No Matplotlib)
                images_html = []
                for idx in indices:
                    img_array = dataset[idx]
                    
                    # Convert numpy array to PIL Image
                    # (Robosuite images are usually already correct, but sometimes flipped)
                    img = Image.fromarray(img_array)
                    
                    # Resize for smaller display if needed
                    img_small = img.resize((128, 128)) 
                    
                    # Hack to display inline in loop
                    print(f"Frame {idx}:")
                    #display(img)
            else:
                print(f"Skipping {demo_name}: 'obs/agentview_rgb' not found.")
                
except Exception as e:
    print(f"An error occurred: {e}")

# %% [markdown]
# # Libraries

# %%
import torch
import os
import torch.nn as nn
import numpy as np
import h5py
import einops
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms

# Verifica disponibilità GPU
print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"✅ GPU Disponibile: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("⚠️ ATTENZIONE: GPU non rilevata. Vai su 'Settings' > 'Accelerator' e seleziona GPU P100 o T4.")
    device = torch.device("cpu")

# Verifica Einops (lo useremo molto nel modello ricorsivo)
print(f"Einops installato correttamente.")

# Configurazione base per riproducibilità (importante per la tesi)
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
print("✅ Ambiente pronto per il modello TinyRecursive.")

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
# import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, replace, field
import json
import wandb
import optuna
from datetime import datetime
from transformers import CLIPTokenizer, CLIPTextModel

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Seed per riproducibilità
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# =========================
# Configuration utilities
# =========================

@dataclass
class TrainingConfig:
    """Container strutturato per gli iperparametri di training/model."""

    lr: float = 3e-4
    hidden_dim: int = 256
    num_recursions: int = 8
    num_slots: int = 4
    epochs: int = 20
    batch_size: int = 64
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = 1.0
    sched_T0: Optional[int] = None
    sched_T_mult: int = 1
    lr_min: float = 1e-6
    warmup_epochs: int = 3
    early_stop_patience: Optional[int] = None
    save_path: str = 'best_model.pt'
    use_pretrained_encoder: bool = True
    freeze_backbone: bool = True
    augmentation: bool = False
    dropout: float = 0.1
    encoder_dropout: float = 0.1
    use_text_prompts: bool = True
    text_encoder_name: str = 'openai/clip-vit-large-patch14'
    train_text_encoder: bool = False
    text_dropout: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def label(self) -> str:
        return f"lr{self.lr}_h{self.hidden_dim}_rec{self.num_recursions}_bs{self.batch_size}"


@dataclass
class HyperparameterSearchSpace:
    lr: List[float]
    hidden_dim: List[int]
    num_recursions: List[int]
    batch_size: List[int]
    weight_decay: List[float]
    pretrained_encoder: List[bool]
    freeze_backbone: List[bool]
    augmentation: List[bool]
    dropout: List[float]
    text_encoder_name: List[str]
    train_text_encoder: List[bool]
    text_dropout: List[float]
    num_slots: List[int] = field(default_factory=lambda: [4])

    def as_optuna_space(self) -> Dict[str, List[Any]]:
        return {
            'lr': self.lr,
            'hidden_dim': self.hidden_dim,
            'num_recursions': self.num_recursions,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
            'pretrained_encoder': self.pretrained_encoder,
            'freeze_backbone': self.freeze_backbone,
            'augmentation': self.augmentation,
            'dropout': self.dropout,
            'text_encoder_name': self.text_encoder_name,
            'train_text_encoder': self.train_text_encoder,
            'text_dropout': self.text_dropout,
            'num_slots': self.num_slots,
        }

# Numero 1
def default_search_space() -> HyperparameterSearchSpace:
    """Restituisce lo spazio di ricerca richiesto dall'utente."""

    return HyperparameterSearchSpace(
        lr=[1e-5, 1e-6, 1e-4, 1e-3],
        hidden_dim=[128, 256, 512],
        num_recursions=[8, 12, 16],
        batch_size=[32, 64, 128, 256, 512],
        weight_decay=[0.1, 0.5, 1.0],
        pretrained_encoder=[False],
        freeze_backbone=[False],
        augmentation=[True, False],
        dropout=[0.1, 0.3, 0.5, 0.7],
        text_encoder_name=[
            'openai/clip-vit-base-patch32',
            'openai/clip-vit-large-patch14'
        ],
        train_text_encoder=[False, True],
        text_dropout=[0.05, 0.1, 0.2, 0.3],
        num_slots=[4, 6, 8]
    )
#Numero 2
'''
def default_search_space() -> HyperparameterSearchSpace:
    """Restituisce lo spazio di ricerca richiesto dall'utente."""

    return HyperparameterSearchSpace(
        lr=[1e-5, 1e-6, 1e-4, 1e-3],
        hidden_dim=[128, 256, 512],
        num_recursions=[8, 12, 16],
        batch_size=[32, 64, 128, 256, 512],
        weight_decay=[0.1, 0.5, 1.0],
        pretrained_encoder=[False],
        freeze_backbone=[True],
        augmentation=[True, False],
        dropout=[0.1, 0.3, 0.5, 0.7],
    )
'''

# Numero 3
'''
def default_search_space() -> HyperparameterSearchSpace:
    """Restituisce lo spazio di ricerca richiesto dall'utente."""

    return HyperparameterSearchSpace(
        lr=[1e-5, 1e-6, 1e-4, 1e-3],
        hidden_dim=[128, 256, 512],
        num_recursions=[8, 12, 16],
        batch_size=[32, 64, 128, 256, 512],
        weight_decay=[0.1, 0.5, 1.0],
        pretrained_encoder=[True],
        freeze_backbone=[False],
        augmentation=[True, False],
        dropout=[0.1, 0.3, 0.5, 0.7],
    )
'''

# Numero 4
'''
def default_search_space() -> HyperparameterSearchSpace:
    """Restituisce lo spazio di ricerca richiesto dall'utente."""

    return HyperparameterSearchSpace(
        lr=[1e-5, 1e-6, 1e-4, 1e-3],
        hidden_dim=[128, 256, 512],
        num_recursions=[8, 12, 16],
        batch_size=[32, 64, 128, 256, 512],
        weight_decay=[0.1, 0.5, 1.0],
        pretrained_encoder=[True],
        freeze_backbone=[True],
        augmentation=[True, False],
        dropout=[0.1, 0.3, 0.5, 0.7],
    )
'''

# %%
import numpy as np
import h5py
import sys
from pathlib import Path
from PIL import Image
from IPython.display import display

# --- HELPER FUNCTIONS (Kept largely the same) ---

def load_images_robust(dataset):
    """
    Carica immagini da dataset HDF5 usando metodo robusto.
    """
    shape = dataset.shape
    
    # METODO 1: Lettura diretta uint8
    try:
        buffer = np.empty(shape, dtype=np.uint8)
        dataset.read_direct(buffer)
        return buffer
    except Exception:
        pass
    
    # METODO 2: Float32 -> Uint8
    try:
        buffer = np.empty(shape, dtype=np.float32)
        dataset.read_direct(buffer)
        if buffer.max() <= 1.0:
            buffer = (buffer * 255).astype(np.uint8)
        else:
            buffer = np.clip(buffer, 0, 255).astype(np.uint8)
        return buffer
    except Exception:
        pass
    
    # METODO 3: Float64 -> Uint8
    try:
        buffer = np.empty(shape, dtype=np.float64)
        dataset.read_direct(buffer)
        if buffer.max() <= 1.0:
            buffer = (buffer * 255).astype(np.uint8)
        else:
            buffer = np.clip(buffer, 0, 255).astype(np.uint8)
        return buffer
    except Exception:
        pass

    # METODO 4: Fallback bytes
    try:
        buffer = np.empty(shape, dtype=np.uint8)
        dataset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, buffer)
        return buffer
    except Exception as e:
        raise RuntimeError(f"Impossibile leggere il dataset: {e}")

def load_actions_robust(dataset):
    """
    Carica azioni da dataset HDF5.
    """
    shape = dataset.shape
    try:
        buffer = np.empty(shape, dtype=np.float32)
        dataset.read_direct(buffer)
        return buffer
    except Exception:
        pass
    
    try:
        buffer = np.empty(shape, dtype=np.float64)
        dataset.read_direct(buffer)
        return buffer.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Impossibile leggere le azioni: {e}")

# --- MODIFIED EXPLORATION FUNCTION (No Matplotlib) ---

def explore_libero_dataset(data_path: Path):
    
    # Trova file
    hdf5_files = list(data_path.glob('**/*.hdf5'))
    
    if not hdf5_files:
        print(f"⚠️ Nessun file HDF5 trovato in {data_path}")
        return []
    
    print(f"✅ Trovati {len(hdf5_files)} file HDF5")
    
    # Analizza il primo file
    demo_file = hdf5_files[0]
    print(f"\n📄 Analizzando: {demo_file.name}")
    
    try:
        with h5py.File(demo_file, 'r') as f:
            if 'data' not in f:
                print("⚠️ Chiave 'data' non trovata")
                return hdf5_files
            
            data_group = f['data']
            demo_keys = list(data_group.keys())
            first_demo_key = demo_keys[0]
            demo_0 = data_group[first_demo_key]
            
            imgs = None
            
            # 1. Caricamento Immagini
            if 'obs' in demo_0:
                obs_group = demo_0['obs']
                
                # Strategia di ricerca chiave immagine
                image_keys = ['agentview_rgb', 'agentview_image', 'rgb', 'image', 'robot0_eye_in_hand_image']
                img_key = next((k for k in image_keys if k in obs_group), None)
                
                # Fallback ricerca generica
                if img_key is None:
                    img_key = next((k for k in obs_group.keys() if 'rgb' in k.lower() or 'image' in k.lower()), None)
                
                if img_key:
                    print(f"\n🖼️ Usando chiave immagini: '{img_key}'")
                    try:
                        imgs = load_images_robust(obs_group[img_key])
                        print(f"  ✅ Immagini caricate: {imgs.shape}")
                    except Exception as e:
                        print(f"  ❌ Errore immagini: {e}")
            
            # 2. Caricamento Azioni
            if 'actions' in demo_0:
                try:
                    actions = load_actions_robust(demo_0['actions'])
                    print(f"\n🎮 Azioni caricate: {actions.shape}")
                    print(f"  Range: [{actions.min():.3f}, {actions.max():.3f}]")
                except Exception as e:
                    print(f"  ❌ Errore azioni: {e}")

            # 3. VISUALIZZAZIONE (Senza Matplotlib)
            if imgs is not None and len(imgs) > 0:
                print("\n🎬 Visualizzazione frame esempio (PIL/IPython):")
                
                num_frames = min(4, len(imgs))
                indices = np.linspace(0, len(imgs) - 1, num_frames, dtype=int)
                
                for idx in indices:
                    img_array = imgs[idx]
                    
                    # Se l'immagine è float [0,1], converti a uint8
                    if img_array.dtype != np.uint8:
                         img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
                    
                    # Crea immagine PIL
                    pil_img = Image.fromarray(img_array)
                    
                    # (Opzionale) Resize per non occupare troppo spazio
                    # pil_img = pil_img.resize((128, 128))
                    
                    print(f"--- Frame {idx} ---")
                    display(pil_img)
            else:
                print("\n⚠️ Nessuna immagine valida da visualizzare")

    except Exception as e:
        print(f"Errore critico durante l'apertura del file: {e}")
    
    return hdf5_files

# Esegui
hdf5_files = explore_libero_dataset(Path('dataset/libero_spatial'))

# %%
class LIBERODataset(Dataset):
    """
    Dataset per dimostrazioni LIBERO
    
    Carica osservazioni visive e azioni da file HDF5.
    Supporta data augmentation e normalizzazione.
    Gestisce automaticamente problemi di dtype non standard nei file HDF5.
    
    Supporta demo-level split: invece di dividere i file, divide le demo
    all'interno di ciascun file (es. 80% train, 20% val per ogni file).
    """
    
    def __init__(
        self,
        hdf5_files: List[Path],
        sequence_length: int = 1,
        image_size: Tuple[int, int] = (128, 128),
        normalize_actions: bool = True,
        augmentation: bool = False,
        max_demos_per_task: Optional[int] = None,
        demo_split_ratio: float = 0.8,
        is_train: bool = True,
        action_stats: Optional[Dict] = None
    ):
        """
        Args:
            hdf5_files: lista di path ai file HDF5
            sequence_length: lunghezza delle sequenze (1 = single-step prediction)
            image_size: dimensioni delle immagini
            normalize_actions: se True, normalizza le azioni con z-score
            augmentation: se True, applica data augmentation
            max_demos_per_task: limite massimo di demo per task (per debugging)
            demo_split_ratio: percentuale di demo per training (default 0.8 = 80%)
            is_train: se True, usa le prime demo_split_ratio% demo; se False, usa il resto
            action_stats: statistiche azioni pre-calcolate (per validation set)
        """
        self.hdf5_files = hdf5_files
        self.sequence_length = sequence_length
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.augmentation = augmentation and is_train
        self.normalize_actions = normalize_actions
        self.demo_split_ratio = demo_split_ratio
        self.is_train = is_train
        
        # Carica tutti i dati in memoria (assumendo dataset gestibile)
        self.data = []
        self.action_stats = action_stats if action_stats is not None else {'mean': None, 'std': None}
        self.samples: List[Tuple[int, int]] = []  # (demo_idx, start_idx)
        
        split_name = "TRAIN" if is_train else "VAL"
        print(f"Loading {len(hdf5_files)} HDF5 files for {split_name} (demo split: {demo_split_ratio:.0%})...")
        all_actions = []
        
        for hdf5_file in hdf5_files:
            try:
                with h5py.File(hdf5_file, 'r') as f:
                    if 'data' not in f:
                        print(f"⚠️ 'data' key not found in {hdf5_file.name}, skipping...")
                        continue
                    
                    demo_keys = list(f['data'].keys())
                    
                    # Limita numero di demo se richiesto
                    if max_demos_per_task is not None:
                        demo_keys = demo_keys[:max_demos_per_task]
                    
                    # Demo-level split: seleziona subset di demo in base a is_train
                    n_demos = len(demo_keys)
                    n_train_demos = int(n_demos * demo_split_ratio)
                    
                    if is_train:
                        # Training: prime n_train_demos demo
                        selected_demo_keys = demo_keys[:n_train_demos]
                    else:
                        # Validation: demo rimanenti
                        selected_demo_keys = demo_keys[n_train_demos:]
                    
                    if len(selected_demo_keys) == 0:
                        print(f"⚠️ No demos selected from {hdf5_file.name} for {split_name}, skipping...")
                        continue
                    
                    task_prompt = self._prompt_from_filename(hdf5_file)

                    for demo_key in selected_demo_keys:
                        try:
                            demo = f[f'data/{demo_key}']
                            
                            # Trova la chiave delle immagini
                            obs_group = demo['obs']
                            img_key = self._find_image_key(obs_group)
                            
                            if img_key is None:
                                print(f"⚠️ No image key found in {hdf5_file.name}/{demo_key}, skipping...")
                                continue
                            
                            # Carica osservazioni con metodo robusto
                            obs = self._load_images_robust(obs_group[img_key])
                            
                            # Carica azioni con metodo robusto
                            actions = self._load_actions_robust(demo['actions'])
                            
                            # Verifica che obs e actions abbiano lunghezze compatibili
                            min_len = min(len(obs), len(actions))
                            if min_len < self.sequence_length:
                                print(f"⚠️ Demo too short ({min_len} < {self.sequence_length}), skipping...")
                                continue
                            
                            obs = obs[:min_len]
                            actions = actions[:min_len]
                            
                            # Aggiungi alla lista
                            self.data.append({
                                'observations': obs,
                                'actions': actions,
                                'prompt': task_prompt
                            })
                            
                            all_actions.append(actions)
                            
                        except Exception as e:
                            print(f"⚠️ Error loading demo {demo_key} from {hdf5_file.name}: {e}")
                            continue
                            
            except Exception as e:
                print(f"❌ Error opening file {hdf5_file}: {e}")
                continue
        
        print(f"✅ Loaded {len(self.data)} demonstrations for {split_name}")
        
        if len(self.data) == 0:
            raise ValueError(f"No valid demonstrations loaded for {split_name}! Check your data files.")
        
        # Calcola statistiche azioni per normalizzazione (solo per training set o se non fornite)
        if self.normalize_actions and len(all_actions) > 0 and action_stats is None:
            all_actions_concat = np.concatenate(all_actions, axis=0)
        
            mean = all_actions_concat.mean(axis=0).astype(np.float32)
            std  = all_actions_concat.std(axis=0).astype(np.float32)
        
            # ⚠️ Floor di sicurezza: evita std troppo piccole che esplodono la normalizzazione
            std_clipped = np.clip(std, 0.1, None)
        
            # Log dettagliato
            print(f"📊 Action statistics computed from {split_name} set:")
            print(f"   Mean:        {np.round(mean, 3)}")
            print(f"   Std (raw):   {np.round(std, 3)}")
            print(f"   Std (clipped to >=0.1): {np.round(std_clipped, 3)}")
        
            self.action_stats['mean'] = mean
            self.action_stats['std']  = std_clipped
        
        elif action_stats is not None:
            print(f"📊 Using provided action statistics")
            self.action_stats = {
                'mean': action_stats['mean'].astype(np.float32),
                'std':  np.clip(action_stats['std'], 0.1, None).astype(np.float32)
            }

        # Costruisci indice delle transizioni per accesso O(1)
        self.samples = self._build_sample_index()
        print(f"📦 Generated {len(self.samples)} transitions for {split_name}")

    @staticmethod
    def _prompt_from_filename(hdf5_file: Path) -> str:
        """Converte il nome del file HDF5 in un prompt naturale."""
        name = hdf5_file.stem
        if name.endswith('_demo'):
            name = name[:-5]
        name = name.replace('_', ' ').replace('-', ' ')
        return ' '.join(name.split()).strip()

    
    def _find_image_key(self, obs_group) -> Optional[str]:
        """Trova la chiave corretta per le immagini nel gruppo obs"""
        # Lista di possibili chiavi per le immagini (in ordine di priorità)
        possible_keys = [
            'agentview_rgb',
            'agentview_image', 
            'rgb',
            'image',
            'robot0_eye_in_hand_image',
            'frontview_image',
            'sideview_image'
        ]
        
        obs_keys = list(obs_group.keys())
        
        # Prima cerca chiavi note
        for key in possible_keys:
            if key in obs_keys:
                return key
        
        # Poi cerca qualsiasi chiave che contiene 'rgb' o 'image'
        for key in obs_keys:
            if 'rgb' in key.lower() or 'image' in key.lower():
                return key
        
        return None
    
    def _load_images_robust(self, dataset) -> np.ndarray:
        """
        Carica immagini da dataset HDF5 usando metodo robusto che bypassa problemi dtype.
        """
        shape = dataset.shape
        
        # METODO 1: Prova lettura diretta come uint8
        try:
            buffer = np.empty(shape, dtype=np.uint8)
            dataset.read_direct(buffer)
            return buffer
        except Exception:
            pass
        
        # METODO 2: Prova come float32 e converti
        try:
            buffer = np.empty(shape, dtype=np.float32)
            dataset.read_direct(buffer)
            if buffer.max() <= 1.0:
                buffer = (buffer * 255).astype(np.uint8)
            else:
                buffer = np.clip(buffer, 0, 255).astype(np.uint8)
            return buffer
        except Exception:
            pass
        
        # METODO 3: Prova come float64
        try:
            buffer = np.empty(shape, dtype=np.float64)
            dataset.read_direct(buffer)
            if buffer.max() <= 1.0:
                buffer = (buffer * 255).astype(np.uint8)
            else:
                buffer = np.clip(buffer, 0, 255).astype(np.uint8)
            return buffer
        except Exception:
            pass
        
        # METODO 4: Lettura raw
        try:
            buffer = np.empty(shape, dtype=np.uint8)
            dataset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, buffer)
            return buffer
        except Exception as e:
            raise RuntimeError(f"Cannot read image dataset: {e}")
    
    def _load_actions_robust(self, dataset) -> np.ndarray:
        """
        Carica azioni da dataset HDF5 usando metodo robusto.
        """
        shape = dataset.shape
        
        # METODO 1: Prova lettura diretta come float32
        try:
            buffer = np.empty(shape, dtype=np.float32)
            dataset.read_direct(buffer)
            return buffer
        except Exception:
            pass
        
        # METODO 2: Prova come float64 e converti
        try:
            buffer = np.empty(shape, dtype=np.float64)
            dataset.read_direct(buffer)
            return buffer.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Cannot read actions dataset: {e}")
    
    def _build_sample_index(self) -> List[Tuple[int, int]]:
        """Pre-calcola gli indici (demo_idx, start_idx) per ogni transizione"""
        indices: List[Tuple[int, int]] = []
        for demo_idx, demo in enumerate(self.data):
            demo_transitions = len(demo['observations']) - self.sequence_length + 1
            if demo_transitions <= 0:
                continue
            indices.extend((demo_idx, start) for start in range(demo_transitions))
        if not indices:
            raise ValueError("Dataset index is empty after preprocessing")
        return indices

    def __len__(self) -> int:
        """Numero totale di transizioni disponibili"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Restituisce una transizione (osservazione, azione).
        
        Args:
            idx: indice della transizione
            
        Returns:
            Dict con 'observations' e 'actions' come tensori
        """
        demo_idx, start_idx = self.samples[idx]
        demo = self.data[demo_idx]
        end_idx = start_idx + self.sequence_length

        obs = demo['observations'][start_idx:end_idx].copy()
        actions = demo['actions'][start_idx:end_idx].copy()

        # Preprocessing
        obs = self._preprocess_obs(obs)
        actions = self._preprocess_actions(actions)

        # Per single-step prediction, restituisci solo primo elemento
        if self.sequence_length == 1:
            obs = obs[0]
            actions = actions[0]

        # Converti HWC -> CHW per PyTorch
        if obs.ndim == 3:  # Single image: (H, W, C) -> (C, H, W)
            obs = np.transpose(obs, (2, 0, 1))
        elif obs.ndim == 4:  # Sequence: (T, H, W, C) -> (T, C, H, W)
            obs = np.transpose(obs, (0, 3, 1, 2))

        return {
            'observations': torch.from_numpy(obs).float(),
            'actions': torch.from_numpy(actions).float(),
            'prompt': demo.get('prompt', '')
        }
    
    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        """Preprocessing delle osservazioni"""
        processed = []
        target_h, target_w = self.image_size
        for img in obs:
            if img.shape[0] != target_h or img.shape[1] != target_w:
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            processed.append(img)
        obs = np.stack(processed, axis=0)

        # Normalizza [0, 255] -> [0, 1]
        obs = obs.astype(np.float32) / 255.0

        # Data augmentation (se abilitato)
        if self.augmentation:
            obs = self._augment_obs(obs)
        
        return obs
    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        """Data augmentation per osservazioni"""
        # Color jitter (brightness)
        if np.random.rand() < 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            obs = np.clip(obs * brightness, 0, 1)
        
        # Contrast adjustment
        if np.random.rand() < 0.3:
            contrast = np.random.uniform(0.8, 1.2)
            mean = obs.mean(axis=(1, 2), keepdims=True)
            obs = np.clip((obs - mean) * contrast + mean, 0, 1)
        
        # Random crop (con padding)
        if np.random.rand() < 0.3:
            crop_ratio = np.random.uniform(0.85, 0.95)
            crop_size_h = int(self.image_size[0] * crop_ratio)
            crop_size_w = int(self.image_size[1] * crop_ratio)
            
            start_y = np.random.randint(0, self.image_size[0] - crop_size_h + 1)
            start_x = np.random.randint(0, self.image_size[1] - crop_size_w + 1)
            
            cropped = []
            for img in obs:
                img_crop = img[start_y:start_y+crop_size_h, start_x:start_x+crop_size_w]
                img_resized = cv2.resize(img_crop, (self.image_size[1], self.image_size[0]))
                cropped.append(img_resized)
            obs = np.stack(cropped)
        
        return obs
    
    def _preprocess_actions(self, actions: np.ndarray) -> np.ndarray:
        """Preprocessing delle azioni con normalizzazione z-score"""
        actions = actions.astype(np.float32)
        if self.action_stats['mean'] is not None:
            actions = (actions - self.action_stats['mean']) / self.action_stats['std']
        
        return actions
    
    def get_action_stats(self) -> Dict[str, np.ndarray]:
        """Restituisce le statistiche delle azioni per denormalizzazione"""
        return self.action_stats.copy()
    
    def denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Denormalizza le azioni per l'esecuzione nel simulatore"""
        if self.action_stats['mean'] is not None:
            return actions * self.action_stats['std'] + self.action_stats['mean']
        return actions

# %%
class PretrainedVisualEncoder(nn.Module):
    """Visual encoder basato su ResNet18 con testa adattiva."""

    def __init__(self, hidden_dim: int = 256, freeze_backbone: bool = True, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim

        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.adapter = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim)
        )
        self.ln = nn.LayerNorm(hidden_dim)

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()

        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        x = (x - self.mean) / self.std
        features = self.backbone(x).flatten(start_dim=1)
        output = self.adapter(features)
        return self.ln(output)


class VisualEncoder(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int] = (3, 128, 128), hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        c, _, _ = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.head(x)


class PromptEncoder(nn.Module):
    """Encodes natural-language task prompts via CLIP ViT-L/14 text tower."""

    def __init__(
        self,
        hidden_dim: int,
        model_name: str = 'openai/clip-vit-large-patch14',
        trainable: bool = False,
        dropout: float = 0.1,
        max_length: int = 77
    ):
        super().__init__()

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        self.max_length = min(max_length, getattr(self.text_model.config, 'max_position_embeddings', max_length))

        if not trainable:
            self.text_model.eval()
            for param in self.text_model.parameters():
                param.requires_grad = False

        text_hidden = self.text_model.config.hidden_size
        self.adapter = nn.Sequential(
            nn.LayerNorm(text_hidden),
            nn.Linear(text_hidden, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self._token_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    def _tokenize(self, prompt: str) -> Dict[str, torch.Tensor]:
        if prompt not in self._token_cache:
            tokens = self.tokenizer(
                prompt,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            )
            self._token_cache[prompt] = {k: v for k, v in tokens.items()}
        cached = self._token_cache[prompt]
        return {k: v.clone() for k, v in cached.items()}

    def forward(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        if not prompts:
            raise ValueError("PromptEncoder received an empty batch of prompts")

        token_batches = [self._tokenize(p) for p in prompts]
        batch = {
            key: torch.cat([tokens[key] for tokens in token_batches], dim=0).to(device)
            for key in token_batches[0]
        }

        outputs = self.text_model(**batch)
        pooled = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, -1, :]
        return self.adapter(pooled)


class RecursiveBlock(nn.Module):
    """Slot-based TRM block with cross- then self-attention updates."""

    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.1):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.slot_norm = nn.LayerNorm(hidden_dim)
        self.cond_norm = nn.LayerNorm(hidden_dim)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, slots: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: tensor (B, S, D) con gli slot correnti.
            cond_tokens: tensor (B, T, D) con i token di condizionamento.
        Returns:
            slots aggiornati (B, S, D).
        """
        bsz, num_slots, dim = slots.shape

        # Cross-attention: gli slot interrogano i token (visivi/testuali).
        slots_norm = self.slot_norm(slots)
        cond_norm = self.cond_norm(cond_tokens)
        cross_out, _ = self.cross_attn(slots_norm, cond_norm, cond_norm)
        gru_input = cross_out.reshape(-1, dim)
        prev_slots = slots.reshape(-1, dim)
        slots = self.gru(gru_input, prev_slots).view(bsz, num_slots, dim)

        # Self-attention tra slot per coordinare le ipotesi.
        attn_in = self.self_attn_norm(slots)
        self_out, _ = self.self_attn(attn_in, attn_in, attn_in)
        slots = slots + self.dropout(self_out)

        # MLP residua per raffinamento.
        mlp_out = self.mlp(self.mlp_norm(slots))
        slots = slots + self.dropout(mlp_out)
        return slots


class TRMPolicy(nn.Module):
    """Policy TRM con slot-attention e fusione vision-language compatibile con la letteratura."""

    def __init__(
        self,
        obs_shape=(3, 128, 128),
        action_dim=7,
        hidden_dim=256,
        num_heads=4,
        num_recursions=8,
        num_slots=4,
        dropout=0.1,
        adaptive_halt=False,
        use_pretrained_encoder=True,
        freeze_backbone=True,
        encoder_dropout=0.1,
        use_text_prompts=True,
        text_encoder_name='openai/clip-vit-large-patch14',
        train_text_encoder=False,
        text_dropout=0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_recursions = num_recursions
        self.num_slots = num_slots
        self.adaptive_halt = adaptive_halt
        self.use_text_prompts = use_text_prompts

        if use_pretrained_encoder:
            self.encoder = PretrainedVisualEncoder(
                hidden_dim=hidden_dim,
                freeze_backbone=freeze_backbone,
                dropout=encoder_dropout
            )
        else:
            self.encoder = VisualEncoder(
                obs_shape=obs_shape,
                hidden_dim=hidden_dim,
                dropout=encoder_dropout
            )

        if self.use_text_prompts:
            self.prompt_encoder = PromptEncoder(
                hidden_dim=hidden_dim,
                model_name=text_encoder_name,
                trainable=train_text_encoder,
                dropout=text_dropout
            )
            self.text_token_adapter = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.prompt_encoder = None
            self.text_token_adapter = None

        self.vision_token_adapter = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.slot_init = nn.Parameter(torch.randn(num_slots, hidden_dim))
        self.slot_conditioning = nn.Linear(hidden_dim, hidden_dim)
        self.slot_readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.recursive_block = RecursiveBlock(hidden_dim, num_heads, dropout)

        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )

        if adaptive_halt:
            self.halt_predictor = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

    def forward(self, obs, prompts: Optional[List[str]] = None, return_all_states: bool = False):
        B = obs.shape[0]
        device = obs.device

        vision_token = self.vision_token_adapter(self.encoder(obs)).unsqueeze(1)
        cond_tokens = [vision_token]

        if self.use_text_prompts:
            if prompts is None:
                raise ValueError("TRMPolicy richiede i prompt testuali quando use_text_prompts=True")
            text_features = self.prompt_encoder(prompts, device=device)
            text_token = self.text_token_adapter(text_features).unsqueeze(1)
            cond_tokens.append(text_token)

        cond_tokens = torch.cat(cond_tokens, dim=1)
        cond_summary = cond_tokens.mean(dim=1)
        slots = self._init_slots(B, device, cond_summary)

        state_trace: Optional[List[torch.Tensor]] = [] if return_all_states else None
        if return_all_states and state_trace is not None:
            state_trace.append(self.slot_readout(slots.mean(dim=1)))

        if self.adaptive_halt:
            pooled, halt_info, adaptive_states = self._forward_adaptive(
                slots,
                cond_tokens,
                B,
                track_states=return_all_states
            )
            if return_all_states and adaptive_states and state_trace is not None:
                state_trace.extend(self.slot_readout(state) for state in adaptive_states)
        else:
            for _ in range(self.num_recursions):
                slots = self.recursive_block(slots, cond_tokens)
                if return_all_states and state_trace is not None:
                    state_trace.append(self.slot_readout(slots.mean(dim=1)))
            pooled = slots.mean(dim=1)

        pooled = self.slot_readout(pooled)
        actions = self.action_head(pooled)

        if return_all_states and state_trace is not None:
            return actions, state_trace
        return actions

    def _forward_adaptive(self, slots, cond_tokens, batch_size, track_states: bool = False):
        """Adaptive Computation Time applicato agli slot."""
        halt_probs = []
        remainders = torch.ones(batch_size, device=slots.device)
        n_updates = torch.zeros(batch_size, device=slots.device)
        accumulated = torch.zeros(batch_size, slots.size(-1), device=slots.device)
        state_trace: Optional[List[torch.Tensor]] = [] if track_states else None

        for _ in range(self.num_recursions):
            slots = self.recursive_block(slots, cond_tokens)
            pooled = slots.mean(dim=1)

            if track_states and state_trace is not None:
                state_trace.append(pooled.detach().clone())

            halt_p = self.halt_predictor(pooled).squeeze(-1)
            halt_probs.append(halt_p)

            still_running = (remainders > 0.01).float()
            accumulated += remainders.unsqueeze(-1) * pooled * still_running.unsqueeze(-1)
            remainders = remainders * (1 - halt_p) * still_running
            n_updates += still_running

            if remainders.max() < 0.01:
                break

        halt_info = {
            'halt_probs': halt_probs,
            'n_updates': n_updates
        }

        if track_states:
            return accumulated, halt_info, state_trace or []
        return accumulated, halt_info, None

    def _init_slots(self, batch_size: int, device: torch.device, cond_summary: torch.Tensor) -> torch.Tensor:
        base = self.slot_init.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        cond_bias = self.slot_conditioning(cond_summary).unsqueeze(1)
        return base + cond_bias


def build_policy_from_config(config: TrainingConfig, obs_shape: Tuple[int, int, int] = (3, 128, 128)) -> TRMPolicy:
    """Costruisce una TRMPolicy coerente con il TrainingConfig."""

    return TRMPolicy(
        obs_shape=obs_shape,
        action_dim=7,
        hidden_dim=config.hidden_dim,
        num_recursions=config.num_recursions,
        num_slots=config.num_slots,
        dropout=config.dropout,
        use_pretrained_encoder=config.use_pretrained_encoder,
        freeze_backbone=config.freeze_backbone,
        encoder_dropout=config.encoder_dropout,
        use_text_prompts=config.use_text_prompts,
        text_encoder_name=config.text_encoder_name,
        train_text_encoder=config.train_text_encoder,
        text_dropout=config.text_dropout
    )

print("ok")

# %%
class BehaviorCloningTrainer:
    """Trainer per Behavior Cloning con configurazione strutturata."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: torch.device,
        use_wandb: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        self.steps_per_epoch = max(len(train_loader), 1)

        # Optimizer e scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        self.scheduler = None
        if config.sched_T0:
            # Cosine warm restarts operanti a livello di iterazioni
            period_iters = max(1, config.sched_T0 * self.steps_per_epoch)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=period_iters,
                T_mult=config.sched_T_mult,
                eta_min=config.lr_min
            )
        else:
            warmup_iters = max(1, config.warmup_epochs * self.steps_per_epoch)
            total_iters = max(1, config.epochs * self.steps_per_epoch - warmup_iters)

            for pg in self.optimizer.param_groups:
                pg['lr'] = config.lr

            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=warmup_iters
            )

            constant_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0,
                total_iters=total_iters
            )

            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, constant_scheduler],
                milestones=[warmup_iters]
            )

        self.use_amp = (self.device.type == 'cuda')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.grad_clip = config.grad_clip
        self.early_stop_patience = config.early_stop_patience
        self._epochs_no_improve = 0

        self.best_val_loss = float('inf')
        self.best_model_path = config.save_path
    
    def train(self):
        """Training loop completo"""
        
        for epoch in range(self.config.epochs):
            # Training
            train_metrics = self._train_epoch(epoch)
            
            # Validation
            val_metrics = self._validate_epoch(epoch)
            print(f"Epoch {epoch}, training loss: {train_metrics['loss']}, validation loss: {val_metrics['loss']}")
            # Logging exaustive
            self._log_metrics(epoch, train_metrics, val_metrics)

            # print(f"Epoch: {epoch}\n\t Training: {train_metrics}\n\t Validation: {val_metrics}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'config': self.config.to_dict()
                }, self.best_model_path)
                print(f"  ✓ Saved best model (val_loss: {val_metrics['loss']:.4f})")
                self._epochs_no_improve = 0
            else:
                self._epochs_no_improve += 1

            if self.early_stop_patience and self._epochs_no_improve >= self.early_stop_patience:
                print("⏹️  Early stopping triggered")
                break
        
        print(f"\n✅ Training completed! Best val loss: {self.best_val_loss:.4f}")
        return self.best_val_loss
    
    def _train_epoch(self, epoch):
        """Training per una epoch"""
        self.model.train()
        
        total_loss = 0
        action_mse = 0
        action_l1 = 0
        
        for step, batch in enumerate(self.train_loader):
            obs = batch['observations'].to(self.device, non_blocking=True)
            target_actions = batch['actions'].to(self.device, non_blocking=True)
            prompts = batch.get('prompt')

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred_actions = self.model(obs, prompts=prompts)
                mse = F.mse_loss(pred_actions, target_actions)
                l1 = F.l1_loss(pred_actions, target_actions)
                loss = 0.7 * mse + 0.3 * l1

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            action_mse += mse.item()
            action_l1 += l1.item()

            if self.scheduler is not None:
                self.scheduler.step()

        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'action_mse': action_mse / n_batches,
            'action_l1': action_l1 / n_batches
        }
    
    def _validate_epoch(self, epoch):
        """Validation per una epoch"""
        self.model.eval()
        
        total_loss = 0
        action_mse = 0
        action_l1 = 0
        
        with torch.no_grad():
            
            for batch in self.val_loader:
                obs = batch['observations'].to(self.device)
                target_actions = batch['actions'].to(self.device)
                prompts = batch.get('prompt')
                
                # Forward pass
                pred_actions = self.model(obs, prompts=prompts)
                
                # Loss
                mse = F.mse_loss(pred_actions, target_actions)
                l1 = F.l1_loss(pred_actions, target_actions)
                loss = 0.7 * mse + 0.3 * l1
                
                total_loss += loss.item()
                action_mse += mse.item()
                action_l1 += l1.item()
                
        
        n_batches = len(self.val_loader)
        return {
            'loss': total_loss / n_batches,
            'action_mse': action_mse / n_batches,
            'action_l1': action_l1 / n_batches
        }
    
    def _log_metrics(self, epoch, train_metrics, val_metrics):
        """Log delle metriche"""
        lr = self.optimizer.param_groups[0]['lr']
        
        # WandB logging
        if self.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/action_mse': train_metrics['action_mse'],
                'train/action_l1': train_metrics.get('action_l1'),
                'val/loss': val_metrics['loss'],
                'val/action_mse': val_metrics['action_mse'],
                'val/action_l1': val_metrics.get('action_l1'),
                'lr': lr
            })

def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    loader_kwargs: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    """Crea data loader consistenti a partire da un template di kwargs."""

    kwargs = loader_kwargs.copy()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs
    )

    return train_loader, val_loader


def _set_dataset_augmentation(dataset, flag: bool):
    """Imposta augmentation temporaneamente e restituisce funzione di restore."""

    if not hasattr(dataset, 'augmentation'):
        return lambda: None

    original = dataset.augmentation
    dataset.augmentation = flag

    def restore():
        dataset.augmentation = original

    return restore


def optuna_random_search(
    train_dataset: Dataset,
    val_dataset: Dataset,
    loader_kwargs: Dict[str, Any],
    device: torch.device,
    quick_epochs: int = 10,
    search_space: Optional[HyperparameterSearchSpace] = None,
    n_trials: int = 50
) -> Tuple[TrainingConfig, List[Dict[str, Any]]]:
    """
    Randomized Optuna search (TPE-based) instead of grid-search.
    Runs a fixed number of trials (n_trials).
    """

    search_space = search_space or default_search_space()

    # --- NEW: TPE Sampler (recommended over random sampling) ---
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        direction='minimize',
        sampler=sampler
    )

    print(f"\n🔍 Starting Optuna random search")
    print(f"  Number of trials: {n_trials}")
    print(f"  Quick epochs: {quick_epochs}")

    def objective(trial: optuna.Trial) -> float:

        # --- Suggest values from the search space ---
        trial_config = TrainingConfig(
            lr=trial.suggest_categorical('lr', search_space.lr),
            hidden_dim=trial.suggest_categorical('hidden_dim', search_space.hidden_dim),
            num_recursions=trial.suggest_categorical('num_recursions', search_space.num_recursions),
            epochs=quick_epochs,
            batch_size=trial.suggest_categorical('batch_size', search_space.batch_size),
            weight_decay=trial.suggest_categorical('weight_decay', search_space.weight_decay),
            use_pretrained_encoder=trial.suggest_categorical('pretrained_encoder', search_space.pretrained_encoder),
            freeze_backbone=trial.suggest_categorical('freeze_backbone', search_space.freeze_backbone),
            augmentation=trial.suggest_categorical('augmentation', search_space.augmentation),
            dropout=trial.suggest_categorical('dropout', search_space.dropout),
            encoder_dropout=trial.suggest_categorical('dropout', search_space.dropout),
            text_encoder_name=trial.suggest_categorical('text_encoder_name', search_space.text_encoder_name),
            train_text_encoder=trial.suggest_categorical('train_text_encoder', search_space.train_text_encoder),
            text_dropout=trial.suggest_categorical('text_dropout', search_space.text_dropout),
            num_slots=trial.suggest_categorical('num_slots', search_space.num_slots),
            grad_clip=1.0,
            save_path=f"optuna_trial_{trial.number}.pt"
        )

        print("-" * 80)
        print(f"[Optuna] Trial {trial.number + 1}/{n_trials}")
        print(json.dumps(trial_config.to_dict(), indent=2))
        sys.stdout.flush()

        # Build dataloaders based on this trial's batch size
        train_loader, val_loader = build_dataloaders(
            train_dataset,
            val_dataset,
            trial_config.batch_size,
            loader_kwargs
        )

        # Build model
        model = build_policy_from_config(trial_config)
        trainer = BehaviorCloningTrainer(
            model,
            train_loader,
            val_loader,
            trial_config,
            device,
            use_wandb=False
        )

        # Turn augmentation on/off safely
        restore_aug = _set_dataset_augmentation(train_dataset, trial_config.augmentation)

        try:
            val_loss = trainer.train()
        finally:
            restore_aug()

        # Store config inside trial for later retrieval
        trial.set_user_attr('config', trial_config.to_dict())

        return val_loss

    # --- NEW: Run fixed-size random search ---
    study.optimize(objective, n_trials=n_trials)

    # Save trial history (full information)
    history = []
    for trial in study.trials:
        history.append({
            'trial_number': trial.number,
            'state': str(trial.state),
            'value': trial.value,
            'params': trial.params,                # all parameters sampled
            'user_attrs': trial.user_attrs,        # user metadata (our config)
            'distributions': {
                k: str(v) for k, v in trial.distributions.items()
            },                                      # searchable distributions
            'system_attrs': trial.system_attrs,    # optuna internal info
            'datetime_start': str(trial.datetime_start),
            'datetime_complete': str(trial.datetime_complete),
        })


    # Best configuration
    best_config = TrainingConfig(**study.best_trial.user_attrs['config'])

    return best_config, history

# %%
def train_final_model(
    base_config: TrainingConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    loader_kwargs: Dict[str, Any],
    device,
    final_epochs: Optional[int] = None,
    use_wandb: bool = False
) -> Tuple[nn.Module, float]:
    """Esegue il training finale a partire dalla configurazione scelta."""

    final_epochs = final_epochs or base_config.epochs
    final_config = replace(
        base_config,
        epochs=final_epochs,
        save_path='final_model.pt',
        early_stop_patience=base_config.early_stop_patience or 10,
        sched_T0=base_config.sched_T0 or max(1, final_epochs // 5)
    )

    print(f"\n{'='*60}")
    print("🎯 FINAL TRAINING")
    print(f"Config: {final_config.label()}")
    print(f"{'='*60}")

    if use_wandb:
        wandb.init(
            project="trm-robotics",
            name=f"final_{final_config.label()}",
            config=final_config.to_dict()
        )

    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        final_config.batch_size,
        loader_kwargs
    )

    model = build_policy_from_config(final_config)
    trainer = BehaviorCloningTrainer(
        model,
        train_loader,
        val_loader,
        final_config,
        device,
        use_wandb=use_wandb
    )

    restore_aug = _set_dataset_augmentation(train_dataset, final_config.augmentation)
    try:
        final_val_loss = trainer.train()
    finally:
        restore_aug()

    if os.path.exists(final_config.save_path):
        checkpoint = torch.load(final_config.save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    if use_wandb:
        wandb.finish()

    print(f"\n✅ Final model trained! Val loss: {final_val_loss:.4f}")

    return model, final_val_loss

class PolicyEvaluator:
    """
    Valutatore per policy robotiche in simulazione LIBERO
    """
    
    def __init__(
        self,
        policy: nn.Module,
        device: torch.device,
        action_stats: Dict = None
    ):
        self.policy = policy.to(device)
        self.policy.eval()
        self.device = device
        self.action_stats = action_stats
    
    def evaluate_on_task(
        self,
        env,
        init_states,
        num_episodes=50,
        max_steps=500,
        record_video=False,
        video_path=None,
        task_prompt: Optional[str] = None
    ):
        """
        Valuta policy su un singolo task
        
        Args:
            env: environment LIBERO
            init_states: stati iniziali per il task
            num_episodes: numero di episodi da valutare
            max_steps: massimo numero di step per episodio
            record_video: se True, registra video
            video_path: path per salvare video
        
        Returns:
            results: dizionario con metriche
        """
        expect_prompt = getattr(self.policy, 'use_text_prompts', False)
        if expect_prompt and not task_prompt:
            raise ValueError("PolicyEvaluator richiede un task_prompt per policy testo-condizionate")

        successes = []
        episode_lengths = []
        frames_buffer = [] if record_video else None
        
        for ep in range(num_episodes):
            # Reset environment
            env.reset()
            env.set_init_state(init_states[ep % len(init_states)])
            
            obs = env.get_observation()
            done = False
            step = 0
            
            episode_frames = [] if record_video and ep < 5 else None
            
            while not done and step < max_steps:
                # Cattura frame
                if episode_frames is not None:
                    frame = env.render(mode='rgb_array')
                    episode_frames.append(frame)
                
                # Preprocessing osservazione
                obs_tensor = self._preprocess_obs(obs)
                
                # Predici azione
                with torch.no_grad():
                    prompt_batch = [task_prompt] if task_prompt else None
                    action = self.policy(obs_tensor, prompts=prompt_batch)
                    action = action.cpu().numpy().squeeze()
                    
                    # Denormalizza azione se necessario
                    if self.action_stats is not None:
                        action = action * self.action_stats['std'] + self.action_stats['mean']
                
                # Step environment
                obs, reward, done, info = env.step(action)
                step += 1
            
            # Registra risultati
            success = info.get('success', False)
            successes.append(success)
            episode_lengths.append(step)
            
            if episode_frames is not None:
                frames_buffer.append(episode_frames)
        
        # Salva video se richiesto
        if record_video and frames_buffer and video_path:
            self._save_videos(frames_buffer, video_path)
        
        # Calcola metriche
        results = {
            'success_rate': np.mean(successes),
            'avg_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            'num_episodes': num_episodes
        }
        
        return results
    
    def _preprocess_obs(self, obs):
        """Preprocessing osservazione per policy"""
        # Estrai immagine RGB
        img = obs['agentview_rgb']
        
        # Normalizza
        img = img.astype(np.float32) / 255.0
        
        # Converti a tensor e aggiungi batch dimension
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
        return img_tensor
    
    def _save_videos(self, frames_buffer, video_path):
        """Salva video degli episodi"""
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        for i, frames in enumerate(frames_buffer):
            path = video_path.replace('.mp4', f'_ep{i}.mp4')
            
            # Usa OpenCV per salvare video
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(path, fourcc, 30.0, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"  Video salvato: {path}")


def generate_evaluation_report(results_dict, output_path):
    """
    Genera report HTML con risultati di valutazione
    
    Args:
        results_dict: dizionario {task_name: results}
        output_path: path per salvare report
    """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>TRM Robotics - Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .success {{ color: green; font-weight: bold; }}
            .fail {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>🤖 TRM Robotics Evaluation Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Results Summary</h2>
        <table>
            <tr>
                <th>Task Name</th>
                <th>Success Rate</th>
                <th>Avg Episode Length</th>
                <th>Std Episode Length</th>
                <th>Num Episodes</th>
            </tr>
    """
    
    # Aggiungi righe per ogni task
    overall_success = []
    
    for task_name, results in results_dict.items():
        success_rate = results['success_rate']
        overall_success.append(success_rate)
        
        success_class = 'success' if success_rate >= 0.5 else 'fail'
        
        html_content += f"""
            <tr>
                <td>{task_name}</td>
                <td class="{success_class}">{success_rate:.2%}</td>
                <td>{results['avg_episode_length']:.1f}</td>
                <td>{results['std_episode_length']:.1f}</td>
                <td>{results['num_episodes']}</td>
            </tr>
        """
    
    # Overall statistics
    mean_success = np.mean(overall_success)
    success_class = 'success' if mean_success >= 0.5 else 'fail'
    
    html_content += f"""
            <tr style="font-weight: bold; background-color: #e0e0e0;">
                <td>OVERALL</td>
                <td class="{success_class}">{mean_success:.2%}</td>
                <td colspan="3">-</td>
            </tr>
        </table>
    </body>
    </html>
    """
    
    # Salva report
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"\n✓ Report salvato: {output_path}")

# %%
def main_pipeline(
    data_path: str = 'dataset/libero_spatial',
    work_dir: str = 'trm_robotics',
    quick_search: bool = True,
    train_final: bool = True,
    use_custom_final_config: bool = False,
    evaluate: bool = True,
    use_wandb: bool = False
):
    """
    Pipeline completa per il progetto TRM Robotics
    
    Args:
        data_path: path ai dati LIBERO
        work_dir: directory di lavoro
        quick_search: se True, esegue hyperparameter search
        train_final: se True, esegue training finale
        evaluate: se True, esegue valutazione in simulazione
        use_wandb: se True, usa WandB per logging
    """
    
    print(f"""
    {'='*80}
    🤖 TinyRecursiveModels per Controllo Robotico
    {'='*80}
    
    Obiettivi:
    1. Adattare architettura TRM per robotica
    2. Training con Behavior Cloning su LIBERO
    3. Valutazione in simulazione con metriche quantitative e qualitative
    
    """)
    
    # Setup directories
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}\n")
    
    # ========== STEP 1: Caricamento Dataset ==========
    print(f"\n{'='*80}")
    print("STEP 1: Caricamento Dataset")
    print(f"{'='*80}")
    
    # Trova file HDF5
    data_path = Path(data_path)
    hdf5_files = list(data_path.glob('**/*.hdf5'))
    
    if not hdf5_files:
        print(f"❌ Nessun file HDF5 trovato in {data_path}")
        print("Assicurati di aver scaricato il dataset LIBERO!")
        return
    
    print(f"✓ Trovati {len(hdf5_files)} file HDF5 (task)")
    
    # Demo-level split: usa TUTTI i file per entrambi i dataset
    # ma dividi le demo all'interno di ciascun file (80% train, 20% val)
    demo_split_ratio = 0.8
    print(f"\n📊 Demo-level split: {demo_split_ratio:.0%} train / {1-demo_split_ratio:.0%} val per ogni task")
    print(f"   Tutti i {len(hdf5_files)} task presenti in entrambi train e val")
    
    # Crea datasets con demo-level split
    print("\nCreating TRAIN dataset...")
    train_dataset = LIBERODataset(
        hdf5_files,  # Usa TUTTI i file
        sequence_length=1,
        image_size=(128, 128),
        augmentation=False,
        max_demos_per_task=50,  # Limita per velocità
        demo_split_ratio=demo_split_ratio,
        is_train=True  # Prime 80% demo per task
    )
    
    # Usa le stesse statistiche del training set per la validation
    train_action_stats = train_dataset.action_stats
    
    print("\nCreating VAL dataset...")
    val_dataset = LIBERODataset(
        hdf5_files,  # Usa TUTTI i file
        sequence_length=1,
        image_size=(128, 128),
        augmentation=False,
        max_demos_per_task=50,  # Stesso limite per coerenza
        demo_split_ratio=demo_split_ratio,
        is_train=False,  # Ultime 20% demo per task
        action_stats=train_action_stats  # Usa statistiche del training
    )
    
    # Data loader template (verranno istanziati on-demand)
    num_workers = min(4, os.cpu_count() or 1)
    use_cuda = torch.cuda.is_available()

    loader_common = {
        'num_workers': num_workers,
        'pin_memory': use_cuda,
        'persistent_workers': num_workers > 0
    }
    if num_workers > 0:
        loader_common['prefetch_factor'] = 2

    print(f"\n✓ Dataset creati con demo-level split")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Salva action stats per denormalizzazione
    action_stats = train_dataset.action_stats
    with open('action_stats.json', 'w') as f:
        json.dump({
            'mean': action_stats['mean'].tolist(),
            'std': action_stats['std'].tolist()
        }, f)
    
    # ========== STEP 2: Hyperparameter Search ==========
    best_config = None
    
    quick_search = False
    if quick_search:
        print(f"\n{'='*80}")
        print("STEP 2: Hyperparameter Search")
        print(f"{'='*80}")
        
        best_config, all_results = optuna_random_search(
            train_dataset,
            val_dataset,
            loader_common,
            device,
            quick_epochs=10
        )
        
        # Salva risultati
        with open('hyperparam_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        with open('best_hyperparams.json', 'w') as f:
            json.dump(best_config.to_dict(), f, indent=2)
    
    # ========== STEP 3: Training Finale ==========
    trained_model = None
    
    if train_final:
        print(f"\n{'='*80}")
        print("STEP 3: Training Finale")
        print(f"{'='*80}")
        
        if best_config is None or use_custom_final_config:
            best_config = TrainingConfig(
                lr=1e-4,
                hidden_dim=128,
                num_recursions=12,
                batch_size=256,
                epochs=20,
                weight_decay=1.0,
                dropout=0.3,
                encoder_dropout=0.1,
                use_pretrained_encoder=True,
                freeze_backbone=False,
                augmentation=True,
                text_encoder_name='openai/clip-vit-large-patch14',
                train_text_encoder = False,
                text_dropout = 0.1,
                use_text_prompts = True,
                num_slots = 4
            )
            print("⚠️  Usando configurazione custom di default")

        trained_model, final_val_loss = train_final_model(
            best_config,
            train_dataset,
            val_dataset,
            loader_common,
            device,
            final_epochs=100,
            use_wandb=use_wandb
        )
    
    # ========== STEP 4: Valutazione ==========
    if evaluate:
        print(f"\n{'='*80}")
        print("STEP 4: Valutazione in Simulazione")
        print(f"{'='*80}")
        
        # Carica modello se non già trainato
        if trained_model is None:
            print("Caricando modello salvato...")
            checkpoint = torch.load('final_model.pt')
            
            if 'config' in checkpoint:
                config_obj = TrainingConfig(**checkpoint['config'])
                trained_model = build_policy_from_config(config_obj)
            else:
                trained_model = TRMPolicy(
                    obs_shape=(3, 128, 128),
                    action_dim=7,
                    hidden_dim=256,
                    num_recursions=8,
                    num_slots=4
                )

            trained_model.load_state_dict(checkpoint['model_state_dict'])
            trained_model = trained_model.to(device)
        
        # Crea evaluator
        evaluator = PolicyEvaluator(trained_model, device, action_stats)
        
        # Nota: La valutazione in simulazione richiede setup di LIBERO environment
        # Questo è un placeholder per la struttura
        print("\n⚠️  La valutazione in simulazione richiede setup completo di LIBERO")
        print("Vedi sezione 'Valutazione in Simulazione' per implementazione completa")
        
        # Pseudocodice per valutazione:
        """
        from libero.libero import benchmark
        
        results_dict = {}
        
        for task_id in range(num_tasks):
            env, init_states = create_eval_env(task_suite, task_id)
            
            results = evaluator.evaluate_on_task(
                env,
                init_states,
                num_episodes=50,
                record_video=True,
                video_path=f'videos/task_{task_id}.mp4'
            )
            
            results_dict[f'task_{task_id}'] = results
            env.close()
        
        # Genera report
        generate_evaluation_report(results_dict, 'evaluation_report.html')
        """
    
    print(f"\n{'='*80}")
    print("✅ Pipeline completata!")
    print(f"{'='*80}")
    print(f"\nFile generati in: {work_dir}")
    print("  - final_model.pt: modello allenato")
    print("  - action_stats.json: statistiche azioni")
    print("  - hyperparam_results.json: risultati hyperparameter search")

# %%


# %%
"""Esegue Optuna HPO (incluso il text-encoder) e avvia il training finale."""
import torch

data_path = 'dataset/libero_spatial'
work_dir = 'trm_robotics'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Using device: {device}\n")

# ========== STEP 1: Caricamento Dataset ==========
print(f"\n{'='*80}")
print("STEP 1: Caricamento Dataset")
print(f"{'='*80}")

data_path = Path(data_path)
hdf5_files = list(data_path.glob('**/*.hdf5'))

if not hdf5_files:
    raise RuntimeError(f"❌ Nessun file HDF5 trovato in {data_path}. Scarica prima il dataset LIBERO.")

print(f"✓ Trovati {len(hdf5_files)} file HDF5 (task)")

demo_split_ratio = 0.8
print(f"\n📊 Demo-level split: {demo_split_ratio:.0%} train / {1-demo_split_ratio:.0%} val per ogni task")
print(f"   Tutti i {len(hdf5_files)} task presenti in entrambi train e val")

print("\nCreating TRAIN dataset...")
train_dataset = LIBERODataset(
    hdf5_files,
    sequence_length=1,
    image_size=(128, 128),
    augmentation=False,
    max_demos_per_task=50,
    demo_split_ratio=demo_split_ratio,
    is_train=True
)

train_action_stats = train_dataset.action_stats

print("\nCreating VAL dataset...")
val_dataset = LIBERODataset(
    hdf5_files,
    sequence_length=1,
    image_size=(128, 128),
    augmentation=False,
    max_demos_per_task=50,
    demo_split_ratio=demo_split_ratio,
    is_train=False,
    action_stats=train_action_stats
)

num_workers = min(4, os.cpu_count() or 1)
use_cuda = torch.cuda.is_available()

loader_common = {
    'num_workers': num_workers,
    'pin_memory': use_cuda,
    'persistent_workers': num_workers > 0
}
if num_workers > 0:
    loader_common['prefetch_factor'] = 2

print(f"\n✓ Dataset creati con demo-level split")
print(f"  Train samples: {len(train_dataset)}")
print(f"  Val samples: {len(val_dataset)}")

action_stats = train_dataset.action_stats
with open('action_stats.json', 'w') as f:
    json.dump({
        'mean': action_stats['mean'].tolist(),
        'std': action_stats['std'].tolist()
    }, f)

# ========== STEP 2: Hyperparameter Search con Optuna ==========
search_trials = 20
quick_epochs = 5
print(f"\n{'='*80}")
print(f"STEP 2: Optuna search ({search_trials} trials, {quick_epochs} quick epochs)")
print(f"{'='*80}")

best_config, search_history = optuna_random_search(
    train_dataset,
    val_dataset,
    loader_common,
    device,
    quick_epochs=quick_epochs,
    search_space=default_search_space(),
    n_trials=search_trials
)

with open('hyperparam_results.json', 'w') as f:
    json.dump(search_history, f, indent=2)

with open('best_hyperparams.json', 'w') as f:
    json.dump(best_config.to_dict(), f, indent=2)

print("\n✅ Optuna search completata. Miglior configurazione:")
print(best_config.to_dict())

# ========== STEP 3: Training con la miglior configurazione ==========
final_config = replace(
    best_config,
    epochs=10,
    save_path='optuna_best.pt'
)

train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, final_config.batch_size, loader_common)
model = build_policy_from_config(final_config, obs_shape=(3, 128, 128)).to(device)
trainer = BehaviorCloningTrainer(model, train_loader, val_loader, final_config, device, use_wandb=False)

print('\nRunning validation before training...')
val_metrics = trainer._validate_epoch(0)
print(f"Initial validation loss: {val_metrics['loss']:.4f}")

print('Starting final training with Optuna config...')
best_val = trainer.train()
print(f"Training finished. Best validation loss saved: {best_val:.4f}")
print(f"Checkpoint saved to: {final_config.save_path}")


