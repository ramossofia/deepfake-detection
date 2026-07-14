import os
import random
from pathlib import Path

import numpy as np
import torch

BASE_PATH = Path.cwd()

_CANDIDATE_ROOTS = [
    os.environ.get('DEEPFAKES_ROOT'),
    BASE_PATH,  
    '/Users/sofiaramos/Desktop/DEEPFAKES',
]

PROJECT_ROOT = next(
    (Path(p) for p in _CANDIDATE_ROOTS if p and Path(p).exists()),
    BASE_PATH,
)

FRAMES_PATH   = PROJECT_ROOT / 'frames'
SPLITS_PATH   = PROJECT_ROOT / 'splits'
DATA_DIR      = PROJECT_ROOT / 'dataset_split'
CSV_PATH      = PROJECT_ROOT / 'FaceForensics++_C23' / 'csv'
OUTPUT_DIR    = PROJECT_ROOT / 'outputs'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



CLASSES = ['real', 'Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap',
           'NeuralTextures', 'DeepFakeDetection']

SPLIT_CLASSES = ['real', 'Deepfakes', 'Face2Face', 'FaceShifter',
                  'FaceSwap', 'NeuralTextures']

FAKE_CLASSES = {'Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap',
                 'NeuralTextures'}

METHODS = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap',
           'NeuralTextures']


def get_device() -> str:
    """Detecta el mejor device disponible: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def set_seed(seed: int = 42) -> None:
    """Fija todas las semillas relevantes para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEFAULT_CONFIG = dict(
    DATA_DIR     = DATA_DIR,
    OUTPUT_DIR   = OUTPUT_DIR,
    SEED         = 42,
    IMG_SIZE     = 299,
    BATCH_SIZE   = 32,
    LR           = 1e-4,
    LR_HEAD      = 1e-3,
    WEIGHT_DECAY = 1e-4,
    EPOCHS       = 15,
    PATIENCE     = 5,
    DROPOUT      = 0.3,
    NUM_WORKERS  = 4,
    DEVICE       = get_device(),
)


def make_config(**overrides) -> dict:
    """Devuelve una copia de DEFAULT_CONFIG con overrides puntuales.

    Ejemplo:
        CONFIG = make_config(MODEL_NAME='efficientnet_b4', BATCH_SIZE=64)
    """
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(overrides)
    cfg['OUTPUT_DIR'].mkdir(parents=True, exist_ok=True)
    return cfg

CMAP_SEQUENTIAL = 'viridis'
COLOR_REAL = '#463480'
COLOR_FAKE = '#9bd93c'

CLASS_COLORS = {
    'real':            '#471365',
    'Deepfakes':       '#3e4989',
    'Face2Face':       '#2b758e',
    'FaceShifter':     '#1f9f88',
    'FaceSwap':        '#52c569',
    'NeuralTextures':  '#bddf26',
}

MODEL_COLORS = {
    'Baseline':     '#482475',
    'BiFPN':        '#238a8d',
    'FNO':          '#9bd93c',
}

def get_palette(n: int, lo: float = 0.05, hi: float = 0.9):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cmap = cm.get_cmap(CMAP_SEQUENTIAL)
    return [mcolors.to_hex(cmap(x)) for x in np.linspace(lo, hi, n)]