from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import FAKE_CLASSES


def build_frame_dataframe(data_dir: Path, splits=('train', 'val', 'test')) -> pd.DataFrame:
    """Recorre `data_dir/<split>/<clase>/**/*.jpg` y arma un DataFrame
    con columnas: path, label (0=real, 1=fake), split, method.

    Reemplaza el loop que se repetía al principio de 03a, 03b, 04, 05, 06a.
    """
    rows = []
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f'[!] No se encontró: {split_dir}')
            continue
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            label = 1 if cls_dir.name in FAKE_CLASSES else 0
            for img_path in cls_dir.rglob('*.jpg'):
                rows.append({
                    'path': str(img_path),
                    'label': label,
                    'split': split,
                    'method': cls_dir.name,
                })
    return pd.DataFrame(rows)


def filter_by_method(df: pd.DataFrame, target_method: str) -> pd.DataFrame:
    """Se queda solo con los frames reales + los de un método de fake
    puntual. Usado en 04 (entrenamiento especializado) y 05 (matriz de
    generalización cruzada) para armar los sub-datasets binarios
    real-vs-<method>.
    """
    return df[(df['label'] == 0) | (df['method'] == target_method)].copy()


def get_transforms(img_size: int = 299, extra_train_transforms=None):
    """Devuelve (train_transforms, val_transforms).

    `extra_train_transforms` permite agregar augmentations puntuales
    (ej. 06a agrega CoarseDropout) sin duplicar toda la lista.
    """
    train_ops = [
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
        A.GaussNoise(p=0.2),
        A.ImageCompression(quality_range=(60, 100), p=0.3),
    ]
    if extra_train_transforms:
        train_ops.extend(extra_train_transforms)
    train_ops += [
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
    train_transforms = A.Compose(train_ops)

    val_transforms = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])
    return train_transforms, val_transforms


class DeepfakeDataset(Dataset):
    """Dataset RGB estándar. Usado por los modelos que no usan la rama
    de frecuencia (Xception, EfficientNet, BiFPN, FNO, matriz cruzada).
    """
    def __init__(self, df: pd.DataFrame, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row['path']).convert('RGB'))
        if self.transforms:
            img = self.transforms(image=img)['image']
        return img, int(row['label'])


def make_loaders(df: pd.DataFrame, config: dict, dataset_cls=DeepfakeDataset,
                  extra_train_transforms=None):
    """Arma (train_loader, val_loader, test_loader) con WeightedRandomSampler
    para balancear real/fake en entrenamiento.

    Reemplaza la función `make_loaders` repetida en 03a y el código
    inline duplicado en 03b/04.
    """
    train_transforms, val_transforms = get_transforms(
        config['IMG_SIZE'], extra_train_transforms=extra_train_transforms
    )

    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df   = df[df['split'] == 'val'].reset_index(drop=True)
    test_df  = df[df['split'] == 'test'].reset_index(drop=True) if 'test' in df['split'].values else None

    class_counts   = train_df['label'].value_counts().sort_index().values
    weights        = 1.0 / class_counts
    sample_weights = train_df['label'].map(lambda x: weights[x]).values
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    pin = (config['DEVICE'] == 'cuda')
    nw = config.get('NUM_WORKERS', 0)

    train_loader = DataLoader(dataset_cls(train_df, train_transforms),
                               batch_size=config['BATCH_SIZE'], sampler=sampler,
                               num_workers=nw, pin_memory=pin)
    val_loader = DataLoader(dataset_cls(val_df, val_transforms),
                             batch_size=config['BATCH_SIZE'], shuffle=False,
                             num_workers=nw, pin_memory=pin)

    test_loader = None
    if test_df is not None and len(test_df) > 0:
        test_loader = DataLoader(dataset_cls(test_df, val_transforms),
                                  batch_size=config['BATCH_SIZE'], shuffle=False,
                                  num_workers=nw, pin_memory=pin)

    msg = f'Train: {len(train_df):,} | Val: {len(val_df):,}'
    if test_df is not None:
        msg += f' | Test: {len(test_df):,}'
    print(msg)

    return train_loader, val_loader, test_loader