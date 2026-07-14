from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .config import CMAP_SEQUENTIAL, MODEL_COLORS


def plot_history(history, model_name: str, output_dir: Path, show=True):
    """Plotea curvas de pérdida, precisión y AUC durante el entrenamiento.
    Reemplaza la función duplicada en 03a, 03b, 04, 06a, 06b, 07, 08.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Curvas de entrenamiento — {model_name}', fontsize=13, fontweight='bold')

    axes[0].plot(history['epoch'], history['train_loss'], label='Train', marker='o', markersize=3)
    axes[0].plot(history['epoch'], history['val_loss'], label='Val', marker='o', markersize=3)
    axes[0].set_title('Loss'); axes[0].set_xlabel('Época')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history['epoch'], history['train_acc'], label='Train', marker='o', markersize=3)
    axes[1].plot(history['epoch'], history['val_acc'], label='Val', marker='o', markersize=3)
    axes[1].set_title('Accuracy'); axes[1].set_xlabel('Época')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(history['epoch'], history['val_auc'], color='#238a8d', marker='o', markersize=3)
    axes[2].axhline(y=history['val_auc'].max(), color='#238a8d', linestyle='--', alpha=0.5,
                     label=f'Best AUC={history["val_auc"].max():.4f}')
    axes[2].set_title('Val AUC'); axes[2].set_xlabel('Época')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'curves_{model_name}.png', dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_confusion_matrix(cm, model_name: str, auc: float, output_dir: Path, show=True):
    """Plotea la matriz de confusión con el colormap viridis (antes cada
    notebook usaba un cmap distinto: 'Blues' en 03a/03b, y había un typo
    'Vidiris' en 11_error_analysis).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=CMAP_SEQUENTIAL,
                xticklabels=['real', 'fake'], yticklabels=['real', 'fake'])
    plt.title(f'{model_name} — Test AUC: {auc:.4f}')
    plt.ylabel('Real'); plt.xlabel('Predicho')
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_{model_name}.png', dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


class GradCAM:
    """Grad-CAM genérico para cualquier capa convolucional de un modelo
    PyTorch. El heatmap resultante tiene el tamaño espacial de la
    activación interna de target_layer; usar overlay_heatmap() para
    superponerlo sobre la imagen original.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x):
        self.model.zero_grad()
        out = self.model(x)
        (out.backward() if out.ndim == 0 else out.sum().backward())

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam).cpu().numpy()
        if cam.max() > 1e-6:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()


def overlay_heatmap(img_rgb, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """Superpone el heatmap Grad-CAM sobre la imagen original, redimensionando
    automáticamente al tamaño de img_rgb.
    """
    h, w = img_rgb.shape[:2]
    hmap_resized = cv2.resize(heatmap, (w, h))
    hmap_uint8 = (hmap_resized * 255).astype(np.uint8)
    hmap_color = cv2.applyColorMap(hmap_uint8, colormap)
    hmap_rgb = cv2.cvtColor(hmap_color, cv2.COLOR_BGR2RGB)
    return (alpha * hmap_rgb + (1 - alpha) * img_rgb).astype(np.uint8)


def get_prob(model, tensor):
    """Devuelve p(fake) en [0,1] para cualquier modelo (single sample)."""
    with torch.no_grad():
        out = model(tensor)
    return torch.sigmoid(out).item()


def load_tensor(img_path, transforms, device):
    """Carga imagen -> (ndarray RGB original, tensor listo para el modelo)."""
    img = np.array(Image.open(img_path).convert('RGB'))
    tensor = transforms(image=img)['image'].unsqueeze(0).to(device)
    return img, tensor


def load_img(path, size: int = 128):
    """Carga y resizea una imagen para mostrarla en una grilla (11_error_analysis)."""
    img = Image.open(path).convert('RGB').resize((size, size))
    return np.array(img)


__all__ = ['plot_history', 'plot_confusion_matrix', 'GradCAM', 'overlay_heatmap',
           'get_prob', 'load_tensor', 'load_img', 'MODEL_COLORS']