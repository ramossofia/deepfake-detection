import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.amp import autocast
from tqdm import tqdm


def _squeeze(out):
    return out.squeeze(1) if out.ndim > 1 else out


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """Entrena el modelo por una época. Si se pasa un GradScaler, entrena
    con AMP (mixed precision); si no, entrenamiento en precisión normal.
    Devuelve (loss_promedio, accuracy_promedio).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb in tqdm(loader, desc='train', leave=False):
        xb, yb = xb.to(device), yb.float().to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with autocast('cuda'):
                out = _squeeze(model(xb))
                loss = criterion(out, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = _squeeze(model(xb))
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(xb)
        preds = (out.sigmoid() > 0.5).long()
        correct += (preds == yb.long()).sum().item()
        total += len(xb)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evalúa el modelo. Devuelve (loss_promedio, accuracy_promedio, AUC)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_targets = [], []

    for xb, yb in tqdm(loader, desc='val  ', leave=False):
        xb, yb = xb.to(device), yb.float().to(device)
        out = _squeeze(model(xb))
        loss = criterion(out, yb)

        total_loss += loss.item() * len(xb)
        preds = (out.sigmoid() > 0.5).long()
        correct += (preds == yb.long()).sum().item()
        total += len(xb)
        all_probs.extend(out.sigmoid().cpu().tolist())
        all_targets.extend(yb.cpu().tolist())

    auc = roc_auc_score(all_targets, all_probs)
    return total_loss / total, correct / total, auc


def train_model(model, model_name, train_loader, val_loader,
                 optimizer, criterion, scheduler, config, scaler=None):
    """Entrena el modelo, guarda el mejor checkpoint por AUC de validación
    y aplica early stopping según config['PATIENCE'] (si está definido).
    Devuelve un DataFrame con el historial por época.
    """
    device, epochs = config['DEVICE'], config['EPOCHS']
    output_dir = config['OUTPUT_DIR']
    patience = config.get('PATIENCE')  # None => sin early stopping

    best_auc, epochs_no_improve, history = 0.0, 0, []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                         'val_loss': val_loss, 'val_acc': val_acc, 'val_auc': val_auc})

        print(f'Epoch {epoch:02d}/{epochs} | '
              f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} | '
              f'val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}', end='')

        if val_auc > best_auc:
            best_auc, epochs_no_improve = val_auc, 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_auc': val_auc, 'config': config},
                       output_dir / f'best_{model_name}.pth')
            print(f'  ✓ checkpoint guardado (AUC={val_auc:.4f})')
        else:
            epochs_no_improve += 1
            if patience is not None:
                print(f'  (sin mejora {epochs_no_improve}/{patience})')
            else:
                print()

        if patience is not None and epochs_no_improve >= patience:
            print(f'\nEarly stopping en época {epoch}.')
            break

    return pd.DataFrame(history)


@torch.no_grad()
def evaluate_test(model, model_name, test_loader, device, output_dir, plot=True):
    """Carga el mejor checkpoint, evalúa en test y (opcionalmente) plotea
    la matriz de confusión. Devuelve (auc, preds, targets, probs).
    """
    ckpt = torch.load(output_dir / f'best_{model_name}.pth', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    probs, preds, targets = [], [], []
    for xb, yb in tqdm(test_loader, desc='Test'):
        out = _squeeze(model(xb.to(device)))
        p = out.sigmoid().cpu().tolist()
        probs.extend(p)
        preds.extend([1 if x > 0.5 else 0 for x in p])
        targets.extend(yb.tolist())

    auc = roc_auc_score(targets, probs)
    print(f'\n── {model_name} — Test AUC: {auc:.4f} ──')
    print(classification_report(targets, preds, target_names=['real', 'fake']))

    if plot:
        from .viz import plot_confusion_matrix
        cm = confusion_matrix(targets, preds)
        plot_confusion_matrix(cm, model_name, auc, output_dir)

    return auc, preds, targets, probs


@torch.no_grad()
def get_auc_for_loader(model, loader, device):
    """Calcula el AUC de un modelo ya cargado sobre un DataLoader dado.
    Usado en la matriz de generalización cruzada (05).
    """
    model.eval()
    all_probs, all_targets = [], []
    for xb, yb in loader:
        out = _squeeze(model(xb.to(device)))
        all_probs.extend(out.sigmoid().cpu().tolist())
        all_targets.extend(yb.tolist())
    return roc_auc_score(all_targets, all_probs)


@torch.no_grad()
def evaluate_by_method(model, df, config, dataset_cls=None, val_transforms=None):
    """Evalúa un modelo ya entrenado, separado por cada método de fake
    (real vs. Deepfakes, real vs. Face2Face, etc.) sobre el split 'test'.
    Devuelve un DataFrame con AUC/Accuracy/F1 por método.
    Usado en 06a/06b/07/08 para comparar generalización entre arquitecturas.
    """
    from torch.utils.data import DataLoader
    from .config import FAKE_CLASSES
    from .data import DeepfakeDataset, get_transforms

    if dataset_cls is None:
        dataset_cls = DeepfakeDataset
    if val_transforms is None:
        _, val_transforms = get_transforms(config['IMG_SIZE'])

    device = config['DEVICE']
    results = []

    for method in sorted(FAKE_CLASSES):
        df_m = df[(df['label'] == 0) | (df['method'] == method)].copy()
        df_m = df_m[df_m['split'] == 'test'].reset_index(drop=True)
        loader = DataLoader(dataset_cls(df_m, val_transforms),
                             batch_size=config['BATCH_SIZE'], shuffle=False,
                             num_workers=config.get('NUM_WORKERS', 0),
                             pin_memory=(device == 'cuda'))

        all_probs, all_labels = [], []
        for xb, yb in tqdm(loader, desc=method, leave=False):
            probs = _squeeze(model(xb.to(device))).sigmoid().cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(yb.tolist())

        preds = [1 if p > 0.5 else 0 for p in all_probs]
        auc = roc_auc_score(all_labels, all_probs)
        acc = sum(p == l for p, l in zip(preds, all_labels)) / len(all_labels)
        tp = sum(p == 1 and l == 1 for p, l in zip(preds, all_labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(preds, all_labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(preds, all_labels))
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)

        results.append({'method': method, 'AUC': auc, 'Accuracy': acc, 'F1': f1})
        print(f'  {method:20s}  AUC={auc:.4f}  Acc={acc:.4f}  F1={f1:.4f}')

    return pd.DataFrame(results)