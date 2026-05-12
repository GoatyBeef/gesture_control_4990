"""
train.py
--------
Trains an MLP classifier on the gesture landmark data collected by data_collector.py.

USAGE:
  python train.py                        # uses default gesture_data.csv
  python train.py --data gesture_data.csv --epochs 100 --lr 0.001

OUTPUT:
  gesture_model.pth       — saved PyTorch model weights
  label_map.json          — maps class index → gesture name
  training_curves.png     — loss & accuracy plots (for your paper)
  confusion_matrix.png    — per-class accuracy heatmap (for your paper)
  training_results.txt    — plain-text summary of all metrics

REQUIREMENTS:
  pip install torch scikit-learn matplotlib seaborn pandas numpy
"""

import os
import json
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # no GUI needed; saves to file
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train gesture MLP classifier")
    p.add_argument("--data",    default="gesture_data.csv", help="Path to collected CSV")
    p.add_argument("--epochs",  type=int,   default=80,    help="Training epochs")
    p.add_argument("--lr",      type=float, default=0.001, help="Learning rate")
    p.add_argument("--batch",   type=int,   default=32,    help="Batch size")
    p.add_argument("--hidden",  type=int,   default=128,   help="Hidden layer size")
    p.add_argument("--test",    type=float, default=0.2,   help="Test split fraction")
    p.add_argument("--seed",    type=int,   default=42,    help="Random seed")
    return p.parse_args()

# ── Model ─────────────────────────────────────────────────────────────────────

class GestureMLP(nn.Module):
    """
    Three-layer MLP for gesture classification from 42 landmark features.

    Architecture:
      Input (42) → FC(hidden) → BN → ReLU → Dropout(0.3)
                 → FC(hidden) → BN → ReLU → Dropout(0.3)
                 → FC(hidden//2) → ReLU
                 → FC(num_classes)
    """
    def __init__(self, input_size=42, hidden_size=128, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),

            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(csv_path):
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"  Total samples : {len(df)}")
    print(f"  Class distribution:")
    for label, count in df["label"].value_counts().items():
        print(f"    {label:15s}: {count}")

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].values

    return X, y_raw

# ── Normalization: make coordinates scale-invariant ───────────────────────────

def normalize_landmarks(X):
    """
    Normalize each sample so that landmark coordinates are relative to the
    wrist (landmark 0) and scaled by the hand span.
    This makes the model robust to hand size and camera distance.
    """
    X_norm = X.copy()
    for i in range(len(X_norm)):
        # Wrist is landmark 0 → columns 0,1
        wrist_x, wrist_y = X_norm[i, 0], X_norm[i, 1]
        # Subtract wrist (center)
        X_norm[i, 0::2] -= wrist_x
        X_norm[i, 1::2] -= wrist_y
        # Scale by max extent
        scale = np.max(np.abs(X_norm[i])) + 1e-6
        X_norm[i] /= scale
    return X_norm

# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct   += (logits.argmax(1) == y_batch).sum().item()
        total     += len(y_batch)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(1) == y_batch).sum().item()
            total      += len(y_batch)
    return total_loss / total, correct / total

# ── Plots ─────────────────────────────────────────────────────────────────────

def save_training_curves(history, out_path="training_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch");  ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training & Validation Loss");  ax1.legend();  ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train Acc", linewidth=2)
    ax2.plot(epochs, history["val_acc"],   label="Val Acc",   linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch");  ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy");  ax2.legend();  ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def save_confusion_matrix(y_true, y_pred, class_names, out_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted");  ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

# ── Hyperparameter sensitivity ─────────────────────────────────────────────────

def run_lr_sweep(X_train, y_train, X_val, y_val, hidden, num_classes, device, epochs=40):
    """
    Train with several learning rates and report final val accuracy.
    Used to produce a hyperparameter sensitivity table for the paper.
    """
    results = {}
    lrs = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    print("\n── LR Sensitivity Sweep ──────────────────────────────────")
    for lr in lrs:
        model = GestureMLP(42, hidden, num_classes).to(device)
        opt   = optim.Adam(model.parameters(), lr=lr)
        crit  = nn.CrossEntropyLoss()
        ds    = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        for _ in range(epochs):
            train_epoch(model, loader, crit, opt, device)
        val_ds  = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        val_loader = DataLoader(val_ds, batch_size=64)
        _, val_acc = eval_epoch(model, val_loader, crit, device)
        results[lr] = val_acc
        print(f"  LR={lr:.4f}  →  val_acc={val_acc:.4f}")
    return results


def run_hidden_sweep(X_train, y_train, X_val, y_val, lr, num_classes, device, epochs=40):
    """Train with several hidden sizes to show architecture sensitivity."""
    results = {}
    sizes = [32, 64, 128, 256, 512]
    print("\n── Hidden Size Sensitivity Sweep ─────────────────────────")
    for h in sizes:
        model = GestureMLP(42, h, num_classes).to(device)
        opt   = optim.Adam(model.parameters(), lr=lr)
        crit  = nn.CrossEntropyLoss()
        ds    = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        for _ in range(epochs):
            train_epoch(model, loader, crit, opt, device)
        val_ds  = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        val_loader = DataLoader(val_ds, batch_size=64)
        _, val_acc = eval_epoch(model, val_loader, crit, device)
        results[h] = val_acc
        print(f"  hidden={h:4d}  →  val_acc={val_acc:.4f}")
    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load & preprocess ────────────────────────────────────────────────────
    X, y_raw = load_data(args.data)
    X = normalize_landmarks(X)

    le = LabelEncoder()
    y  = le.fit_transform(y_raw).astype(np.int64)
    class_names = list(le.classes_)
    num_classes = len(class_names)

    # Save label map so the live recognizer can decode predictions
    label_map = {int(i): name for i, name in enumerate(class_names)}
    with open("label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\nClasses ({num_classes}): {class_names}")

    # Train / val / test split  (60 / 20 / 20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=args.seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=args.seed, stratify=y_temp)

    print(f"\nSplit: train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    # ── Dataloaders ──────────────────────────────────────────────────────────
    def make_loader(X_, y_, shuffle=False, batch=args.batch):
        ds = TensorDataset(torch.tensor(X_), torch.tensor(y_))
        return DataLoader(ds, batch_size=batch, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val)
    test_loader  = make_loader(X_test,  y_test)

    # ── Model, loss, optimizer ───────────────────────────────────────────────
    model     = GestureMLP(42, args.hidden, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {args.epochs} epochs  |  LR={args.lr}  |  batch={args.batch}\n")

    # ── Training loop ────────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state   = None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = eval_epoch(model,  val_loader,   criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
                  f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")

    print(f"\nTraining time: {time.time() - t0:.1f}s")
    print(f"Best val accuracy: {best_val_acc:.4f}")

    # ── Evaluate on test set ─────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    test_acc = accuracy_score(all_true, all_preds)
    test_f1  = f1_score(all_true, all_preds, average="weighted")
    report   = classification_report(all_true, all_preds, target_names=class_names)

    print(f"\nTest Accuracy : {test_acc:.4f}")
    print(f"Test F1 (weighted): {test_f1:.4f}")
    print("\nClassification Report:")
    print(report)

    # ── Save model ───────────────────────────────────────────────────────────
    torch.save({
        "model_state_dict": best_state,
        "input_size":  42,
        "hidden_size": args.hidden,
        "num_classes": num_classes,
        "label_map":   label_map,
    }, "gesture_model.pth")
    print("Saved: gesture_model.pth")

    # ── Plots ────────────────────────────────────────────────────────────────
    save_training_curves(history)
    save_confusion_matrix(all_true, all_preds, class_names)

    # ── Hyperparameter sweeps ─────────────────────────────────────────────────
    lr_results     = run_lr_sweep(X_train, y_train, X_val, y_val,
                                  args.hidden, num_classes, device)
    hidden_results = run_hidden_sweep(X_train, y_train, X_val, y_val,
                                      args.lr, num_classes, device)

    # ── Save text summary ────────────────────────────────────────────────────
    with open("training_results.txt", "w") as f:
        f.write("=== Gesture MLP Training Results ===\n\n")
        f.write(f"Data file   : {args.data}\n")
        f.write(f"Epochs      : {args.epochs}\n")
        f.write(f"LR          : {args.lr}\n")
        f.write(f"Batch size  : {args.batch}\n")
        f.write(f"Hidden size : {args.hidden}\n")
        f.write(f"Classes     : {class_names}\n")
        f.write(f"Train / Val / Test: {len(X_train)} / {len(X_val)} / {len(X_test)}\n\n")
        f.write(f"Best val accuracy : {best_val_acc:.4f}\n")
        f.write(f"Test accuracy     : {test_acc:.4f}\n")
        f.write(f"Test F1 (weighted): {test_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("LR Sensitivity (40 epochs each):\n")
        for lr, acc in lr_results.items():
            f.write(f"  LR={lr}  val_acc={acc:.4f}\n")
        f.write("\nHidden Size Sensitivity (40 epochs each):\n")
        for h, acc in hidden_results.items():
            f.write(f"  hidden={h}  val_acc={acc:.4f}\n")

    print("\nSaved: training_results.txt")
    print("\nAll done! Files produced:")
    print("  gesture_model.pth      — model weights (use in gesture_recognizer.py)")
    print("  label_map.json         — class index → gesture name")
    print("  training_curves.png    — loss & accuracy plots for paper")
    print("  confusion_matrix.png   — per-class accuracy heatmap for paper")
    print("  training_results.txt   — full metrics summary")


if __name__ == "__main__":
    main()