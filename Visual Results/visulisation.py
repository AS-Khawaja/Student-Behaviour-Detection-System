#!/usr/bin/env python3
"""
visualize_from_outputs.py

Create visualizations from training outputs (saved CSVs / pickles) without retraining.

Produces:
 - confusion matrix (true vs pred)
 - classification report (text and CSV)
 - per-class metrics bar chart (precision/recall/f1)
 - label distributions (true / pred)
 - agreement histogram
 - headpose scatter (yaw vs pitch) colored by true/pred
 - facemesh_pc PCA / t-SNE visualization
 - prediction example grids (random correct / incorrect)
 - (optional) training curves if history.json exists

Output images saved into visuals_from_outputs/
"""
import os
import sys
import json
import time
import random
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib
from PIL import Image

sns.set(style="whitegrid")
OUT_DIR = "visuals_from_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Filenames (change if needed)
DATASET_CSV = "dataset_with_metadata_features.csv"
TEST_PRED_CSV = "test_predictions.csv"
FACEMESH_PKL = "facemesh_pca.pkl"        # optional
SCALER_PKL = "metadata_scaler.pkl"       # optional
HISTORY_JSON = "history.json"            # optional training history (loss/acc lists)

# Utility helpers
def safe_read_csv(path):
    if not os.path.exists(path):
        print(f"[WARN] {path} not found.")
        return None
    return pd.read_csv(path)

def save_fig(fname):
    p = os.path.join(OUT_DIR, fname)
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
    print("Saved:", p)

# 1) Load CSVs
df_all = safe_read_csv(DATASET_CSV)
df_test = safe_read_csv(TEST_PRED_CSV)

if df_test is None and df_all is None:
    print("No dataset/test CSVs found. Place dataset_with_metadata_features.csv and/or test_predictions.csv in the working folder.")
    sys.exit(1)

# If test_predictions exists, prefer it for y_true/y_pred and sample examples
if df_test is not None:
    df = df_test.copy()
    # expected columns: 'true', 'pred', 'true_label', 'pred_label' (your file uses those names)
    if 'true' in df.columns and 'pred' in df.columns:
        y_true = df['true'].astype(int).values
        y_pred = df['pred'].astype(int).values
        true_label_names = None
        pred_label_names = None
        try:
            # inverse transform names may be present
            if 'true_label' in df.columns and 'pred_label' in df.columns:
                true_label_names = df['true_label'].astype(str).values
                pred_label_names = df['pred_label'].astype(str).values
        except Exception:
            pass
    else:
        print("[WARN] test_predictions.csv missing 'true'/'pred' integer columns. Trying 'true_label'/'pred_label' strings.")
        if 'true_label' in df.columns and 'pred_label' in df.columns:
            true_label_names = df['true_label'].astype(str).values
            pred_label_names = df['pred_label'].astype(str).values
            y_true = None; y_pred = None
        else:
            print("[ERROR] Can't find true/pred in test_predictions.csv. Exiting.")
            sys.exit(1)
else:
    df = df_all.copy()
    print("[INFO] No test_predictions.csv found. Using dataset CSV only (no preds available).")
    y_true = None
    y_pred = None
    true_label_names = None
    pred_label_names = None

# If label name mapping present in saved model metadata (optional), try to load it
label_classes = None
# try to find 'le_classes' in a JSON or in saved model metadata if available - skipped to avoid errors.

# ---------- Plot: Label distribution (majority) ----------
if 'maj_attention' in df.columns or 'maj_emotion' in df.columns:
    if 'maj_attention' in df.columns:
        plt.figure(figsize=(8,5))
        vc = df['maj_attention'].value_counts(dropna=True)
        vc.plot(kind='bar')
        plt.title("Distribution: maj_attention (whole set / test subset)")
        save_fig("maj_attention_distribution.png")
    if 'maj_emotion' in df.columns:
        plt.figure(figsize=(10,5))
        vc = df['maj_emotion'].value_counts(dropna=True)
        vc.plot(kind='bar')
        plt.title("Distribution: maj_emotion")
        save_fig("maj_emotion_distribution.png")

# ---------- Agreement histogram ----------
if 'agreement_emotion' in df.columns:
    plt.figure(figsize=(7,4))
    sns.histplot(df['agreement_emotion'].dropna(), bins=10)
    plt.title("Agreement among labelers (emotion)")
    plt.xlabel("Agreement fraction")
    save_fig("agreement_emotion_hist.png")

# ---------- Confusion matrix & classification report (if preds present) ----------
if y_true is not None and y_pred is not None:
    # class names
    if true_label_names is not None:
        # try to get ordered classes
        classes = list(pd.Categorical(true_label_names).categories)
    else:
        # fallback: unique labels from df
        classes = sorted(list(map(str, np.unique(np.concatenate([y_true.astype(str), y_pred.astype(str)])))))
    # if integer-coded and 'pred_label' exists, use those names
    labels_for_plot = None
    if 'true_label' in df.columns:
        labels_for_plot = list(df['true_label'].unique())
    else:
        labels_for_plot = [str(c) for c in sorted(np.unique(np.concatenate([y_true, y_pred])))]
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion matrix (test set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    save_fig("confusion_matrix_test.png")
    # classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_txt = classification_report(y_true, y_pred, zero_division=0, target_names=None)
    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report_txt)
    # save numeric report to CSV
    rep_df = pd.DataFrame(report).transpose()
    rep_df.to_csv(os.path.join(OUT_DIR, "classification_report.csv"), index=True)
    print("Saved classification report CSV and text.")
    # per-class bar chart for precision/recall/f1
    prf = rep_df.loc[[r for r in rep_df.index if r not in ['accuracy','macro avg','weighted avg']], ['precision','recall','f1-score']].fillna(0)
    prf.plot.bar(rot=45, figsize=(10,6))
    plt.title("Per-class: precision / recall / f1")
    save_fig("per_class_prf.png")
else:
    print("[INFO] Predictions not available (no test_predictions.csv). Skipping confusion matrix/PRF plots.")

# ---------- Headpose scatter (yaw vs pitch) colored by maj_attention / true_label ----------
if ('yaw' in df.columns) and ('pitch' in df.columns):
    # if predictions available, create two plots: colored by true_label and pred_label (if present)
    plt.figure(figsize=(8,6))
    hue_col = None
    if 'maj_attention' in df.columns:
        hue_col = 'maj_attention'  # mostly ground truth label
    sns.scatterplot(data=df, x="yaw", y="pitch", hue=hue_col, alpha=0.7)
    plt.title("Headpose: yaw vs pitch (colored by maj_attention)")
    save_fig("headpose_yaw_pitch_by_maj_attention.png")

    if ('true_label' in df.columns) and ('pred_label' in df.columns):
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df, x="yaw", y="pitch", hue="true_label", style="pred_label", alpha=0.7, legend='brief')
        plt.title("Headpose: yaw vs pitch (true_label: color, pred_label: marker)")
        save_fig("headpose_yaw_pitch_true_vs_pred.png")

# ---------- Facemesh visualization (PCA or t-SNE) ----------
# gather facemesh_pc_* columns if present
facemask_cols = [c for c in df.columns if c.startswith("facemesh_pc_")]
if facemask_cols:
    X_fm = df[facemask_cols].fillna(0).values
    # PCA to 2D
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X_fm)
    plt.figure(figsize=(7,6))
    if 'true_label' in df.columns:
        sns.scatterplot(x=Xp[:,0], y=Xp[:,1], hue=df['true_label'].astype(str), alpha=0.7, s=40)
        plt.title("Facemesh PCA 2D (colored by true_label)")
    else:
        plt.scatter(Xp[:,0], Xp[:,1], s=10)
        plt.title("Facemesh PCA 2D")
    save_fig("facemesh_pca2d_true.png")

    # t-SNE (slower) - do only if dataset smallish (<3000) or user allows
    try:
        if len(df) <= 3000:
            ts = TSNE(n_components=2, perplexity=30, n_iter=1000, init='pca', random_state=42)
            Xts = ts.fit_transform(X_fm)
            plt.figure(figsize=(7,6))
            if 'true_label' in df.columns:
                sns.scatterplot(x=Xts[:,0], y=Xts[:,1], hue=df['true_label'].astype(str), alpha=0.8, s=30)
                plt.title("Facemesh t-SNE (true_label)")
            else:
                plt.scatter(Xts[:,0], Xts[:,1], s=8)
                plt.title("Facemesh t-SNE")
            save_fig("facemesh_tsne2d_true.png")
        else:
            print("[INFO] dataset large >3000 rows; skipping t-SNE to avoid long runtime. If you want it, set a smaller sample or increase resources.")
    except Exception as e:
        print("t-SNE failed:", e)
else:
    print("[INFO] No facemesh_pc_* columns found — skipping facemesh visualizations.")

# ---------- Prediction examples grid (correct vs incorrect) ----------
def show_image_grid(rows_df, title, fname, n_cols=4, img_root="."):
    n = min(len(rows_df), 12)
    if n == 0:
        print(f"[INFO] No examples to show for {title}")
        return
    n_rows = (n + n_cols - 1) // n_cols
    plt.figure(figsize=(4*n_cols, 3*n_rows))
    for i, (_, r) in enumerate(rows_df.sample(n=n, random_state=42).iterrows()):
        ax = plt.subplot(n_rows, n_cols, i+1)
        img_path = r.get("image_path", None)
        if img_path is None or (isinstance(img_path, float) and np.isnan(img_path)):
            ax.text(0.5, 0.5, "no image", ha='center', va='center')
        else:
            # resolve path
            p = Path(img_root) / Path(str(img_path))
            if not p.exists():
                # try without root
                p = Path(str(img_path))
            try:
                im = Image.open(p).convert("RGB")
                ax.imshow(im)
            except Exception as e:
                ax.text(0.5, 0.5, "open failed", ha='center', va='center')
        true_lab = r.get("true_label", r.get("maj_attention", r.get("maj_emotion", "")))
        pred_lab = r.get("pred_label", r.get("self_attention", r.get("self_emotion", "")))
        ax.set_title(f"T:{true_lab}\nP:{pred_lab}", fontsize=9)
        ax.axis("off")
    plt.suptitle(title)
    save_fig(fname)

# correct examples
if ('true' in df.columns) and ('pred' in df.columns):
    correct = df[df['true'] == df['pred']]
    incorrect = df[df['true'] != df['pred']]
    show_image_grid(correct, "Random Correct Predictions (test)", "examples_correct.png")
    show_image_grid(incorrect, "Random Incorrect Predictions (test)", "examples_incorrect.png")
else:
    print("[INFO] true/pred numeric columns missing; trying to use true_label/pred_label strings for examples.")
    if 'true_label' in df.columns and 'pred_label' in df.columns:
        correct = df[df['true_label'] == df['pred_label']]
        incorrect = df[df['true_label'] != df['pred_label']]
        show_image_grid(correct, "Random Correct Predictions (test)", "examples_correct.png")
        show_image_grid(incorrect, "Random Incorrect Predictions (test)", "examples_incorrect.png")
    else:
        print("[INFO] No prediction labels found for example grids. Skipping.")

# ---------- Training curves (if history.json exists) ----------
if os.path.exists(HISTORY_JSON):
    try:
        with open(HISTORY_JSON, "r") as f:
            history = json.load(f)
        # expect keys: loss, val_loss, acc, val_acc, val_f1 or similar
        epochs = range(1, len(history.get("loss", history.get("train_loss", []))) + 1)
        plt.figure(); plt.plot(epochs, history.get("loss", history.get("train_loss", [])), label="train_loss")
        plt.plot(epochs, history.get("val_loss", []), label="val_loss")
        plt.legend(); plt.title("Loss"); save_fig("training_loss_curve.png")
        if "acc" in history or "train_acc" in history:
            plt.figure(); plt.plot(epochs, history.get("acc", history.get("train_acc", [])), label="train_acc")
            plt.plot(epochs, history.get("val_acc", []), label="val_acc")
            plt.legend(); plt.title("Accuracy"); save_fig("training_acc_curve.png")
        if "val_f1" in history:
            plt.figure(); plt.plot(epochs, history.get("val_f1", []), label="val_f1"); plt.legend(); plt.title("Validation F1"); save_fig("training_val_f1_curve.png")
    except Exception as e:
        print("[WARN] Failed to load/plot history.json:", e)
else:
    print("[INFO] No history.json found. If you saved training metrics (loss/acc) to a JSON, place it as history.json to plot training curves.")

print("All done. Visuals saved to", OUT_DIR)
