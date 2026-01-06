#!/usr/bin/env python3
"""
Colab-friendly preprocessing script for Student Behaviour Detection (Experiment 03).

Features:
- Auto-detects input CSV (searches recursively for group_01_experiment_03_subject_01.csv)
- Robust metadata resolution:
    * Accepts JSON text in CSV cell
    * Accepts Windows absolute paths / local absolute paths by matching basename
    * Recursively searches project folder for JSON files if needed
- Aggregates 4 labelers by majority vote (emotion & attention)
- Extracts metadata fields: headpose (pitch, yaw, roll), bounding box, age, gender, dominant emotion
- Defensive NaN handling and scaling
- Saves cleaned CSV and visualization PNGs in preproc_visuals/
- Designed to run in Google Colab or local Linux/Mac
"""

import os
import glob
import json
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix

# --------------- Config ---------------
# If you want to hardcode a path instead of auto-detection, set this variable:
INPUT_CSV = "/content/project/group_01_experiment_03_subject_01_colab_paths.csv"  # e.g. "/content/project/group_01_experiment_03_subject_01.csv"
OUTPUT_CSV = "cleaned_subject01.csv"
SAMPLE_FRACTION = 0.15   # set to 1.0 to use whole CSV
MAKE_SAMPLE = True       # set False to process full dataset
VIS_DIR = "preproc_visuals"
os.makedirs(VIS_DIR, exist_ok=True)
# --------------- End Config ---------------


def find_csv():
    """Find the CSV file automatically if INPUT_CSV is None or not found."""
    if INPUT_CSV:
        if os.path.exists(INPUT_CSV):
            return INPUT_CSV
        else:
            print(f"Configured INPUT_CSV set but not found: {INPUT_CSV}")
    # Search common names recursively
    candidates = glob.glob("**/group_01_experiment_03_subject_01.csv", recursive=True)
    if candidates:
        # use first match
        return candidates[0]
    # fallback: search for any csv with 'experiment_03' in name
    candidates = glob.glob("**/*experiment_03*.csv", recursive=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError("Could not find the CSV file. Place 'group_01_experiment_03_subject_01.csv' in the project folder.")


def safe_load_json(path):
    """Load JSON safely; return dict or None."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception:
        return None


def majority_label(values):
    """Return majority label among a list (ties resolved deterministically)."""
    if not values:
        return None
    cnt = Counter([v for v in values if pd.notna(v) and v is not None])
    if not cnt:
        return None
    most_common = cnt.most_common()
    top_count = most_common[0][1]
    top_candidates = [lab for lab, c in most_common if c == top_count]
    return sorted(top_candidates)[0]


def extract_from_metadata(meta):
    """
    Extract features from metadata JSON structure.
    Returns a dict with keys for expected features (NaN if missing).
    """
    out = {
        "pitch": np.nan, "yaw": np.nan, "roll": np.nan,
        "bbox_x0": np.nan, "bbox_y0": np.nan, "bbox_x1": np.nan, "bbox_y1": np.nan,
        "bbox_w": np.nan, "bbox_h": np.nan,
        "meta_dominant_emotion": None,
        "age_est": np.nan,
        "gender_est": None
    }
    if meta is None:
        return out

    person = meta.get("person", {}) if isinstance(meta, dict) else {}
    face = person.get("face", {}) if isinstance(person, dict) and isinstance(person.get("face", {}), dict) else {}

    hp = face.get("headpose")
    pose = hp.get("pose") if isinstance(hp, dict) and isinstance(hp.get("pose"), dict) else {}

    #pose = face.get("headpose", {}).get("pose", {}) if isinstance(face.get("headpose", {}), dict) else {}
    out["pitch"] = pose.get("pitch", np.nan)
    out["yaw"] = pose.get("yaw", np.nan)
    out["roll"] = pose.get("roll", np.nan)

    bbox = face.get("bounding_box", {}) or {}
    if bbox:
        x0 = bbox.get("x0"); y0 = bbox.get("y0"); x1 = bbox.get("x1"); y1 = bbox.get("y1")
        out["bbox_x0"] = x0 if x0 is not None else np.nan
        out["bbox_y0"] = y0 if y0 is not None else np.nan
        out["bbox_x1"] = x1 if x1 is not None else np.nan
        out["bbox_y1"] = y1 if y1 is not None else np.nan
        try:
            out["bbox_w"] = (out["bbox_x1"] - out["bbox_x0"]) if pd.notna(out["bbox_x1"]) and pd.notna(out["bbox_x0"]) else np.nan
            out["bbox_h"] = (out["bbox_y1"] - out["bbox_y0"]) if pd.notna(out["bbox_y1"]) and pd.notna(out["bbox_y0"]) else np.nan
        except Exception:
            out["bbox_w"], out["bbox_h"] = np.nan, np.nan

    emotion = face.get("emotion", {}) or {}
    out["meta_dominant_emotion"] = emotion.get("dominant_emotion") if isinstance(emotion, dict) else None

    age = face.get("age")
    out["age_est"] = age if age is not None else np.nan

    gender = face.get("gender", {}) or {}
    out["gender_est"] = gender.get("gender_name") if isinstance(gender, dict) else None

    return out


def aggregate_labelers(row_dict):
    """
    Aggregate 4 labelers and compute majority and agreement metrics.
    Handles labeler column names with two-digit indices (labeler_01, labeler_02, ...).
    """
    emotions = []
    attentions = []
    # note: labeler columns in your CSV are 'labeler_01 emotion', 'labeler_02 emotion', etc.
    for i in range(1, 5):
        idx = f"{i:02d}"  # produces '01', '02', '03', '04'
        emo_col = f"labeler_{idx} emotion"
        att_col = f"labeler_{idx} attention"
        e = None
        a = None
        if emo_col in row_dict:
            e = row_dict.get(emo_col)
        if att_col in row_dict:
            a = row_dict.get(att_col)
        # keep None if missing or NaN
        emotions.append(e if pd.notna(e) else None)
        attentions.append(a if pd.notna(a) else None)

    maj_emo = majority_label(emotions)
    maj_att = majority_label(attentions)

    valid_emotions = [e for e in emotions if e is not None]
    agreement_emo = (valid_emotions.count(maj_emo) / len(valid_emotions)) if valid_emotions else np.nan
    valid_attentions = [a for a in attentions if a is not None]
    agreement_att = (valid_attentions.count(maj_att) / len(valid_attentions)) if valid_attentions else np.nan

    return {
        "labeler_emotions": emotions,
        "labeler_attentions": attentions,
        "maj_emotion": maj_emo,
        "maj_attention": maj_att,
        "agreement_emotion": agreement_emo,
        "agreement_attention": agreement_att
    }



def safe_encode_with_le(le, x):
    """Encode with LabelEncoder safely; return np.nan on failure."""
    try:
        if pd.isna(x) or x is None:
            return np.nan
        return int(le.transform([x])[0])
    except Exception:
        return np.nan


def main():
    # Find CSV
    csv_path = find_csv()
    print("Using CSV:", csv_path)

    # Load CSV
    print("Loading CSV...")
    df = pd.read_csv(csv_path)
    print("CSV shape:", df.shape)

    # ---------- Find the uploaded metadata directory (preferred) ----------
# We'll try to find a folder named 'metadata' that contains JSON files (uploaded to Colab).
    metadata_dirs = []
    for d in glob.glob("**/metadata", recursive=True):
    # ensure it actually contains JSON files
        js = glob.glob(os.path.join(d, "*.json"))
        if js:
            metadata_dirs.append(d)

    if metadata_dirs:
    # choose the metadata dir with most jsons (likely correct)
        metadata_dir = max(metadata_dirs, key=lambda d: len(glob.glob(os.path.join(d, "*.json"))))
        print(f"Using metadata_dir: {metadata_dir} (contains {len(glob.glob(os.path.join(metadata_dir, '*.json')))} json files)")
    else:
        metadata_dir = None
        print("No top-level metadata/ folder found containing JSONs. Will do recursive search per-file (slower).")


    # Optional sampling
    if MAKE_SAMPLE and 0 < SAMPLE_FRACTION < 1.0:
        df = df.sample(frac=SAMPLE_FRACTION, random_state=42).reset_index(drop=True)
        print(f"Sampling {SAMPLE_FRACTION*100:.1f}% -> new shape {df.shape}")

    # Process rows
    new_rows = []
    print("Processing rows and metadata JSONs...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        r = row.to_dict()

        meta = None
        meta_path_raw = r.get("metadata")

        # If the metadata cell contains inline JSON text, parse it directly
        if isinstance(meta_path_raw, str) and meta_path_raw.strip().startswith("{"):
            try:
                meta = json.loads(meta_path_raw)
            except Exception:
                meta = None

        if meta is None and isinstance(meta_path_raw, str) and meta_path_raw.strip():
            basename = os.path.basename(meta_path_raw).strip()
            found = []

            # 1) Prefer direct lookup inside metadata_dir (fast & reliable)
            if metadata_dir:
                candidate = os.path.join(metadata_dir, basename)
                if os.path.exists(candidate):
                    found.append(candidate)

            # 2) Also try CSV folder's metadata subfolder (if different)
            csv_dir = os.path.dirname(csv_path) or "."
            candidate2 = os.path.join(csv_dir, "metadata", basename)
            if os.path.exists(candidate2):
                found.append(candidate2)

            # 3) If still not found, do recursive search across the workspace
            if not found:
                matches = glob.glob(f"**/{basename}", recursive=True)
                matches = [m for m in matches if os.path.isfile(m)]
                if matches:
                    found.extend(matches)

            if not found:
                # no JSON found — keep original for debugging and leave meta None
                print(f"Warning: metadata file for row idx {idx} not found for original path: {meta_path_raw}")
                meta = None
                meta_path = meta_path_raw
            else:
                if len(found) > 1:
                    print(f"Warning: multiple metadata matches found for '{basename}', using first: {found[0]}")
                meta_path = found[0]
                meta = safe_load_json(meta_path)

        # If metadata was not a string or empty, meta remains None

        meta_feats = extract_from_metadata(meta)
        agg = aggregate_labelers(r)

        # self labels
        self_emo = r.get("self_labeling emotion")
        self_att = r.get("self_labeling attention")

        out = {
            "time": r.get("time"),
            "subject": r.get("subject"),
            "experiment": r.get("experiment"),
            "image_path": r.get("image_path"),
            "metadata_path": meta_path if 'meta_path' in locals() else meta_path_raw,
            # aggregated labeler outputs
            "maj_emotion": agg["maj_emotion"],
            "maj_attention": agg["maj_attention"],
            "agreement_emotion": agg["agreement_emotion"],
            "agreement_attention": agg["agreement_attention"],
            # self labels
            "self_emotion": self_emo,
            "self_attention": self_att,
            # metadata features
            "pitch": meta_feats["pitch"],
            "yaw": meta_feats["yaw"],
            "roll": meta_feats["roll"],
            "bbox_x0": meta_feats["bbox_x0"],
            "bbox_y0": meta_feats["bbox_y0"],
            "bbox_x1": meta_feats["bbox_x1"],
            "bbox_y1": meta_feats["bbox_y1"],
            "bbox_w": meta_feats["bbox_w"],
            "bbox_h": meta_feats["bbox_h"],
            "meta_dominant_emotion": meta_feats["meta_dominant_emotion"],
            "age_est": meta_feats["age_est"],
            "gender_est": meta_feats["gender_est"]
        }
        new_rows.append(out)

    cleaned = pd.DataFrame(new_rows)
    print("Cleaned dataframe shape:", cleaned.shape)

    # Drop rows with no aggregated labelers at all (both maj_emotion and maj_attention missing)
    cleaned = cleaned.dropna(subset=["maj_emotion", "maj_attention"], how="all").reset_index(drop=True)
    print("After dropping rows missing both maj_emotion & maj_attention:", cleaned.shape)

    # Fill numeric missing with median where appropriate
    numeric_cols = ["pitch", "yaw", "roll", "bbox_w", "bbox_h", "age_est", "agreement_emotion", "agreement_attention"]
    for c in numeric_cols:
        if c in cleaned.columns:
            med = cleaned[c].median(skipna=True)
            if np.isnan(med):
                # If median is nan (no numeric data), skip
                continue
            cleaned[c] = cleaned[c].fillna(med)

    # --- Encoding categorical labels ---
    le_emotion = LabelEncoder()
    le_attention = LabelEncoder()

    # Build pool for emotions from available columns
    pools = []
    for col in ["maj_emotion", "self_emotion", "meta_dominant_emotion"]:
        if col in cleaned.columns:
            pools.append(cleaned[col].dropna().astype(str))
    if pools:
        all_emotions = pd.concat(pools).unique().tolist()
    else:
        all_emotions = []

    # Fit if we have any emotions
    if all_emotions:
        try:
            le_emotion.fit(all_emotions)
        except Exception:
            # fallback: no fit
            le_emotion = None

    if le_emotion:
        cleaned["maj_emotion_encoded"] = cleaned["maj_emotion"].apply(lambda x: safe_encode_with_le(le_emotion, x))
        cleaned["self_emotion_encoded"] = cleaned["self_emotion"].apply(lambda x: safe_encode_with_le(le_emotion, x))
        cleaned["meta_emotion_encoded"] = cleaned["meta_dominant_emotion"].apply(lambda x: safe_encode_with_le(le_emotion, x))
    else:
        cleaned["maj_emotion_encoded"] = np.nan
        cleaned["self_emotion_encoded"] = np.nan
        cleaned["meta_emotion_encoded"] = np.nan

    # Attention encoding
    att_pools = []
    for col in ["maj_attention", "self_attention"]:
        if col in cleaned.columns:
            att_pools.append(cleaned[col].dropna().astype(str))
    all_atts = pd.concat(att_pools).unique().tolist() if att_pools else []
    if all_atts:
        try:
            le_attention.fit(all_atts)
        except Exception:
            le_attention = None

    if le_attention:
        cleaned["maj_attention_encoded"] = cleaned["maj_attention"].apply(lambda x: safe_encode_with_le(le_attention, x))
        cleaned["self_attention_encoded"] = cleaned["self_attention"].apply(lambda x: safe_encode_with_le(le_attention, x))
    else:
        cleaned["maj_attention_encoded"] = np.nan
        cleaned["self_attention_encoded"] = np.nan

    # --- Normalize numeric features (defensive) ---
    scaler = MinMaxScaler()
    scale_cols = [c for c in ["pitch", "yaw", "roll", "bbox_w", "bbox_h", "age_est", "agreement_emotion", "agreement_attention"] if c in cleaned.columns]

    if scale_cols and len(cleaned) > 0:
        sub = cleaned[scale_cols].copy()
        cols_with_data = [c for c in scale_cols if sub[c].notna().any()]
        if len(cols_with_data) == 0:
            print("Warning: no numeric columns with data to scale — skipping MinMax scaling.")
        else:
            # fill NaNs with median for scaling
            for c in cols_with_data:
                med = sub[c].median(skipna=True)
                if np.isnan(med):
                    sub[c] = sub[c].fillna(0)
                else:
                    sub[c] = sub[c].fillna(med)
            try:
                cleaned[cols_with_data] = scaler.fit_transform(sub[cols_with_data])
            except Exception as e:
                print("Warning: scaling failed:", e)
    else:
        print("No numeric columns found for scaling. Skipping MinMax scaling.")

    # Save cleaned CSV
    cleaned.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved cleaned CSV -> {OUTPUT_CSV}")

    # ---------------- Visualizations (defensive) ----------------
    print("Creating visualizations...")

    def safe_savefig(fname):
        path = os.path.join(VIS_DIR, fname)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Saved: {path}")

    # Emotion distribution (majority)
    if "maj_emotion" in cleaned.columns and cleaned["maj_emotion"].notna().any():
        vc = cleaned["maj_emotion"].value_counts(dropna=True)
        fig = plt.figure(figsize=(8,5))
        ax = vc.plot(kind='bar')
        ax.set_title("Distribution of majority labeler emotions")
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Count")
        safe_savefig("maj_emotion_distribution.png")
    else:
        print("Warning: maj_emotion is empty or missing — skipping emotion distribution plot.")

    # Attention pie
    if "maj_attention" in cleaned.columns and cleaned["maj_attention"].notna().any():
        fig = plt.figure(figsize=(6,6))
        cleaned["maj_attention"].value_counts(dropna=True).plot(kind='pie', autopct='%1.1f%%')
        plt.title("Majority labeler attention distribution")
        plt.ylabel("")
        safe_savefig("maj_attention_pie.png")
    else:
        print("Warning: maj_attention is empty or missing — skipping attention pie chart.")

    # Headpose scatter
    if ("yaw" in cleaned.columns) and ("pitch" in cleaned.columns) and cleaned["yaw"].notna().any() and cleaned["pitch"].notna().any():
        fig = plt.figure(figsize=(8,6))
        try:
            sns.scatterplot(data=cleaned, x="yaw", y="pitch", hue="maj_attention", alpha=0.8)
            plt.title("Headpose: yaw vs pitch (colored by majority attention)")
            safe_savefig("headpose_yaw_vs_pitch.png")
        except Exception as e:
            plt.close()
            print("Warning: failed to create headpose scatterplot:", e)
    else:
        print("Warning: yaw/pitch missing or empty — skipping headpose scatterplot.")

    # Agreement histogram
    if "agreement_emotion" in cleaned.columns and cleaned["agreement_emotion"].notna().any():
        fig = plt.figure(figsize=(7,4))
        sns.histplot(cleaned["agreement_emotion"].dropna(), bins=5)
        plt.title("Histogram of emotion agreement among labelers (fraction)")
        plt.xlabel("Agreement fraction")
        safe_savefig("agreement_emotion_hist.png")
    else:
        print("Warning: agreement_emotion missing or empty — skipping agreement histogram.")

    # Confusion matrix: maj vs self (if encoded and present)
    if "maj_emotion_encoded" in cleaned.columns and "self_emotion_encoded" in cleaned.columns:
        cm_df = cleaned[["maj_emotion_encoded", "self_emotion_encoded"]].dropna()
        if not cm_df.empty:
            y_true = cm_df["maj_emotion_encoded"].astype(int)
            y_pred = cm_df["self_emotion_encoded"].astype(int)
            try:
                cm = confusion_matrix(y_true, y_pred)
                fig = plt.figure(figsize=(8,6))
                sns.heatmap(cm, annot=True, fmt='d')
                plt.title("Confusion matrix: majority_labelers vs self_labeling (encoded)")
                plt.xlabel("self_labeling (pred)")
                plt.ylabel("majority_labelers (true)")
                safe_savefig("confusion_majority_vs_self_emotion.png")
            except Exception as e:
                plt.close()
                print("Warning: could not compute confusion matrix:", e)
        else:
            print("Warning: Not enough rows with both maj_emotion_encoded and self_emotion_encoded — skipping confusion matrix.")
    else:
        print("Warning: maj_emotion_encoded or self_emotion_encoded columns missing — skipping confusion matrix.")

    print(f"Saved visuals to {VIS_DIR}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
