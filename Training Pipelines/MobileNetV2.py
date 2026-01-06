"""
deliverable2_pipeline.py

End-to-end pipeline for Deliverable 2 - Student Behaviour Detection

Modifications:
- If metadata_path starts with "metadata_matched/", that prefix is removed before resolving the JSON path
- Robust metadata extraction (handles missing fields, optional facemesh PCA saving)
- Saves facemesh PCA (joblib) and metadata scaler
- Defensive handling of missing images / metadata

Usage:
  python deliverable2_pipeline.py
"""

import os
import json
import glob
import math
import joblib
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# -----------------------------
# CONFIG - edit paths & params
# -----------------------------
CONFIG = {
    # I/O
    "master_csv_pattern": "images_master*.csv",  # now supports multiple CSVs
    "labelers_dir": "labelers_jsons",
    "metadata_root": ".",
    "image_root": ".",

    # facemesh PCA
    "facemesh_pca_n": 30,

    # training
    "batch_size": 32,
    "num_epochs": 8,
    "learning_rate": 1e-4,
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # outputs
    "out_dir": "deliverable2_outputs",
    "save_model_name": "hybrid_model.pth",
    "random_seed": 42
}

os.makedirs(CONFIG["out_dir"], exist_ok=True)

# reproducibility
np.random.seed(CONFIG["random_seed"])
torch.manual_seed(CONFIG["random_seed"])

# --------------------------------
# Utility: parse time strings (HH:MM:SS:micro)
# --------------------------------
def parse_time_str(tstr):
    """
    parse strings like "10:56:17:027710" into seconds since midnight (float)
    """
    try:
        parts = str(tstr).strip().split(":")
        if len(parts) == 4:
            hh, mm, ss, micro = parts
            dt = datetime.strptime(f"{hh}:{mm}:{ss}.{micro}", "%H:%M:%S.%f")
        elif len(parts) == 3:
            dt = datetime.strptime(tstr, "%H:%M:%S")
        else:
            # fallback: try to parse as full timestamp
            dt = datetime.fromisoformat(str(tstr))
        seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
        return seconds
    except Exception:
        try:
            return float(tstr)
        except Exception as e:
            raise ValueError(f"Can't parse time string: {tstr}") from e

# --------------------------------
# 1) LABEL AGGREGATION (optional)
# --------------------------------
def aggregate_labelers_to_csv(master_csv, labelers_dir):
    df = pd.read_csv(master_csv)
    if 'maj_emotion' in df.columns and 'maj_attention' in df.columns:
        print("Master CSV already has aggregated labels, skipping re-aggregation.")
        return df

    json_files = sorted(glob.glob(os.path.join(labelers_dir, "*.json")))
    if len(json_files) == 0:
        raise FileNotFoundError(f"No labeler JSON files found in {labelers_dir}.")

    print("Loading labeler JSONs:", json_files)
    labeler_annotations = []
    labeler_names = []
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
        ann_map = {}
        for rec in data:
            if 'datetime' not in rec:
                continue
            tsec = parse_time_str(rec['datetime'])
            att = rec.get('attention', None)
            emo = rec.get('emotion', None)
            ann_map[tsec] = {"attention": att, "emotion": emo}
        labeler_annotations.append(ann_map)
        labeler_names.append(os.path.splitext(os.path.basename(jf))[0])

    def nearest_label_for_time(tsec, ann_map, tolerance=1.0):
        if len(ann_map) == 0:
            return None, None
        keys = np.array(list(ann_map.keys()))
        idx = np.argmin(np.abs(keys - tsec))
        if abs(keys[idx] - tsec) <= tolerance:
            rec = ann_map[keys[idx]]
            return rec.get('emotion', None), rec.get('attention', None)
        else:
            return None, None

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Aggregating labels"):
        tstr = row['time']
        tsec = parse_time_str(tstr)
        emo_votes = []
        att_votes = []
        self_emo = None
        self_att = None
        for name, ann_map in zip(labeler_names, labeler_annotations):
            emo, att = nearest_label_for_time(tsec, ann_map, tolerance=1.0)
            emo_votes.append(emo)
            att_votes.append(att)
            if 'self' in name.lower() or 'self_label' in name.lower():
                self_emo = emo
                self_att = att

        def majority(votes):
            votes = [v for v in votes if (v is not None and (str(v).strip() != ""))]
            if len(votes) == 0:
                return None, 0
            vals, counts = np.unique(votes, return_counts=True)
            maxidx = np.argmax(counts)
            maj = vals[maxidx]
            agreement = counts[maxidx] / len(votes)
            return maj, agreement

        maj_emotion, agreement_emotion = majority(emo_votes)
        maj_attention, agreement_attention = majority(att_votes)
        rows.append({
            **row.to_dict(),
            "maj_emotion": maj_emotion,
            "maj_attention": maj_attention,
            "agreement_emotion": agreement_emotion,
            "agreement_attention": agreement_attention,
            "self_emotion": self_emo,
            "self_attention": self_att
        })

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(CONFIG["out_dir"], "master_with_labels.csv")
    df_out.to_csv(out_path, index=False)
    print("Saved aggregated CSV to", out_path)
    return df_out

# --------------------------------
# 2) METADATA FEATURE EXTRACTION (robust)
# --------------------------------
def _safe_num(v):
    try:
        if v is None:
            return np.nan
        return float(v)
    except:
        return np.nan

def extract_features_from_metadata(df, facemesh_pca_n=30, metadata_root=".", save_pca_path=None):
    """
    Robust metadata extraction. Strips leading 'metadata_matched/' prefix if present in metadata_path.
    Saves facemesh PCA to save_pca_path if provided.
    """
    feats = []
    facemesh_list = []
    n_with_facemesh = 0
    example_missing = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting metadata features"):
        mpath = row.get('metadata_path', "")
        if isinstance(mpath, str) and mpath.startswith("metadata_matched/"):
            # strip that prefix as requested by user
            mpath = mpath[len("metadata_matched/"):]

        if not os.path.isabs(mpath):
            mpath = os.path.join(metadata_root, mpath)

        if not os.path.exists(mpath):
            # missing file
            feats.append({
                "pitch": np.nan, "yaw": np.nan, "roll": np.nan,
                "bbox_x0": np.nan, "bbox_y0": np.nan, "bbox_x1": np.nan, "bbox_y1": np.nan,
                "bbox_w": np.nan, "bbox_h": np.nan, "bbox_area": np.nan,
                "age_est": np.nan, "gender_est": None, "gender_score": np.nan,
                "meta_dominant_emotion": None
            })
            facemesh_list.append(np.array([]))
            example_missing.append(mpath)
            continue

        with open(mpath, 'r') as f:
            try:
                meta = json.load(f)
            except Exception:
                feats.append({
                    "pitch": np.nan, "yaw": np.nan, "roll": np.nan,
                    "bbox_x0": np.nan, "bbox_y0": np.nan, "bbox_x1": np.nan, "bbox_y1": np.nan,
                    "bbox_w": np.nan, "bbox_h": np.nan, "bbox_area": np.nan,
                    "age_est": np.nan, "gender_est": None, "gender_score": np.nan,
                    "meta_dominant_emotion": None
                })
                facemesh_list.append(np.array([]))
                example_missing.append(mpath)
                continue

        # locate face dict
        face = None
        if isinstance(meta, dict):
            if 'person' in meta and isinstance(meta['person'], dict):
                face = meta['person'].get('face', None)
            if face is None:
                face = meta.get('face', None)
        if not isinstance(face, dict):
            feats.append({
                "pitch": np.nan, "yaw": np.nan, "roll": np.nan,
                "bbox_x0": np.nan, "bbox_y0": np.nan, "bbox_x1": np.nan, "bbox_y1": np.nan,
                "bbox_w": np.nan, "bbox_h": np.nan, "bbox_area": np.nan,
                "age_est": np.nan, "gender_est": None, "gender_score": np.nan,
                "meta_dominant_emotion": None
            })
            facemesh_list.append(np.array([]))
            example_missing.append(mpath)
            continue

        # headpose
        pitch = yaw = roll = np.nan
        hp = face.get("headpose", {}) if isinstance(face.get("headpose", {}), dict) else {}
        if "pose" in hp and isinstance(hp["pose"], dict):
            pose = hp["pose"]
            pitch = _safe_num(pose.get("pitch", np.nan))
            yaw = _safe_num(pose.get("yaw", np.nan))
            roll = _safe_num(pose.get("roll", np.nan))

        # bbox
        bb = face.get("bounding_box", {})
        x0 = _safe_num(bb.get("x0", np.nan))
        y0 = _safe_num(bb.get("y0", np.nan))
        x1 = _safe_num(bb.get("x1", np.nan))
        y1 = _safe_num(bb.get("y1", np.nan))
        bw = (x1 - x0) if (not np.isnan(x1) and not np.isnan(x0)) else np.nan
        bh = (y1 - y0) if (not np.isnan(y1) and not np.isnan(y0)) else np.nan
        area = (bw * bh) if (not np.isnan(bw) and not np.isnan(bh)) else np.nan

        # age & gender
        age = face.get("age", np.nan)
        gender_info = face.get("gender", {})
        if isinstance(gender_info, dict):
            gname = gender_info.get("gender_name", None)
            gscore = _safe_num(gender_info.get("gender_score", np.nan))
        else:
            gname = gender_info
            gscore = np.nan

        # dominant emotion and probs
        dom_emo = None
        emo = face.get("emotion", {})
        if isinstance(emo, dict):
            dom_emo = emo.get("dominant_emotion", None)
            prob = emo.get("probability_emotion", {})
        else:
            prob = {}

        # facemesh: support list of dicts or list of lists
        facemesh = face.get("facemesh", [])
        coords = []
        if isinstance(facemesh, list) and len(facemesh) > 0:
            for pt in facemesh:
                if isinstance(pt, dict):
                    x = _safe_num(pt.get("x", 0.0)); y = _safe_num(pt.get("y", 0.0)); z = _safe_num(pt.get("z", 0.0))
                elif isinstance(pt, (list, tuple)) and len(pt) >= 3:
                    x, y, z = pt[0], pt[1], pt[2]
                else:
                    continue
                coords.extend([x, y, z])
        if len(coords) > 0:
            facemesh_list.append(np.array(coords, dtype=np.float32))
            n_with_facemesh += 1
        else:
            facemesh_list.append(np.array([]))

        emo_flat = {f"emo_{k}": _safe_num(v) for k,v in (prob.items() if isinstance(prob, dict) else [])}

        feats.append({
            "pitch": pitch, "yaw": yaw, "roll": roll,
            "bbox_x0": x0, "bbox_y0": y0, "bbox_x1": x1, "bbox_y1": y1,
            "bbox_w": bw, "bbox_h": bh, "bbox_area": area,
            "age_est": _safe_num(age), "gender_est": gname, "gender_score": gscore,
            "meta_dominant_emotion": dom_emo,
            **emo_flat
        })

    # report
    print(f"Metadata files processed: {len(df)}, with facemesh present in {n_with_facemesh} files")
    if len(example_missing) > 0:
        print("Examples of missing/invalid metadata files:", example_missing[:5])

    # PCA over facemesh
    max_len = max([arr.size for arr in facemesh_list]) if len(facemesh_list) > 0 else 0
    if max_len == 0:
        print("No facemesh vectors found. Creating zero facemesh PCA features.")
        pca_components = np.zeros((len(df), facemesh_pca_n), dtype=np.float32)
        pca = None
    else:
        padded = np.zeros((len(facemesh_list), max_len), dtype=np.float32)
        for i, arr in enumerate(facemesh_list):
            if arr.size == 0:
                continue
            padded[i, :arr.size] = arr
        padded = np.nan_to_num(padded)
        n_comp = min(facemesh_pca_n, padded.shape[1], padded.shape[0])
        pca = PCA(n_components=n_comp)
        pca_feats = pca.fit_transform(padded)
        if pca_feats.shape[1] < facemesh_pca_n:
            pad_cols = np.zeros((pca_feats.shape[0], facemesh_pca_n - pca_feats.shape[1]), dtype=np.float32)
            pca_components = np.hstack([pca_feats, pad_cols])
        else:
            pca_components = pca_feats[:, :facemesh_pca_n]

        if save_pca_path:
            joblib.dump(pca, save_pca_path)
            print("Saved facemesh PCA to:", save_pca_path)

    feats_df = pd.DataFrame(feats)
    feats_df.index = df.index

    # attach PCA columns
    for i in range(facemesh_pca_n):
        feats_df[f"facemesh_pc_{i}"] = pca_components[:, i]

    out_df = pd.concat([df.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)
    return out_df

# --------------------------------
# 3) Dataset preparation: images + metadata -> DataLoader
# --------------------------------
class StudentBehaviorDataset(Dataset):
    def __init__(self, df, image_root=".", metadata_features=None, transform=None, label_col="label_encoded",
                 target_encoder=None, crop_face_bbox=True, expected_metadata_dim=None):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform
        self.metadata_features = metadata_features or []
        self.label_col = label_col
        self.target_encoder = target_encoder
        self.crop_face_bbox = crop_face_bbox
        if expected_metadata_dim is None:
            self.expected_metadata_dim = len(self.metadata_features)
        else:
            self.expected_metadata_dim = int(expected_metadata_dim)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # image load
        image_path = row['image_path']
        if isinstance(image_path, str) and image_path.startswith("metadata_matched/"):
            # unlikely but strip if present
            image_path = image_path[len("metadata_matched/"):]
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.image_root, image_path)
        if not os.path.exists(image_path):
            img = Image.new("RGB", (224, 224))
        else:
            img = Image.open(image_path).convert("RGB")

        # optional bbox crop
        if self.crop_face_bbox:
            try:
                x0 = int(row.get('bbox_x0', 0) or 0)
                y0 = int(row.get('bbox_y0', 0) or 0)
                x1 = int(row.get('bbox_x1', img.width) or img.width)
                y1 = int(row.get('bbox_y1', img.height) or img.height)
                if x1 > x0 and y1 > y0 and 0 <= x0 < img.width and 0 <= y0 < img.height:
                    img = img.crop((max(0, x0), max(0, y0), min(img.width, x1), min(img.height, y1)))
            except Exception:
                pass

        if self.transform:
            img = self.transform(img)

        # metadata vector
        if len(self.metadata_features) == 0:
            meta_array = np.zeros((self.expected_metadata_dim,), dtype=np.float32)
        else:
            meta_array = self.df.loc[idx, self.metadata_features].fillna(0).values.astype(np.float32)
            if meta_array.ndim > 1:
                meta_array = meta_array.reshape(-1)
            cur_len = meta_array.size
            if cur_len < self.expected_metadata_dim:
                pad_len = self.expected_metadata_dim - cur_len
                meta_array = np.concatenate([meta_array, np.zeros((pad_len,), dtype=np.float32)])
            elif cur_len > self.expected_metadata_dim:
                meta_array = meta_array[:self.expected_metadata_dim]

        meta_tensor = torch.tensor(meta_array, dtype=torch.float32)

        # label handling
        y = row.get(self.label_col, None)
        if pd.isna(y):
            y_enc = -1
        else:
            if self.target_encoder and not pd.api.types.is_integer_dtype(self.df[self.label_col]):
                y_enc = self.target_encoder.transform([str(y)])[0]
            else:
                try:
                    y_enc = int(y)
                except:
                    try:
                        y_enc = int(float(y))
                    except:
                        y_enc = 0

        label_tensor = torch.tensor(y_enc, dtype=torch.long)
        return img, meta_tensor, label_tensor

# --------------------------------
# 4) Hybrid PyTorch model (MobileNetV2 backbone)
# --------------------------------
class HybridNet(nn.Module):
    """
    Hybrid network using MobileNetV2 as image backbone and a small MLP for metadata.
    Image features from MobileNetV2 are pooled and concatenated with metadata MLP output.
    """
    def __init__(self, metadata_dim, num_classes, pretrained=True, dropout=0.3):
        super().__init__()
        # Load MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=pretrained)

        # mobilenet.features produces a feature map; we'll apply adaptive pooling to get a fixed vector
        # Keep the feature extractor (all layers except the classifier)
        self.backbone = nn.Sequential(
            mobilenet.features,
            nn.AdaptiveAvgPool2d((1,1))  # produce [B, C, 1, 1]
        )
        # backbone feature dim is the input features of mobilenet classifier linear layer
        try:
            backbone_feat_dim = mobilenet.classifier[1].in_features
        except Exception:
            # fallback to a commonly used dimension
            backbone_feat_dim = 1280

        # small metadata MLP
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # classifier that combines image + metadata features
        self.classifier = nn.Sequential(
            nn.Linear(backbone_feat_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, meta):
        # img: [B, 3, H, W]
        x = self.backbone(img)               # [B, C, 1, 1]
        x = x.view(x.size(0), -1)            # [B, C]
        m = self.metadata_mlp(meta)          # [B, 64]
        out = torch.cat([x, m], dim=1)       # [B, C+64]
        out = self.classifier(out)
        return out

# --------------------------------
# 5) Training and evaluation utils
# --------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    for imgs, metas, labels in loader:
        imgs = imgs.to(device)
        metas = metas.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs, metas)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(labels.detach().cpu().numpy().tolist())
    epoch_loss = running_loss / max(1, len(loader.dataset))
    acc = accuracy_score(all_targets, all_preds) if len(all_targets)>0 else 0.0
    return epoch_loss, acc

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, metas, labels in loader:
            imgs = imgs.to(device)
            metas = metas.to(device)
            labels = labels.to(device)
            logits = model(imgs, metas)
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.detach().cpu().numpy().tolist())
    epoch_loss = running_loss / max(1, len(loader.dataset))
    acc = accuracy_score(all_targets, all_preds) if len(all_targets)>0 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0) if len(all_targets)>0 else (0,0,0,0)
    return epoch_loss, acc, prec, rec, f1, all_targets, all_preds

# --------------------------------
# MAIN
# --------------------------------
def main():
    # 0) Load all master CSVs
    csv_files = sorted(glob.glob(CONFIG["master_csv_pattern"]))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found matching pattern {CONFIG['master_csv_pattern']}")
    print("Found CSVs:", csv_files)

    df_list = []
    for f in csv_files:
        tmp = pd.read_csv(f)
        tmp['source_csv'] = os.path.basename(f)  # optional: keep track of source CSV
        df_list.append(tmp)
    df_master = pd.concat(df_list, ignore_index=True)
    print(f"Combined {len(csv_files)} CSVs, total rows: {len(df_master)}")

    labelers_dir = CONFIG["labelers_dir"]
    metadata_root = CONFIG["metadata_root"]
    image_root = CONFIG["image_root"]
    out_dir = CONFIG["out_dir"]

    # 1) Aggregate labels (if needed)
    if os.path.exists(labelers_dir) and len(glob.glob(os.path.join(labelers_dir, "*.json"))) > 0:
        try:
            df = aggregate_labelers_to_csv(df_master, labelers_dir)
        except Exception as e:
            print("Aggregation failed or not needed:", e)
            df = df_master
    else:
        df = df_master

    # ensure required columns
    if 'image_path' not in df.columns or 'metadata_path' not in df.columns:
        raise ValueError("master CSV must contain 'image_path' and 'metadata_path' columns.")

    # 2) If master CSV already contains needed metadata columns, use them; otherwise extract from JSONs
    required_cols = ['pitch','yaw','roll','bbox_x0','bbox_y0','bbox_x1','bbox_y1','bbox_w','bbox_h','age_est','gender_score','meta_dominant_emotion']
    facemesh_cols = [f"facemesh_pc_{i}" for i in range(CONFIG["facemesh_pca_n"])]
    have_required = all(c in df.columns for c in required_cols)
    have_facemesh = all(c in df.columns for c in facemesh_cols)


    df_features = df.copy()

    if have_required:
        print("Using metadata columns from master CSV.")
        # ensure numeric dtype for numeric columns
        for c in ['pitch','yaw','roll','bbox_x0','bbox_y0','bbox_x1','bbox_y1','bbox_w','bbox_h','age_est','gender_score','bbox_area']:
            if c in df_features.columns:
                df_features[c] = pd.to_numeric(df_features[c], errors='coerce')

        if not have_facemesh:
            print("Facemesh PCA columns missing in CSV — computing from metadata JSONs if available.")
            # drop old metadata columns to avoid duplicates
            drop_cols = ['pitch','yaw','roll','bbox_x0','bbox_y0','bbox_x1','bbox_y1','bbox_w','bbox_h',
                         'bbox_area','age_est','gender_est','gender_score','meta_dominant_emotion']
            df_features = df_features.drop(columns=[c for c in drop_cols if c in df_features.columns])
            # compute facemesh only if JSONs exist
            any_jsons = any(os.path.exists(os.path.join(metadata_root, p.replace("metadata_matched/",""))) for p in df_features['metadata_path'].dropna().unique())
            if any_jsons:
                df_features = extract_features_from_metadata(df_features, facemesh_pca_n=CONFIG["facemesh_pca_n"],
                                                             metadata_root=metadata_root,
                                                             save_pca_path=os.path.join(out_dir, "facemesh_pca.pkl"))
            else:
                print("No JSONs available to compute facemesh — setting facemesh_pc_* to 0.")
                for c in facemesh_cols:
                    df_features[c] = 0.0
    else:
        print("Required metadata columns missing in CSV; extracting from metadata JSONs.")
        # drop old metadata columns to avoid duplicates
        drop_cols = ['pitch','yaw','roll','bbox_x0','bbox_y0','bbox_x1','bbox_y1','bbox_w','bbox_h',
                     'bbox_area','age_est','gender_est','gender_score','meta_dominant_emotion']
        df_features = df_features.drop(columns=[c for c in drop_cols if c in df_features.columns])
        df_features = extract_features_from_metadata(df_features, facemesh_pca_n=CONFIG["facemesh_pca_n"],
                                                     metadata_root=metadata_root,
                                                     save_pca_path=os.path.join(out_dir, "facemesh_pca.pkl"))

    # Save intermediate dataset
    df_features.to_csv(os.path.join(out_dir, "dataset_with_metadata_features.csv"), index=False)
    print("Saved dataset with metadata features.")

    # 3) Prepare labels
    label_col = None
    if 'maj_attention' in df_features.columns:
        label_col = 'maj_attention'
    elif 'maj_emotion' in df_features.columns:
        label_col = 'maj_emotion'
    else:
        if 'self_attention' in df_features.columns:
            label_col = 'self_attention'
        else:
            raise ValueError("No label column found (maj_attention, maj_emotion, or self_attention). Please provide labels.")

    df_features = df_features[~df_features[label_col].isnull()].reset_index(drop=True)

    # 4) encode labels
    le = LabelEncoder()
    df_features['label_encoded'] = le.fit_transform(df_features[label_col].astype(str))
    num_classes = len(le.classes_)
    print("Label classes:", le.classes_)

    # 5) metadata numeric feature columns to include
    metadata_cols = ['pitch','yaw','roll','bbox_area','age_est','gender_score'] + facemesh_cols
    metadata_cols = [c for c in metadata_cols if c in df_features.columns]
    print("Using metadata columns:", metadata_cols)

    # 6) split dataset (frame-level)
    train_df, test_df = train_test_split(df_features, test_size=0.2, stratify=df_features['label_encoded'], random_state=CONFIG["random_seed"])
    train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df['label_encoded'], random_state=CONFIG["random_seed"])
    print(f"Splits - train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # 7) scale metadata features
    scaler = StandardScaler()
    scaler.fit(train_df[metadata_cols].fillna(0.0).values)
    joblib.dump(scaler, os.path.join(out_dir, "metadata_scaler.pkl"))
    for df_ in (train_df, val_df, test_df):
        df_[metadata_cols] = scaler.transform(df_[metadata_cols].fillna(0.0).values)

    expected_dim = len(metadata_cols)

    img_transforms = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_ds = StudentBehaviorDataset(train_df, image_root=image_root,
                                 metadata_features=metadata_cols, transform=img_transforms,
                                 label_col='label_encoded', target_encoder=None,
                                 expected_metadata_dim=expected_dim)

    val_ds = StudentBehaviorDataset(val_df, image_root=image_root,
                                 metadata_features=metadata_cols, transform=img_transforms,
                                 label_col='label_encoded', target_encoder=None,
                                 expected_metadata_dim=expected_dim)

    test_ds = StudentBehaviorDataset(test_df, image_root=image_root,
                                 metadata_features=metadata_cols, transform=img_transforms,
                                 label_col='label_encoded', target_encoder=None,
                                 expected_metadata_dim=expected_dim)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # 8) build model
    model = HybridNet(metadata_dim=len(metadata_cols), num_classes=num_classes, pretrained=True)
    model = model.to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # 9) training loop
    best_val_f1 = 0.0
    for epoch in range(CONFIG["num_epochs"]):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, CONFIG["device"])
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = eval_epoch(model, val_loader, criterion, CONFIG["device"])
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}  Train loss {train_loss:.4f} acc {train_acc:.4f}  Val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}")

        # save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'le_classes': le.classes_,
                'metadata_cols': metadata_cols
            }, os.path.join(out_dir, CONFIG["save_model_name"]))
            print("Saved best model.")

    # 10) final evaluation on test set
    test_loss, test_acc, test_prec, test_rec, test_f1, y_true, y_pred = eval_epoch(model, test_loader, criterion, CONFIG["device"])
    print("\nTEST RESULTS:")
    print(f"Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", classification_report(y_true, y_pred, zero_division=0, target_names=[str(c) for c in le.classes_]))

    # save test predictions
    test_df = test_df.reset_index(drop=True)
    test_df['pred'] = y_pred
    test_df['true'] = y_true
    test_df['pred_label'] = le.inverse_transform(test_df['pred'])
    test_df['true_label'] = le.inverse_transform(test_df['true'])
    test_df.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)
    print("Saved test predictions.")

if __name__ == "__main__":
    main()
