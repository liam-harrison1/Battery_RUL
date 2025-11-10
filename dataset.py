# -*- coding: utf-8 -*-
"""
Data loading and preprocessing utilities
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class SeqDataset(Dataset):
    """PyTorch Dataset for sequences"""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)  # (N, L, F)
        self.y = torch.from_numpy(y)  # (N,)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]


def fit_standard_scaler(X):
    """
    Fit StandardScaler on training data
    
    Args:
        X: (N, L, F) numpy array
    Returns:
        StandardScaler object
    """
    N, L, F = X.shape
    sc = StandardScaler()
    sc.fit(X.reshape(N * L, F))
    return sc


def apply_standard_scaler(sc, X):
    """
    Apply fitted StandardScaler to data
    
    Args:
        sc: fitted StandardScaler
        X: (N, L, F) numpy array
    Returns:
        Normalized data
    """
    N, L, F = X.shape
    Xn = sc.transform(X.reshape(N * L, F)).reshape(N, L, F).astype(np.float32)
    return Xn


def load_seq_splits(seq_dir):
    """
    Load preprocessed sequences
    
    Args:
        seq_dir: directory containing .npy files
    Returns:
        (Xtr, ytr), (Xva, yva), (Xte, yte)
    """
    Xtr = np.load(os.path.join(seq_dir, "seq_X_train.npy"))
    ytr = np.load(os.path.join(seq_dir, "seq_y_train.npy"))
    Xva = np.load(os.path.join(seq_dir, "seq_X_val.npy"))
    yva = np.load(os.path.join(seq_dir, "seq_y_val.npy"))
    Xte = np.load(os.path.join(seq_dir, "seq_X_test.npy"))
    yte = np.load(os.path.join(seq_dir, "seq_y_test.npy"))
    return (Xtr, ytr), (Xva, yva), (Xte, yte)


def load_feature_list(seq_dir):
    """Load feature names from JSON"""
    with open(os.path.join(seq_dir, "features.json"), "r", encoding="utf-8") as f:
        return json.load(f)["feature_cols"]


def add_cycle_frac(df):
    """Add normalized cycle fraction to dataframe"""
    out = []
    for bid, g in df.groupby("battery_id"):
        g = g.sort_values("cycle_index").copy()
        m = float(g["cycle_index"].max())
        g["cycle_frac"] = g["cycle_index"] / (m if m > 0 else 1.0)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def make_seq(df, feature_cols, L, stride, min_len):
    """
    Create sliding window sequences from tabular data
    
    Args:
        df: DataFrame with columns [battery_id, cycle_index, features..., y]
        feature_cols: list of feature column names
        L: window length
        stride: sliding stride
        min_len: minimum battery cycles to include
    Returns:
        X: (N, L, F) sequences
        y: (N,) labels (RUL at last timestep)
        meta: DataFrame with metadata
    """
    X_list, y_list, meta_rows = [], [], []
    
    for bid, g in df.groupby("battery_id"):
        g = g.sort_values("cycle_index").reset_index(drop=True)
        if len(g) < min_len:
            continue
        
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        y = g["y"].to_numpy(dtype=np.float32)
        cyc = g["cycle_index"].to_numpy()
        
        N = (len(g) - L) // stride + 1
        for i in range(max(0, N)):
            s = i * stride
            e = s + L
            if e > len(g):
                break
            
            X_list.append(feats[s:e])
            y_list.append(y[e - 1])  # Label at last timestep
            meta_rows.append({
                "battery_id": bid,
                "start_cycle": int(cyc[s]),
                "end_cycle": int(cyc[e - 1])
            })
    
    if not X_list:
        return (
            np.empty((0, L, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            pd.DataFrame(columns=["battery_id", "start_cycle", "end_cycle"])
        )
    
    return np.stack(X_list, axis=0), np.array(y_list), pd.DataFrame(meta_rows)