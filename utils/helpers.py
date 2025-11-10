# -*- coding: utf-8 -*-
"""
Helper functions for visualization and logging
"""
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def setup_logger(log_file=None):
    """
    Setup logger for the project
    
    Args:
        log_file: path to log file (optional)
    Returns:
        logger object
    """
    logger = logging.getLogger("RUL_Prediction")
    logger.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


def save_curves(log_csv, out_dir):
    """
    Plot and save training curves
    
    Args:
        log_csv: path to training log CSV file
        out_dir: output directory for plots
    """
    df = pd.read_csv(log_csv)
    
    # Loss + validation curve
    plt.figure(figsize=(7, 4))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_mae"], label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(alpha=0.8)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200)
    plt.close()
    
    # Validation MAE only
    plt.figure(figsize=(7, 4))
    plt.plot(df["epoch"], df["val_mae"], label="Val MAE", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Validation MAE")
    plt.legend()
    plt.grid(alpha=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_mae_curve.png"), dpi=200)
    plt.close()


def scatter_plot(y_true, y_pred, out_path, title="Pred vs True"):
    """
    Create scatter plot of predictions vs ground truth
    
    Args:
        y_true: ground truth values
        y_pred: predicted values
        out_path: output file path
        title: plot title
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=8, alpha=0.6, edgecolors='k', linewidths=0.5)
    
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel("True RUL", fontsize=12)
    plt.ylabel("Predicted RUL", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(alpha=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def quantile_coverage(y_true, q_lo, q_md, q_hi, out_path):
    """
    Visualize quantile prediction intervals
    
    Args:
        y_true: ground truth
        q_lo: lower quantile predictions
        q_md: median predictions
        q_hi: upper quantile predictions
        out_path: output path for histogram
    Returns:
        coverage: fraction of true values within interval
    """
    cover = np.mean((y_true >= q_lo) & (y_true <= q_hi))
    
    plt.figure(figsize=(6, 4))
    width = q_hi - q_lo
    plt.hist(width, bins=40, edgecolor='k', alpha=0.85)
    plt.title(f"Interval Width (Coverage={cover:.3f})")
    plt.xlabel("q0.9 - q0.1")
    plt.ylabel("Count")
    plt.grid(alpha=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    
    return float(cover)


def plot_battery_predictions(pred_df, out_dir):
    """
    Plot individual battery prediction curves
    
    Args:
        pred_df: DataFrame with columns [battery_id, end_cycle, y_true, y_pred, q_lo, q_hi]
        out_dir: output directory
    """
    os.makedirs(out_dir, exist_ok=True)
    
    for bid, g in pred_df.groupby("battery_id"):
        g = g.sort_values("end_cycle")
        
        plt.figure(figsize=(7, 4))
        plt.plot(g["end_cycle"], g["y_true"], label="True RUL", marker='o', markersize=3)
        plt.plot(g["end_cycle"], g["y_pred"], label="Pred (q50)", marker='s', markersize=3)
        plt.fill_between(g["end_cycle"], g["q_lo"], g["q_hi"], 
                        alpha=0.2, label="q10~q90")
        
        plt.xlabel("Cycle")
        plt.ylabel("RUL")
        plt.title(f"Battery {bid}")
        plt.legend()
        plt.grid(alpha=0.8)
        plt.tight_layout()
        
        out_path = os.path.join(out_dir, f"{bid}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()