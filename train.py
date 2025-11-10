# -*- coding: utf-8 -*-
"""
Training script for SAETR model
Combines training and validation with best model selection
"""
import os
import sys
import csv
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from datetime import datetime
from model import SAETR, pinball_loss
from dataset import (
    load_seq_splits, fit_standard_scaler, 
    apply_standard_scaler, SeqDataset
)
from utils.helpers import setup_logger, save_curves


def make_y_transformer(mode, y_train):
    """
    Create transformation functions for target variable
    
    Args:
        mode: 'none', 'standardize', or 'log1p'
        y_train: training labels for fitting
    Returns:
        (transform_fn, inverse_fn), config_dict
    """
    if mode == "none":
        return (lambda y: y, lambda y: y), {}
    
    if mode == "standardize":
        mu = float(np.mean(y_train))
        sd = float(np.std(y_train) + 1e-8)
        return (
            lambda y: (y - mu) / sd,
            lambda y: y * sd + mu
        ), {"y_mu": mu, "y_sd": sd}
    
    if mode == "log1p":
        return (
            lambda y: np.log1p(np.clip(y, a_min=0, a_max=None)),
            lambda y: np.expm1(y)
        ), {}
    
    raise ValueError("y_transform must be one of ['none', 'standardize', 'log1p']")


def train_epoch(model, dataloader, optimizer, device, alpha_recon, beta_sparse):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        
        q, xrec, z, sparse_loss = model(xb)
        
        # Combined loss
        loss_main = pinball_loss(q, yb, qs=(0.1, 0.5, 0.9))
        loss_rec = nn.SmoothL1Loss()(xrec, xb)
        # 直接使用模型返回的稀疏损失
        loss = loss_main + alpha_recon * loss_rec + beta_sparse * sparse_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, X_val, y_val, y_inv_fn, device):
    """Validate model and return MAE on original scale"""
    model.eval()
    with torch.no_grad():
        xv = torch.from_numpy(X_val).to(device)
        qv, _, _ = model(xv)
        pred_md = y_inv_fn(qv[:, 1].cpu().numpy())  # Use median quantile
        val_mae = np.mean(np.abs(pred_md - y_val))
    return val_mae


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 获取实验的根目录
    base_dir = config['paths']['experiment_base_dir']
    
    # 2. 创建带时间戳的本次运行的专属目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"train_run_{timestamp}")
    
    # 3. 在专属目录内，定义好 checkpoints 和 results 目录的路径
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    results_dir = os.path.join(run_dir, "results")
    
    # 4. 为了让后续代码能继续工作，我们将新路径写回到 config 字典中
    config['paths']['checkpoint_dir'] = checkpoint_dir
    config['paths']['results_dir'] = results_dir


    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    
    logger = setup_logger(os.path.join(config['paths']['results_dir'], 'train.log'))
    logger.info("="*60)
    logger.info("Starting training with configuration:")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")
    logger.info("="*60)
    
    # 1) Load sequences
    logger.info("Loading data...")
    processed_dir = config['paths']['processed_dir']
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_seq_splits(processed_dir)
    logger.info(f"Train: X{Xtr.shape}, y{ytr.shape}")
    logger.info(f"Val:   X{Xva.shape}, y{yva.shape}")
    logger.info(f"Test:  X{Xte.shape}, y{yte.shape}")
    
    # 2) Feature standardization
    logger.info("Fitting StandardScaler on training data...")
    sc = fit_standard_scaler(Xtr)
    Xtr = apply_standard_scaler(sc, Xtr)
    Xva = apply_standard_scaler(sc, Xva)
    
    # Save scaler
    scaler_path = os.path.join(config['paths']['checkpoint_dir'], "scaler.json")
    with open(scaler_path, "w") as f:
        json.dump({
            "mean": sc.mean_.tolist(),
            "scale": sc.scale_.tolist()
        }, f)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # 3) Target transformation
    y_mode = config['train_params']['y_transform']
    (y_tf, y_inv), extra_cfg = make_y_transformer(y_mode, ytr)
    ytr_t = y_tf(ytr).astype(np.float32)
    yva_t = y_tf(yva).astype(np.float32)
    logger.info(f"Applied y_transform: {y_mode}")
    
    # DataLoaders
    batch_size = config['train_params']['batch_size']
    dl_tr = DataLoader(SeqDataset(Xtr, ytr_t), batch_size=batch_size, 
                       shuffle=True, drop_last=True)
    dl_va = DataLoader(SeqDataset(Xva, yva_t), batch_size=batch_size, 
                       shuffle=False)
    
    L, F = Xtr.shape[1], Xtr.shape[2]
    logger.info(f"Sequence length: {L}, Features: {F}")
    
    # 4) Model and optimizer
    logger.info("Initializing model...")
    model = SAETR(
        F,
        sae_dim=config['model_params']['sae_dim'],
        d_model=config['model_params']['d_model'],
        nhead=config['model_params']['nhead'],
        nlayers=config['model_params']['nlayers'],
        use_cnn=config['model_params']['use_cnn'],
        cnn_channels=config['model_params']['cnn_channels'],
        qs=tuple(config['model_params']['quantiles'])
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train_params']['learning_rate'],
        weight_decay=config['train_params']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5) Training loop
    best_val_mae = 1e9
    bad_epochs = 0
    patience = config['train_params']['patience']
    best_model_path = os.path.join(config['paths']['checkpoint_dir'], "best_model.pth")
    log_csv = os.path.join(config['paths']['results_dir'], "training_log.csv")
    
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_mae", "lr"])
    
    logger.info("Starting training...")
    for epoch in range(1, config['train_params']['epochs'] + 1):
        # Train
        train_loss = train_epoch(
            model, dl_tr, optimizer, device,
            config['train_params']['alpha_recon'],
            config['train_params']['beta_sparse']
        )
        
        # Validate
        val_mae = validate(model, Xva, yva, y_inv, device)
        
        # Learning rate scheduling
        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_mae, current_lr])
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            bad_epochs = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
            }, best_model_path)
            logger.info(f"Epoch {epoch:03d} | Loss={train_loss:.4f} | Val MAE={val_mae:.4f} ✓ Best")
        else:
            bad_epochs += 1
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch:03d} | Loss={train_loss:.4f} | Val MAE={val_mae:.4f} | lr={current_lr:.2e}")
        
        # Early stopping
        if bad_epochs >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Save training configuration
    train_config = {
        'model_params': config['model_params'],
        'train_params': config['train_params'],
        'y_transform': y_mode,
        'best_val_mae': float(best_val_mae)
    }
    train_config.update(extra_cfg)  # Add y_mu/y_sd if standardized
    
    config_path = os.path.join(config['paths']['checkpoint_dir'], "train_config.json")
    with open(config_path, "w") as f:
        json.dump(train_config, f, indent=2)
    
    logger.info("="*60)
    logger.info(f"Training completed!")
    logger.info(f"Best validation MAE: {best_val_mae:.4f}")
    logger.info(f"Model saved to: {best_model_path}")
    logger.info("="*60)
    
    # Generate training curves
    logger.info("Generating training curves...")
    save_curves(log_csv, config['paths']['results_dir'])
    logger.info("Training curves saved to results directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAETR model for battery RUL prediction")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    args = parser.parse_args()
    
    main(args)