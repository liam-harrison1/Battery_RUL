# -*- coding: utf-8 -*-
"""
Training script for baseline models and SAETR with experiment tracking
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
from baseline_models import RNNModel, LSTMModel, GRUModel, TransformerModel
from dataset import (
    load_seq_splits, fit_standard_scaler,
    apply_standard_scaler, SeqDataset
)
from utils.helpers import setup_logger, save_curves

# 实验记录管理
EXPERIMENT_FILE = "experiments.json"

def load_experiments():
    """加载现有实验记录"""
    if os.path.exists(EXPERIMENT_FILE):
        with open(EXPERIMENT_FILE, 'r') as f:
            return json.load(f)
    return []

def record_experiment(config, run_dir, model_name, best_val_mae):
    """记录实验信息到JSON文件"""
    experiments = load_experiments()
    exp_entry = {
        "id": len(experiments) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "val_mae": float(best_val_mae),
        "path": os.path.abspath(run_dir),
        "config": config
    }
    experiments.append(exp_entry)
    
    with open(EXPERIMENT_FILE, 'w') as f:
        json.dump(experiments, f, indent=2)
    return exp_entry["id"]


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


def train_epoch(model, dataloader, optimizer, device, model_type, alpha_recon=0, beta_sparse=0):
    """
    Train for one epoch
    
    Args:
        model: neural network model
        dataloader: training data loader
        optimizer: optimizer
        device: 'cuda' or 'cpu'
        model_type: 'rnn', 'lstm', 'gru', 'transformer', or 'saetr'
        alpha_recon: reconstruction loss weight (only for SAETR)
        beta_sparse: sparsity loss weight (only for SAETR)
    """
    model.train()
    total_loss = 0.0
    
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        
        # Different forward pass and loss for SAETR vs baseline models
        if model_type == 'saetr':
            # SAETR returns (q, xrec, z)
            q, xrec, z = model(xb)
            
            # Combined loss: main + reconstruction + sparsity
            loss_main = pinball_loss(q, yb, qs=model.qs)
            loss_rec = nn.SmoothL1Loss()(xrec, xb)
            loss_sp = z.abs().mean()
            loss = loss_main + alpha_recon * loss_rec + beta_sparse * loss_sp
        else:
            # Baseline models only return q
            q = model(xb)
            
            # Only main loss for baseline models
            loss = pinball_loss(q, yb, qs=model.qs)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, X_val, y_val, y_inv_fn, device, model_type):
    """
    Validate model and return MAE on original scale
    
    Args:
        model: neural network model
        X_val: validation features
        y_val: validation labels (original scale)
        y_inv_fn: inverse transform function
        device: 'cuda' or 'cpu'
        model_type: 'rnn', 'lstm', 'gru', 'transformer', or 'saetr'
    """
    model.eval()
    with torch.no_grad():
        xv = torch.from_numpy(X_val).to(device)
        
        # Different forward pass for SAETR vs baseline models
        if model_type == 'saetr':
            qv, _, _ = model(xv)
        else:
            qv = model(xv)
        
        # Use median quantile (index 1) for prediction
        pred_md = y_inv_fn(qv[:, 1].cpu().numpy())
        val_mae = np.mean(np.abs(pred_md - y_val))
    
    return val_mae


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create experiment directory with model name
    base_dir = config['paths']['experiment_base_dir']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"train_run_{args.model}_{timestamp}")
    
    # Define checkpoint and results directories
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    results_dir = os.path.join(run_dir, "results")
    
    # Update config with new paths
    config['paths']['checkpoint_dir'] = checkpoint_dir
    config['paths']['results_dir'] = results_dir
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    logger = setup_logger(os.path.join(results_dir, 'train.log'))
    logger.info("="*60)
    logger.info(f"Starting training: Model = {args.model.upper()}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")
    logger.info("="*60)
    
    # Load sequences
    logger.info("Loading data...")
    processed_dir = config['paths']['processed_dir']
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_seq_splits(processed_dir)
    logger.info(f"Train: X{Xtr.shape}, y{ytr.shape}")
    logger.info(f"Val:   X{Xva.shape}, y{yva.shape}")
    logger.info(f"Test:  X{Xte.shape}, y{yte.shape}")
    
    # Feature standardization
    logger.info("Fitting StandardScaler on training data...")
    sc = fit_standard_scaler(Xtr)
    Xtr = apply_standard_scaler(sc, Xtr)
    Xva = apply_standard_scaler(sc, Xva)
    
    # Save scaler
    scaler_path = os.path.join(checkpoint_dir, "scaler.json")
    with open(scaler_path, "w") as f:
        json.dump({
            "mean": sc.mean_.tolist(),
            "scale": sc.scale_.tolist()
        }, f)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # Target transformation
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
    
    # Model initialization based on model type
    logger.info(f"Initializing {args.model.upper()} model...")
    quantiles = tuple(config['model_params']['quantiles'])
    
    if args.model == 'saetr':
        model = SAETR(
            F,
            sae_dim=config['model_params']['sae_dim'],
            d_model=config['model_params']['d_model'],
            nhead=config['model_params']['nhead'],
            nlayers=config['model_params']['nlayers'],
            use_cnn=config['model_params']['use_cnn'],
            cnn_channels=config['model_params']['cnn_channels'],
            qs=quantiles
        ).to(device)
    elif args.model == 'rnn':
        model = RNNModel(
            F,
            hidden_dim=128,
            num_layers=2,
            dropout=0.15,
            qs=quantiles
        ).to(device)
    elif args.model == 'lstm':
        model = LSTMModel(
            F,
            hidden_dim=128,
            num_layers=2,
            dropout=0.15,
            qs=quantiles
        ).to(device)
    elif args.model == 'gru':
        model = GRUModel(
            F,
            hidden_dim=128,
            num_layers=2,
            dropout=0.15,
            qs=quantiles
        ).to(device)
    elif args.model == 'transformer':
        model = TransformerModel(
            F,
            d_model=128,
            nhead=4,
            nlayers=2,
            dropout=0.15,
            qs=quantiles
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train_params']['learning_rate'],
        weight_decay=config['train_params']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_val_mae = 1e9
    bad_epochs = 0
    patience = config['train_params']['patience']
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    log_csv = os.path.join(results_dir, "training_log.csv")
    
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_mae", "lr"])
    
    logger.info("Starting training...")
    
    # Get loss weights for SAETR (only used if model_type == 'saetr')
    alpha_recon = config['train_params'].get('alpha_recon', 0)
    beta_sparse = config['train_params'].get('beta_sparse', 0)
    
    for epoch in range(1, config['train_params']['epochs'] + 1):
        # Train
        train_loss = train_epoch(
            model, dl_tr, optimizer, device, args.model,
            alpha_recon, beta_sparse
        )
        
        # Validate on transformed scale (for model selection)
        val_mae_transformed = validate(model, Xva, yva_t, lambda y: y, device, args.model)
        
        # Also calculate MAE on original scale (for reporting)
        val_mae_original = validate(model, Xva, yva, y_inv, device, args.model)
        logger.info(f"Validation MAE (transformed scale): {val_mae_transformed:.4f}, (original scale): {val_mae_original:.4f}")
        
        # Use transformed scale MAE for model selection
        val_mae = val_mae_transformed
        
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
        'model_name': args.model,  # Critical: save model type for prediction script
        'model_params': config['model_params'],
        'train_params': config['train_params'],
        'y_transform': y_mode,
        'best_val_mae': float(best_val_mae)
    }
    train_config.update(extra_cfg)  # Add y_mu/y_sd if standardized
    
    config_path = os.path.join(checkpoint_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(train_config, f, indent=2)
    
    logger.info("="*60)
    logger.info(f"Training completed!")
    logger.info(f"Model: {args.model.upper()}")
    logger.info(f"Best validation MAE: {best_val_mae:.4f}")
    logger.info(f"Model saved to: {best_model_path}")
    logger.info("="*60)
    
    # 记录实验信息
    exp_id = record_experiment(train_config, run_dir, args.model, best_val_mae)
    logger.info(f"Experiment recorded with ID: {exp_id}")
    
    # Generate training curves
    logger.info("Generating training curves...")
    save_curves(log_csv, results_dir)
    logger.info("Training curves saved to results directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train baseline models or SAETR for battery RUL prediction"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=['rnn', 'lstm', 'gru', 'transformer', 'saetr'],
        help="Model to train: rnn, lstm, gru, transformer, or saetr"
    )
    args = parser.parse_args()
    
    main(args)