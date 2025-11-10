# -*- coding: utf-8 -*-
"""
Prediction script for baseline models and SAETR
Evaluates trained model on test set
Supports RNN, LSTM, GRU, Transformer, and SAETR models
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import pandas as pd
import yaml

from datetime import datetime
from model import SAETR
from baseline_models import RNNModel, LSTMModel, GRUModel, TransformerModel
from dataset import load_seq_splits
from utils.metrics import eval_metrics, calculate_mape
from utils.helpers import (
    setup_logger, scatter_plot, quantile_coverage, 
    plot_battery_predictions
)


def load_scaler(checkpoint_dir):
    """Load saved StandardScaler"""
    scaler_path = os.path.join(checkpoint_dir, "scaler.json")
    with open(scaler_path, "r") as f:
        js = json.load(f)
    
    class Scaler:
        pass
    
    sc = Scaler()
    sc.mean_ = np.array(js["mean"])
    sc.scale_ = np.array(js["scale"])
    
    def transform(X):
        N, L, F = X.shape
        return ((X.reshape(N*L, F) - sc.mean_) / sc.scale_).reshape(N, L, F)
    
    sc.transform = transform
    return sc


def inv_y(mode, v, cfg):
    """Inverse transform for target variable"""
    if mode == "standardize":
        mu = cfg.get("y_mu", 0.0)
        sd = cfg.get("y_sd", 1.0)
        return v * sd + mu
    
    if mode == "log1p":
        return np.expm1(v)
    
    return v


def conformal_calibrate(q_lo, q_md, q_hi, y_true, tau=0.8):
    """
    Conformal calibration for prediction intervals
    
    Args:
        q_lo, q_md, q_hi: quantile predictions
        y_true: ground truth
        tau: target coverage rate
    Returns:
        alpha: calibration constant
    """
    rad = np.maximum(q_md - q_lo, q_hi - q_md)
    miss = np.maximum(np.abs(y_true - q_md) - rad, 0.0)
    alpha = np.quantile(miss, tau)
    return float(alpha)


def forward_predict(model, X, device, model_name):
    """
    Forward pass for predictions
    
    Args:
        model: trained model
        X: input features
        device: 'cuda' or 'cpu'
        model_name: 'rnn', 'lstm', 'gru', 'transformer', or 'saetr'
    Returns:
        q: quantile predictions (N, num_quantiles)
    """
    model.eval()
    with torch.no_grad():
        te = torch.from_numpy(X).to(device)
        
        # Different forward pass for SAETR vs baseline models
        if model_name == 'saetr':
            q, _, _ = model(te)
        else:
            q = model(te)
        
        return q.cpu().numpy()


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 根据实验ID或最新实验获取路径
    if not hasattr(args, 'exp_id') and not args.latest:
        raise ValueError("必须指定 --exp-id 或 --latest 参数")
    
    # 加载实验记录
    if not os.path.exists('experiments.json'):
        raise FileNotFoundError("未找到实验记录文件 experiments.json")
    
    with open('experiments.json', 'r') as f:
        experiments = json.load(f)
    
    # 获取目标实验
    if args.latest:
        target_exp = max(experiments, key=lambda x: x['id'])
    else:
        target_exp = next((e for e in experiments if e['id'] == args.exp_id), None)
    
    if not target_exp:
        raise ValueError(f"未找到ID为 {args.exp_id} 的实验记录")
    
    run_dir = target_exp['path']
    
    # 确保配置文件参数存在
    if not hasattr(args, 'config'):
        args.config = 'configs/config.yaml'
    
    # 初始化日志系统
    results_dir = os.path.join(run_dir, "predictions")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "per_battery_traces"), exist_ok=True)
    logger = setup_logger(os.path.join(results_dir, 'predict.log'))
    logger.info(f"使用实验记录 [ID:{target_exp['id']}] {target_exp['model']} (MAE={target_exp['val_mae']:.4f})")
    
    # Define checkpoint directory
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    logger.info("="*60)
    logger.info("Starting prediction on test set")
    logger.info(f"Device: {device}")
    logger.info(f"Run directory: {run_dir}")
    logger.info("="*60)
    
    # Load training config to determine model type
    train_config_path = os.path.join(checkpoint_dir, "train_config.json")
    if not os.path.exists(train_config_path):
        raise FileNotFoundError(f"Training config not found: {train_config_path}")
    
    with open(train_config_path, "r") as f:
        train_cfg = json.load(f)
    
    model_name = train_cfg.get('model_name', 'saetr')  # Default to saetr for backward compatibility
    y_mode = train_cfg.get("y_transform", "none")
    
    logger.info(f"Model type: {model_name.upper()}")
    logger.info(f"Target transform mode: {y_mode}")
    
    # Load scaler
    logger.info("Loading scaler...")
    sc = load_scaler(checkpoint_dir)
    
    # Load data
    logger.info("Loading sequences...")
    processed_dir = config['paths']['processed_dir']
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_seq_splits(processed_dir)
    
    # Load metadata
    meta_te = pd.read_csv(os.path.join(processed_dir, "meta_test.csv"))
    logger.info(f"Test set: X{Xte.shape}, y{yte.shape}, meta={len(meta_te)} rows")
    
    # Standardize
    Xva = sc.transform(Xva).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    
    # Load model based on model_name
    logger.info(f"Loading {model_name.upper()} model...")
    F = Xte.shape[2]
    quantiles = tuple(train_cfg['model_params']['quantiles'])
    
    if model_name == 'saetr':
        model = SAETR(
            F,
            sae_dim=train_cfg['model_params']['sae_dim'],
            d_model=train_cfg['model_params']['d_model'],
            nhead=train_cfg['model_params']['nhead'],
            nlayers=train_cfg['model_params']['nlayers'],
            use_cnn=train_cfg['model_params']['use_cnn'],
            cnn_channels=train_cfg['model_params'].get('cnn_channels', 64),
            qs=quantiles
        ).to(device)
    elif model_name == 'rnn':
        model = RNNModel(
            F,
            hidden_dim=128,
            num_layers=2,
            dropout=0.15,
            qs=quantiles
        ).to(device)
    elif model_name == 'lstm':
        model = LSTMModel(
            F,
            hidden_dim=128,
            num_layers=2,
            dropout=0.15,
            qs=quantiles
        ).to(device)
    elif model_name == 'gru':
        model = GRUModel(
            F,
            hidden_dim=128,
            num_layers=2,
            dropout=0.15,
            qs=quantiles
        ).to(device)
    elif model_name == 'transformer':
        model = TransformerModel(
            F,
            d_model=128,
            nhead=4,
            nlayers=2,
            dropout=0.15,
            qs=quantiles
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    # Load checkpoint
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, "best_model.pth"),
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"Loaded model from epoch {checkpoint['epoch']} with val MAE={checkpoint['val_mae']:.4f}")
    
    # Conformal calibration on validation set
    logger.info("Performing conformal calibration on validation set...")
    tau = config['eval_params']['tau']
    q_val = forward_predict(model, Xva, device, model_name)
    qlo_v = inv_y(y_mode, q_val[:, 0], train_cfg)
    qmd_v = inv_y(y_mode, q_val[:, 1], train_cfg)
    qhi_v = inv_y(y_mode, q_val[:, 2], train_cfg)
    
    alpha = conformal_calibrate(qlo_v, qmd_v, qhi_v, yva, tau=tau)
    logger.info(f"Conformal alpha (tau={tau}): {alpha:.4f}")
    
    # Predict on test set
    logger.info("Predicting on test set...")
    q_te = forward_predict(model, Xte, device, model_name)
    
    pred_lo = inv_y(y_mode, q_te[:, 0], train_cfg) - alpha
    pred_md = inv_y(y_mode, q_te[:, 1], train_cfg)
    pred_hi = inv_y(y_mode, q_te[:, 2], train_cfg) + alpha
    
    # 检查预测值范围是否合理
    logger.info(f"预测值范围: {np.min(pred_md):.2f} ~ {np.max(pred_md):.2f}")
    logger.info(f"真实值范围: {np.min(yte):.2f} ~ {np.max(yte):.2f}")
    
    # 计算指标前确保使用原始尺度
    if y_mode == "standardize":
        logger.info(f"逆变换参数: mu={train_cfg.get('y_mu', 0):.2f}, sd={train_cfg.get('y_sd', 1):.2f}")
    
    # Calculate metrics
    mae, rmse = eval_metrics(yte, pred_md)
    mape = calculate_mape(yte, pred_md)
    
    logger.info("="*60)
    logger.info(f"Test Set Performance ({model_name.upper()}):")
    logger.info(f"  MAE:  {mae:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAPE: {mape:.2f}%")
    logger.info("="*60)
    
    # Scatter plot
    scatter_path = os.path.join(results_dir, "test_scatter_pred_vs_true.png")
    scatter_plot(yte, pred_md, scatter_path,
                title=f"{model_name.upper()}: Prediction vs True (MAE={mae:.3f}, RMSE={rmse:.3f})")
    logger.info(f"Saved scatter plot to {scatter_path}")
    
    # Quantile coverage
    coverage_path = os.path.join(results_dir, "quantile_coverage.png")
    cover = quantile_coverage(yte, pred_lo, pred_md, pred_hi, coverage_path)
    logger.info(f"Prediction interval coverage: {cover:.3f}")
    logger.info(f"Saved coverage plot to {coverage_path}")
    
    # Save predictions
    pred_df = meta_te.copy()
    pred_df["y_true"] = yte
    pred_df["y_pred"] = pred_md
    pred_df["q_lo"] = pred_lo
    pred_df["q_hi"] = pred_hi
    pred_df["abs_error"] = np.abs(yte - pred_md)
    pred_df["rel_error"] = np.abs((yte - pred_md) / (yte + 1e-8)) * 100
    
    pred_csv = os.path.join(results_dir, "predictions_test.csv")
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved detailed predictions to {pred_csv}")
    
    # Per-battery statistics
    battery_stats = pred_df.groupby("battery_id").agg({
        "abs_error": ["mean", "std", "max"],
        "rel_error": ["mean", "max"],
        "y_true": "count"
    }).reset_index()
    battery_stats.columns = ["battery_id", "mae", "std_ae", "max_ae", 
                             "mape", "max_ape", "num_predictions"]
    
    battery_stats_path = os.path.join(results_dir, "battery_statistics.csv")
    battery_stats.to_csv(battery_stats_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved per-battery statistics to {battery_stats_path}")
    
    # Plot individual battery traces
    logger.info("Generating per-battery prediction curves...")
    traces_dir = os.path.join(results_dir, "per_battery_traces")
    plot_battery_predictions(pred_df, traces_dir)
    logger.info(f"Saved {len(pred_df['battery_id'].unique())} battery curves to {traces_dir}")
    
    # Summary report
    summary_path = os.path.join(results_dir, "summary_report.txt")
    with open(summary_path, "w") as f:
        f.write("="*60 + "\n")
        f.write(f"Battery RUL Prediction - Test Set Summary\n")
        f.write(f"Model: {model_name.upper()}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test samples: {len(yte)}\n")
        f.write(f"Unique batteries: {pred_df['battery_id'].nunique()}\n\n")
        f.write("Overall Performance:\n")
        f.write(f"  MAE:  {mae:.4f} cycles\n")
        f.write(f"  RMSE: {rmse:.4f} cycles\n")
        f.write(f"  MAPE: {mape:.2f}%\n\n")
        f.write(f"Prediction Interval (tau={tau}):\n")
        f.write(f"  Coverage: {cover:.3f}\n")
        f.write(f"  Conformal alpha: {alpha:.4f}\n\n")
        f.write("Top 5 Batteries by MAE:\n")
        top5_worst = battery_stats.nlargest(5, "mae")
        for idx, row in top5_worst.iterrows():
            f.write(f"  {row['battery_id']}: MAE={row['mae']:.2f}, "
                   f"MAPE={row['mape']:.2f}%\n")
        f.write("\n")
        f.write("Top 5 Batteries by Accuracy:\n")
        top5_best = battery_stats.nsmallest(5, "mae")
        for idx, row in top5_best.iterrows():
            f.write(f"  {row['battery_id']}: MAE={row['mae']:.2f}, "
                   f"MAPE={row['mape']:.2f}%\n")
    
    logger.info(f"Saved summary report to {summary_path}")
    logger.info("="*60)
    logger.info("Prediction completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict RUL on test set using trained baseline or SAETR model"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--exp-id",
        type=int,
        help="Experiment ID from experiments.json (use 'cli.py list' to view)"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest experiment for prediction"
    )
    args = parser.parse_args()
    
    main(args)