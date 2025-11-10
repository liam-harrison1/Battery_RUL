# -*- coding: utf-8 -*-
"""
Early RUL Prediction Module (Enhanced & Fixed)
Uses only first N cycles to predict entire battery lifespan
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict

from model import SAETR
from dataset import load_seq_splits, apply_standard_scaler
from utils.metrics import eval_metrics, calculate_mape
from utils.helpers import setup_logger


class EarlyPredictor:
    """
    Early RUL prediction using partial cycle data
    
    Key Innovation:
    - Train on full data but test on early cycles only
    - Compare early predictions with full-cycle predictions
    """
    
    def __init__(self, model, scaler, config, device='cpu'):
        self.model = model
        self.scaler = scaler
        self.config = config
        self.device = device
        self.model.eval()
    
    def extract_early_cycles(self, X, y, meta, max_cycles=100):
        """
        Extract sequences from first N cycles only
        
        Args:
            X: (N, L, F) all sequences
            y: (N,) all labels
            meta: DataFrame with start_cycle, end_cycle, battery_id
            max_cycles: only keep sequences ending before this cycle
        
        Returns:
            X_early, y_early, meta_early
        """
        # Filter sequences where end_cycle <= max_cycles
        mask = meta['end_cycle'] <= max_cycles
        
        X_early = X[mask]
        y_early = y[mask]
        meta_early = meta[mask].reset_index(drop=True)
        
        return X_early, y_early, meta_early
    
    def predict_with_sequences(self, X, return_all_quantiles=False):
        """
        Make predictions on sequence data
        
        Args:
            X: (N, L, F) sequence data
            return_all_quantiles: if True, return all quantiles
        
        Returns:
            predictions: array of predictions
        """
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).to(self.device)
            q, _, _ = self.model(X_tensor)
            q = q.cpu().numpy()
        
        if return_all_quantiles:
            return q  # (N, 3)
        else:
            return q[:, 1]  # Return median only
    
    def evaluate_early_prediction(self, X_test, y_test, meta_test, 
                                  X_full, y_full, meta_full,
                                  early_cycle_thresholds=[30, 50, 100]):
        """
        Comprehensive early prediction evaluation with comparison to full-cycle predictions
        
        Args:
            X_test, y_test, meta_test: EARLY cycle test data (filtered)
            X_full, y_full, meta_full: FULL test data (for comparison)
            early_cycle_thresholds: list of cutoff cycles to test
        
        Returns:
            results: dict with metrics for each threshold
        """
        results = {}
        
        # First, get baseline performance on full data
        print(f"\n{'='*70}")
        print("BASELINE: Full-Cycle Prediction Performance")
        print(f"{'='*70}")
        
        q_full = self.predict_with_sequences(X_full, return_all_quantiles=True)
        pred_full = q_full[:, 1]
        
        mae_full, rmse_full = eval_metrics(y_full, pred_full)
        mape_full = calculate_mape(y_full, pred_full)
        coverage_full = np.mean((y_full >= q_full[:, 0]) & (y_full <= q_full[:, 2]))
        
        print(f"  Total samples: {len(X_full)}")
        print(f"  Total batteries: {meta_full['battery_id'].nunique()}")
        print(f"  MAE:  {mae_full:.4f} cycles")
        print(f"  RMSE: {rmse_full:.4f} cycles")
        print(f"  MAPE: {mape_full:.2f}%")
        print(f"  Coverage: {coverage_full:.3f}")
        
        results['full'] = {
            'num_samples': len(X_full),
            'num_batteries': meta_full['battery_id'].nunique(),
            'mae': mae_full,
            'rmse': rmse_full,
            'mape': mape_full,
            'coverage': coverage_full
        }
        
        # Now evaluate at different early cycle thresholds
        print(f"\n{'='*70}")
        print("EARLY PREDICTION EVALUATION")
        print(f"{'='*70}")
        
        for threshold in early_cycle_thresholds:
            print(f"\n{'─'*70}")
            print(f"Using First {threshold} Cycles")
            print(f"{'─'*70}")
            
            # Extract early cycles
            X_early, y_early, meta_early = self.extract_early_cycles(
                X_test, y_test, meta_test, max_cycles=threshold
            )
            
            if len(X_early) == 0:
                print(f"  ⚠ WARNING: No sequences available for threshold {threshold}")
                print(f"     (All sequences end after cycle {threshold})")
                continue
            
            # Predict
            q_early = self.predict_with_sequences(X_early, return_all_quantiles=True)
            pred_early = q_early[:, 1]
            
            # Calculate metrics
            mae_early, rmse_early = eval_metrics(y_early, pred_early)
            mape_early = calculate_mape(y_early, pred_early)
            coverage_early = np.mean((y_early >= q_early[:, 0]) & (y_early <= q_early[:, 2]))
            
            # Calculate degradation compared to full-cycle
            mae_degradation = ((mae_early - mae_full) / mae_full) * 100
            rmse_degradation = ((rmse_early - rmse_full) / rmse_full) * 100
            
            results[threshold] = {
                'num_samples': len(X_early),
                'num_batteries': meta_early['battery_id'].nunique(),
                'mae': mae_early,
                'rmse': rmse_early,
                'mape': mape_early,
                'coverage': coverage_early,
                'mae_degradation_pct': mae_degradation,
                'rmse_degradation_pct': rmse_degradation
            }
            
            # Print results
            print(f"  Samples: {len(X_early)} ({len(X_early)/len(X_full)*100:.1f}% of full)")
            print(f"  Batteries: {meta_early['battery_id'].nunique()}")
            print(f"\n  Performance Metrics:")
            print(f"    MAE:  {mae_early:.4f} cycles (vs {mae_full:.4f} full, {mae_degradation:+.1f}%)")
            print(f"    RMSE: {rmse_early:.4f} cycles (vs {rmse_full:.4f} full, {rmse_degradation:+.1f}%)")
            print(f"    MAPE: {mape_early:.2f}% (vs {mape_full:.2f}% full)")
            print(f"    Coverage: {coverage_early:.3f} (vs {coverage_full:.3f} full)")
            
            # Quality assessment
            if mae_degradation < 10:
                quality = "✓ EXCELLENT"
            elif mae_degradation < 25:
                quality = "○ GOOD"
            elif mae_degradation < 50:
                quality = "△ ACCEPTABLE"
            else:
                quality = "✗ POOR"
            
            print(f"\n  Quality Assessment: {quality}")
        
        return results
    
    def create_per_battery_comparison(self, X_test, y_test, meta_test,
                                     X_full, y_full, meta_full,
                                     early_cycle_cutoff=50):
        """
        Create detailed per-battery comparison between early and full predictions
        
        Returns:
            comparison_df: DataFrame with per-battery results
        """
        print(f"\n{'='*70}")
        print(f"Per-Battery Analysis (Early: {early_cycle_cutoff} cycles vs Full)")
        print(f"{'='*70}\n")
        
        # Get predictions for early cycles
        X_early, y_early, meta_early = self.extract_early_cycles(
            X_test, y_test, meta_test, max_cycles=early_cycle_cutoff
        )
        
        q_early = self.predict_with_sequences(X_early, return_all_quantiles=True)
        pred_early = q_early[:, 1]
        
        # Get predictions for full data
        q_full = self.predict_with_sequences(X_full, return_all_quantiles=True)
        pred_full = q_full[:, 1]
        
        # Merge predictions with metadata
        meta_early['pred_rul'] = pred_early
        meta_early['true_rul'] = y_early
        meta_early['error'] = np.abs(pred_early - y_early)
        meta_early['data_type'] = 'early'
        
        meta_full['pred_rul'] = pred_full
        meta_full['true_rul'] = y_full
        meta_full['error'] = np.abs(pred_full - y_full)
        meta_full['data_type'] = 'full'
        
        # Combine and group by battery
        comparison_list = []
        
        for battery_id in meta_full['battery_id'].unique():
            # Early data for this battery
            early_bat = meta_early[meta_early['battery_id'] == battery_id]
            full_bat = meta_full[meta_full['battery_id'] == battery_id]
            
            if len(early_bat) == 0:
                continue
            
            row = {
                'battery_id': battery_id,
                'early_samples': len(early_bat),
                'full_samples': len(full_bat),
                'early_mae': early_bat['error'].mean(),
                'full_mae': full_bat['error'].mean(),
                'early_rmse': np.sqrt(np.mean(early_bat['error']**2)),
                'full_rmse': np.sqrt(np.mean(full_bat['error']**2)),
            }
            
            # Calculate degradation
            row['mae_degradation'] = row['early_mae'] - row['full_mae']
            row['mae_degradation_pct'] = (row['mae_degradation'] / row['full_mae']) * 100
            
            comparison_list.append(row)
        
        comparison_df = pd.DataFrame(comparison_list)
        comparison_df = comparison_df.sort_values('mae_degradation_pct')
        
        # Print summary
        print(f"{'Battery ID':<15} {'Early MAE':<12} {'Full MAE':<12} {'Degradation':<15} {'Status'}")
        print(f"{'-'*70}")
        
        for _, row in comparison_df.iterrows():
            status = "✓" if row['mae_degradation_pct'] < 25 else "△" if row['mae_degradation_pct'] < 50 else "✗"
            print(f"{row['battery_id']:<15} {row['early_mae']:<12.4f} {row['full_mae']:<12.4f} "
                  f"{row['mae_degradation_pct']:>6.1f}%        {status}")
        
        print(f"\n{'-'*70}")
        print(f"{'AVERAGE':<15} {comparison_df['early_mae'].mean():<12.4f} "
              f"{comparison_df['full_mae'].mean():<12.4f} "
              f"{comparison_df['mae_degradation_pct'].mean():>6.1f}%")
        
        return comparison_df


def visualize_early_vs_full_comparison(results, save_path):
    """
    Create comprehensive visualization comparing early and full predictions
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Early Prediction vs Full-Cycle Prediction Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Prepare data
    thresholds = [k for k in sorted(results.keys()) if k != 'full']
    full_mae = results['full']['mae']
    full_rmse = results['full']['rmse']
    
    # Plot 1: MAE Comparison
    early_mae = [results[t]['mae'] for t in thresholds]
    
    axes[0, 0].plot(thresholds, early_mae, marker='o', linewidth=2.5, 
                   markersize=10, label='Early Prediction', color='#E74C3C')
    axes[0, 0].axhline(y=full_mae, color='#27AE60', linestyle='--', 
                      linewidth=2, label=f'Full-Cycle Baseline ({full_mae:.4f})')
    axes[0, 0].fill_between(thresholds, full_mae, early_mae, 
                           alpha=0.2, color='#E74C3C')
    axes[0, 0].set_xlabel('Early Cycles Used', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('MAE (cycles)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Mean Absolute Error', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3, linestyle='--')
    
    # Plot 2: RMSE Comparison
    early_rmse = [results[t]['rmse'] for t in thresholds]
    
    axes[0, 1].plot(thresholds, early_rmse, marker='s', linewidth=2.5, 
                   markersize=10, label='Early Prediction', color='#3498DB')
    axes[0, 1].axhline(y=full_rmse, color='#27AE60', linestyle='--', 
                      linewidth=2, label=f'Full-Cycle Baseline ({full_rmse:.4f})')
    axes[0, 1].fill_between(thresholds, full_rmse, early_rmse, 
                           alpha=0.2, color='#3498DB')
    axes[0, 1].set_xlabel('Early Cycles Used', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('RMSE (cycles)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Root Mean Square Error', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3, linestyle='--')
    
    # Plot 3: Performance Degradation
    degradation = [results[t]['mae_degradation_pct'] for t in thresholds]
    
    colors = ['#27AE60' if d < 10 else '#F39C12' if d < 25 else '#E74C3C' for d in degradation]
    bars = axes[1, 0].bar(range(len(thresholds)), degradation, color=colors, 
                         alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].axhline(y=25, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 0].set_xticks(range(len(thresholds)))
    axes[1, 0].set_xticklabels(thresholds)
    axes[1, 0].set_xlabel('Early Cycles Used', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('MAE Increase (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Performance Degradation vs Full-Cycle', fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, degradation)):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Sample Count and Coverage
    sample_counts = [results[t]['num_samples'] for t in thresholds]
    coverages = [results[t]['coverage'] * 100 for t in thresholds]
    
    ax4_twin = axes[1, 1].twinx()
    
    line1 = axes[1, 1].plot(thresholds, sample_counts, marker='o', linewidth=2.5,
                           markersize=10, color='#9B59B6', label='Sample Count')
    line2 = ax4_twin.plot(thresholds, coverages, marker='^', linewidth=2.5,
                         markersize=10, color='#E67E22', label='Coverage')
    
    axes[1, 1].set_xlabel('Early Cycles Used', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Number of Samples', fontsize=12, fontweight='bold', color='#9B59B6')
    ax4_twin.set_ylabel('Prediction Interval Coverage (%)', fontsize=12, fontweight='bold', color='#E67E22')
    axes[1, 1].set_title('Data Availability & Prediction Quality', fontsize=13, fontweight='bold')
    axes[1, 1].tick_params(axis='y', labelcolor='#9B59B6')
    ax4_twin.tick_params(axis='y', labelcolor='#E67E22')
    axes[1, 1].grid(alpha=0.3, linestyle='--')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axes[1, 1].legend(lines, labels, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Visualization saved to: {save_path}")


def create_detailed_comparison_table(results, save_path):
    """
    Create detailed comparison table
    """
    rows = []
    
    # Full baseline
    full = results['full']
    rows.append({
        'Data Type': 'Full Cycle (Baseline)',
        'Cycles Used': 'All',
        'Samples': full['num_samples'],
        'Batteries': full['num_batteries'],
        'MAE': full['mae'],
        'RMSE': full['rmse'],
        'MAPE (%)': full['mape'],
        'Coverage': full['coverage'],
        'vs Baseline (%)': 0.0
    })
    
    # Early predictions
    for threshold in sorted([k for k in results.keys() if k != 'full']):
        r = results[threshold]
        rows.append({
            'Data Type': f'Early Prediction',
            'Cycles Used': threshold,
            'Samples': r['num_samples'],
            'Batteries': r['num_batteries'],
            'MAE': r['mae'],
            'RMSE': r['rmse'],
            'MAPE (%)': r['mape'],
            'Coverage': r['coverage'],
            'vs Baseline (%)': r['mae_degradation_pct']
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, float_format='%.4f')
    
    print(f"\n✓ Detailed table saved to: {save_path}")
    
    return df


def main():
    """
    Main function for early prediction evaluation
    """
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Early RUL Prediction Evaluation')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file path')
    parser.add_argument('--early_cycles', type=int, nargs='+', 
                       default=[30, 50, 75, 100],
                       help='List of early cycle thresholds to evaluate')
    parser.add_argument('--data_split', default='test', choices=['test', 'val'],
                       help='Which data split to use for evaluation')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*70)
    print("EARLY RUL PREDICTION EVALUATION SYSTEM")
    print("="*70)
    print(f"Device: {device}")
    print(f"Data Split: {args.data_split}")
    print(f"Early Cycle Thresholds: {args.early_cycles}")
    print("="*70 + "\n")
    
    # Setup paths
    experiment_base = config['paths']['experiment_base_dir']
    
    if not os.path.exists(experiment_base):
        print(f"✗ ERROR: Experiment directory not found: {experiment_base}")
        print("  Please run training first (train.py)")
        return
    
    all_runs = sorted([d for d in os.listdir(experiment_base) if d.startswith('train_run_')])
    
    if len(all_runs) == 0:
        print(f"✗ ERROR: No training runs found in {experiment_base}")
        print("  Please run training first (train.py)")
        return
    
    latest_run = os.path.join(experiment_base, all_runs[-1])
    checkpoint_dir = os.path.join(latest_run, "checkpoints")
    
    print(f"Loading from: {latest_run}\n")
    
    # Create early prediction results directory
    early_pred_dir = os.path.join(latest_run, f"early_prediction_{args.data_split}")
    os.makedirs(early_pred_dir, exist_ok=True)
    
    logger = setup_logger(os.path.join(early_pred_dir, 'early_prediction.log'))
    logger.info("="*70)
    logger.info("Early RUL Prediction Evaluation")
    logger.info(f"Device: {device}")
    logger.info(f"Data Split: {args.data_split}")
    logger.info(f"Early cycle thresholds: {args.early_cycles}")
    logger.info("="*70)
    
    # Load scaler
    scaler_path = os.path.join(checkpoint_dir, "scaler.json")
    
    if not os.path.exists(scaler_path):
        print(f"✗ ERROR: Scaler not found: {scaler_path}")
        return
    
    with open(scaler_path) as f:
        scaler_data = json.load(f)
    
    class Scaler:
        def __init__(self, mean, scale):
            self.mean_ = np.array(mean)
            self.scale_ = np.array(scale)
        
        def transform(self, X):
            N, L, F = X.shape
            return ((X.reshape(N*L, F) - self.mean_) / self.scale_).reshape(N, L, F).astype(np.float32)
    
    scaler = Scaler(scaler_data['mean'], scaler_data['scale'])
    
    # Load model config
    train_config_path = os.path.join(checkpoint_dir, "train_config.json")
    
    if not os.path.exists(train_config_path):
        print(f"✗ ERROR: Training config not found: {train_config_path}")
        return
    
    with open(train_config_path) as f:
        train_cfg = json.load(f)
    
    # Load data
    processed_dir = config['paths']['processed_dir']
    
    if not os.path.exists(processed_dir):
        print(f"✗ ERROR: Processed data directory not found: {processed_dir}")
        print("  Please run preprocessing first (preprocess.py)")
        return
    
    print("Loading sequence data...")
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_seq_splits(processed_dir)
    
    # Select data split
    if args.data_split == 'test':
        X_eval = Xte
        y_eval = yte
        split_name = 'test'
    else:
        X_eval = Xva
        y_eval = yva
        split_name = 'val'
    
    print(f"✓ Loaded {split_name} data: X{X_eval.shape}, y{y_eval.shape}")
    
    # Load metadata
    meta_path = os.path.join(processed_dir, f"meta_{split_name}.csv")
    
    if not os.path.exists(meta_path):
        print(f"✗ ERROR: Metadata not found: {meta_path}")
        return
    
    meta_eval = pd.read_csv(meta_path)
    print(f"✓ Loaded metadata: {len(meta_eval)} rows, {meta_eval['battery_id'].nunique()} batteries")
    
    # Initialize model
    F = X_eval.shape[2]
    
    print(f"\nInitializing SAETR model (F={F})...")
    model = SAETR(
        F,
        sae_dim=train_cfg['model_params']['sae_dim'],
        d_model=train_cfg['model_params']['d_model'],
        nhead=train_cfg['model_params']['nhead'],
        nlayers=train_cfg['model_params']['nlayers'],
        use_cnn=train_cfg['model_params']['use_cnn'],
        cnn_channels=train_cfg['model_params'].get('cnn_channels', 64),
        qs=tuple(train_cfg['model_params']['quantiles'])
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"✗ ERROR: Model checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']} (val_mae={checkpoint['val_mae']:.4f})")
    
    # Standardize data
    print("\nStandardizing data...")
    X_eval = scaler.transform(X_eval)
    print("✓ Data standardized")
    
    # Create predictor
    predictor = EarlyPredictor(model, scaler, config, device)
    
    # Evaluate at different early cycle thresholds
    results = predictor.evaluate_early_prediction(
        X_eval, y_eval, meta_eval,  # For early filtering
        X_eval, y_eval, meta_eval,  # Full data for comparison
        early_cycle_thresholds=args.early_cycles
    )
    
    # Save numerical results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}\n")
    
    # 1. Summary table
    table_path = os.path.join(early_pred_dir, "comparison_table.csv")
    table_df = create_detailed_comparison_table(results, table_path)
    
    # 2. Visualizations
    viz_path = os.path.join(early_pred_dir, "early_vs_full_comparison.png")
    visualize_early_vs_full_comparison(results, viz_path)
    
    # 3. Per-battery analysis (if 50 cycles threshold was used)
    if 50 in args.early_cycles:
        print(f"\n{'='*70}")
        print("PER-BATTERY DETAILED ANALYSIS")
        print(f"{'='*70}")
        
        battery_comparison = predictor.create_per_battery_comparison(
            X_eval, y_eval, meta_eval,
            X_eval, y_eval, meta_eval,
            early_cycle_cutoff=50
        )
        
        battery_path = os.path.join(early_pred_dir, "per_battery_comparison.csv")
        battery_comparison.to_csv(battery_path, index=False, float_format='%.4f')
        print(f"\n✓ Per-battery comparison saved to: {battery_path}")
    
    # 4. Text report
    report_path = os.path.join(early_pred_dir, "EVALUATION_REPORT.txt")
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EARLY RUL PREDICTION EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: SAETR (Sparse Autoencoder + Transformer)\n")
        f.write(f"Data Split: {args.data_split}\n")
        f.write(f"Total Samples: {len(X_eval)}\n")
        f.write(f"Total Batteries: {meta_eval['battery_id'].nunique()}\n\n")
        
        f.write("BASELINE PERFORMANCE (Full-Cycle Data):\n")
        f.write("-"*70 + "\n")
        full = results['full']
        f.write(f"  MAE:  {full['mae']:.4f} cycles\n")
        f.write(f"  RMSE: {full['rmse']:.4f} cycles\n")
        f.write(f"  MAPE: {full['mape']:.2f}%\n")
        f.write(f"  Prediction Interval Coverage: {full['coverage']:.3f}\n\n")
        
        f.write("EARLY PREDICTION PERFORMANCE:\n")
        f.write("-"*70 + "\n\n")
        
        for threshold in sorted([k for k in results.keys() if k != 'full']):
            r = results[threshold]
            f.write(f"Using First {threshold} Cycles:\n")
            f.write(f"  Samples: {r['num_samples']} ({r['num_samples']/full['num_samples']*100:.1f}% of full data)\n")
            f.write(f"  Batteries: {r['num_batteries']}\n")
            f.write(f"  MAE:  {r['mae']:.4f} cycles ({r['mae_degradation_pct']:+.1f}% vs baseline)\n")
            f.write(f"  RMSE: {r['rmse']:.4f} cycles ({r['rmse_degradation_pct']:+.1f}% vs baseline)\n")
            f.write(f"  MAPE: {r['mape']:.2f}%\n")
            f.write(f"  Coverage: {r['coverage']:.3f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("-"*70 + "\n\n")
        
        # Find best threshold
        early_thresholds = [k for k in results.keys() if k != 'full']
        if early_thresholds:
            best_threshold = min(early_thresholds, key=lambda k: results[k]['mae_degradation_pct'])
            best_result = results[best_threshold]
            
            f.write(f"1. Optimal Early Cycle Threshold: {best_threshold} cycles\n")
            f.write(f"   - MAE: {best_result['mae']:.4f} cycles\n")
            f.write(f"   - Only {best_result['mae_degradation_pct']:.1f}% worse than full-cycle baseline\n")
            f.write(f"   - Uses {best_threshold/meta_eval['end_cycle'].max()*100:.1f}% of battery lifetime data\n\n")
            
            # Find worst threshold
            worst_threshold = max(early_thresholds, key=lambda k: results[k]['mae_degradation_pct'])
            worst_result = results[worst_threshold]
            
            f.write(f"2. Earliest Prediction ({worst_threshold} cycles):\n")
            f.write(f"   - MAE: {worst_result['mae']:.4f} cycles\n")
            f.write(f"   - {worst_result['mae_degradation_pct']:.1f}% worse than baseline\n")
            f.write(f"   - Trade-off: Earlier warning vs. prediction accuracy\n\n")
        
        f.write("3. Data Efficiency Analysis:\n")
        for threshold in sorted(early_thresholds):
            r = results[threshold]
            samples_pct = r['num_samples'] / full['num_samples'] * 100
            mae_increase = r['mae_degradation_pct']
            efficiency = samples_pct / (100 + mae_increase) * 100
            f.write(f"   - {threshold} cycles: {samples_pct:.1f}% data → {mae_increase:+.1f}% error (efficiency: {efficiency:.1f})\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("-"*70 + "\n\n")
        
        if early_thresholds:
            # Find sweet spot (good accuracy, reasonably early)
            acceptable_thresholds = [k for k in early_thresholds if results[k]['mae_degradation_pct'] < 25]
            
            if acceptable_thresholds:
                recommended = min(acceptable_thresholds)
                rec_result = results[recommended]
                
                f.write(f"✓ RECOMMENDED: Use {recommended} early cycles\n\n")
                f.write(f"  Rationale:\n")
                f.write(f"  - Achieves MAE of {rec_result['mae']:.4f} cycles\n")
                f.write(f"  - Only {rec_result['mae_degradation_pct']:.1f}% worse than full-cycle prediction\n")
                f.write(f"  - Enables prediction at {recommended/meta_eval['end_cycle'].max()*100:.1f}% of battery life\n")
                f.write(f"  - Provides sufficient early warning for maintenance planning\n\n")
            else:
                f.write(f"⚠ WARNING: All early predictions show >25% degradation\n")
                f.write(f"  - Consider using more early cycles for reliable predictions\n")
                f.write(f"  - Or improve model with additional features\n\n")
        
        f.write("Additional Considerations:\n")
        f.write("  • Monitor prediction intervals to assess uncertainty\n")
        f.write("  • Update predictions as more cycle data becomes available\n")
        f.write("  • Consider ensemble methods for critical applications\n")
        f.write("  • Validate on different battery chemistries/manufacturers\n")
    
    print(f"\n✓ Comprehensive report saved to: {report_path}")
    
    # 5. Print summary to console
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}\n")
    
    print("Performance Comparison:")
    print(f"{'Cycles Used':<15} {'MAE':<12} {'RMSE':<12} {'vs Baseline':<15} {'Quality'}")
    print("-"*70)
    print(f"{'Full (Baseline)':<15} {full['mae']:<12.4f} {full['rmse']:<12.4f} {'---':<15} {'Reference'}")
    
    for threshold in sorted([k for k in results.keys() if k != 'full']):
        r = results[threshold]
        if r['mae_degradation_pct'] < 10:
            quality = "✓ Excellent"
        elif r['mae_degradation_pct'] < 25:
            quality = "○ Good"
        elif r['mae_degradation_pct'] < 50:
            quality = "△ Acceptable"
        else:
            quality = "✗ Poor"
        
        print(f"{str(threshold):<15} {r['mae']:<12.4f} {r['rmse']:<12.4f} "
              f"{r['mae_degradation_pct']:>6.1f}%        {quality}")
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETED SUCCESSFULLY")
    print(f"Results saved to: {early_pred_dir}")
    print("="*70 + "\n")
    
    logger.info("="*70)
    logger.info("Early prediction evaluation completed successfully!")
    logger.info("="*70)


if __name__ == "__main__":
    main()