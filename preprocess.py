# -*- coding: utf-8 -*-
"""
Data preprocessing script
Converts tabular CSV data to sequences for model training
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import yaml

from dataset import add_cycle_frac, make_seq
from utils.helpers import setup_logger


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = config['paths']['data_dir']
    out_dir = config['paths']['processed_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    logger = setup_logger(os.path.join(out_dir, 'preprocess.log'))
    logger.info("="*60)
    logger.info("Starting data preprocessing")
    logger.info(f"Input directory: {data_dir}")
    logger.info(f"Output directory: {out_dir}")
    logger.info("="*60)
    
    # Load raw data
    def load_split(split_name):
        path = os.path.join(data_dir, f"tabular_{split_name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        logger.info(f"Loading {path}...")
        return pd.read_csv(path)
    
    df_train = add_cycle_frac(load_split("train"))
    df_val = add_cycle_frac(load_split("val"))
    df_test = add_cycle_frac(load_split("test"))
    
    logger.info(f"Train: {len(df_train)} rows, {df_train['battery_id'].nunique()} batteries")
    logger.info(f"Val:   {len(df_val)} rows, {df_val['battery_id'].nunique()} batteries")
    logger.info(f"Test:  {len(df_test)} rows, {df_test['battery_id'].nunique()} batteries")
    
    # Extract feature columns
    feature_cols = [c for c in df_train.columns 
                   if c not in ("battery_id", "cycle_index", "y")]
    logger.info(f"Number of features: {len(feature_cols)}")
    
    # Save feature list
    features_json = os.path.join(out_dir, "features.json")
    with open(features_json, "w", encoding="utf-8") as f:
        json.dump({"feature_cols": feature_cols}, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved feature list to {features_json}")
    
    # Get preprocessing parameters
    window = config['data_params']['window']
    stride = config['data_params']['stride']
    min_cycles = config['data_params']['min_cycles']
    
    logger.info(f"Sequence parameters: window={window}, stride={stride}, min_cycles={min_cycles}")
    
    # Create sequences for each split
    for split_name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        logger.info(f"\\nProcessing {split_name} set...")
        
        X, y, meta = make_seq(df, feature_cols, window, stride, min_cycles)
        
        # Save sequences
        np.save(os.path.join(out_dir, f"seq_X_{split_name}.npy"), X)
        np.save(os.path.join(out_dir, f"seq_y_{split_name}.npy"), y)
        meta.to_csv(os.path.join(out_dir, f"meta_{split_name}.csv"), 
                   index=False, encoding="utf-8-sig")
        
        logger.info(f"  Sequences: X{X.shape}, y{y.shape}")
        logger.info(f"  Metadata: {len(meta)} rows")
        logger.info(f"  Batteries included: {meta['battery_id'].nunique()}")
    
    logger.info("="*60)
    logger.info("Preprocessing completed successfully!")
    logger.info(f"Output saved to: {out_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess tabular data into sequences")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    args = parser.parse_args()
    
    main(args)