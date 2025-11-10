# -*- coding: utf-8 -*-
"""
Evaluation metrics
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def eval_metrics(y_true, y_pred):
    """
    Calculate MAE and RMSE
    
    Args:
        y_true: ground truth values
        y_pred: predicted values
    Returns:
        mae, rmse
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    mae = mean_absolute_error(y_true, y_pred)
    
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return mae, rmse


def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error
    
    Args:
        y_true: ground truth values
        y_pred: predicted values
    Returns:
        MAPE (%)
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return mape