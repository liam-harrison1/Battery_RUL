# -*- coding: utf-8 -*-
"""
Baseline Models for Battery RUL Prediction
Implements RNN, LSTM, GRU, and Transformer models for comparison with SAETR
"""
import torch
import torch.nn as nn
from model import QuantileHead, SinusoidalPositionalEncoding, TransEncoder


class RNNModel(nn.Module):
    """
    RNN-based model for RUL prediction
    Uses vanilla RNN with QuantileHead for multi-quantile regression
    """
    def __init__(self, F, hidden_dim=128, num_layers=2, dropout=0.15, qs=(0.1, 0.5, 0.9)):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=F,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.head = QuantileHead(hidden_dim, qs=qs)
        self.qs = qs
    
    def forward(self, x):
        """
        Args:
            x: (B, L, F) - input sequences
        Returns:
            q: (B, num_quantiles) - predicted quantiles
        """
        # RNN forward pass
        # output: (B, L, hidden_dim), h_n: (num_layers, B, hidden_dim)
        _, h_n = self.rnn(x)
        
        # Use last layer's hidden state
        h_last = h_n[-1]  # (B, hidden_dim)
        
        # Predict quantiles
        q = self.head(h_last)  # (B, num_quantiles)
        
        return q


class LSTMModel(nn.Module):
    """
    LSTM-based model for RUL prediction
    Uses LSTM with QuantileHead for multi-quantile regression
    """
    def __init__(self, F, hidden_dim=128, num_layers=2, dropout=0.15, qs=(0.1, 0.5, 0.9)):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=F,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.head = QuantileHead(hidden_dim, qs=qs)
        self.qs = qs
    
    def forward(self, x):
        """
        Args:
            x: (B, L, F) - input sequences
        Returns:
            q: (B, num_quantiles) - predicted quantiles
        """
        # LSTM forward pass
        # output: (B, L, hidden_dim), (h_n, c_n): both (num_layers, B, hidden_dim)
        _, (h_n, _) = self.lstm(x)
        
        # Use last layer's hidden state
        h_last = h_n[-1]  # (B, hidden_dim)
        
        # Predict quantiles
        q = self.head(h_last)  # (B, num_quantiles)
        
        return q


class GRUModel(nn.Module):
    """
    GRU-based model for RUL prediction
    Uses GRU with QuantileHead for multi-quantile regression
    """
    def __init__(self, F, hidden_dim=128, num_layers=2, dropout=0.15, qs=(0.1, 0.5, 0.9)):
        super().__init__()
        self.gru = nn.GRU(
            input_size=F,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.head = QuantileHead(hidden_dim, qs=qs)
        self.qs = qs
    
    def forward(self, x):
        """
        Args:
            x: (B, L, F) - input sequences
        Returns:
            q: (B, num_quantiles) - predicted quantiles
        """
        # GRU forward pass
        # output: (B, L, hidden_dim), h_n: (num_layers, B, hidden_dim)
        _, h_n = self.gru(x)
        
        # Use last layer's hidden state
        h_last = h_n[-1]  # (B, hidden_dim)
        
        # Predict quantiles
        q = self.head(h_last)  # (B, num_quantiles)
        
        return q


class TransformerModel(nn.Module):
    """
    Transformer-based model for RUL prediction (without Sparse Autoencoder)
    Uses pure Transformer encoder with QuantileHead for multi-quantile regression
    Reuses SinusoidalPositionalEncoding and TransEncoder from model.py
    """
    def __init__(self, F, d_model=128, nhead=4, nlayers=2, dim_ff=256, 
                 dropout=0.15, qs=(0.1, 0.5, 0.9)):
        super().__init__()
        # Project input features to d_model dimension
        self.proj = nn.Linear(F, d_model)
        
        # Reuse positional encoding from SAETR model
        self.pos = SinusoidalPositionalEncoding(d_model)
        
        # Reuse Transformer encoder from SAETR model
        self.tr = TransEncoder(d_model, nhead, nlayers, dim_ff=dim_ff, dropout=dropout)
        
        # Quantile regression head
        self.head = QuantileHead(d_model, qs=qs)
        self.qs = qs
    
    def forward(self, x):
        """
        Args:
            x: (B, L, F) - input sequences
        Returns:
            q: (B, num_quantiles) - predicted quantiles
        """
        # Project input to d_model dimension
        h = self.proj(x)  # (B, L, d_model)
        
        # Add positional encoding
        h = self.pos(h)  # (B, L, d_model)
        
        # Transformer encoder (returns last timestep encoding)
        h_last = self.tr(h)  # (B, d_model)
        
        # Predict quantiles
        q = self.head(h_last)  # (B, num_quantiles)
        
        return q