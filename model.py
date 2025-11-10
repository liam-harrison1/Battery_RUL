# -*- coding: utf-8 -*-
"""
SAETR Model: Sparse Autoencoder + Transformer for Battery RUL Prediction
"""
import math
import torch
import torch.nn as nn


class SAE(nn.Module):
    """Sparse Autoencoder"""
    def __init__(self, in_dim, hid=64, out_dim=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hid), 
            nn.ReLU(), 
            nn.Linear(hid, out_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(out_dim, hid), 
            nn.ReLU(), 
            nn.Linear(hid, in_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, L, F) - batch_size, sequence_length, feature_dim
        Returns:
            z: (B, L, out_dim) - encoded features
            xrec: (B, L, F) - reconstructed input
            sparse_loss: scalar tensor - L1 regularization term
        """
        z = self.enc(x)
        xrec = self.dec(z)
        sparse_loss = torch.mean(torch.abs(z))  # L1 regularization
        return z, xrec, sparse_loss


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer"""
    def __init__(self, d_model, max_len=4000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """Add positional encoding to input"""
        L = x.size(1)
        return x + self.pe[:, :L, :]


class TransEncoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self, d_model, nhead, nlayers, dim_ff=256, dropout=0.15):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_ff, 
            batch_first=True, 
            dropout=dropout
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, d_model) - last timestep encoding
        """
        h = self.enc(x)
        return h[:, -1, :]  # Return last timestep


class QuantileHead(nn.Module):
    """Quantile regression head"""
    def __init__(self, in_dim, qs=(0.1, 0.5, 0.9)):
        super().__init__()
        self.qs = qs
        self.net = nn.Sequential(
            nn.Linear(in_dim, max(64, in_dim // 2)), 
            nn.ReLU(),
            nn.Linear(max(64, in_dim // 2), len(qs))
        )
    
    def forward(self, h):
        return self.net(h)


def pinball_loss(pred, target, qs=(0.1, 0.5, 0.9)):
    """
    Quantile regression loss (pinball loss)
    
    Args:
        pred: (B, num_quantiles) - predicted quantiles
        target: (B,) - true values
        qs: tuple of quantiles
    Returns:
        loss: scalar
    """
    target = target.unsqueeze(1).repeat(1, len(qs))
    e = target - pred
    loss = 0.0
    for i, q in enumerate(qs):
        loss += torch.mean(torch.maximum(q * e[:, i], (q - 1) * e[:, i]))
    return loss / len(qs)


class SAETR(nn.Module):
    """
    Sparse Autoencoder + Transformer model for RUL prediction
    """
    def __init__(self, F, sae_dim=32, d_model=128, nhead=4, nlayers=2,
                 use_cnn=True, cnn_channels=64, qs=(0.1, 0.5, 0.9)):
        super().__init__()
        self.sae = SAE(F, hid=2 * sae_dim, out_dim=sae_dim)
        self.use_cnn = use_cnn
        
        if use_cnn:
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=sae_dim, out_channels=cnn_channels, 
                         kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(cnn_channels, sae_dim, kernel_size=3, padding=1),
                nn.ReLU()
            )
        
        self.proj = nn.Linear(sae_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.tr = TransEncoder(d_model, nhead, nlayers)
        self.head = QuantileHead(d_model, qs=qs)
        self.qs = qs

    def forward(self, x):
        """
        Args:
            x: (B, L, F) - input sequences
        Returns:
            q: (B, num_quantiles) - predicted quantiles
            xrec: (B, L, F) - reconstructed input
            z: (B, L, sae_dim) - encoded features
        """
        z, xrec, sparse_loss = self.sae(x)
        
        if self.use_cnn:
            z = z.transpose(1, 2)  # (B, C, L)
            z = self.cnn(z)
            z = z.transpose(1, 2)  # (B, L, C)
        
        h = self.proj(z)
        h = self.pos(h)
        h_last = self.tr(h)
        q = self.head(h_last)
        
        return q, xrec, z, sparse_loss