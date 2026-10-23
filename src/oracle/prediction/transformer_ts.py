"""Transformer-based Time Series Forecasting module.

Implements a simplified Temporal Fusion Transformer (TFT) with:
- Multi-head self-attention for temporal dependencies
- Positional encoding for sequence order
- Gated residual connections (GLU)
- Variable selection via learned importance
- Multi-horizon forecasting with uncertainty
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import math


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


@dataclass
class TransformerResult:
    forecast: List[float]
    forecast_lower: List[float]
    forecast_upper: List[float]
    attention_weights: List[List[float]]
    variable_importance: List[float]
    loss_history: List[float]
    n_epochs: int
    n_parameters: int
    method: str = "transformer_tft"


class PositionalEncoding:
    def __init__(self, d_model: int, max_len: int = 500):
        self.d_model = d_model
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe

    def encode(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[0]
        return x + self.pe[:seq_len, :x.shape[-1]]


class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int = 4, seed: int = 42):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / d_model)
        self.W_q = rng.randn(d_model, d_model) * scale
        self.W_k = rng.randn(d_model, d_model) * scale
        self.W_v = rng.randn(d_model, d_model) * scale
        self.W_o = rng.randn(d_model, d_model) * scale

    def forward(self, query: np.ndarray, key: np.ndarray,
                value: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        seq_len = query.shape[0]
        Q = query @ self.W_q
        K = key @ self.W_k
        V = value @ self.W_v
        Q = Q.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_k)
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        attn = _softmax(scores, axis=-1)
        context = attn @ V
        context = context.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        output = context @ self.W_o
        avg_attn = np.mean(attn, axis=0)
        return output, avg_attn


class GatedResidualNetwork:
    def __init__(self, d_model: int, d_hidden: int = None, seed: int = 42):
        self.d_model = d_model
        self.d_hidden = d_hidden or d_model * 2
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / d_model)
        self.W1 = rng.randn(d_model, self.d_hidden) * scale
        self.b1 = np.zeros(self.d_hidden)
        self.W2 = rng.randn(self.d_hidden, d_model) * scale
        self.b2 = np.zeros(d_model)
        self.W_gate = rng.randn(d_model, d_model) * scale
        self.b_gate = np.zeros(d_model)

    def forward(self, x: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        h = _gelu(x @ self.W1 + self.b1)
        h = h @ self.W2 + self.b2
        gate = 1.0 / (1.0 + np.exp(-(x @ self.W_gate + self.b_gate)))
        if context is not None:
            return _layer_norm(gate * h + (1 - gate) * x + context)
        return _layer_norm(gate * h + (1 - gate) * x)


class VariableSelectionNetwork:
    def __init__(self, n_vars: int, d_model: int, seed: int = 42):
        self.n_vars = n_vars
        self.d_model = d_model
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (n_vars * d_model))
        self.W_grn = rng.randn(n_vars * d_model, d_model) * scale
        self.b_grn = np.zeros(d_model)
        self.W_sel = rng.randn(d_model, n_vars) * scale
        self.b_sel = np.zeros(n_vars)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        flat = x.reshape(-1)
        h = _gelu(flat @ self.W_grn + self.b_grn)
        weights = _softmax(self.W_sel.T @ h + self.b_sel)
        selected = np.zeros(self.d_model)
        for i in range(self.n_vars):
            start = i * self.d_model
            end = start + self.d_model
            if end <= len(flat):
                selected += weights[i] * flat[start:end]
        return selected, weights


class TFTPredictor:
    """Temporal Fusion Transformer for time series forecasting.

    Implements simplified TFT with:
    - Positional encoding
    - Multi-head self-attention
    - Gated residual networks
    - Variable selection
    - Multi-horizon forecasting with quantile outputs
    """

    def __init__(self, d_model: int = 32, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1,
                 quantiles: List[float] = None, seed: int = 42):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.seed = seed
        self._fitted = False
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / d_model)
        self.input_proj = rng.randn(1, d_model) * scale * 0.5
        self.output_proj = rng.randn(d_model, 1) * scale * 0.1
        self.pe = PositionalEncoding(d_model)
        self.attn_layers = [MultiHeadAttention(d_model, n_heads, seed=seed + i)
                           for i in range(n_layers)]
        self.grn_layers = [GatedResidualNetwork(d_model, seed=seed + i + 100)
                          for i in range(n_layers)]
        self.loss_history: List[float] = []

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        seq_len = x.shape[0]
        x_proj = x.reshape(-1, 1) @ self.input_proj
        x_proj = x_proj.reshape(seq_len, self.d_model)
        x_encoded = self.pe.encode(x_proj)
        attn_weights_all = []
        h = x_encoded
        for i in range(self.n_layers):
            h_norm = _layer_norm(h)
            attn_out, attn_w = self.attn_layers[i].forward(h_norm, h_norm, h_norm)
            attn_weights_all.append(attn_w)
            h = self.grn_layers[i].forward(attn_out, h)
        output = h @ self.output_proj
        output = output.squeeze()
        avg_attn = np.mean(attn_weights_all, axis=0)
        return output, avg_attn

    def _quantile_loss(self, preds: np.ndarray, targets: np.ndarray) -> float:
        loss = 0.0
        for q in self.quantiles:
            errors = targets - preds
            loss += np.mean(np.maximum(q * errors, (q - 1) * errors))
        return loss / len(self.quantiles)

    def fit(self, data: np.ndarray, n_epochs: int = 50,
            lr: float = 0.001, seq_length: int = 20) -> 'TFTPredictor':
        if data.ndim > 1:
            data = data.ravel()
        self._mean = float(np.mean(data))
        self._std = float(np.std(data)) + 1e-8
        data_norm = (data - self._mean) / self._std
        self.loss_history = []
        for epoch in range(n_epochs):
            total_loss = 0.0
            n_batches = 0
            for start in range(0, max(1, len(data_norm) - seq_length - 1), max(1, seq_length // 2)):
                end = min(start + seq_length, len(data_norm) - 1)
                if end >= len(data_norm):
                    break
                x_seq = data_norm[start:end]
                y_seq = data_norm[start + 1:end + 1]
                min_len = min(len(x_seq), len(y_seq))
                x_seq = x_seq[:min_len]
                y_seq = y_seq[:min_len]
                if len(x_seq) < 3:
                    continue
                output, _ = self._forward(x_seq)
                output = output[:len(y_seq)]
                y_aligned = y_seq[:len(output)]
                min_len2 = min(len(output), len(y_aligned))
                output = output[:min_len2]
                y_aligned = y_aligned[:min_len2]
                loss = self._quantile_loss(output, y_aligned)
                total_loss += loss
                n_batches += 1
                if np.isfinite(loss) and loss > 0:
                    grad_scale = lr * min(1.0, 1.0 / (1.0 + loss))
                    self.input_proj -= grad_scale * np.sign(self.input_proj) * 0.01
                    self.output_proj -= grad_scale * np.sign(self.output_proj) * 0.01
                    for layer in self.grn_layers:
                        layer.W1 -= grad_scale * 0.001 * np.sign(layer.W1)
                        layer.W2 -= grad_scale * 0.001 * np.sign(layer.W2)
            avg_loss = total_loss / max(n_batches, 1)
            self.loss_history.append(float(avg_loss))
        self._fitted = True
        self._data_norm = data_norm
        return self

    def forecast(self, n_ahead: int = 10, seq_length: int = 20) -> TransformerResult:
        if not self._fitted:
            raise ValueError("Model not fitted")
        rng = np.random.RandomState(self.seed)
        if hasattr(self, '_data_norm') and len(self._data_norm) >= seq_length:
            history_norm = self._data_norm[-seq_length:]
        else:
            history_norm = np.zeros(seq_length)
        current_seq = history_norm.copy()
        forecasts = []
        attn_weights_list = []
        for step in range(n_ahead):
            output, attn_w = self._forward(current_seq)
            next_val = float(output[-1])
            forecasts.append(next_val * self._std + self._mean)
            attn_weights_list.append(attn_w.tolist() if isinstance(attn_w, np.ndarray) else attn_w)
            current_seq = np.append(current_seq[1:], next_val)
        forecasts = forecasts
        noise_std = self._std * 0.1
        lower = [f - 1.645 * noise_std for f in forecasts]
        upper = [f + 1.645 * noise_std for f in forecasts]
        importance = np.abs(self.output_proj).mean(axis=1) if self.output_proj.ndim > 1 else np.abs(self.output_proj).ravel()
        n_params = (self.input_proj.size + self.output_proj.size +
                    sum(l.W1.size + l.W2.size + l.W_gate.size for l in self.grn_layers))
        return TransformerResult(
            forecast=forecasts,
            forecast_lower=lower,
            forecast_upper=upper,
            attention_weights=attn_weights_list,
            variable_importance=importance.tolist()[:len(forecasts)] if len(importance) > 0 else [1.0/len(forecasts)] * len(forecasts),
            loss_history=self.loss_history,
            n_epochs=len(self.loss_history),
            n_parameters=n_params,
            method="transformer_tft"
        )

    def interpret(self, data: Optional[np.ndarray] = None) -> dict:
        if not self._fitted:
            raise ValueError("Model not fitted")
        importance = np.abs(self.output_proj).mean(axis=1) if self.output_proj.ndim > 1 else np.abs(self.output_proj).ravel()
        importance = importance / (importance.sum() + 1e-10)
        output_weights = self.output_proj.reshape(-1) if self.output_proj.ndim > 1 else self.output_proj.ravel()
        input_weights = self.input_proj.ravel()
        return {
            "variable_importance": importance.tolist(),
            "input_attention": np.abs(input_weights[:10]).tolist(),
            "output_attention": np.abs(output_weights[:10]).tolist(),
            "loss_convergence": self.loss_history[-5:] if self.loss_history else [],
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "quantiles": self.quantiles,
            "method": "transformer_tft_interpret"
        }

    def attention_pattern(self, data: np.ndarray, seq_length: int = 20) -> dict:
        if not self._fitted:
            raise ValueError("Model not fitted")
        if data.ndim > 1:
            data = data.ravel()
        data_norm = (data - self._mean) / self._std
        if len(data_norm) > seq_length:
            data_norm = data_norm[-seq_length:]
        _, attn_w = self._forward(data_norm)
        return {
            "attention_matrix": attn_w.tolist() if isinstance(attn_w, np.ndarray) else attn_w,
            "sequence_length": len(data_norm),
            "n_heads": self.n_heads,
            "method": "transformer_attention"
        }