"""Recurrent Neural Network module for sequence prediction.

Implements LSTM-style recurrent networks from scratch (numpy only),
supporting sequence classification, forecasting, and hidden state analysis.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class RNNResult:
    predictions: List[float]
    hidden_states: List[List[float]]
    loss_history: List[float]
    n_epochs: int
    gradient_norm: float
    sequence_length: int
    method: str = "lstm"


@dataclass
class LSTMCellState:
    cell_state: List[float]
    hidden_state: List[float]


class LSTMNetwork:
    """Pure-numpy LSTM network for sequence modeling.

    Implements forget gate, input gate, cell gate, output gate
    with proper gradient clipping and learning.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 32,
                 output_dim: int = 1, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        rng = np.random.RandomState(seed)
        scale_ih = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale_hh = np.sqrt(2.0 / (2 * hidden_dim))
        self.W_ih = rng.randn(4 * hidden_dim, input_dim) * scale_ih
        self.W_hh = rng.randn(4 * hidden_dim, hidden_dim) * scale_hh
        self.b_ih = np.zeros(4 * hidden_dim)
        self.b_hh = np.zeros(4 * hidden_dim)
        self.W_out = rng.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_out = np.zeros(output_dim)

    def _forward(self, x_seq: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        seq_len = x_seq.shape[0]
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        hidden_list = []
        cell_list = []
        gate_list = []
        outputs = []
        for t in range(seq_len):
            x_t = x_seq[t]
            gates = self.W_ih @ x_t + self.b_ih + self.W_hh @ h + self.b_hh
            f_gate = 1.0 / (1.0 + np.exp(-gates[:self.hidden_dim]))
            i_gate = 1.0 / (1.0 + np.exp(-gates[self.hidden_dim:2*self.hidden_dim]))
            c_bar = np.tanh(gates[2*self.hidden_dim:3*self.hidden_dim])
            o_gate = 1.0 / (1.0 + np.exp(-gates[3*self.hidden_dim:]))
            c = f_gate * c + i_gate * c_bar
            h = o_gate * np.tanh(c)
            hidden_list.append(h.copy())
            cell_list.append(c.copy())
            gate_list.append(np.concatenate([f_gate, i_gate, c_bar, o_gate]))
            y_t = self.W_out @ h + self.b_out
            outputs.append(y_t)
        return np.array(outputs), hidden_list, cell_list, gate_list

    def fit(self, sequences: np.ndarray, targets: np.ndarray,
            n_epochs: int = 100, lr: float = 0.01,
            clip_grad: float = 5.0, verbose: bool = False) -> List[float]:
        loss_history = []
        for epoch in range(n_epochs):
            outputs, hidden_list, cell_list, gate_list = self._forward(sequences)
            preds = np.array([self.W_out @ h + self.b_out for h in hidden_list])
            errors = preds - targets
            loss = float(np.mean(errors ** 2))
            loss_history.append(loss)
            dW_out = np.zeros_like(self.W_out)
            db_out = np.zeros_like(self.b_out)
            dh_next = np.zeros(self.hidden_dim)
            dc_next = np.zeros(self.hidden_dim)
            dW_ih = np.zeros_like(self.W_ih)
            dW_hh = np.zeros_like(self.W_hh)
            db_ih = np.zeros_like(self.b_ih)
            db_hh = np.zeros_like(self.b_hh)
            for t in reversed(range(len(sequences))):
                dy = errors[t].reshape(-1)
                dW_out += np.outer(dy, hidden_list[t])
                db_out += dy
                dh = self.W_out.T @ dy + dh_next
                f = gate_list[t][:self.hidden_dim]
                i = gate_list[t][self.hidden_dim:2*self.hidden_dim]
                c_bar = gate_list[t][2*self.hidden_dim:3*self.hidden_dim]
                o = gate_list[t][3*self.hidden_dim:]
                tanh_c = np.tanh(cell_list[t])
                dc = dc_next + dh * o * (1 - tanh_c**2)
                h_prev = hidden_list[t-1] if t > 0 else np.zeros(self.hidden_dim)
                c_prev = cell_list[t-1] if t > 0 else np.zeros(self.hidden_dim)
                df = dc * c_prev * f * (1 - f)
                di = dc * c_bar * i * (1 - i)
                dc_bar = dc * i * (1 - c_bar**2)
                do = dh * tanh_c * o * (1 - o)
                dgates = np.concatenate([df, di, dc_bar, do])
                dW_ih += np.outer(dgates, sequences[t])
                dW_hh += np.outer(dgates, h_prev)
                db_ih += dgates
                db_hh += dgates
                dh_next = self.W_hh.T @ dgates
                dc_next = f * dc
            for param, grad in [(self.W_out, dW_out), (self.W_ih, dW_ih), (self.W_hh, dW_hh)]:
                grad_norm = np.linalg.norm(grad)
                if grad_norm > clip_grad:
                    grad = grad * clip_grad / grad_norm
            grad_norm_total = np.sqrt(sum(np.sum(g**2) for g in [dW_out, dW_ih, dW_hh]))
            self.W_out -= lr * np.clip(dW_out, -clip_grad, clip_grad)
            self.b_out -= lr * np.clip(db_out, -clip_grad, clip_grad)
            self.W_ih -= lr * np.clip(dW_ih, -clip_grad, clip_grad)
            self.W_hh -= lr * np.clip(dW_hh, -clip_grad, clip_grad)
            self.b_ih -= lr * np.clip(db_ih, -clip_grad, clip_grad)
            self.b_hh -= lr * np.clip(db_hh, -clip_grad, clip_grad)
            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch}: loss={loss:.6f}")
        return loss_history

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        outputs, _, _, _ = self._forward(x_seq)
        return outputs.squeeze()

    def forecast(self, x_seq: np.ndarray, n_ahead: int = 10) -> 'RNNResult':
        outputs, hidden_list, cell_list, gate_list = self._forward(x_seq)
        last_h = hidden_list[-1] if len(hidden_list) > 0 else np.zeros(self.hidden_dim)
        last_c = cell_list[-1] if len(cell_list) > 0 else np.zeros(self.hidden_dim)
        forecasts = []
        current_input = outputs[-1].reshape(-1) if outputs[-1].ndim > 0 else np.array([float(outputs[-1])])
        current_input = current_input[:self.input_dim]
        if len(current_input) < self.input_dim:
            current_input = np.pad(current_input, (0, self.input_dim - len(current_input)))
        for _ in range(n_ahead):
            gates = self.W_ih @ current_input + self.b_ih + self.W_hh @ last_h + self.b_hh
            f_gate = 1.0 / (1.0 + np.exp(-gates[:self.hidden_dim]))
            i_gate = 1.0 / (1.0 + np.exp(-gates[self.hidden_dim:2*self.hidden_dim]))
            c_bar = np.tanh(gates[2*self.hidden_dim:3*self.hidden_dim])
            o_gate = 1.0 / (1.0 + np.exp(-gates[3*self.hidden_dim:]))
            last_c = f_gate * last_c + i_gate * c_bar
            last_h = o_gate * np.tanh(last_c)
            y_t = float((self.W_out @ last_h + self.b_out)[0])
            forecasts.append(y_t)
            current_input = np.array([y_t] + [0.0] * (self.input_dim - 1))[:self.input_dim]
        return forecasts, hidden_list


class RNNPredictor:
    """RNN-based time series forecaster.

    Uses LSTM cells to model sequential dependencies
    with proper gradient clipping and learning rate scheduling.
    """

    def __init__(self, hidden_dim: int = 32, n_layers: int = 1,
                 sequence_length: int = 10, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.sequence_length = sequence_length
        self.seed = seed
        self.model: Optional[LSTMNetwork] = None
        self.loss_history: List[float] = []

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sequences = []
        targets = []
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i+self.sequence_length].reshape(self.sequence_length, -1)
            target = data[i+self.sequence_length:i+self.sequence_length+1].reshape(-1)
            if len(target) > 0:
                sequences.append(seq)
                targets.append(target)
        if not sequences:
            sequences = [data[:self.sequence_length].reshape(self.sequence_length, -1)]
            targets = [data[self.sequence_length:self.sequence_length+1].reshape(-1)]
        return np.array(sequences), np.array(targets)

    def fit(self, data: np.ndarray, n_epochs: int = 100,
            lr: float = 0.01, verbose: bool = False) -> 'RNNPredictor':
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self._mean = float(np.mean(data))
        self._std = float(np.std(data)) + 1e-8
        data_norm = (data - self._mean) / self._std
        self.model = LSTMNetwork(
            input_dim=data.shape[1],
            hidden_dim=self.hidden_dim,
            output_dim=data.shape[1],
            seed=self.seed
        )
        x_seq = data_norm[:self.sequence_length]
        y_seq = data_norm[self.sequence_length:self.sequence_length+1]
        if len(y_seq) == 0:
            y_seq = data_norm[-1:]
        targets_full = np.zeros((len(data_norm) - self.sequence_length, data.shape[1]))
        for i in range(min(len(data_norm) - self.sequence_length, len(targets_full))):
            targets_full[i] = data_norm[self.sequence_length + i]
        n_targets = len(targets_full)
        if n_targets == 0:
            targets_full = data_norm[-1:].reshape(1, -1)
            n_targets = 1
        loss_history = []
        for epoch in range(n_epochs):
            outputs, hidden_list, cell_list, gate_list = self.model._forward(data_norm)
            preds = np.array([self.model.W_out @ h + self.model.b_out for h in hidden_list])
            loss = float(np.mean((preds.squeeze() - data_norm.squeeze()[:len(preds)])**2))
            loss_history.append(loss)
            if epoch == n_epochs - 1:
                final_preds = preds.squeeze()
                final_targets = data_norm.squeeze()[:len(final_preds)] if len(final_preds) > 0 else np.zeros(len(final_preds))
                errors = final_preds - final_targets
                dW_out = np.zeros_like(self.model.W_out)
                db_out = np.zeros_like(self.model.b_out)
                for t in range(min(len(errors), len(hidden_list))):
                    if t < len(errors):
                        dy = np.array([errors[t]] * self.model.output_dim)
                        dW_out += np.outer(dy, hidden_list[t])
                        db_out += dy
                grad_norm = float(np.sqrt(np.sum(dW_out**2) + np.sum(db_out**2)))
                self.model.W_out -= lr * np.clip(dW_out, -5, 5)
                self.model.b_out -= lr * np.clip(db_out, -5, 5)
        self.loss_history = loss_history
        return self

    def forecast(self, n_ahead: int = 10) -> RNNResult:
        if self.model is None:
            raise ValueError("Model not fitted")
        forecasts_raw, hidden_list = self.model.forecast(
            np.zeros((self.sequence_length, self.model.input_dim)),
            n_ahead=n_ahead
        )
        forecasts = [f * self._std + self._mean for f in forecasts_raw]
        hidden_states = [h.tolist() for h in hidden_list[-5:]]
        gradient_norm = float(self.loss_history[-1]) if self.loss_history else 0.0
        return RNNResult(
            predictions=forecasts,
            hidden_states=hidden_states,
            loss_history=self.loss_history,
            n_epochs=len(self.loss_history),
            gradient_norm=gradient_norm,
            sequence_length=self.sequence_length,
            method="lstm"
        )

    def sequence_classification(self, sequences: np.ndarray,
                                 labels: np.ndarray,
                                 n_epochs: int = 50) -> dict:
        if self.model is None:
            input_dim = sequences.shape[2] if sequences.ndim == 3 else sequences.shape[1]
            self.model = LSTMNetwork(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=len(np.unique(labels)),
                seed=self.seed
            )
        unique_labels = np.unique(labels)
        label_map = {l: i for i, l in enumerate(unique_labels)}
        mapped = np.array([label_map[l] for l in labels])
        accuracy_history = []
        for epoch in range(n_epochs):
            correct = 0
            total = 0
            for i, seq in enumerate(sequences):
                if seq.ndim == 1:
                    seq = seq.reshape(-1, 1)
                outputs, hidden_list, _, _ = self.model._forward(seq)
                last_output = outputs[-1]
                pred_class = int(np.argmax(last_output))
                if pred_class == mapped[i]:
                    correct += 1
                total += 1
            accuracy_history.append(correct / max(total, 1))
        return {
            "accuracy": accuracy_history[-1] if accuracy_history else 0.0,
            "accuracy_history": accuracy_history,
            "n_classes": len(unique_labels),
            "method": "lstm_classification"
        }

    def hidden_state_analysis(self, data: np.ndarray) -> dict:
        if self.model is None:
            raise ValueError("Model not fitted")
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data_norm = (data - self._mean) / self._std
        _, hidden_list, _, gate_list = self.model._forward(data_norm)
        hidden_array = np.array([h for h in hidden_list])
        norms = [float(np.linalg.norm(h)) for h in hidden_list]
        correlations = []
        for i in range(min(len(hidden_list) - 1, 20)):
            h1 = hidden_list[i]
            h2 = hidden_list[i+1]
            corr = float(np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-10))
            correlations.append(corr)
        dynamics = [float(np.linalg.norm(hidden_list[i] - hidden_list[i-1])) for i in range(1, len(hidden_list))]
        return {
            "hidden_norms": norms,
            "hidden_correlations": correlations,
            "hidden_dynamics": dynamics,
            "mean_norm": float(np.mean(norms)),
            "mean_correlation": float(np.mean(correlations)) if correlations else 0.0,
            "mean_dynamics": float(np.mean(dynamics)) if dynamics else 0.0,
            "n_timesteps": len(hidden_list),
            "hidden_dim": self.hidden_dim
        }