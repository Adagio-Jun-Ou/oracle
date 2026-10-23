from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class TransferEntropyResult:
    transfer_entropy: float
    source: str
    target: str
    lag: int
    significance: float
    method: str


@dataclass
class MutualInfoResult:
    mutual_information: float
    normalized_mi: float
    method: str
    n_bins: int


@dataclass
class GrangerCausalityResult:
    f_statistic: float
    p_value: float
    lag: int
    direction: str
    significant: bool
    method: str


@dataclass
class ConditionalEntropyResult:
    conditional_entropy: float
    mutual_information: float
    normalized: float
    method: str


class InformationTheoreticPredictor:
    def __init__(self, seed: int = 42):
        self.seed = seed

    @staticmethod
    def transfer_entropy(source: np.ndarray, target: np.ndarray,
                          lag: int = 1, n_shuffles: int = 100) -> TransferEntropyResult:
        n = min(len(source), len(target))
        if n < lag + 5:
            return TransferEntropyResult(transfer_entropy=0.0, source="x", target="y", lag=lag, significance=1.0, method="transfer_entropy")
        source = source[:n]
        target = target[:n]
        t_future = target[lag:]
        t_past = target[:-lag]
        s_past = source[:-lag]
        def _discretize(x, n_bins=8):
            percentiles = np.linspace(0, 100, n_bins + 1)
            edges = np.percentile(x, percentiles)
            return np.digitize(x, edges[1:-1])
        t_f_d = _discretize(t_future)
        t_p_d = _discretize(t_past)
        s_p_d = _discretize(s_past)
        n_bins = max(np.max(t_f_d), np.max(t_p_d), np.max(s_p_d)) + 1
        te = 0.0
        for i_val in range(n_bins):
            mask_i = t_f_d == i_val
            p_i = np.mean(mask_i) + 1e-10
            for j_val in range(n_bins):
                mask_j = t_p_d == j_val
                mask_ij = mask_i & mask_j
                p_ij = np.mean(mask_ij) + 1e-10
                p_j = np.mean(mask_j) + 1e-10
                for k_val in range(n_bins):
                    mask_k = s_p_d == k_val
                    mask_ijk = mask_ij & (s_p_d == k_val)
                    mask_jk = mask_j & mask_k
                    p_ijk = np.mean(mask_ijk) + 1e-10
                    p_jk = np.mean(mask_jk) + 1e-10
                    p_ik_given_jk = p_ijk / p_jk
                    p_i_given_j = p_ij / p_j
                    if p_ik_given_jk > 0 and p_i_given_j > 0:
                        te += p_ijk * np.log(p_ik_given_jk / p_i_given_j)
        rng = np.random.RandomState(42)
        n_sig = 0
        for _ in range(n_shuffles):
            s_shuffled = s_p_d.copy()
            rng.shuffle(s_shuffled)
            te_shuffled = 0.0
            for i_val in range(n_bins):
                mask_i = t_f_d == i_val
                p_i = np.mean(mask_i) + 1e-10
                for j_val in range(n_bins):
                    mask_j = t_p_d == j_val
                    mask_ij = mask_i & mask_j
                    p_ij = np.mean(mask_ij) + 1e-10
                    p_j = np.mean(mask_j) + 1e-10
                    for k_val in range(n_bins):
                        mask_k = s_shuffled == k_val
                        mask_ijk = mask_ij & mask_k
                        mask_jk = mask_j & mask_k
                        p_ijk = np.mean(mask_ijk) + 1e-10
                        p_jk = np.mean(mask_jk) + 1e-10
                        p_ik_given_jk = p_ijk / p_jk
                        p_i_given_j = p_ij / p_j
                        if p_ik_given_jk > 0 and p_i_given_j > 0:
                            te_shuffled += p_ijk * np.log(p_ik_given_jk / p_i_given_j)
            if te > te_shuffled:
                n_sig += 1
        significance = 1 - n_sig / max(n_shuffles, 1)
        return TransferEntropyResult(
            transfer_entropy=float(max(0, te)),
            source="x", target="y",
            lag=lag,
            significance=float(significance),
            method="transfer_entropy",
        )

    @staticmethod
    def mutual_information(x: np.ndarray, y: np.ndarray,
                           n_bins: int = 10) -> MutualInfoResult:
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return MutualInfoResult(mutual_information=0.0, normalized_mi=0.0, method="discrete_mi", n_bins=n_bins)
        x_edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
        y_edges = np.percentile(y, np.linspace(0, 100, n_bins + 1))
        x_d = np.digitize(x, x_edges[1:-1])
        y_d = np.digitize(y, y_edges[1:-1])
        mi = 0.0
        for i in range(n_bins):
            mask_x = x_d == i
            p_x = np.mean(mask_x) + 1e-10
            for j in range(n_bins):
                mask_y = y_d == j
                p_y = np.mean(mask_y) + 1e-10
                mask_xy = mask_x & mask_y
                p_xy = np.mean(mask_xy) + 1e-10
                mi += p_xy * np.log(p_xy / (p_x * p_y))
        h_x = -sum(np.mean(x_d == i) * np.log(np.mean(x_d == i) + 1e-10) for i in range(n_bins))
        h_y = -sum(np.mean(y_d == j) * np.log(np.mean(y_d == j) + 1e-10) for j in range(n_bins))
        normalized = mi / max(min(h_x, h_y), 1e-10)
        return MutualInfoResult(
            mutual_information=float(max(0, mi)),
            normalized_mi=float(min(1.0, max(0.0, normalized))),
            method="discrete_mi",
            n_bins=n_bins,
        )

    @staticmethod
    def granger_causality(x: np.ndarray, y: np.ndarray,
                          lag: int = 3) -> GrangerCausalityResult:
        n = min(len(x), len(y))
        if n < lag + 5:
            return GrangerCausalityResult(f_statistic=0.0, p_value=1.0, lag=lag, direction="x->y", significant=False, method="granger")
        x = x[:n]
        y = y[:n]
        Y = y[lag:]
        n_obs = len(Y)
        Y_lagged = np.column_stack([y[lag - i - 1:n - i - 1] for i in range(lag)])
        X_lagged = np.column_stack([x[lag - i - 1:n - i - 1] for i in range(lag)])
        Y_aug = np.column_stack([Y_lagged, X_lagged])
        X_restricted = np.column_stack([np.ones(n_obs), Y_lagged])
        X_unrestricted = np.column_stack([np.ones(n_obs), Y_aug])
        beta_r = np.linalg.lstsq(X_restricted, Y, rcond=None)[0]
        beta_u = np.linalg.lstsq(X_unrestricted, Y, rcond=None)[0]
        rss_r = float(np.sum((Y - X_restricted @ beta_r) ** 2))
        rss_u = float(np.sum((Y - X_unrestricted @ beta_u) ** 2))
        if rss_r <= rss_u or rss_r < 1e-10:
            return GrangerCausalityResult(f_statistic=0.0, p_value=1.0, lag=lag, direction="x->y", significant=False, method="granger")
        f_stat = ((rss_r - rss_u) / lag) / (rss_u / max(n_obs - 2 * lag - 1, 1))
        from scipy import stats as sp_stats
        p_value = float(1 - sp_stats.f.cdf(f_stat, lag, max(n_obs - 2 * lag - 1, 1)))
        significant = p_value < 0.05
        return GrangerCausalityResult(
            f_statistic=float(f_stat),
            p_value=float(p_value),
            lag=lag,
            direction="x->y",
            significant=significant,
            method="granger",
        )

    @staticmethod
    def conditional_entropy(x: np.ndarray, y: np.ndarray,
                            n_bins: int = 8) -> ConditionalEntropyResult:
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]
        x_edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
        y_edges = np.percentile(y, np.linspace(0, 100, n_bins + 1))
        x_d = np.digitize(x, x_edges[1:-1])
        y_d = np.digitize(y, y_edges[1:-1])
        h_x = -sum(np.mean(x_d == i) * np.log(np.mean(x_d == i) + 1e-10) for i in range(n_bins))
        mi = InformationTheoreticPredictor.mutual_information(x, y, n_bins=n_bins)
        cond_ent = h_x - mi.mutual_information
        h_x_pos = max(h_x, 1e-10)
        normalized = cond_ent / h_x_pos if h_x_pos > 0 else 1.0
        return ConditionalEntropyResult(
            conditional_entropy=float(max(0, cond_ent)),
            mutual_information=float(mi.mutual_information),
            normalized=float(min(1, max(0, normalized))),
            method="conditional_entropy",
        )

    @staticmethod
    def active_info_storage(series: np.ndarray, history: int = 3,
                             n_bins: int = 6) -> dict:
        n = len(series)
        if n < history + 5:
            return {"ais": 0.0, "history": history, "method": "active_info_storage"}
        edges = np.percentile(series, np.linspace(0, 100, n_bins + 1))
        s_d = np.digitize(series, edges[1:-1])
        present = s_d[history:]
        past_cols = [s_d[history - i - 1:n - i - 1] for i in range(history)]
        past_joint = np.zeros(n - history, dtype=int)
        base = n_bins
        for i, col in enumerate(past_cols):
            past_joint = past_joint * base + col
        mi = InformationTheoreticPredictor.mutual_information(
            past_joint.astype(float), present.astype(float), n_bins=max(n_bins, len(np.unique(past_joint)))
        )
        return {
            "ais": float(mi.mutual_information),
            "normalized_ais": float(mi.normalized_mi),
            "history": history,
            "method": "active_info_storage",
        }