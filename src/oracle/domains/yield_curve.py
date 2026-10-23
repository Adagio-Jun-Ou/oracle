import numpy as np
from scipy.optimize import minimize
from typing import Optional, Tuple


class YieldCurveModel:

    def __init__(self):
        self.name = "yield_curve_model"

    @staticmethod
    def nelson_siegel(
        beta0: float = 0.05,
        beta1: float = -0.02,
        beta2: float = 0.03,
        tau: float = 2.0,
        maturities: Optional[list] = None,
    ) -> dict:
        if maturities is None:
            maturities = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
        T = np.array(maturities)
        T = np.where(T == 0, 1e-10, T)
        factor1 = np.ones_like(T)
        factor2 = (1 - np.exp(-T / tau)) / (T / tau)
        factor3 = (1 - np.exp(-T / tau)) / (T / tau) - np.exp(-T / tau)
        rates = beta0 * factor1 + beta1 * factor2 + beta2 * factor3
        return {
            "maturities": maturities,
            "rates": [round(float(r) * 100, 4) for r in rates],
            "beta0": beta0,
            "beta1": beta1,
            "beta2": beta2,
            "tau": tau,
            "model": "nelson_siegel",
            "long_rate": round(float(beta0) * 100, 4),
            "short_rate": round(float((beta0 + beta1)) * 100, 4),
            "curvature": round(float(beta2) * 100, 4),
        }

    @staticmethod
    def nelson_siegel_svensson(
        beta0: float = 0.05,
        beta1: float = -0.02,
        beta2: float = 0.03,
        beta3: float = -0.01,
        tau1: float = 2.0,
        tau2: float = 5.0,
        maturities: Optional[list] = None,
    ) -> dict:
        if maturities is None:
            maturities = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
        T = np.array(maturities)
        T = np.where(T == 0, 1e-10, T)
        factor1 = np.ones_like(T)
        factor2 = (1 - np.exp(-T / tau1)) / (T / tau1)
        factor3 = (1 - np.exp(-T / tau1)) / (T / tau1) - np.exp(-T / tau1)
        factor4 = (1 - np.exp(-T / tau2)) / (T / tau2) - np.exp(-T / tau2)
        rates = beta0 * factor1 + beta1 * factor2 + beta2 * factor3 + beta3 * factor4
        return {
            "maturities": maturities,
            "rates": [round(float(r) * 100, 4) for r in rates],
            "beta0": beta0, "beta1": beta1, "beta2": beta2, "beta3": beta3,
            "tau1": tau1, "tau2": tau2,
            "model": "nelson_siegel_svensson",
        }

    @staticmethod
    def fit_nelson_siegel(
        observed_maturities: list,
        observed_rates: list,
        initial_tau: float = 2.0,
    ) -> dict:
        T = np.array(observed_maturities)
        y = np.array(observed_rates)
        T = np.where(T == 0, 1e-10, T)

        def objective(params):
            b0, b1, b2, tau = params
            if tau <= 0:
                return 1e10
            f1 = np.ones_like(T)
            f2 = (1 - np.exp(-T / tau)) / (T / tau)
            f3 = (1 - np.exp(-T / tau)) / (T / tau) - np.exp(-T / tau)
            fitted = b0 * f1 + b1 * f2 + b2 * f3
            return np.sum((y - fitted) ** 2)

        x0 = [np.mean(y), y[0] - np.mean(y) if len(y) > 1 else 0, 0.01, initial_tau]
        bounds = [(-0.1, 0.2), (-0.2, 0.2), (-0.1, 0.1), (0.1, 10.0)]
        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
        if not result.success:
            return {"error": "Optimization failed", "success": False}
        b0, b1, b2, tau = result.x
        fitted = YieldCurveModel.nelson_siegel(b0, b1, b2, tau, observed_maturities)
        residuals = np.array(y) - np.array([r / 100 for r in fitted["rates"]])
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        return {
            "success": True,
            "fitted_params": {"beta0": round(float(b0), 6), "beta1": round(float(b1), 6), "beta2": round(float(b2), 6), "tau": round(float(tau), 4)},
            "fitted_rates": fitted["rates"],
            "observed_rates": [round(float(r) * 100, 4) for r in y],
            "rmse_bps": round(rmse * 10000, 2),
            "maturities": observed_maturities,
        }

    @staticmethod
    def spot_rates_from_bonds(
        prices: list,
        coupons: list,
        maturities: list,
        face_value: float = 1000,
    ) -> dict:
        n = len(prices)
        spots = np.zeros(n)
        for i in range(n):
            P = prices[i]
            C = coupons[i] * face_value
            T = maturities[i]
            pv_coupons = 0.0
            for j in range(i):
                pv_coupons += C * np.exp(-spots[j] * maturities[j])
            remaining = P - pv_coupons
            total_cf = C + face_value
            if remaining > 0 and total_cf > 0:
                spots[i] = -np.log(remaining / total_cf) / T
            else:
                spots[i] = 0.0
        return {
            "maturities": maturities,
            "spot_rates": [round(float(s) * 100, 4) for s in spots],
            "prices": prices,
            "coupons": [round(float(c) * 100, 4) for c in coupons],
            "face_value": face_value,
        }

    @staticmethod
    def forward_rates(
        spot_rates: list,
        maturities: list,
    ) -> dict:
        n = len(spot_rates)
        if n < 2:
            return {"forward_rates": [], "maturities": []}
        fwd_rates = []
        fwd_maturities = []
        for i in range(n - 1):
            t1 = maturities[i]
            t2 = maturities[i + 1]
            r1 = spot_rates[i]
            r2 = spot_rates[i + 1]
            if t1 > 0 and t2 > t1:
                fwd = (r2 * t2 - r1 * t1) / (t2 - t1)
                fwd_rates.append(round(float(fwd) * 100, 4))
                fwd_maturities.append(f"{t1}y-{t2}y")
        return {
            "forward_rates": fwd_rates,
            "forward_maturities": fwd_maturities,
            "spot_rates": [round(float(r) * 100, 4) for r in spot_rates],
            "spot_maturities": maturities,
        }

    @staticmethod
    def yield_curve_scenario(
        shift_bps: float = 0,
        steepen_bps: float = 0,
        twist_bps: float = 0,
        butterfly_bps: float = 0,
        base_beta0: float = 0.05,
        base_beta1: float = -0.02,
        base_beta2: float = 0.03,
        base_tau: float = 2.0,
        maturities: Optional[list] = None,
    ) -> dict:
        if maturities is None:
            maturities = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
        bps = 0.0001
        shifted_beta0 = base_beta0 + shift_bps * bps
        shifted_beta1 = base_beta1 + steepen_bps * bps
        shifted_beta2 = base_beta2 + butterfly_bps * bps
        shifted_tau = base_tau + twist_bps * 0.1 if base_tau + twist_bps * 0.1 > 0.1 else 0.1
        base = YieldCurveModel.nelson_siegel(base_beta0, base_beta1, base_beta2, base_tau, maturities)
        shifted = YieldCurveModel.nelson_siegel(shifted_beta0, shifted_beta1, shifted_beta2, shifted_tau, maturities)
        base_rates = np.array([r / 100 for r in base["rates"]])
        shifted_rates = np.array([r / 100 for r in shifted["rates"]])
        changes = shifted_rates - base_rates
        return {
            "base_curve": base,
            "shifted_curve": shifted,
            "changes_bps": [round(float(c) * 10000, 2) for c in changes],
            "scenario": {
                "parallel_shift_bps": shift_bps,
                "steepening_bps": steepen_bps,
                "twist_bps": twist_bps,
                "butterfly_bps": butterfly_bps,
            },
        }