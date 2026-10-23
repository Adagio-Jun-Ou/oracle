import numpy as np
from scipy.stats import norm
from typing import Optional


class OptionsPricer:

    def __init__(self):
        self.name = "options_pricer"

    @staticmethod
    def black_scholes(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        if T <= 0:
            if option_type == "call":
                return max(S - K, 0.0)
            else:
                return max(K - S, 0.0)
        if sigma <= 0:
            intrinsic = (S - K) if option_type == "call" else (K - S)
            return max(intrinsic, 0.0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        return max(float(price), 0.0)

    @staticmethod
    def greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> dict:
        if T <= 0:
            if option_type == "call":
                intrinsic = max(S - K, 0.0)
            else:
                intrinsic = max(K - S, 0.0)
            return {
                "delta": 1.0 if intrinsic > 0 else 0.0,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
                "rho": 0.0,
            }
        if sigma <= 0:
            return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        pdf_d1 = norm.pdf(d1)
        sqrt_T = np.sqrt(T)
        if option_type == "call":
            delta = norm.cdf(d1)
            theta = (-S * pdf_d1 * sigma / (2 * sqrt_T)
                     - r * K * np.exp(-r * T) * norm.cdf(d2))
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        elif option_type == "put":
            delta = norm.cdf(d1) - 1.0
            theta = (-S * pdf_d1 * sigma / (2 * sqrt_T)
                     + r * K * np.exp(-r * T) * norm.cdf(-d2))
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        gamma = pdf_d1 / (S * sigma * sqrt_T)
        vega = S * sqrt_T * pdf_d1 / 100
        return {
            "delta": round(float(delta), 6),
            "gamma": round(float(gamma), 6),
            "theta": round(float(theta), 6),
            "vega": round(float(vega), 6),
            "rho": round(float(rho), 6),
        }

    @staticmethod
    def implied_volatility(
        S: float,
        K: float,
        T: float,
        r: float,
        market_price: float,
        option_type: str = "call",
        max_iterations: int = 100,
        tolerance: float = 1e-8,
    ) -> Optional[float]:
        if T <= 0:
            return None
        intrinsic = (S - K) if option_type == "call" else (K - S)
        if market_price <= max(intrinsic, 0.0) + 1e-10:
            return None
        sigma = 0.3
        pricer = OptionsPricer()
        for _ in range(max_iterations):
            price = pricer.black_scholes(S, K, T, r, sigma, option_type)
            diff = price - market_price
            if abs(diff) < tolerance:
                return round(sigma, 6)
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            vega = S * np.sqrt(T) * norm.pdf(d1)
            if abs(vega) < 1e-12:
                return None
            sigma -= diff / vega
            if sigma <= 0.001:
                sigma = 0.001
            if sigma > 10.0:
                sigma = 10.0
        return None

    @staticmethod
    def put_call_parity(
        S: float,
        K: float,
        T: float,
        r: float,
        call_price: Optional[float] = None,
        put_price: Optional[float] = None,
        sigma: Optional[float] = None,
    ) -> dict:
        if call_price is not None and put_price is not None:
            parity_diff = call_price - put_price - (S - K * np.exp(-r * T))
            return {
                "call_price": call_price,
                "put_price": put_price,
                "parity_difference": round(float(parity_diff), 6),
                "parity_holds": bool(abs(parity_diff) < 0.01 * S),
                "strike": K,
                "spot": S,
                "time_to_expiry": T,
                "risk_free_rate": r,
                "forward": round(float(K * np.exp(-r * T)), 6),
            }
        pricer = OptionsPricer()
        if sigma is None:
            sigma = 0.2
        if call_price is not None:
            put_from_parity = call_price - S + K * np.exp(-r * T)
            put_bs = pricer.black_scholes(S, K, T, r, sigma, "put")
            return {
                "put_price_from_parity": round(float(put_from_parity), 6),
                "put_price_black_scholes": round(float(put_bs), 6),
                "call_price": call_price,
                "call_put_difference": round(float(call_price - put_bs), 6),
                "strike": K,
                "spot": S,
                "time_to_expiry": T,
                "risk_free_rate": r,
            }
        if put_price is not None:
            call_from_parity = put_price + S - K * np.exp(-r * T)
            call_bs = pricer.black_scholes(S, K, T, r, sigma, "call")
            return {
                "call_price_from_parity": round(float(call_from_parity), 6),
                "call_price_black_scholes": round(float(call_bs), 6),
                "put_price": put_price,
                "call_put_difference": round(float(call_bs - put_price), 6),
                "strike": K,
                "spot": S,
                "time_to_expiry": T,
                "risk_free_rate": r,
            }
        call_bs = pricer.black_scholes(S, K, T, r, sigma, "call")
        put_bs = pricer.black_scholes(S, K, T, r, sigma, "put")
        parity_diff = call_bs - put_bs - (S - K * np.exp(-r * T))
        return {
            "call_price": round(float(call_bs), 6),
            "put_price": round(float(put_bs), 6),
            "parity_difference": round(float(parity_diff), 6),
            "parity_holds": bool(abs(parity_diff) < 0.01 * S),
            "strike": K,
            "spot": S,
            "time_to_expiry": T,
            "risk_free_rate": r,
        }

    @staticmethod
    def binomial_tree(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        steps: int = 50,
        option_type: str = "call",
        american: bool = False,
    ) -> dict:
        dt = T / steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp(r * dt) - d) / (u - d)
        if not (0 < p < 1):
            p = 0.5
        discount = np.exp(-r * dt)
        prices = np.zeros(steps + 1)
        for i in range(steps + 1):
            prices[i] = S * (u ** (steps - i)) * (d**i)
        if option_type == "call":
            option_values = np.maximum(prices - K, 0.0)
        else:
            option_values = np.maximum(K - prices, 0.0)
        for step in range(steps - 1, -1, -1):
            for i in range(step + 1):
                continuation = discount * (p * option_values[i] + (1 - p) * option_values[i + 1])
                if american:
                    exercise = max(S * (u ** (step - i)) * (d**i) - K, 0.0) if option_type == "call" else max(K - S * (u ** (step - i)) * (d**i), 0.0)
                    option_values[i] = max(continuation, exercise)
                else:
                    option_values[i] = continuation
        price = float(option_values[0])
        pricer = OptionsPricer()
        bs_price = pricer.black_scholes(S, K, T, r, sigma, option_type)
        return {
            "binomial_price": round(price, 6),
            "black_scholes_price": round(bs_price, 6),
            "price_difference": round(abs(price - bs_price), 6),
            "steps": steps,
            "option_type": option_type,
            "american": american,
            "up_factor": round(float(u), 6),
            "down_factor": round(float(d), 6),
            "risk_neutral_prob": round(float(p), 6),
        }

    @staticmethod
    def option_chain(
        S: float,
        strikes: list,
        T: float,
        r: float,
        sigma: float,
    ) -> dict:
        pricer = OptionsPricer()
        chain = []
        for K in strikes:
            call_price = pricer.black_scholes(S, K, T, r, sigma, "call")
            put_price = pricer.black_scholes(S, K, T, r, sigma, "put")
            call_greeks = pricer.greeks(S, K, T, r, sigma, "call")
            put_greeks = pricer.greeks(S, K, T, r, sigma, "put")
            iv_call = pricer.implied_volatility(S, K, T, r, call_price, "call")
            iv_put = pricer.implied_volatility(S, K, T, r, put_price, "put")
            chain.append({
                "strike": K,
                "call": {
                    "price": round(call_price, 4),
                    "delta": call_greeks["delta"],
                    "gamma": call_greeks["gamma"],
                    "theta": call_greeks["theta"],
                    "vega": call_greeks["vega"],
                    "implied_volatility": iv_call,
                },
                "put": {
                    "price": round(put_price, 4),
                    "delta": put_greeks["delta"],
                    "gamma": put_greeks["gamma"],
                    "theta": put_greeks["theta"],
                    "vega": put_greeks["vega"],
                    "implied_volatility": iv_put,
                },
            })
        return {
            "spot": S,
            "time_to_expiry": T,
            "risk_free_rate": r,
            "volatility": sigma,
            "chain": chain,
        }

    @staticmethod
    def volatility_surface(
        S: float,
        strikes: list,
        maturities: list,
        r: float,
        base_sigma: float = 0.2,
        skew: float = 0.1,
        term_structure_slope: float = 0.05,
    ) -> dict:
        surface = []
        for T in maturities:
            row = []
            for K in strikes:
                moneyness = np.log(K / S)
                skew_adj = skew * moneyness
                term_adj = term_structure_slope * np.sqrt(T)
                sigma = max(base_sigma + skew_adj + term_adj, 0.01)
                row.append(round(sigma, 4))
            surface.append(row)
        return {
            "spot": S,
            "strikes": strikes,
            "maturities": maturities,
            "surface": surface,
            "base_volatility": base_sigma,
            "skew": skew,
            "term_structure_slope": term_structure_slope,
        }

    @staticmethod
    def strategy_pnl(
        S: float,
        strategy: list,
        price_range: Optional[list] = None,
        n_points: int = 100,
    ) -> dict:
        if price_range is None:
            price_range = [S * 0.7, S * 1.3]
        prices = np.linspace(price_range[0], price_range[1], n_points)
        total_pnl = np.zeros(n_points)
        leg_details = []
        for leg in strategy:
            leg_type = leg.get("type", "call")
            K = leg.get("strike", S)
            premium = leg.get("premium", 0.0)
            quantity = leg.get("quantity", 1)
            direction = leg.get("direction", "long")
            if direction == "short":
                quantity = -quantity
            if leg_type == "call":
                payoff = np.maximum(prices - K, 0.0) - premium
            else:
                payoff = np.maximum(K - prices, 0.0) - premium
            total_pnl += quantity * payoff
            leg_details.append({
                "type": leg_type,
                "strike": K,
                "premium": premium,
                "quantity": quantity,
                "direction": direction,
            })
        max_profit = float(np.max(total_pnl))
        max_loss = float(np.min(total_pnl))
        breakevens = []
        for i in range(1, len(total_pnl)):
            if total_pnl[i - 1] * total_pnl[i] < 0:
                breakevens.append(round(float((prices[i - 1] + prices[i]) / 2), 2))
        return {
            "prices": [round(float(p), 2) for p in prices],
            "pnl": [round(float(p), 4) for p in total_pnl],
            "max_profit": round(max_profit, 4) if max_profit != float("inf") else "unlimited",
            "max_loss": round(max_loss, 4) if max_loss != float("-inf") else "unlimited",
            "breakeven_points": breakevens,
            "legs": leg_details,
        }

    @staticmethod
    def greeks_surface(
        S: float,
        K_range: list,
        T_range: list,
        r: float,
        sigma: float,
        greek: str = "delta",
        option_type: str = "call",
    ) -> dict:
        pricer = OptionsPricer()
        surface = []
        for T in T_range:
            row = []
            for K in K_range:
                g = pricer.greeks(S, K, T, r, sigma, option_type)
                row.append(g.get(greek, 0.0))
            surface.append(row)
        return {
            "spot": S,
            "strikes": K_range,
            "maturities": T_range,
            "sigma": sigma,
            "greek": greek,
            "option_type": option_type,
            "surface": surface,
        }