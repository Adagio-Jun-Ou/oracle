"""Prediction Market Simulator — Simulated market pricing for events with information aggregation.

Core Insight: Prediction markets aggregate dispersed information efficiently. When many
participants bet on outcomes, prices converge to well-calibrated probabilities. This module
simulates market dynamics to produce probability estimates that are often better than
any individual forecaster, following the wisdom-of-crowds principle.

Implements:
- PredictionMarket: Simulated market with automated market makers
- SimulatedTrader: Diverse trader archetypes (superforecaster, noise trader, insider, contrarian)
- MarketSimulator: Runs market simulation and produces aggregated probability estimate
- MarketResult: Final probability, price history, trader statistics, calibration
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class MarketTrade:
    trader_type: str
    direction: str
    quantity: float
    price: float
    round: int
    information_strength: float


class SimulatedTrader:
    """Base simulated trader with configurable strategy."""

    def __init__(self, name: str, trader_type: str, accuracy: float = 0.6,
                 bias: float = 0.0, aggressiveness: float = 0.5,
                 information_access: float = 0.7):
        self.name = name
        self.trader_type = trader_type
        self.accuracy = accuracy
        self.bias = bias
        self.aggressiveness = aggressiveness
        self.information_access = information_access
        self.position: float = 0.0
        self.trades: List[MarketTrade] = []
        self.profit: float = 0.0

    def generate_signal(self, true_probability: float, market_price: float) -> Tuple[str, float]:
        signal_with_noise = true_probability * self.information_access + np.random.normal(0, 0.05 * (1 - self.information_access))
        signal = signal_with_noise + self.bias * 0.1
        signal = max(0.01, min(0.99, signal))
        if signal > market_price + 0.02:
            return "buy", min(1.0, (signal - market_price) * self.aggressiveness * 2)
        elif signal < market_price - 0.02:
            return "sell", min(1.0, (market_price - signal) * self.aggressiveness * 2)
        return "hold", 0.0

    def decide(self, true_probability: float, market_price: float, round_num: int) -> Optional[MarketTrade]:
        direction, strength = self.generate_signal(true_probability, market_price)
        if direction == "hold" or strength < 0.01:
            return None
        noise = np.random.normal(0, 0.1 * (1 - self.accuracy))
        adjusted_strength = max(0.01, min(1.0, strength + noise))
        quantity = adjusted_strength * 10
        price = market_price
        if direction == "buy":
            self.position += quantity
        else:
            self.position -= quantity
        trade = MarketTrade(
            trader_type=self.trader_type, direction=direction,
            quantity=quantity, price=price, round=round_num,
            information_strength=adjusted_strength,
        )
        self.trades.append(trade)
        return trade


class SuperforecasterTrader(SimulatedTrader):
    def __init__(self, name: str = "superforecaster"):
        super().__init__(name, "superforecaster", accuracy=0.75, bias=0.02,
                         aggressiveness=0.6, information_access=0.9)


class NoiseTrader(SimulatedTrader):
    def __init__(self, name: str = "noise"):
        super().__init__(name, "noise", accuracy=0.45, bias=0.0,
                         aggressiveness=0.4, information_access=0.3)


class InsiderTrader(SimulatedTrader):
    def __init__(self, name: str = "insider"):
        super().__init__(name, "insider", accuracy=0.85, bias=0.05,
                         aggressiveness=0.7, information_access=0.95)


class ContrarianTrader(SimulatedTrader):
    def __init__(self, name: str = "contrarian"):
        super().__init__(name, "contrarian", accuracy=0.55, bias=-0.03,
                         aggressiveness=0.5, information_access=0.6)

    def generate_signal(self, true_probability: float, market_price: float) -> Tuple[str, float]:
        direction, strength = super().generate_signal(true_probability, market_price)
        if direction == "buy":
            return "sell", strength * 0.5
        elif direction == "sell":
            return "buy", strength * 0.5
        return "hold", 0.0


class MomentumTrader(SimulatedTrader):
    def __init__(self, name: str = "momentum"):
        super().__init__(name, "momentum", accuracy=0.5, bias=0.0,
                         aggressiveness=0.7, information_access=0.4)

    def decide(self, true_probability: float, market_price: float, round_num: int) -> Optional[MarketTrade]:
        if round_num < 3:
            return None
        self.bias = (market_price - 0.5) * 0.1
        return super().decide(true_probability, market_price, round_num)


class PredictionMarket:
    """Automated market maker using logarithmic market scoring rule (LMSR)."""

    def __init__(self, event: str, initial_liquidity: float = 100.0):
        self.event = event
        self.initial_liquidity = initial_liquidity
        self.liquidity = initial_liquidity
        self.yes_shares = 0.0
        self.no_shares = 0.0
        self.price_history: List[float] = [0.5]
        self.volume_history: List[float] = [0.0]
        self.trades: List[MarketTrade] = []

    def current_price(self) -> float:
        exp_yes = math.exp(self.yes_shares / self.liquidity)
        exp_no = math.exp(self.no_shares / self.liquidity)
        total = exp_yes + exp_no
        price = exp_yes / total
        return max(0.01, min(0.99, price))

    def cost_yes(self, n_shares: float) -> float:
        cost_before = self.liquidity * math.log(math.exp(self.yes_shares / self.liquidity) + math.exp(self.no_shares / self.liquidity))
        cost_after = self.liquidity * math.log(math.exp((self.yes_shares + n_shares) / self.liquidity) + math.exp(self.no_shares / self.liquidity))
        return max(0, cost_after - cost_before)

    def execute_trade(self, trade: MarketTrade) -> float:
        if trade.direction == "buy":
            n_shares = trade.quantity
            cost = self.cost_yes(n_shares)
            self.yes_shares += n_shares
        else:
            n_shares = trade.quantity
            cost = self.cost_yes(n_shares)
            self.no_shares += n_shares
        new_price = self.current_price()
        self.price_history.append(new_price)
        self.volume_history.append(trade.quantity)
        self.trades.append(trade)
        return new_price


@dataclass
class MarketResult:
    event: str
    final_probability: float
    confidence_interval: Tuple[float, float]
    price_history: List[float]
    n_traders: int
    n_rounds: int
    superforecaster_consensus: float
    market_efficiency: float
    calibration_error: float
    trader_pnl: Dict[str, float]
    method: str = "prediction_market"


class MarketSimulator:
    """Runs a full prediction market simulation with diverse trader types."""

    def __init__(self, n_superforecasters: int = 5, n_noise: int = 10,
                 n_insiders: int = 2, n_contrarians: int = 3,
                 n_momentum: int = 5, n_rounds: int = 50):
        self.traders: List[SimulatedTrader] = []
        for i in range(n_superforecasters):
            self.traders.append(SuperforecasterTrader(f"sf_{i}"))
        for i in range(n_noise):
            self.traders.append(NoiseTrader(f"noise_{i}"))
        for i in range(n_insiders):
            self.traders.append(InsiderTrader(f"insider_{i}"))
        for i in range(n_contrarians):
            self.traders.append(ContrarianTrader(f"contrarian_{i}"))
        for i in range(n_momentum):
            self.traders.append(MomentumTrader(f"momentum_{i}"))
        self.n_rounds = n_rounds

    def simulate(self, event: str, true_probability: float,
                  initial_price: float = 0.5) -> MarketResult:
        market = PredictionMarket(event)
        market.yes_shares = math.log(initial_price / (1 - initial_price + 0.01)) * market.liquidity
        market.price_history = [market.current_price()]
        for round_num in range(self.n_rounds):
            np.random.shuffle(self.traders)
            for trader in self.traders:
                trade = trader.decide(true_probability=0.5, market_price=market.current_price(), round_num=round_num)
                if trade is not None:
                    market.execute_trade(trade)
        final_prob = market.current_price()
        sf_prices = [t.trades[-1].price for t in self.traders if t.trader_type == "superforecaster" and t.trades]
        sf_consensus = np.mean(sf_prices) if sf_prices else final_prob
        price_arr = np.array(market.price_history)
        efficiency = 1.0 - np.mean(np.abs(np.diff(price_arr))) / max(0.01, np.mean(price_arr))
        calibration_error = abs(final_prob - true_probability)
        trader_pnl = {}
        for trader in self.traders:
            if trader.trader_type not in trader_pnl:
                trader_pnl[trader.trader_type] = 0
            pnl = sum(t.quantity for t in trader.trades if t.direction == "buy") - sum(t.quantity for t in trader.trades if t.direction == "sell")
            trader_pnl[trader.trader_type] += pnl
        ci_lower = max(0.01, final_prob - 1.96 * math.sqrt(final_prob * (1 - final_prob) / max(1, self.n_rounds * len(self.traders))))
        ci_upper = min(0.99, final_prob + 1.96 * math.sqrt(final_prob * (1 - final_prob) / max(1, self.n_rounds * len(self.traders))))
        return MarketResult(
            event=event,
            final_probability=round(final_prob, 4),
            confidence_interval=(round(ci_lower, 4), round(ci_upper, 4)),
            price_history=[round(p, 4) for p in market.price_history[-50:]],
            n_traders=len(self.traders),
            n_rounds=self.n_rounds,
            superforecaster_consensus=round(sf_consensus, 4),
            market_efficiency=round(max(0, min(1, efficiency)), 4),
            calibration_error=round(calibration_error, 4),
            trader_pnl={k: round(v, 2) for k, v in trader_pnl.items()},
        )

    def simulate_multiple_events(self, events: Dict[str, float]) -> Dict[str, MarketResult]:
        results = {}
        for event, prob in events.items():
            results[event] = self.simulate(event, prob)
        return results