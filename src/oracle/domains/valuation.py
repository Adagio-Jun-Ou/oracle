import numpy as np
from typing import Optional


class ValuationModels:

    def __init__(self):
        self.name = "valuation_models"

    @staticmethod
    def dividend_discount_model(
        current_dividend: float,
        growth_rate: float,
        discount_rate: float,
        terminal_growth: Optional[float] = None,
        stages: int = 1,
        years_per_stage: Optional[list] = None,
    ) -> dict:
        if discount_rate <= growth_rate:
            growth_rate = discount_rate - 0.001
        if terminal_growth is None:
            terminal_growth = min(growth_rate * 0.5, discount_rate * 0.4)
        if terminal_growth >= discount_rate:
            terminal_growth = discount_rate * 0.4
        if years_per_stage is None:
            years_per_stage = [5] * stages
        total_value = 0.0
        year = 0
        dividend = current_dividend
        growth_rates = []
        for stage in range(stages):
            stage_growth = growth_rate * (1 - stage * 0.3) if stages > 1 else growth_rate
            stage_growth = max(stage_growth, 0)
            for y in range(years_per_stage[stage]):
                year += 1
                dividend = dividend * (1 + stage_growth)
                pv = dividend / (1 + discount_rate) ** year
                total_value += pv
                growth_rates.append(stage_growth)
        terminal_value = dividend * (1 + terminal_growth) / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / (1 + discount_rate) ** year
        total_value += pv_terminal
        return {
            "intrinsic_value": round(float(total_value), 4),
            "current_dividend": current_dividend,
            "growth_rate": round(growth_rate, 6),
            "terminal_growth": round(terminal_growth, 6),
            "discount_rate": round(discount_rate, 6),
            "terminal_value": round(float(terminal_value), 4),
            "pv_terminal": round(float(pv_terminal), 4),
            "pv_dividends": round(float(total_value - pv_terminal), 4),
            "stages": stages,
            "years_per_stage": years_per_stage,
            "terminal_value_pct": round(float(pv_terminal / total_value * 100), 2) if total_value > 0 else 0,
        }

    @staticmethod
    def discounted_cash_flow(
        fcf_current: float,
        growth_rates: list,
        terminal_growth: float,
        wacc: float,
        net_debt: float = 0,
        shares_outstanding: float = 1,
    ) -> dict:
        total_fcf_pv = 0.0
        year_fcfs = []
        fcf = fcf_current
        for year, g in enumerate(growth_rates, 1):
            fcf = fcf * (1 + g)
            pv = fcf / (1 + wacc) ** year
            total_fcf_pv += pv
            year_fcfs.append({
                "year": year,
                "fcf": round(float(fcf), 4),
                "growth": round(float(g), 4),
                "pv": round(float(pv), 4),
            })
        terminal_fcf = fcf * (1 + terminal_growth)
        terminal_value = terminal_fcf / (wacc - terminal_growth) if wacc > terminal_growth else 0
        pv_terminal = terminal_value / (1 + wacc) ** len(growth_rates)
        enterprise_value = total_fcf_pv + pv_terminal
        equity_value = enterprise_value - net_debt
        per_share = equity_value / shares_outstanding if shares_outstanding > 0 else equity_value
        return {
            "enterprise_value": round(float(enterprise_value), 2),
            "equity_value": round(float(equity_value), 2),
            "per_share_value": round(float(per_share), 4),
            "total_fcf_pv": round(float(total_fcf_pv), 2),
            "terminal_value": round(float(terminal_value), 2),
            "pv_terminal": round(float(pv_terminal), 2),
            "net_debt": net_debt,
            "shares_outstanding": shares_outstanding,
            "wacc": round(wacc, 4),
            "terminal_growth": round(terminal_growth, 4),
            "year_fcfs": year_fcfs,
            "terminal_value_pct": round(float(pv_terminal / enterprise_value * 100), 2) if enterprise_value > 0 else 0,
        }

    @staticmethod
    def residual_income_model(
        book_value: float,
        earnings_per_share: float,
        cost_of_equity: float,
        growth_rates: list,
        terminal_persistence: float = 0.5,
    ) -> dict:
        current_value = book_value
        residual_incomes = []
        ri_total = 0.0
        eps = earnings_per_share
        for year, g in enumerate(growth_rates, 1):
            eps = eps * (1 + g)
            expected_return = book_value * cost_of_equity
            ri = eps - expected_return
            pv_ri = ri / (1 + cost_of_equity) ** year
            ri_total += pv_ri
            residual_incomes.append({
                "year": year,
                "eps": round(float(eps), 4),
                "expected_return": round(float(expected_return), 4),
                "residual_income": round(float(ri), 4),
                "pv_ri": round(float(pv_ri), 4),
            })
            book_value += eps - earnings_per_share * cost_of_equity
        terminal_ri = residual_incomes[-1]["residual_income"] * terminal_persistence if residual_incomes else 0
        terminal_ri_value = terminal_ri / (cost_of_equity * (1 + cost_of_equity) ** len(growth_rates))
        total_value = book_value + ri_total + terminal_ri_value
        return {
            "intrinsic_value": round(float(total_value), 4),
            "book_value": round(float(book_value), 4),
            "ri_contribution": round(float(ri_total), 4),
            "terminal_ri_value": round(float(terminal_ri_value), 4),
            "residual_incomes": residual_incomes,
            "terminal_persistence": terminal_persistence,
            "cost_of_equity": round(cost_of_equity, 4),
        }

    @staticmethod
    def eva_model(
        nopat: float,
        invested_capital: float,
        wacc: float,
        growth_rates: list,
        terminal_roic: Optional[float] = None,
    ) -> dict:
        if terminal_roic is None:
            terminal_roic = wacc + 0.01
        current_eva = nopat - invested_capital * wacc
        eva_total = 0.0
        eva_list = []
        eva = current_eva
        ic = invested_capital
        for year, g in enumerate(growth_rates, 1):
            eva = eva * (1 + g)
            ic = ic * (1 + g * 0.3)
            pv_eva = eva / (1 + wacc) ** year
            eva_total += pv_eva
            eva_list.append({
                "year": year,
                "eva": round(float(eva), 4),
                "roic": round(float(nopat / ic), 4) if ic > 0 else 0,
                "spread": round(float((nopat / ic - wacc) * 100), 2) if ic > 0 else 0,
                "pv_eva": round(float(pv_eva), 4),
            })
        terminal_eva = eva_list[-1]["eva"] * 1.0 if eva_list else current_eva
        terminal_value = terminal_eva / (wacc - 0.02) if wacc > 0.02 else 0
        pv_terminal = terminal_value / (1 + wacc) ** len(growth_rates)
        total_value = invested_capital + eva_total + pv_terminal
        return {
            "enterprise_value": round(float(total_value), 2),
            "invested_capital": round(float(invested_capital), 2),
            "eva_contribution": round(float(eva_total), 2),
            "terminal_value": round(float(pv_terminal), 2),
            "current_eva": round(float(current_eva), 4),
            "current_roic": round(float(nopat / invested_capital * 100), 2) if invested_capital > 0 else 0,
            "wacc": round(wacc, 4),
            "eva_years": eva_list,
            "terminal_value_pct": round(float(pv_terminal / total_value * 100), 2) if total_value > 0 else 0,
        }

    @staticmethod
    def SensitivityAnalysis(
        model: str = "ddm",
        base_params: Optional[dict] = None,
        param_ranges: Optional[dict] = None,
    ) -> dict:
        vm = ValuationModels()
        if base_params is None:
            base_params = {"current_dividend": 2.0, "growth_rate": 0.05, "discount_rate": 0.10}
        if param_ranges is None:
            param_ranges = {
                "growth_rate": [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
                "discount_rate": [0.08, 0.09, 0.10, 0.11, 0.12, 0.13],
            }
        base_result = vm.dividend_discount_model(**base_params)
        base_value = base_result["intrinsic_value"]
        sensitivities = {}
        for param_name, values in param_ranges.items():
            results = []
            for v in values:
                params = base_params.copy()
                params[param_name] = v
                try:
                    if model == "ddm":
                        r = vm.dividend_discount_model(**params)
                    else:
                        r = vm.dividend_discount_model(**params)
                    results.append({"param_value": round(v, 4), "value": r["intrinsic_value"]})
                except Exception:
                    results.append({"param_value": round(v, 4), "value": None})
            sensitivities[param_name] = results
        return {
            "base_value": base_value,
            "base_params": {k: round(v, 4) for k, v in base_params.items()},
            "sensitivities": sensitivities,
            "model": model,
        }