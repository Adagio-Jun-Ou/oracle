import numpy as np
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class SEIRState:
    susceptible: float = 0.0
    exposed: float = 0.0
    infectious: float = 0.0
    recovered: float = 0.0
    deceased: float = 0.0
    vaccinated: float = 0.0
    hospitalized: float = 0.0


@dataclass
class EpidemicPrediction:
    peak_infected: float = 0.0
    peak_day: int = 0
    total_infected: float = 0.0
    total_deceased: float = 0.0
    total_recovered: float = 0.0
    r0: float = 0.0
    herd_immunity_threshold: float = 0.0
    days_to_peak: int = 0
    duration_days: int = 0
    daily_curve: list = field(default_factory=list)


class EpidemiologyModel:

    def __init__(self):
        self.name = "epidemiology_model"

    @staticmethod
    def seir(
        population: float = 1e6,
        initial_infected: float = 100,
        beta: float = 0.3,
        sigma: float = 1 / 5.2,
        gamma: float = 1 / 12,
        mu: float = 0.005,
        vaccination_rate: float = 0.0,
        vaccine_efficacy: float = 0.9,
        hospitalization_rate: float = 0.05,
        days: int = 365,
        intervention_day: Optional[int] = None,
        intervention_beta: Optional[float] = None,
    ) -> dict:
        N = population
        S = N - initial_infected
        E = 0.0
        I = initial_infected
        R = 0.0
        D = 0.0
        V = 0.0
        H = 0.0
        daily_S, daily_E, daily_I, daily_R, daily_D, daily_H = [S], [E], [I], [R], [D], [H]
        r0 = beta / gamma
        herd_threshold = 1 - 1 / r0
        for day in range(days):
            current_beta = beta
            if intervention_day is not None and day >= intervention_day and intervention_beta is not None:
                current_beta = intervention_beta
            daily_vax = min(S * (vaccination_rate / 365), S)
            S -= daily_vax
            V += daily_vax * vaccine_efficacy
            new_exposed = current_beta * S * I / N
            new_infectious = sigma * E
            new_recovered = gamma * I * (1 - mu)
            new_deceased = gamma * I * mu
            new_hospitalized = new_infectious * hospitalization_rate
            S = max(S - new_exposed, 0)
            E = max(E + new_exposed - new_infectious, 0)
            I = max(I + new_infectious - new_recovered - new_deceased, 0)
            R += new_recovered
            D += new_deceased
            H = max(H + new_hospitalized - new_recovered * 0.3, 0)
            daily_S.append(S)
            daily_E.append(E)
            daily_I.append(I)
            daily_R.append(R)
            daily_D.append(D)
            daily_H.append(H)
        peak_infected = max(daily_I)
        peak_day = daily_I.index(peak_infected)
        total_infected = max(daily_I) + sum(daily_R) / max(len(daily_R), 1) * (days / days)
        return {
            "peak_infected": round(float(peak_infected), 0),
            "peak_day": peak_day,
            "total_deceased": round(float(D), 0),
            "total_recovered": round(float(R), 0),
            "total_vaccinated": round(float(V), 0),
            "r0": round(float(r0), 4),
            "herd_immunity_threshold": round(float(herd_threshold), 4),
            "days_to_peak": peak_day,
            "duration_days": days,
            "population": float(population),
            "parameters": {
                "beta": beta, "sigma": sigma, "gamma": gamma, "mu": mu,
                "vaccination_rate": vaccination_rate, "vaccine_efficacy": vaccine_efficacy,
                "hospitalization_rate": hospitalization_rate,
            },
            "daily": {
                "susceptible": [round(float(x), 0) for x in daily_S[::7]],
                "exposed": [round(float(x), 0) for x in daily_E[::7]],
                "infectious": [round(float(x), 0) for x in daily_I[::7]],
                "recovered": [round(float(x), 0) for x in daily_R[::7]],
                "deceased": [round(float(x), 0) for x in daily_D[::7]],
                "hospitalized": [round(float(x), 0) for x in daily_H[::7]],
            },
            "intervention_effect": None,
        }

    @staticmethod
    def pandemic_scenarios(
        population: float = 1e6,
        base_r0: float = 2.5,
        ifr: float = 0.005,
        scenarios: Optional[dict] = None,
    ) -> dict:
        epi = EpidemiologyModel()
        gamma = 1 / 12
        if scenarios is None:
            scenarios = {
                "no_intervention": {"beta": base_r0 * gamma, "intervention_day": None, "intervention_beta": None},
                "moderate_lockdown": {"beta": base_r0 * gamma, "intervention_day": 30, "intervention_beta": 1.2 * gamma},
                "strict_lockdown": {"beta": base_r0 * gamma, "intervention_day": 14, "intervention_beta": 0.7 * gamma},
                "vaccination_50pct": {"beta": base_r0 * gamma, "vaccination_rate": 0.50},
                "vaccination_80pct": {"beta": base_r0 * gamma, "vaccination_rate": 0.80},
            }
        results = {}
        for name, params in scenarios.items():
            result = epi.seir(
                population=population,
                beta=params.get("beta", base_r0 * gamma),
                mu=ifr,
                intervention_day=params.get("intervention_day"),
                intervention_beta=params.get("intervention_beta"),
                vaccination_rate=params.get("vaccination_rate", 0.0),
            )
            results[name] = {
                "peak_infected": result["peak_infected"],
                "peak_day": result["peak_day"],
                "total_deceased": result["total_deceased"],
                "total_recovered": result["total_recovered"],
                "r0": result["r0"],
            }
        return {"population": population, "base_r0": base_r0, "ifr": ifr, "scenarios": results}

    @staticmethod
    def health_risk_score(
        age: int = 40,
        bmi: float = 25.0,
        smoking: bool = False,
        diabetes: bool = False,
        hypertension: bool = False,
        cardiovascular: bool = False,
        respiratory: bool = False,
        immunocompromised: bool = False,
        region_risk: float = 0.5,
        healthcare_access: float = 0.8,
    ) -> dict:
        base_risk = 0.01
        if 18 <= age <= 30:
            age_factor = 0.3
        elif 30 < age <= 45:
            age_factor = 0.5
        elif 45 < age <= 60:
            age_factor = 1.0
        elif 60 < age <= 75:
            age_factor = 2.5
        else:
            age_factor = 5.0
        bmi_factor = 1.0 + max(0, (bmi - 25) / 15)
        comorbidity_count = sum([smoking, diabetes, hypertension, cardiovascular, respiratory, immunocompromised])
        comorbidity_factor = 1.0 + comorbidity_count * 0.5
        region_factor = 1.0 + (1.0 - region_risk) * 0.5
        access_factor = 1.0 + (1.0 - healthcare_access) * 0.8
        total_risk = base_risk * age_factor * bmi_factor * comorbidity_factor * region_factor * access_factor
        total_risk = min(total_risk, 0.95)
        severity = "low" if total_risk < 0.02 else "moderate" if total_risk < 0.05 else "high" if total_risk < 0.15 else "very_high" if total_risk < 0.30 else "critical"
        risks = {
            "cardiovascular": min(base_risk * age_factor * 3 * (2 if diabetes or hypertension else 1) * (1.5 if smoking else 1), 0.9),
            "respiratory": min(base_risk * age_factor * 2 * (2 if respiratory or smoking else 1), 0.8),
            "metabolic": min(base_risk * age_factor * 2.5 * (2 if diabetes else 1) * bmi_factor, 0.85),
            "infection": min(base_risk * age_factor * (3 if immunocompromised else 1) * region_factor, 0.9),
        }
        recommendations = []
        if smoking:
            recommendations.append("Smoking cessation - highest impact intervention")
        if bmi > 30:
            recommendations.append("Weight management - target BMI < 30")
        if diabetes:
            recommendations.append("Blood glucose monitoring and HbA1c control")
        if hypertension:
            recommendations.append("Blood pressure monitoring and management")
        if age > 60:
            recommendations.append("Prioritize vaccination and preventive screening")
        if healthcare_access < 0.5:
            recommendations.append("Improve healthcare access - telemedicine options")
        if comorbidity_count >= 3:
            recommendations.append("High comorbidity burden - integrated care pathway recommended")
        if not recommendations:
            recommendations.append("Continue current health maintenance")
        return {
            "total_risk_score": round(float(total_risk), 4),
            "severity": severity,
            "age_factor": round(age_factor, 2),
            "bmi_factor": round(bmi_factor, 2),
            "comorbidity_factor": round(comorbidity_factor, 2),
            "comorbidity_count": comorbidity_count,
            "region_factor": round(region_factor, 2),
            "access_factor": round(access_factor, 2),
            "specific_risks": {k: round(v, 4) for k, v in risks.items()},
            "recommendations": recommendations,
        }

    @staticmethod
    def disease_spread_network(
        n_nodes: int = 100,
        n_edges: int = 200,
        initial_infected: int = 3,
        transmission_prob: float = 0.15,
        recovery_prob: float = 0.05,
        days: int = 60,
        seed: int = 42,
    ) -> dict:
        np.random.seed(seed)
        adjacency = {i: set() for i in range(n_nodes)}
        edges_added = 0
        while edges_added < n_edges:
            i, j = np.random.randint(0, n_nodes, 2)
            if i != j and j not in adjacency[i]:
                adjacency[i].add(j)
                adjacency[j].add(i)
                edges_added += 1
        states = np.zeros(n_nodes, dtype=int)
        initially = np.random.choice(n_nodes, min(initial_infected, n_nodes), replace=False)
        states[initially] = 1
        daily_counts = []
        for day in range(days):
            new_states = states.copy()
            for node in range(n_nodes):
                if states[node] == 1:
                    for neighbor in adjacency[node]:
                        if states[neighbor] == 0 and np.random.random() < transmission_prob:
                            new_states[neighbor] = 1
                    if np.random.random() < recovery_prob:
                        new_states[node] = 2
            states = new_states
            infected = int(np.sum(states == 1))
            susceptible = int(np.sum(states == 0))
            recovered = int(np.sum(states == 2))
            daily_counts.append({"day": day, "susceptible": susceptible, "infected": infected, "recovered": recovered})
        return {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "average_degree": round(2 * n_edges / n_nodes, 2),
            "peak_infected": max(d["infected"] for d in daily_counts),
            "peak_day": max(range(len(daily_counts)), key=lambda i: daily_counts[i]["infected"]),
            "final_infected_pct": round(daily_counts[-1]["recovered"] / n_nodes * 100, 2),
            "daily_counts": daily_counts[::3],
            "parameters": {"transmission_prob": transmission_prob, "recovery_prob": recovery_prob},
        }

    @staticmethod
    def intervention_optimizer(
        population: float = 1e6,
        initial_infected: float = 100,
        beta: float = 0.3,
        gamma: float = 1 / 12,
        mu: float = 0.005,
        days: int = 365,
        candidate_intervention_days: Optional[list] = None,
        candidate_intervention_betas: Optional[list] = None,
        vaccination_rates: Optional[list] = None,
    ) -> dict:
        if candidate_intervention_days is None:
            candidate_intervention_days = [7, 14, 21, 30, 45, 60, 90]
        if candidate_intervention_betas is None:
            candidate_intervention_betas = [beta * f for f in [0.1, 0.2, 0.3, 0.5, 0.7]]
        if vaccination_rates is None:
            vaccination_rates = [0.0, 0.3, 0.5, 0.8]
        epi = EpidemiologyModel()
        best = {"score": float("inf"), "config": None, "result": None}
        all_results = []
        baseline = epi.seir(population=population, initial_infected=initial_infected, beta=beta, gamma=gamma, mu=mu, days=days)
        baseline_deaths = baseline["total_deceased"]
        baseline_peak = baseline["peak_infected"]
        for int_day in candidate_intervention_days:
            for int_beta in candidate_intervention_betas:
                result = epi.seir(
                    population=population, initial_infected=initial_infected,
                    beta=beta, gamma=gamma, mu=mu, days=days,
                    intervention_day=int_day, intervention_beta=int_beta,
                )
                deaths_prevented = baseline_deaths - result["total_deceased"]
                peak_reduction = (baseline_peak - result["peak_infected"]) / max(baseline_peak, 1)
                delay_days = result["peak_day"] - baseline["peak_day"]
                score = -(deaths_prevented * 0.5 + peak_reduction * population * 0.3 + delay_days * 1000 * 0.2)
                config_key = f"day{int_day}_beta{int_beta:.3f}"
                entry = {
                    "intervention_day": int_day,
                    "intervention_beta": round(int_beta, 4),
                    "intervention_strength": round((1 - int_beta / beta) * 100, 1),
                    "deaths_prevented": round(deaths_prevented, 0),
                    "peak_reduction_pct": round(peak_reduction * 100, 1),
                    "peak_delay_days": delay_days,
                    "score": round(score, 0),
                }
                all_results.append(entry)
                if score < best["score"]:
                    best = {"score": score, "config": entry, "result": result}
        for vax_rate in vaccination_rates:
            if vax_rate > 0:
                result = epi.seir(
                    population=population, initial_infected=initial_infected,
                    beta=beta, gamma=gamma, mu=mu, days=days,
                    vaccination_rate=vax_rate,
                )
                deaths_prevented = baseline_deaths - result["total_deceased"]
                peak_reduction = (baseline_peak - result["peak_infected"]) / max(baseline_peak, 1)
                all_results.append({
                    "intervention_day": 0,
                    "vaccination_rate": vax_rate,
                    "deaths_prevented": round(deaths_prevented, 0),
                    "peak_reduction_pct": round(peak_reduction * 100, 1),
                    "score": round(-(deaths_prevented * 0.5 + peak_reduction * population * 0.3), 0),
                })
        all_results.sort(key=lambda x: x["score"])
        optimal = best["config"]
        return {
            "baseline": {"deaths": baseline_deaths, "peak_infected": baseline_peak, "peak_day": baseline["peak_day"]},
            "optimal_intervention": optimal,
            "ranking": all_results[:10],
            "tradeoffs": {
                "early_aggressive": {"description": "Intervene early with strong measures", "deaths_prevented": all_results[0]["deaths_prevented"] if all_results else 0, "constraint": "Maximum economic disruption"},
                "delayed_moderate": {"description": "Wait for data, moderate intervention", "deaths_prevented": all_results[len(all_results)//2]["deaths_prevented"] if all_results else 0, "constraint": "Higher peak, more deaths delayed"},
                "vaccination_focused": {"description": "Minimal NPI, maximize vaccination", "deaths_prevented": 0, "constraint": "Only effective with high vax rate"},
            },
            "recommendation": f"Optimal: Intervene on day {optimal['intervention_day']} with {optimal.get('intervention_strength', 0):.0f}% transmission reduction" if optimal else "Insufficient data for recommendation",
        }