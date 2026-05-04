from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def censored_exponential_mle(times: np.ndarray, events: np.ndarray) -> Dict[str, float]:
    total_time = float(np.sum(times))
    observed_events = int(np.sum(events))
    if total_time <= 0.0:
        return {"lambda_hat": np.nan, "tau_hat": np.nan, "events": observed_events, "total_time": total_time}
    if observed_events <= 0:
        return {"lambda_hat": 0.0, "tau_hat": np.inf, "events": observed_events, "total_time": total_time}
    lambda_hat = observed_events / total_time
    tau_hat = total_time / observed_events
    return {"lambda_hat": lambda_hat, "tau_hat": tau_hat, "events": observed_events, "total_time": total_time}


def kaplan_meier_curve(times: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if times.size == 0:
        return np.array([0.0]), np.array([1.0])
    order = np.argsort(times)
    times_sorted = times[order]
    events_sorted = events[order].astype(bool)
    unique_times = np.unique(times_sorted)
    n_at_risk = len(times_sorted)
    survival = 1.0
    curve_times = [0.0]
    curve_survival = [1.0]
    for current_time in unique_times:
        mask = times_sorted == current_time
        deaths = int(np.sum(events_sorted[mask]))
        censored = int(np.sum(~events_sorted[mask]))
        if deaths > 0 and n_at_risk > 0:
            survival *= 1.0 - deaths / n_at_risk
            curve_times.append(float(current_time))
            curve_survival.append(float(survival))
        n_at_risk -= deaths + censored
    return np.asarray(curve_times, dtype=float), np.asarray(curve_survival, dtype=float)


def kaplan_meier_median(times: np.ndarray, events: np.ndarray) -> float:
    curve_times, curve_survival = kaplan_meier_curve(times, events)
    below = np.where(curve_survival <= 0.5)[0]
    if below.size == 0:
        return float("inf")
    return float(curve_times[below[0]])


def bootstrap_lifetime_ci(
    times: np.ndarray,
    events: np.ndarray,
    n_bootstrap: int,
    seed: int,
    metric: str = "mle",
) -> Dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    n_samples = len(times)
    if n_samples == 0:
        return {"lower": np.nan, "upper": np.nan, "samples": np.asarray([], dtype=float)}

    estimates = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled_indices = rng.integers(0, n_samples, size=n_samples)
        sampled_times = times[sampled_indices]
        sampled_events = events[sampled_indices]
        if metric == "kaplan_meier_median":
            estimates[index] = kaplan_meier_median(sampled_times, sampled_events)
        else:
            estimates[index] = censored_exponential_mle(sampled_times, sampled_events)["tau_hat"]

    finite = np.isfinite(estimates)
    if not np.any(finite):
        return {"lower": np.nan, "upper": np.nan, "samples": estimates}
    finite_estimates = estimates[finite]
    lower = float(np.percentile(finite_estimates, 2.5))
    upper = float(np.percentile(finite_estimates, 97.5))
    return {"lower": lower, "upper": upper, "samples": estimates}


def survival_linearity_r_squared(times: np.ndarray, events: np.ndarray) -> float:
    curve_times, curve_survival = kaplan_meier_curve(times, events)
    mask = (curve_times > 0.0) & (curve_survival > 0.0) & np.isfinite(curve_survival)
    x = curve_times[mask]
    y = np.log(curve_survival[mask])
    if x.size < 3:
        return np.nan
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = slope * x + intercept
    residual = np.sum((y - fitted) ** 2)
    total = np.sum((y - np.mean(y)) ** 2)
    if total <= 0.0:
        return np.nan
    return float(1.0 - residual / total)


def fit_log_linear_relationship(x: np.ndarray, lifetimes: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(lifetimes) & (lifetimes > 0.0)
    x_clean = x[mask]
    y_clean = np.log(lifetimes[mask])
    if x_clean.size < 2:
        return {"slope": np.nan, "intercept": np.nan, "r_squared": np.nan}
    slope, intercept = np.polyfit(x_clean, y_clean, deg=1)
    fitted = slope * x_clean + intercept
    residual = np.sum((y_clean - fitted) ** 2)
    total = np.sum((y_clean - np.mean(y_clean)) ** 2)
    r_squared = np.nan if total <= 0.0 else float(1.0 - residual / total)
    return {"slope": float(slope), "intercept": float(intercept), "r_squared": r_squared}


def slope_uncertainty_envelope(
    x: np.ndarray,
    lower_lifetimes: np.ndarray,
    upper_lifetimes: np.ndarray,
) -> Dict[str, float]:
    lower_fit = fit_log_linear_relationship(x, lower_lifetimes)
    upper_fit = fit_log_linear_relationship(x, upper_lifetimes)
    return {
        "slope_lower": lower_fit["slope"],
        "slope_upper": upper_fit["slope"],
        "intercept_lower": lower_fit["intercept"],
        "intercept_upper": upper_fit["intercept"],
    }


def fixed_slope_intercept(x: np.ndarray, lifetimes: np.ndarray, fixed_slope: float) -> float:
    mask = np.isfinite(x) & np.isfinite(lifetimes) & (lifetimes > 0.0)
    if not np.any(mask):
        return np.nan
    y = np.log(lifetimes[mask])
    return float(np.mean(y - fixed_slope * x[mask]))


def interpolate_crossing(x: np.ndarray, y: np.ndarray, target_y: float) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if x_clean.size < 2:
        return np.nan
    for index in range(len(x_clean) - 1):
        left_y = y_clean[index]
        right_y = y_clean[index + 1]
        if (left_y >= target_y >= right_y) or (left_y <= target_y <= right_y):
            left_x = x_clean[index]
            right_x = x_clean[index + 1]
            if right_y == left_y:
                return float(left_x)
            fraction = (target_y - left_y) / (right_y - left_y)
            return float(left_x + fraction * (right_x - left_x))
    return np.nan
