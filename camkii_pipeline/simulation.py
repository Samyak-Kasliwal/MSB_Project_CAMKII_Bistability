from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import NumericalConfig, SimulationSettings

try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback exists for users who skip numba
    def njit(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def decorator(function: Any) -> Any:
            return function

        return decorator


TRANSITION_NONE = -1
TRANSITION_UP_TO_DOWN = 0
TRANSITION_DOWN_TO_UP = 1
TRANSITION_FIRST_PHOSPHORYLATION = 2

EXCHANGE_NONE = 0
EXCHANGE_A_MONOMER = 1
EXCHANGE_B_MONOMER = 2
EXCHANGE_B_DIMER = 3


INVALID_REASON_MAP = {
    0: "",
    1: "zero_total_propensity",
    2: "propensity_overflow",
    3: "state_bounds_violation",
    4: "running_sum_mismatch",
    5: "negative_time",
    6: "exchange_nonconvergent",
}


@dataclass
class ConditionSpec:
    run_id: str
    model_label: str
    transition_label: str
    M: int
    n_trajectories: int
    t_max: float
    gamma: float = 0.0
    delta_subunit: float = 0.0
    delta_holo: float = 0.0
    r1_value: float = 0.0
    epsilon_value: float = 0.0
    k1_value: float = 0.0
    k2_value: float = 0.0
    kh1_uM: float = 0.0
    kh2_uM: float = 0.0
    hill_exponent: float = 2.0
    rest_ca_uM: float = 0.1
    ltp_ca_uM: float = 1.0
    exchange_code: int = EXCHANGE_NONE
    ca_noise_enabled: bool = False
    use_full_model: bool = False
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NotebookConditionSpec:
    run_id: str
    model_label: str
    transition_label: str
    M: int
    n_trajectories: int
    t_max: float
    N: int = 12
    theta: int = 6
    r1: float = 1.5
    Vmax: float = 2.0
    Km: float = 2.0
    gamma: float = 0.01
    delta: float = 1.0e-4
    autophosphorylation_step: int = 1
    calcium_spike_enabled: bool = False
    calcium_spike_start_seconds: float = 500.0
    calcium_spike_duration_seconds: float = 20.0
    calcium_spike_period_seconds: float = 1_000.0
    calcium_spike_count: int = 20
    calcium_spike_r1_multiplier: float = 8.0
    initial_mode: str = "up"
    sweep_parameter: str = ""
    sweep_value: float = float("nan")
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def transition_code_from_label(label: str) -> int:
    normalized = label.strip().lower()
    if normalized == "up_to_down":
        return TRANSITION_UP_TO_DOWN
    if normalized == "down_to_up":
        return TRANSITION_DOWN_TO_UP
    if normalized == "first_phosphorylation":
        return TRANSITION_FIRST_PHOSPHORYLATION
    raise ValueError(f"Unsupported transition label '{label}'.")


def notebook_initial_state(condition: NotebookConditionSpec) -> np.ndarray:
    mode = condition.initial_mode.strip().lower()
    if mode == "up":
        return np.full(condition.M, condition.N, dtype=np.int64)
    if mode == "down":
        return np.zeros(condition.M, dtype=np.int64)
    if mode == "half":
        return np.full(condition.M, condition.theta, dtype=np.int64)
    if mode == "threshold_minus_one":
        return np.full(condition.M, max(condition.theta - 1, 0), dtype=np.int64)
    if mode == "single_on":
        return np.ones(condition.M, dtype=np.int64)
    raise ValueError(f"Unsupported notebook initial_mode '{condition.initial_mode}'.")


def notebook_propensity_sanity(condition: NotebookConditionSpec) -> Dict[str, float | int]:
    state = np.full(condition.M, condition.theta, dtype=np.int64)
    n_tot = int(state.sum())
    r1_total = 0.0
    r2_total = 0.0
    r4_total = 0.0
    r3_total = 0.0

    for alpha in range(condition.M):
        n_alpha = int(state[alpha])
        r1_total += condition.r1 * (condition.N - n_alpha) * n_alpha / condition.N
        r2_total += condition.Vmax * n_alpha / (condition.Km + n_tot) if n_tot > 0 else 0.0
        r4_total += condition.delta * n_alpha * n_tot / (condition.M * condition.N)

    for alpha in range(condition.M):
        for beta in range(condition.M):
            if alpha == beta:
                continue
            r3_total += (condition.gamma / condition.M) * int(state[alpha]) * (condition.N - int(state[beta]))

    return {
        "M": condition.M,
        "N": condition.N,
        "theta": condition.theta,
        "channels": 3 * condition.M + condition.M * (condition.M - 1),
        "n_tot": n_tot,
        "r1_total": r1_total,
        "r2_total": r2_total,
        "r3_total": r3_total,
        "r4_total": r4_total,
        "total_propensity": r1_total + r2_total + r3_total + r4_total,
    }


def notebook_fixed_points_single(
    N: int,
    r1: float,
    Vmax: float,
    Km: float,
    delta: float,
) -> Dict[str, Any]:
    a_value = r1 + delta
    b_value = -(r1 * N - Km * (r1 + delta))
    c_value = N * Vmax
    discriminant = b_value ** 2 - 4.0 * a_value * c_value

    fixed_points: List[float] = []
    if discriminant > 0.0:
        root = math.sqrt(discriminant)
        candidates = [(-b_value + root) / (2.0 * a_value), (-b_value - root) / (2.0 * a_value)]
        fixed_points = sorted(value for value in candidates if 0.0 < value < N)

    crit_a = Km ** 2
    crit_b = -(2.0 * r1 * N * Km + 4.0 * N * Vmax)
    crit_c = (r1 * N) ** 2
    crit_disc = crit_b ** 2 - 4.0 * crit_a * crit_c
    delta_crit = float("nan")
    if crit_disc >= 0.0:
        candidates = [
            (-crit_b + sign * math.sqrt(crit_disc)) / (2.0 * crit_a) - r1
            for sign in (1.0, -1.0)
        ]
        positive = [value for value in candidates if value > 0.0]
        if positive:
            delta_crit = min(positive)

    return {
        "N": N,
        "r1": r1,
        "Vmax": Vmax,
        "Km": Km,
        "delta": delta,
        "fixed_points": fixed_points,
        "discriminant": float(discriminant),
        "is_bistable": len(fixed_points) == 2,
        "delta_crit": float(delta_crit),
    }


@njit(cache=True)
def _hill_fraction(ca_value: float, k_half: float, hill_exponent: float) -> float:
    numerator = ca_value ** hill_exponent
    denominator = k_half ** hill_exponent + numerator
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


@njit(cache=True)
def _notebook_fraction_up_count(state: np.ndarray, theta: int) -> int:
    count = 0
    for index in range(state.shape[0]):
        if int(state[index]) >= theta:
            count += 1
    return count


@njit(cache=True)
def _notebook_is_up(up_count: int, m_count: int) -> int:
    if 2 * up_count >= m_count:
        return 1
    return 0


@njit(cache=True)
def _notebook_r1_rate(r1_value: float, n_value: int, n_subunits: int) -> float:
    return r1_value * (n_subunits - n_value) * n_value / n_subunits


@njit(cache=True)
def _notebook_spike_active(
    t_now: float,
    enabled: int,
    start_seconds: float,
    duration_seconds: float,
    period_seconds: float,
    spike_count: int,
) -> int:
    if enabled == 0 or duration_seconds <= 0.0 or spike_count <= 0:
        return 0
    if t_now < start_seconds:
        return 0
    if period_seconds <= 0.0:
        if t_now < start_seconds + duration_seconds:
            return 1
        return 0
    spike_index = int(math.floor((t_now - start_seconds) / period_seconds))
    if spike_index < 0 or spike_index >= spike_count:
        return 0
    spike_start = start_seconds + spike_index * period_seconds
    if t_now >= spike_start and t_now < spike_start + duration_seconds:
        return 1
    return 0


@njit(cache=True)
def _notebook_next_spike_boundary(
    t_now: float,
    stop_time: float,
    enabled: int,
    start_seconds: float,
    duration_seconds: float,
    period_seconds: float,
    spike_count: int,
) -> float:
    if enabled == 0 or duration_seconds <= 0.0 or spike_count <= 0:
        return stop_time

    eps = 1.0e-12
    if period_seconds <= 0.0:
        spike_end = start_seconds + duration_seconds
        if t_now + eps < start_seconds:
            return min(start_seconds, stop_time)
        if t_now + eps < spike_end:
            return min(spike_end, stop_time)
        return stop_time

    if t_now + eps < start_seconds:
        return min(start_seconds, stop_time)

    spike_index = int(math.floor((t_now - start_seconds) / period_seconds))
    if spike_index < 0:
        return min(start_seconds, stop_time)
    if spike_index >= spike_count:
        return stop_time

    spike_start = start_seconds + spike_index * period_seconds
    spike_end = spike_start + duration_seconds
    if t_now + eps < spike_end:
        return min(spike_end, stop_time)

    next_index = spike_index + 1
    if next_index < spike_count:
        next_start = start_seconds + next_index * period_seconds
        if t_now + eps < next_start:
            return min(next_start, stop_time)
    return stop_time


@njit(cache=True)
def _simulate_notebook_core(
    initial_n: np.ndarray,
    stop_time: float,
    transition_code: int,
    n_subunits: int,
    theta: int,
    r1_value: float,
    vmax_value: float,
    km_value: float,
    gamma_value: float,
    delta_value: float,
    autophosphorylation_step: int,
    calcium_spike_enabled: int,
    calcium_spike_start_seconds: float,
    calcium_spike_duration_seconds: float,
    calcium_spike_period_seconds: float,
    calcium_spike_count: int,
    calcium_spike_r1_multiplier: float,
    propensity_overflow_limit: float,
    state_tolerance: float,
    seed: int,
) -> tuple[int, float, float, int, int, int, int, np.ndarray]:
    np.random.seed(seed)

    state = initial_n.copy()
    m_count = state.shape[0]
    n_tot = 0
    for index in range(m_count):
        n_tot += int(state[index])

    t_now = 0.0
    transition_time = -1.0
    event_count = 0
    invalid_code = 0
    previous_up_count = _notebook_fraction_up_count(state, theta)

    while t_now < stop_time:
        current_r1 = r1_value
        if _notebook_spike_active(
            t_now,
            calcium_spike_enabled,
            calcium_spike_start_seconds,
            calcium_spike_duration_seconds,
            calcium_spike_period_seconds,
            calcium_spike_count,
        ) == 1:
            current_r1 = r1_value * calcium_spike_r1_multiplier

        total_r1 = 0.0
        diagonal_exchange = 0.0
        for alpha in range(m_count):
            n_alpha = int(state[alpha])
            total_r1 += _notebook_r1_rate(current_r1, n_alpha, n_subunits)
            diagonal_exchange += n_alpha * (n_subunits - n_alpha)

        total_r2 = 0.0
        if n_tot > 0:
            total_r2 = vmax_value * n_tot / (km_value + n_tot)

        total_exchange = (gamma_value / m_count) * (n_tot * (m_count * n_subunits - n_tot) - diagonal_exchange)
        if total_exchange < 0.0:
            total_exchange = 0.0

        total_turnover = 0.0
        if n_tot > 0:
            total_turnover = delta_value * n_tot * n_tot / (m_count * n_subunits)

        a0 = total_r1 + total_r2 + total_exchange + total_turnover
        if a0 <= 0.0:
            break
        if a0 >= propensity_overflow_limit:
            invalid_code = 2
            break

        u_wait = np.random.random()
        if u_wait < 1.0e-15:
            u_wait = 1.0e-15
        tau_step = -math.log(u_wait) / a0

        next_boundary = _notebook_next_spike_boundary(
            t_now,
            stop_time,
            calcium_spike_enabled,
            calcium_spike_start_seconds,
            calcium_spike_duration_seconds,
            calcium_spike_period_seconds,
            calcium_spike_count,
        )
        if t_now + tau_step > next_boundary:
            t_now = next_boundary
            continue

        if t_now + tau_step > stop_time:
            t_now = stop_time
            break

        event_threshold = np.random.random() * a0
        previous_n_tot = n_tot
        previous_up_count = _notebook_fraction_up_count(state, theta)
        t_now += tau_step

        if event_threshold < total_r1:
            cumulative = 0.0
            selected_alpha = 0
            for alpha in range(m_count):
                cumulative += _notebook_r1_rate(current_r1, int(state[alpha]), n_subunits)
                selected_alpha = alpha
                if event_threshold <= cumulative:
                    break
            old_n = int(state[selected_alpha])
            new_n = old_n + autophosphorylation_step
            if new_n > n_subunits:
                new_n = n_subunits
            state[selected_alpha] = new_n
            n_tot += new_n - old_n
        elif event_threshold < total_r1 + total_r2:
            local_threshold = event_threshold - total_r1
            cumulative = 0.0
            selected_alpha = 0
            if n_tot <= 0:
                invalid_code = 1
                break
            for alpha in range(m_count):
                cumulative += total_r2 * int(state[alpha]) / n_tot
                selected_alpha = alpha
                if local_threshold <= cumulative:
                    break
            state[selected_alpha] -= 1
            n_tot -= 1
        elif event_threshold < total_r1 + total_r2 + total_exchange:
            local_threshold = event_threshold - total_r1 - total_r2
            exchange_scale = gamma_value / m_count
            total_accept = m_count * n_subunits - n_tot
            cumulative = 0.0
            selected_alpha = 0
            fallback_alpha = 0
            for alpha in range(m_count):
                donor = int(state[alpha])
                donor_rate = exchange_scale * donor * (total_accept - (n_subunits - donor))
                if donor_rate > 0.0:
                    fallback_alpha = alpha
                cumulative += donor_rate
                selected_alpha = alpha
                if local_threshold <= cumulative:
                    break
            if int(state[selected_alpha]) <= 0:
                selected_alpha = fallback_alpha
            donor_before = int(state[selected_alpha])
            donor_rate_selected = exchange_scale * donor_before * (total_accept - (n_subunits - donor_before))
            donor_threshold = local_threshold - (cumulative - donor_rate_selected)
            cumulative_beta = 0.0
            selected_beta = 0
            fallback_beta = 0
            for beta in range(m_count):
                if beta == selected_alpha:
                    continue
                rate = exchange_scale * donor_before * (n_subunits - int(state[beta]))
                if rate > 0.0:
                    fallback_beta = beta
                cumulative_beta += rate
                selected_beta = beta
                if donor_threshold <= cumulative_beta:
                    break
            if selected_beta == selected_alpha or int(state[selected_beta]) >= n_subunits:
                selected_beta = fallback_beta
            state[selected_alpha] -= 1
            state[selected_beta] += 1
        else:
            local_threshold = event_threshold - total_r1 - total_r2 - total_exchange
            cumulative = 0.0
            selected_alpha = 0
            if n_tot <= 0:
                invalid_code = 1
                break
            for alpha in range(m_count):
                cumulative += total_turnover * int(state[alpha]) / n_tot
                selected_alpha = alpha
                if local_threshold <= cumulative:
                    break
            state[selected_alpha] -= 1
            n_tot -= 1

        event_count += 1

        if t_now < 0.0:
            invalid_code = 5
            break

        recomputed_total = 0
        for alpha in range(m_count):
            n_alpha = int(state[alpha])
            if n_alpha < 0 or n_alpha > n_subunits:
                invalid_code = 3
                break
            recomputed_total += n_alpha
        if invalid_code != 0:
            break
        if math.fabs(recomputed_total - n_tot) > state_tolerance:
            invalid_code = 4
            break

        current_up_count = _notebook_fraction_up_count(state, theta)
        if transition_code == TRANSITION_FIRST_PHOSPHORYLATION:
            if previous_n_tot == 0 and n_tot > 0:
                transition_time = t_now
                return 1, transition_time, t_now, n_tot, current_up_count, event_count, invalid_code, state
        elif transition_code == TRANSITION_UP_TO_DOWN:
            if _notebook_is_up(previous_up_count, m_count) == 1 and _notebook_is_up(current_up_count, m_count) == 0:
                transition_time = t_now
                return 1, transition_time, t_now, n_tot, current_up_count, event_count, invalid_code, state
        elif transition_code == TRANSITION_DOWN_TO_UP:
            if _notebook_is_up(previous_up_count, m_count) == 0 and _notebook_is_up(current_up_count, m_count) == 1:
                transition_time = t_now
                return 1, transition_time, t_now, n_tot, current_up_count, event_count, invalid_code, state

    final_up_count = _notebook_fraction_up_count(state, theta)
    return 0, transition_time, t_now, n_tot, final_up_count, event_count, invalid_code, state


@njit(cache=True)
def _simulate_simple_core(
    initial_n: np.ndarray,
    stop_time: float,
    transition_code: int,
    n_subunits: int,
    theta: int,
    r1_base: float,
    epsilon_value: float,
    k_cat: float,
    effective_km_count: float,
    gamma_value: float,
    exchange_code: int,
    delta_subunit: float,
    delta_holo: float,
    ca_noise_enabled: int,
    ca_noise_low: float,
    ca_noise_high: float,
    ca_noise_interval: float,
    ca_noise_kh1: float,
    hill_exponent: float,
    propensity_overflow_limit: float,
    max_consecutive_exchange_events: int,
    state_tolerance: float,
    use_kahan: int,
    seed: int,
) -> tuple[int, float, float, float, int, int, int, int, np.ndarray]:
    np.random.seed(seed)

    state = initial_n.copy()
    m_count = state.shape[0]
    n_tot = 0
    for index in range(m_count):
        n_tot += int(state[index])

    t_now = 0.0
    next_ca_update = ca_noise_interval
    current_r1 = r1_base
    trigger_time = -1.0
    confirmation_time = -1.0
    pending_trigger = False
    events_since_trigger = 0
    event_count = 0
    consecutive_exchange_events = 0
    invalid_code = 0
    warning_code = 0

    trigger_threshold = theta * m_count
    confirm_low_threshold = 0.2 * m_count * n_subunits
    confirm_high_threshold = 0.4 * m_count * n_subunits

    while t_now < stop_time:
        total_r1 = 0.0
        total_r2 = 0.0
        total_subturn = 0.0
        total_holoturn = 0.0
        diagonal_exchange = 0.0
        donor_sum = 0.0
        accept_sum = 0.0
        diagonal_dimer = 0.0

        c_r1 = 0.0
        c_r2 = 0.0
        c_subturn = 0.0
        c_holoturn = 0.0
        c_diag_exchange = 0.0
        c_donor = 0.0
        c_accept = 0.0
        c_diag_dimer = 0.0

        dephospho_denominator = effective_km_count + n_tot

        for alpha in range(m_count):
            n_alpha = int(state[alpha])
            r1_alpha = current_r1 * (n_subunits - n_alpha) * (n_alpha / n_subunits + epsilon_value)
            r2_alpha = 0.0
            subturn_alpha = 0.0
            holo_alpha = delta_holo
            if dephospho_denominator > 0.0:
                r2_alpha = k_cat * n_alpha / dephospho_denominator
            if delta_subunit > 0.0:
                subturn_alpha = delta_subunit * n_alpha

            if use_kahan == 1:
                y_value = r1_alpha - c_r1
                t_sum = total_r1 + y_value
                c_r1 = (t_sum - total_r1) - y_value
                total_r1 = t_sum

                y_value = r2_alpha - c_r2
                t_sum = total_r2 + y_value
                c_r2 = (t_sum - total_r2) - y_value
                total_r2 = t_sum

                y_value = subturn_alpha - c_subturn
                t_sum = total_subturn + y_value
                c_subturn = (t_sum - total_subturn) - y_value
                total_subturn = t_sum

                y_value = holo_alpha - c_holoturn
                t_sum = total_holoturn + y_value
                c_holoturn = (t_sum - total_holoturn) - y_value
                total_holoturn = t_sum
            else:
                total_r1 += r1_alpha
                total_r2 += r2_alpha
                total_subturn += subturn_alpha
                total_holoturn += holo_alpha

            if exchange_code == EXCHANGE_A_MONOMER or exchange_code == EXCHANGE_B_MONOMER:
                diag_term = n_alpha * (n_subunits - n_alpha)
                if use_kahan == 1:
                    y_value = diag_term - c_diag_exchange
                    t_sum = diagonal_exchange + y_value
                    c_diag_exchange = (t_sum - diagonal_exchange) - y_value
                    diagonal_exchange = t_sum
                else:
                    diagonal_exchange += diag_term
            elif exchange_code == EXCHANGE_B_DIMER:
                donor_weight = n_alpha // 2
                accept_weight = (n_subunits - n_alpha) // 2
                diag_term = donor_weight * accept_weight
                if use_kahan == 1:
                    y_value = donor_weight - c_donor
                    t_sum = donor_sum + y_value
                    c_donor = (t_sum - donor_sum) - y_value
                    donor_sum = t_sum

                    y_value = accept_weight - c_accept
                    t_sum = accept_sum + y_value
                    c_accept = (t_sum - accept_sum) - y_value
                    accept_sum = t_sum

                    y_value = diag_term - c_diag_dimer
                    t_sum = diagonal_dimer + y_value
                    c_diag_dimer = (t_sum - diagonal_dimer) - y_value
                    diagonal_dimer = t_sum
                else:
                    donor_sum += donor_weight
                    accept_sum += accept_weight
                    diagonal_dimer += diag_term

        total_exchange = 0.0
        if exchange_code == EXCHANGE_A_MONOMER:
            total_exchange = gamma_value * (n_tot * (m_count * n_subunits - n_tot) - diagonal_exchange)
        elif exchange_code == EXCHANGE_B_MONOMER:
            total_exchange = (gamma_value / m_count) * (n_tot * (m_count * n_subunits - n_tot) - diagonal_exchange)
        elif exchange_code == EXCHANGE_B_DIMER:
            total_exchange = (gamma_value / m_count) * (donor_sum * accept_sum - diagonal_dimer)

        if total_exchange < 0.0:
            total_exchange = 0.0

        a0 = total_r1 + total_r2 + total_exchange + total_subturn + total_holoturn
        if a0 <= 0.0:
            invalid_code = 1
            break
        if a0 >= propensity_overflow_limit:
            invalid_code = 2
            break

        u_wait = np.random.random()
        if u_wait < 1.0e-15:
            u_wait = 1.0e-15
        tau_step = -math.log(u_wait) / a0
        if t_now + tau_step > stop_time:
            t_now = stop_time
            break

        event_threshold = np.random.random() * a0
        previous_n_tot = n_tot
        t_now += tau_step

        if event_threshold < total_r1:
            cumulative = 0.0
            selected_alpha = 0
            for alpha in range(m_count):
                n_alpha = int(state[alpha])
                rate = current_r1 * (n_subunits - n_alpha) * (n_alpha / n_subunits + epsilon_value)
                cumulative += rate
                selected_alpha = alpha
                if event_threshold <= cumulative:
                    break
            state[selected_alpha] += 1
            n_tot += 1
            consecutive_exchange_events = 0
        elif event_threshold < total_r1 + total_r2:
            local_threshold = event_threshold - total_r1
            cumulative = 0.0
            selected_alpha = 0
            for alpha in range(m_count):
                n_alpha = int(state[alpha])
                rate = 0.0
                if dephospho_denominator > 0.0:
                    rate = k_cat * n_alpha / dephospho_denominator
                cumulative += rate
                selected_alpha = alpha
                if local_threshold <= cumulative:
                    break
            state[selected_alpha] -= 1
            n_tot -= 1
            consecutive_exchange_events = 0
        elif event_threshold < total_r1 + total_r2 + total_exchange:
            local_threshold = event_threshold - total_r1 - total_r2
            selected_alpha = 0
            selected_beta = 0

            if exchange_code == EXCHANGE_A_MONOMER or exchange_code == EXCHANGE_B_MONOMER:
                exchange_scale = gamma_value
                if exchange_code == EXCHANGE_B_MONOMER:
                    exchange_scale = gamma_value / m_count
                total_accept = m_count * n_subunits - n_tot
                cumulative = 0.0
                last_valid_alpha = 0
                for alpha in range(m_count):
                    donor_weight = int(state[alpha])
                    accept_excluding_self = total_accept - (n_subunits - int(state[alpha]))
                    rate = exchange_scale * donor_weight * accept_excluding_self
                    if rate > 0.0:
                        last_valid_alpha = alpha
                    cumulative += rate
                    selected_alpha = alpha
                    if local_threshold <= cumulative:
                        break
                else:
                    selected_alpha = last_valid_alpha
                donor_weight = int(state[selected_alpha])
                donor_threshold = local_threshold - (cumulative - exchange_scale * donor_weight * (total_accept - (n_subunits - int(state[selected_alpha]))))
                cumulative = 0.0
                selected_beta = 0 if selected_alpha != 0 else 1
                for beta in range(m_count):
                    if beta == selected_alpha:
                        continue
                    rate = exchange_scale * donor_weight * (n_subunits - int(state[beta]))
                    if rate > 0.0:
                        selected_beta = beta
                    cumulative += rate
                    if donor_threshold <= cumulative:
                        selected_beta = beta
                        break
                state[selected_alpha] -= 1
                state[selected_beta] += 1
            else:
                exchange_scale = gamma_value / m_count
                cumulative = 0.0
                last_valid_alpha = 0
                for alpha in range(m_count):
                    donor_weight = int(state[alpha]) // 2
                    accept_excluding_self = accept_sum - ((n_subunits - int(state[alpha])) // 2)
                    rate = exchange_scale * donor_weight * accept_excluding_self
                    if rate > 0.0:
                        last_valid_alpha = alpha
                    cumulative += rate
                    selected_alpha = alpha
                    if local_threshold <= cumulative:
                        break
                else:
                    selected_alpha = last_valid_alpha
                donor_weight = int(state[selected_alpha]) // 2
                donor_threshold = local_threshold - (cumulative - exchange_scale * donor_weight * (accept_sum - ((n_subunits - int(state[selected_alpha])) // 2)))
                cumulative = 0.0
                selected_beta = 0 if selected_alpha != 0 else 1
                for beta in range(m_count):
                    if beta == selected_alpha:
                        continue
                    rate = exchange_scale * donor_weight * ((n_subunits - int(state[beta])) // 2)
                    if rate > 0.0:
                        selected_beta = beta
                    cumulative += rate
                    if donor_threshold <= cumulative:
                        selected_beta = beta
                        break
                state[selected_alpha] -= 2
                state[selected_beta] += 2

            consecutive_exchange_events += 1
            if consecutive_exchange_events > max_consecutive_exchange_events:
                invalid_code = 6
                warning_code = 1
                break
        elif event_threshold < total_r1 + total_r2 + total_exchange + total_subturn:
            local_threshold = event_threshold - total_r1 - total_r2 - total_exchange
            cumulative = 0.0
            selected_alpha = 0
            for alpha in range(m_count):
                rate = delta_subunit * int(state[alpha])
                cumulative += rate
                selected_alpha = alpha
                if local_threshold <= cumulative:
                    break
            state[selected_alpha] -= 1
            n_tot -= 1
            consecutive_exchange_events = 0
        else:
            local_threshold = event_threshold - total_r1 - total_r2 - total_exchange - total_subturn
            cumulative = 0.0
            selected_alpha = 0
            for alpha in range(m_count):
                cumulative += delta_holo
                selected_alpha = alpha
                if local_threshold <= cumulative:
                    break
            n_tot -= int(state[selected_alpha])
            state[selected_alpha] = 0
            consecutive_exchange_events = 0

        event_count += 1

        if t_now < 0.0:
            invalid_code = 5
            break

        recomputed_total = 0
        for alpha in range(m_count):
            n_alpha = int(state[alpha])
            if n_alpha < 0 or n_alpha > n_subunits:
                invalid_code = 3
                break
            recomputed_total += n_alpha
        if invalid_code != 0:
            break
        if math.fabs(recomputed_total - n_tot) > state_tolerance:
            invalid_code = 4
            break

        if ca_noise_enabled == 1 and t_now >= next_ca_update:
            while next_ca_update <= t_now:
                current_ca = ca_noise_low + (ca_noise_high - ca_noise_low) * np.random.random()
                current_r1 = r1_base * _hill_fraction(current_ca, ca_noise_kh1, hill_exponent)
                next_ca_update += ca_noise_interval

        if transition_code == TRANSITION_NONE:
            continue

        if transition_code == TRANSITION_FIRST_PHOSPHORYLATION:
            if previous_n_tot == 0 and n_tot > 0:
                trigger_time = t_now
                confirmation_time = t_now
                return 1, trigger_time, confirmation_time, t_now, n_tot, event_count, invalid_code, warning_code, state
            continue

        if transition_code == TRANSITION_UP_TO_DOWN:
            if not pending_trigger and previous_n_tot >= trigger_threshold and n_tot < trigger_threshold:
                pending_trigger = True
                trigger_time = t_now
            if pending_trigger:
                if n_tot <= confirm_low_threshold:
                    confirmation_time = t_now
                    return 1, trigger_time, confirmation_time, t_now, n_tot, event_count, invalid_code, warning_code, state
                if n_tot >= trigger_threshold:
                    pending_trigger = False
            continue

        if transition_code == TRANSITION_DOWN_TO_UP:
            if not pending_trigger and previous_n_tot < trigger_threshold and n_tot >= trigger_threshold:
                pending_trigger = True
                trigger_time = t_now
                events_since_trigger = 0
            elif pending_trigger:
                events_since_trigger += 1
                if n_tot < trigger_threshold:
                    pending_trigger = False
                    events_since_trigger = 0
                elif events_since_trigger >= 1 and n_tot >= confirm_high_threshold:
                    confirmation_time = t_now
                    return 1, trigger_time, confirmation_time, t_now, n_tot, event_count, invalid_code, warning_code, state

    return 0, trigger_time, confirmation_time, t_now, n_tot, event_count, invalid_code, warning_code, state


@njit(cache=True)
def _simulate_full_core(
    initial_n: np.ndarray,
    stop_time: float,
    transition_code: int,
    n_subunits: int,
    theta: int,
    k1_value: float,
    k2_value: float,
    kh1_uM: float,
    kh2_uM: float,
    hill_exponent: float,
    ca_value: float,
    k_cat: float,
    effective_km_count: float,
    delta_holo: float,
    propensity_overflow_limit: float,
    state_tolerance: float,
    use_kahan: int,
    seed: int,
) -> tuple[int, float, float, float, int, int, int, np.ndarray]:
    np.random.seed(seed)

    state = initial_n.copy()
    m_count = state.shape[0]
    n_tot = 0
    for index in range(m_count):
        n_tot += int(state[index])

    t_now = 0.0
    trigger_time = -1.0
    confirmation_time = -1.0
    pending_trigger = False
    events_since_trigger = 0
    event_count = 0
    invalid_code = 0

    trigger_threshold = theta * m_count
    confirm_low_threshold = 0.2 * m_count * n_subunits
    confirm_high_threshold = 0.4 * m_count * n_subunits

    hill_1 = _hill_fraction(ca_value, kh1_uM, hill_exponent)
    hill_2 = _hill_fraction(ca_value, kh2_uM, hill_exponent)

    while t_now < stop_time:
        total_rf1 = 0.0
        total_rf2 = 0.0
        total_dephospho = 0.0
        total_holoturn = 0.0
        c_rf1 = 0.0
        c_rf2 = 0.0
        c_dephospho = 0.0
        c_holoturn = 0.0
        dephospho_denominator = effective_km_count + n_tot

        for alpha in range(m_count):
            n_alpha = int(state[alpha])
            rf1 = 0.0
            rf2 = 0.0
            if n_alpha == 0:
                rf1 = k1_value * n_subunits * hill_1
            elif n_alpha < n_subunits:
                rf2 = k2_value * (n_subunits - n_alpha) * (n_alpha / n_subunits) * hill_2

            dephospho = 0.0
            if dephospho_denominator > 0.0:
                dephospho = k_cat * n_alpha / dephospho_denominator

            if use_kahan == 1:
                y_value = rf1 - c_rf1
                t_sum = total_rf1 + y_value
                c_rf1 = (t_sum - total_rf1) - y_value
                total_rf1 = t_sum

                y_value = rf2 - c_rf2
                t_sum = total_rf2 + y_value
                c_rf2 = (t_sum - total_rf2) - y_value
                total_rf2 = t_sum

                y_value = dephospho - c_dephospho
                t_sum = total_dephospho + y_value
                c_dephospho = (t_sum - total_dephospho) - y_value
                total_dephospho = t_sum

                y_value = delta_holo - c_holoturn
                t_sum = total_holoturn + y_value
                c_holoturn = (t_sum - total_holoturn) - y_value
                total_holoturn = t_sum
            else:
                total_rf1 += rf1
                total_rf2 += rf2
                total_dephospho += dephospho
                total_holoturn += delta_holo

        a0 = total_rf1 + total_rf2 + total_dephospho + total_holoturn
        if a0 <= 0.0:
            invalid_code = 1
            break
        if a0 >= propensity_overflow_limit:
            invalid_code = 2
            break

        u_wait = np.random.random()
        if u_wait < 1.0e-15:
            u_wait = 1.0e-15
        tau_step = -math.log(u_wait) / a0
        if t_now + tau_step > stop_time:
            t_now = stop_time
            break

        event_threshold = np.random.random() * a0
        previous_n_tot = n_tot
        t_now += tau_step

        if event_threshold < total_rf1:
            cumulative = 0.0
            selected_alpha = 0
            for alpha in range(m_count):
                rf1 = 0.0
                if int(state[alpha]) == 0:
                    rf1 = k1_value * n_subunits * hill_1
                cumulative += rf1
                selected_alpha = alpha
                if event_threshold <= cumulative:
                    break
            state[selected_alpha] = 1
            n_tot += 1
        elif event_threshold < total_rf1 + total_rf2:
            local_threshold = event_threshold - total_rf1
            cumulative = 0.0
            selected_alpha = 0
            for alpha in range(m_count):
                n_alpha = int(state[alpha])
                rf2 = 0.0
                if n_alpha >= 1 and n_alpha < n_subunits:
                    rf2 = k2_value * (n_subunits - n_alpha) * (n_alpha / n_subunits) * hill_2
                cumulative += rf2
                selected_alpha = alpha
                if local_threshold <= cumulative:
                    break
            state[selected_alpha] += 1
            n_tot += 1
        elif event_threshold < total_rf1 + total_rf2 + total_dephospho:
            local_threshold = event_threshold - total_rf1 - total_rf2
            cumulative = 0.0
            selected_alpha = 0
            for alpha in range(m_count):
                dephospho = 0.0
                if dephospho_denominator > 0.0:
                    dephospho = k_cat * int(state[alpha]) / dephospho_denominator
                cumulative += dephospho
                selected_alpha = alpha
                if local_threshold <= cumulative:
                    break
            state[selected_alpha] -= 1
            n_tot -= 1
        else:
            local_threshold = event_threshold - total_rf1 - total_rf2 - total_dephospho
            cumulative = 0.0
            selected_alpha = 0
            for alpha in range(m_count):
                cumulative += delta_holo
                selected_alpha = alpha
                if local_threshold <= cumulative:
                    break
            n_tot -= int(state[selected_alpha])
            state[selected_alpha] = 0

        event_count += 1

        if t_now < 0.0:
            invalid_code = 5
            break

        recomputed_total = 0
        for alpha in range(m_count):
            n_alpha = int(state[alpha])
            if n_alpha < 0 or n_alpha > n_subunits:
                invalid_code = 3
                break
            recomputed_total += n_alpha
        if invalid_code != 0:
            break
        if math.fabs(recomputed_total - n_tot) > state_tolerance:
            invalid_code = 4
            break

        if transition_code == TRANSITION_NONE:
            continue

        if transition_code == TRANSITION_UP_TO_DOWN:
            if not pending_trigger and previous_n_tot >= trigger_threshold and n_tot < trigger_threshold:
                pending_trigger = True
                trigger_time = t_now
            if pending_trigger:
                if n_tot <= confirm_low_threshold:
                    confirmation_time = t_now
                    return 1, trigger_time, confirmation_time, t_now, n_tot, event_count, invalid_code, state
                if n_tot >= trigger_threshold:
                    pending_trigger = False
            continue

        if transition_code == TRANSITION_DOWN_TO_UP:
            if not pending_trigger and previous_n_tot < trigger_threshold and n_tot >= trigger_threshold:
                pending_trigger = True
                trigger_time = t_now
                events_since_trigger = 0
            elif pending_trigger:
                events_since_trigger += 1
                if n_tot < trigger_threshold:
                    pending_trigger = False
                    events_since_trigger = 0
                elif events_since_trigger >= 1 and n_tot >= confirm_high_threshold:
                    confirmation_time = t_now
                    return 1, trigger_time, confirmation_time, t_now, n_tot, event_count, invalid_code, state

    return 0, trigger_time, confirmation_time, t_now, n_tot, event_count, invalid_code, state


@njit(cache=True)
def _simple_r1_rate(current_r1: float, n_value: int, n_subunits: int, epsilon_value: float) -> float:
    return current_r1 * (n_subunits - n_value) * (n_value / n_subunits + epsilon_value)


@njit(cache=True)
def _recompute_simple_r1_rates(
    state: np.ndarray,
    rates: np.ndarray,
    current_r1: float,
    n_subunits: int,
    epsilon_value: float,
) -> float:
    total = 0.0
    for alpha in range(state.shape[0]):
        rate = _simple_r1_rate(current_r1, int(state[alpha]), n_subunits, epsilon_value)
        rates[alpha] = rate
        total += rate
    return total


@njit(cache=True)
def _recompute_full_rates(
    state: np.ndarray,
    rf1_rates: np.ndarray,
    rf2_rates: np.ndarray,
    k1_value: float,
    k2_value: float,
    n_subunits: int,
    hill_1: float,
    hill_2: float,
) -> tuple[float, float]:
    total_rf1 = 0.0
    total_rf2 = 0.0
    for alpha in range(state.shape[0]):
        n_alpha = int(state[alpha])
        rf1 = 0.0
        rf2 = 0.0
        if n_alpha == 0:
            rf1 = k1_value * n_subunits * hill_1
        elif n_alpha < n_subunits:
            rf2 = k2_value * (n_subunits - n_alpha) * (n_alpha / n_subunits) * hill_2
        rf1_rates[alpha] = rf1
        rf2_rates[alpha] = rf2
        total_rf1 += rf1
        total_rf2 += rf2
    return total_rf1, total_rf2


@njit(cache=True)
def _simulate_simple_core_fast(
    initial_n: np.ndarray,
    stop_time: float,
    transition_code: int,
    n_subunits: int,
    theta: int,
    r1_base: float,
    epsilon_value: float,
    k_cat: float,
    effective_km_count: float,
    gamma_value: float,
    exchange_code: int,
    delta_subunit: float,
    delta_holo: float,
    ca_noise_enabled: int,
    ca_noise_low: float,
    ca_noise_high: float,
    ca_noise_interval: float,
    ca_noise_kh1: float,
    hill_exponent: float,
    propensity_overflow_limit: float,
    max_consecutive_exchange_events: int,
    state_tolerance: float,
    use_kahan: int,
    seed: int,
) -> tuple[int, float, float, float, int, int, int, int, np.ndarray]:
    np.random.seed(seed)

    state = initial_n.copy()
    m_count = state.shape[0]
    n_tot = 0
    for index in range(m_count):
        n_tot += int(state[index])

    t_now = 0.0
    next_ca_update = ca_noise_interval
    current_r1 = r1_base
    trigger_time = -1.0
    confirmation_time = -1.0
    pending_trigger = False
    events_since_trigger = 0
    event_count = 0
    consecutive_exchange_events = 0
    invalid_code = 0
    warning_code = 0

    trigger_threshold = theta * m_count
    confirm_low_threshold = 0.2 * m_count * n_subunits
    confirm_high_threshold = 0.4 * m_count * n_subunits

    r1_rates = np.zeros(m_count, dtype=np.float64)
    total_r1 = _recompute_simple_r1_rates(state, r1_rates, current_r1, n_subunits, epsilon_value)

    diagonal_exchange = 0.0
    donor_sum = 0.0
    accept_sum = 0.0
    diagonal_dimer = 0.0
    for alpha in range(m_count):
        n_alpha = int(state[alpha])
        if exchange_code == EXCHANGE_A_MONOMER or exchange_code == EXCHANGE_B_MONOMER:
            diagonal_exchange += n_alpha * (n_subunits - n_alpha)
        elif exchange_code == EXCHANGE_B_DIMER:
            donor_weight = n_alpha // 2
            accept_weight = (n_subunits - n_alpha) // 2
            donor_sum += donor_weight
            accept_sum += accept_weight
            diagonal_dimer += donor_weight * accept_weight

    while t_now < stop_time:
        total_r2 = 0.0
        denominator = effective_km_count + n_tot
        if denominator > 0.0:
            total_r2 = k_cat * n_tot / denominator
        total_subturn = delta_subunit * n_tot
        total_holoturn = delta_holo * m_count

        total_exchange = 0.0
        if exchange_code == EXCHANGE_A_MONOMER:
            total_exchange = gamma_value * (n_tot * (m_count * n_subunits - n_tot) - diagonal_exchange)
        elif exchange_code == EXCHANGE_B_MONOMER:
            total_exchange = (gamma_value / m_count) * (n_tot * (m_count * n_subunits - n_tot) - diagonal_exchange)
        elif exchange_code == EXCHANGE_B_DIMER:
            total_exchange = (gamma_value / m_count) * (donor_sum * accept_sum - diagonal_dimer)
        if total_exchange < 0.0:
            total_exchange = 0.0

        a0 = total_r1 + total_r2 + total_exchange + total_subturn + total_holoturn
        if a0 <= 0.0:
            invalid_code = 1
            break
        if a0 >= propensity_overflow_limit:
            invalid_code = 2
            break

        u_wait = np.random.random()
        if u_wait < 1.0e-15:
            u_wait = 1.0e-15
        tau_step = -math.log(u_wait) / a0
        if t_now + tau_step > stop_time:
            t_now = stop_time
            break

        event_threshold = np.random.random() * a0
        previous_n_tot = n_tot
        t_now += tau_step

        if event_threshold < total_r1:
            cumulative = 0.0
            selected_alpha = 0
            for alpha in range(m_count):
                cumulative += r1_rates[alpha]
                selected_alpha = alpha
                if event_threshold <= cumulative:
                    break
            old_n = int(state[selected_alpha])
            new_n = old_n + 1
            state[selected_alpha] = new_n
            n_tot += 1
            total_r1 += _simple_r1_rate(current_r1, new_n, n_subunits, epsilon_value) - r1_rates[selected_alpha]
            r1_rates[selected_alpha] = _simple_r1_rate(current_r1, new_n, n_subunits, epsilon_value)
            if exchange_code == EXCHANGE_A_MONOMER or exchange_code == EXCHANGE_B_MONOMER:
                diagonal_exchange += new_n * (n_subunits - new_n) - old_n * (n_subunits - old_n)
            elif exchange_code == EXCHANGE_B_DIMER:
                old_donor = old_n // 2
                new_donor = new_n // 2
                old_accept = (n_subunits - old_n) // 2
                new_accept = (n_subunits - new_n) // 2
                donor_sum += new_donor - old_donor
                accept_sum += new_accept - old_accept
                diagonal_dimer += new_donor * new_accept - old_donor * old_accept
            consecutive_exchange_events = 0
        elif event_threshold < total_r1 + total_r2:
            local_threshold = event_threshold - total_r1
            cumulative = 0.0
            selected_alpha = 0
            if n_tot <= 0:
                invalid_code = 1
                break
            for alpha in range(m_count):
                cumulative += total_r2 * int(state[alpha]) / n_tot
                selected_alpha = alpha
                if local_threshold <= cumulative:
                    break
            old_n = int(state[selected_alpha])
            new_n = old_n - 1
            state[selected_alpha] = new_n
            n_tot -= 1
            total_r1 += _simple_r1_rate(current_r1, new_n, n_subunits, epsilon_value) - r1_rates[selected_alpha]
            r1_rates[selected_alpha] = _simple_r1_rate(current_r1, new_n, n_subunits, epsilon_value)
            if exchange_code == EXCHANGE_A_MONOMER or exchange_code == EXCHANGE_B_MONOMER:
                diagonal_exchange += new_n * (n_subunits - new_n) - old_n * (n_subunits - old_n)
            elif exchange_code == EXCHANGE_B_DIMER:
                old_donor = old_n // 2
                new_donor = new_n // 2
                old_accept = (n_subunits - old_n) // 2
                new_accept = (n_subunits - new_n) // 2
                donor_sum += new_donor - old_donor
                accept_sum += new_accept - old_accept
                diagonal_dimer += new_donor * new_accept - old_donor * old_accept
            consecutive_exchange_events = 0
        elif event_threshold < total_r1 + total_r2 + total_exchange:
            local_threshold = event_threshold - total_r1 - total_r2
            selected_alpha = 0
            selected_beta = 0
            if exchange_code == EXCHANGE_A_MONOMER or exchange_code == EXCHANGE_B_MONOMER:
                exchange_scale = gamma_value
                if exchange_code == EXCHANGE_B_MONOMER:
                    exchange_scale = gamma_value / m_count
                total_accept = m_count * n_subunits - n_tot
                cumulative = 0.0
                for alpha in range(m_count):
                    donor_rate = exchange_scale * int(state[alpha]) * (total_accept - (n_subunits - int(state[alpha])))
                    cumulative += donor_rate
                    selected_alpha = alpha
                    if local_threshold <= cumulative:
                        break
                donor_before = int(state[selected_alpha])
                donor_threshold = local_threshold - (cumulative - exchange_scale * donor_before * (total_accept - (n_subunits - donor_before)))
                cumulative = 0.0
                for beta in range(m_count):
                    if beta == selected_alpha:
                        continue
                    rate = exchange_scale * donor_before * (n_subunits - int(state[beta]))
                    cumulative += rate
                    selected_beta = beta
                    if donor_threshold <= cumulative:
                        break
                old_alpha = donor_before
                old_beta = int(state[selected_beta])
                new_alpha = old_alpha - 1
                new_beta = old_beta + 1
                state[selected_alpha] = new_alpha
                state[selected_beta] = new_beta
                total_r1 += _simple_r1_rate(current_r1, new_alpha, n_subunits, epsilon_value) - r1_rates[selected_alpha]
                total_r1 += _simple_r1_rate(current_r1, new_beta, n_subunits, epsilon_value) - r1_rates[selected_beta]
                r1_rates[selected_alpha] = _simple_r1_rate(current_r1, new_alpha, n_subunits, epsilon_value)
                r1_rates[selected_beta] = _simple_r1_rate(current_r1, new_beta, n_subunits, epsilon_value)
                diagonal_exchange += new_alpha * (n_subunits - new_alpha) - old_alpha * (n_subunits - old_alpha)
                diagonal_exchange += new_beta * (n_subunits - new_beta) - old_beta * (n_subunits - old_beta)
            else:
                exchange_scale = gamma_value / m_count
                cumulative = 0.0
                for alpha in range(m_count):
                    donor_units = int(state[alpha]) // 2
                    donor_rate = exchange_scale * donor_units * (accept_sum - ((n_subunits - int(state[alpha])) // 2))
                    cumulative += donor_rate
                    selected_alpha = alpha
                    if local_threshold <= cumulative:
                        break
                donor_before = int(state[selected_alpha])
                donor_units_before = donor_before // 2
                donor_threshold = local_threshold - (
                    cumulative
                    - exchange_scale * donor_units_before * (accept_sum - ((n_subunits - donor_before) // 2))
                )
                cumulative = 0.0
                for beta in range(m_count):
                    if beta == selected_alpha:
                        continue
                    rate = exchange_scale * donor_units_before * ((n_subunits - int(state[beta])) // 2)
                    cumulative += rate
                    selected_beta = beta
                    if donor_threshold <= cumulative:
                        break
                old_alpha = donor_before
                old_beta = int(state[selected_beta])
                new_alpha = old_alpha - 2
                new_beta = old_beta + 2
                state[selected_alpha] = new_alpha
                state[selected_beta] = new_beta
                total_r1 += _simple_r1_rate(current_r1, new_alpha, n_subunits, epsilon_value) - r1_rates[selected_alpha]
                total_r1 += _simple_r1_rate(current_r1, new_beta, n_subunits, epsilon_value) - r1_rates[selected_beta]
                r1_rates[selected_alpha] = _simple_r1_rate(current_r1, new_alpha, n_subunits, epsilon_value)
                r1_rates[selected_beta] = _simple_r1_rate(current_r1, new_beta, n_subunits, epsilon_value)
                old_alpha_donor = old_alpha // 2
                new_alpha_donor = new_alpha // 2
                old_alpha_accept = (n_subunits - old_alpha) // 2
                new_alpha_accept = (n_subunits - new_alpha) // 2
                old_beta_donor = old_beta // 2
                new_beta_donor = new_beta // 2
                old_beta_accept = (n_subunits - old_beta) // 2
                new_beta_accept = (n_subunits - new_beta) // 2
                donor_sum += (new_alpha_donor - old_alpha_donor) + (new_beta_donor - old_beta_donor)
                accept_sum += (new_alpha_accept - old_alpha_accept) + (new_beta_accept - old_beta_accept)
                diagonal_dimer += new_alpha_donor * new_alpha_accept - old_alpha_donor * old_alpha_accept
                diagonal_dimer += new_beta_donor * new_beta_accept - old_beta_donor * old_beta_accept
            consecutive_exchange_events += 1
            if consecutive_exchange_events > max_consecutive_exchange_events:
                invalid_code = 6
                warning_code = 1
                break
        elif event_threshold < total_r1 + total_r2 + total_exchange + total_subturn:
            local_threshold = event_threshold - total_r1 - total_r2 - total_exchange
            cumulative = 0.0
            selected_alpha = 0
            if n_tot <= 0:
                invalid_code = 1
                break
            for alpha in range(m_count):
                cumulative += total_subturn * int(state[alpha]) / n_tot
                selected_alpha = alpha
                if local_threshold <= cumulative:
                    break
            old_n = int(state[selected_alpha])
            new_n = old_n - 1
            state[selected_alpha] = new_n
            n_tot -= 1
            total_r1 += _simple_r1_rate(current_r1, new_n, n_subunits, epsilon_value) - r1_rates[selected_alpha]
            r1_rates[selected_alpha] = _simple_r1_rate(current_r1, new_n, n_subunits, epsilon_value)
            if exchange_code == EXCHANGE_A_MONOMER or exchange_code == EXCHANGE_B_MONOMER:
                diagonal_exchange += new_n * (n_subunits - new_n) - old_n * (n_subunits - old_n)
            elif exchange_code == EXCHANGE_B_DIMER:
                old_donor = old_n // 2
                new_donor = new_n // 2
                old_accept = (n_subunits - old_n) // 2
                new_accept = (n_subunits - new_n) // 2
                donor_sum += new_donor - old_donor
                accept_sum += new_accept - old_accept
                diagonal_dimer += new_donor * new_accept - old_donor * old_accept
            consecutive_exchange_events = 0
        else:
            local_threshold = event_threshold - total_r1 - total_r2 - total_exchange - total_subturn
            selected_alpha = 0
            cumulative = 0.0
            for alpha in range(m_count):
                cumulative += delta_holo
                selected_alpha = alpha
                if local_threshold <= cumulative:
                    break
            old_n = int(state[selected_alpha])
            new_n = 0
            state[selected_alpha] = 0
            n_tot -= old_n
            total_r1 += _simple_r1_rate(current_r1, new_n, n_subunits, epsilon_value) - r1_rates[selected_alpha]
            r1_rates[selected_alpha] = _simple_r1_rate(current_r1, new_n, n_subunits, epsilon_value)
            if exchange_code == EXCHANGE_A_MONOMER or exchange_code == EXCHANGE_B_MONOMER:
                diagonal_exchange += new_n * (n_subunits - new_n) - old_n * (n_subunits - old_n)
            elif exchange_code == EXCHANGE_B_DIMER:
                old_donor = old_n // 2
                new_donor = 0
                old_accept = (n_subunits - old_n) // 2
                new_accept = n_subunits // 2
                donor_sum += new_donor - old_donor
                accept_sum += new_accept - old_accept
                diagonal_dimer += new_donor * new_accept - old_donor * old_accept
            consecutive_exchange_events = 0

        event_count += 1

        if t_now < 0.0:
            invalid_code = 5
            break

        recomputed_total = 0
        for alpha in range(m_count):
            n_alpha = int(state[alpha])
            if n_alpha < 0 or n_alpha > n_subunits:
                invalid_code = 3
                break
            recomputed_total += n_alpha
        if invalid_code != 0:
            break
        if math.fabs(recomputed_total - n_tot) > state_tolerance:
            invalid_code = 4
            break

        if ca_noise_enabled == 1 and t_now >= next_ca_update:
            while next_ca_update <= t_now:
                current_ca = ca_noise_low + (ca_noise_high - ca_noise_low) * np.random.random()
                current_r1 = r1_base * _hill_fraction(current_ca, ca_noise_kh1, hill_exponent)
                next_ca_update += ca_noise_interval
            total_r1 = _recompute_simple_r1_rates(state, r1_rates, current_r1, n_subunits, epsilon_value)

        if transition_code == TRANSITION_NONE:
            continue
        if transition_code == TRANSITION_FIRST_PHOSPHORYLATION:
            if previous_n_tot == 0 and n_tot > 0:
                trigger_time = t_now
                confirmation_time = t_now
                return 1, trigger_time, confirmation_time, t_now, n_tot, event_count, invalid_code, warning_code, state
            continue
        if transition_code == TRANSITION_UP_TO_DOWN:
            if not pending_trigger and previous_n_tot >= trigger_threshold and n_tot < trigger_threshold:
                pending_trigger = True
                trigger_time = t_now
            if pending_trigger:
                if n_tot <= confirm_low_threshold:
                    confirmation_time = t_now
                    return 1, trigger_time, confirmation_time, t_now, n_tot, event_count, invalid_code, warning_code, state
                if n_tot >= trigger_threshold:
                    pending_trigger = False
            continue
        if transition_code == TRANSITION_DOWN_TO_UP:
            if not pending_trigger and previous_n_tot < trigger_threshold and n_tot >= trigger_threshold:
                pending_trigger = True
                trigger_time = t_now
                events_since_trigger = 0
            elif pending_trigger:
                events_since_trigger += 1
                if n_tot < trigger_threshold:
                    pending_trigger = False
                    events_since_trigger = 0
                elif events_since_trigger >= 1 and n_tot >= confirm_high_threshold:
                    confirmation_time = t_now
                    return 1, trigger_time, confirmation_time, t_now, n_tot, event_count, invalid_code, warning_code, state

    return 0, trigger_time, confirmation_time, t_now, n_tot, event_count, invalid_code, warning_code, state


@njit(cache=True)
def _simulate_full_core_fast(
    initial_n: np.ndarray,
    stop_time: float,
    transition_code: int,
    n_subunits: int,
    theta: int,
    k1_value: float,
    k2_value: float,
    kh1_uM: float,
    kh2_uM: float,
    hill_exponent: float,
    ca_value: float,
    k_cat: float,
    effective_km_count: float,
    delta_holo: float,
    propensity_overflow_limit: float,
    state_tolerance: float,
    use_kahan: int,
    seed: int,
) -> tuple[int, float, float, float, int, int, int, np.ndarray]:
    np.random.seed(seed)

    state = initial_n.copy()
    m_count = state.shape[0]
    n_tot = 0
    for index in range(m_count):
        n_tot += int(state[index])

    t_now = 0.0
    trigger_time = -1.0
    confirmation_time = -1.0
    pending_trigger = False
    events_since_trigger = 0
    event_count = 0
    invalid_code = 0

    trigger_threshold = theta * m_count
    confirm_low_threshold = 0.2 * m_count * n_subunits
    confirm_high_threshold = 0.4 * m_count * n_subunits
    hill_1 = _hill_fraction(ca_value, kh1_uM, hill_exponent)
    hill_2 = _hill_fraction(ca_value, kh2_uM, hill_exponent)

    rf1_rates = np.zeros(m_count, dtype=np.float64)
    rf2_rates = np.zeros(m_count, dtype=np.float64)
    total_rf1, total_rf2 = _recompute_full_rates(state, rf1_rates, rf2_rates, k1_value, k2_value, n_subunits, hill_1, hill_2)

    while t_now < stop_time:
        total_dephospho = 0.0
        denominator = effective_km_count + n_tot
        if denominator > 0.0:
            total_dephospho = k_cat * n_tot / denominator
        total_holoturn = delta_holo * m_count

        a0 = total_rf1 + total_rf2 + total_dephospho + total_holoturn
        if a0 <= 0.0:
            invalid_code = 1
            break
        if a0 >= propensity_overflow_limit:
            invalid_code = 2
            break

        u_wait = np.random.random()
        if u_wait < 1.0e-15:
            u_wait = 1.0e-15
        tau_step = -math.log(u_wait) / a0
        if t_now + tau_step > stop_time:
            t_now = stop_time
            break

        event_threshold = np.random.random() * a0
        previous_n_tot = n_tot
        t_now += tau_step

        if event_threshold < total_rf1:
            cumulative = 0.0
            selected_alpha = 0
            for alpha in range(m_count):
                cumulative += rf1_rates[alpha]
                selected_alpha = alpha
                if event_threshold <= cumulative:
                    break
            old_n = int(state[selected_alpha])
            new_n = 1
            state[selected_alpha] = new_n
            n_tot += 1
            total_rf1 += (0.0 - rf1_rates[selected_alpha])
            total_rf2 += (k2_value * (n_subunits - new_n) * (new_n / n_subunits) * hill_2 - rf2_rates[selected_alpha])
            rf1_rates[selected_alpha] = 0.0
            rf2_rates[selected_alpha] = k2_value * (n_subunits - new_n) * (new_n / n_subunits) * hill_2
        elif event_threshold < total_rf1 + total_rf2:
            local_threshold = event_threshold - total_rf1
            cumulative = 0.0
            selected_alpha = 0
            for alpha in range(m_count):
                cumulative += rf2_rates[alpha]
                selected_alpha = alpha
                if local_threshold <= cumulative:
                    break
            old_n = int(state[selected_alpha])
            new_n = old_n + 1
            state[selected_alpha] = new_n
            n_tot += 1
            new_rf1 = 0.0
            new_rf2 = 0.0
            if new_n < n_subunits:
                new_rf2 = k2_value * (n_subunits - new_n) * (new_n / n_subunits) * hill_2
            total_rf1 += new_rf1 - rf1_rates[selected_alpha]
            total_rf2 += new_rf2 - rf2_rates[selected_alpha]
            rf1_rates[selected_alpha] = new_rf1
            rf2_rates[selected_alpha] = new_rf2
        elif event_threshold < total_rf1 + total_rf2 + total_dephospho:
            local_threshold = event_threshold - total_rf1 - total_rf2
            cumulative = 0.0
            selected_alpha = 0
            if n_tot <= 0:
                invalid_code = 1
                break
            for alpha in range(m_count):
                cumulative += total_dephospho * int(state[alpha]) / n_tot
                selected_alpha = alpha
                if local_threshold <= cumulative:
                    break
            old_n = int(state[selected_alpha])
            new_n = old_n - 1
            state[selected_alpha] = new_n
            n_tot -= 1
            new_rf1 = 0.0
            new_rf2 = 0.0
            if new_n == 0:
                new_rf1 = k1_value * n_subunits * hill_1
            elif new_n < n_subunits:
                new_rf2 = k2_value * (n_subunits - new_n) * (new_n / n_subunits) * hill_2
            total_rf1 += new_rf1 - rf1_rates[selected_alpha]
            total_rf2 += new_rf2 - rf2_rates[selected_alpha]
            rf1_rates[selected_alpha] = new_rf1
            rf2_rates[selected_alpha] = new_rf2
        else:
            local_threshold = event_threshold - total_rf1 - total_rf2 - total_dephospho
            cumulative = 0.0
            selected_alpha = 0
            for alpha in range(m_count):
                cumulative += delta_holo
                selected_alpha = alpha
                if local_threshold <= cumulative:
                    break
            old_n = int(state[selected_alpha])
            state[selected_alpha] = 0
            n_tot -= old_n
            new_rf1 = k1_value * n_subunits * hill_1
            total_rf1 += new_rf1 - rf1_rates[selected_alpha]
            total_rf2 += 0.0 - rf2_rates[selected_alpha]
            rf1_rates[selected_alpha] = new_rf1
            rf2_rates[selected_alpha] = 0.0

        event_count += 1

        if t_now < 0.0:
            invalid_code = 5
            break

        recomputed_total = 0
        for alpha in range(m_count):
            n_alpha = int(state[alpha])
            if n_alpha < 0 or n_alpha > n_subunits:
                invalid_code = 3
                break
            recomputed_total += n_alpha
        if invalid_code != 0:
            break
        if math.fabs(recomputed_total - n_tot) > state_tolerance:
            invalid_code = 4
            break

        if transition_code == TRANSITION_NONE:
            continue
        if transition_code == TRANSITION_UP_TO_DOWN:
            if not pending_trigger and previous_n_tot >= trigger_threshold and n_tot < trigger_threshold:
                pending_trigger = True
                trigger_time = t_now
            if pending_trigger:
                if n_tot <= confirm_low_threshold:
                    confirmation_time = t_now
                    return 1, trigger_time, confirmation_time, t_now, n_tot, event_count, invalid_code, state
                if n_tot >= trigger_threshold:
                    pending_trigger = False
            continue
        if transition_code == TRANSITION_DOWN_TO_UP:
            if not pending_trigger and previous_n_tot < trigger_threshold and n_tot >= trigger_threshold:
                pending_trigger = True
                trigger_time = t_now
                events_since_trigger = 0
            elif pending_trigger:
                events_since_trigger += 1
                if n_tot < trigger_threshold:
                    pending_trigger = False
                    events_since_trigger = 0
                elif events_since_trigger >= 1 and n_tot >= confirm_high_threshold:
                    confirmation_time = t_now
                    return 1, trigger_time, confirmation_time, t_now, n_tot, event_count, invalid_code, state

    return 0, trigger_time, confirmation_time, t_now, n_tot, event_count, invalid_code, state


def _simple_ltp_r1_value(settings: SimulationSettings, base_r1: float) -> float:
    if not settings.map_simple_model_ltp_to_hill_scaled_r1:
        return base_r1
    resting_hill = settings.resting_hill_first_step()
    if resting_hill <= 0.0:
        return base_r1
    return base_r1 * settings.ltp_hill_first_step() / resting_hill


def initialize_simple_up_state(
    condition: ConditionSpec,
    settings: SimulationSettings,
    numerical: NumericalConfig,
    seed: int,
) -> np.ndarray:
    initial_state = np.full(condition.M, settings.n_subunits, dtype=np.int64)
    use_kahan = int(condition.M >= numerical.kahan_required_m_threshold)

    _, _, _, _, _, _, invalid_code, _, equilibrated = _simulate_simple_core_fast(
        initial_state,
        500.0,
        TRANSITION_NONE,
        settings.n_subunits,
        settings.theta,
        _simple_ltp_r1_value(settings, condition.r1_value),
        condition.epsilon_value,
        settings.k_cat,
        settings.effective_km_count,
        condition.gamma,
        condition.exchange_code,
        condition.delta_subunit,
        condition.delta_holo,
        0,
        settings.ca_noise_low_uM,
        settings.ca_noise_high_uM,
        settings.ca_noise_refresh_seconds,
        settings.kh1_uM,
        settings.hill_exponent,
        numerical.propensity_overflow_limit,
        numerical.max_consecutive_exchange_events,
        numerical.state_tolerance,
        use_kahan,
        seed,
    )
    if invalid_code != 0:
        return equilibrated

    extension_seed = seed + 10_000
    current_state = equilibrated
    threshold_total = condition.M * settings.theta
    for _ in range(100):
        if int(np.sum(current_state)) >= threshold_total:
            break
        _, _, _, _, _, _, invalid_code, _, current_state = _simulate_simple_core_fast(
            current_state,
            200.0,
            TRANSITION_NONE,
            settings.n_subunits,
            settings.theta,
            condition.r1_value,
            condition.epsilon_value,
            settings.k_cat,
            settings.effective_km_count,
            condition.gamma,
            condition.exchange_code,
            condition.delta_subunit,
            condition.delta_holo,
            int(condition.ca_noise_enabled),
            settings.ca_noise_low_uM,
            settings.ca_noise_high_uM,
            settings.ca_noise_refresh_seconds,
            settings.kh1_uM,
            settings.hill_exponent,
            numerical.propensity_overflow_limit,
            numerical.max_consecutive_exchange_events,
            numerical.state_tolerance,
            use_kahan,
            extension_seed,
        )
        extension_seed += 1
        if invalid_code != 0:
            break
    return current_state


def initialize_full_up_state(
    condition: ConditionSpec,
    settings: SimulationSettings,
    numerical: NumericalConfig,
    seed: int,
) -> np.ndarray:
    initial_state = np.full(condition.M, settings.n_subunits, dtype=np.int64)
    use_kahan = int(condition.M >= numerical.kahan_required_m_threshold)

    _, _, _, _, _, _, invalid_code, equilibrated = _simulate_full_core_fast(
        initial_state,
        500.0,
        TRANSITION_NONE,
        settings.n_subunits,
        settings.theta,
        condition.k1_value,
        condition.k2_value,
        condition.kh1_uM,
        condition.kh2_uM,
        condition.hill_exponent,
        condition.ltp_ca_uM,
        settings.k_cat,
        settings.effective_km_count,
        condition.delta_holo,
        numerical.propensity_overflow_limit,
        numerical.state_tolerance,
        use_kahan,
        seed,
    )
    if invalid_code != 0:
        return equilibrated

    current_state = equilibrated
    extension_seed = seed + 20_000
    threshold_total = condition.M * settings.theta
    for _ in range(100):
        if int(np.sum(current_state)) >= threshold_total:
            break
        _, _, _, _, _, _, invalid_code, current_state = _simulate_full_core_fast(
            current_state,
            200.0,
            TRANSITION_NONE,
            settings.n_subunits,
            settings.theta,
            condition.k1_value,
            condition.k2_value,
            condition.kh1_uM,
            condition.kh2_uM,
            condition.hill_exponent,
            condition.rest_ca_uM,
            settings.k_cat,
            settings.effective_km_count,
            condition.delta_holo,
            numerical.propensity_overflow_limit,
            numerical.state_tolerance,
            use_kahan,
            extension_seed,
        )
        extension_seed += 1
        if invalid_code != 0:
            break
    return current_state


def simulate_trajectory_worker(
    condition: ConditionSpec,
    settings: SimulationSettings,
    numerical: NumericalConfig,
    seed: int,
    trajectory_index: int,
) -> Dict[str, Any]:
    transition_code = transition_code_from_label(condition.transition_label)
    use_kahan = int(condition.M >= numerical.kahan_required_m_threshold)

    if condition.transition_label == "up_to_down":
        if condition.use_full_model:
            initial_state = initialize_full_up_state(condition, settings, numerical, seed + 1)
        else:
            initial_state = initialize_simple_up_state(condition, settings, numerical, seed + 1)
    else:
        initial_state = np.zeros(condition.M, dtype=np.int64)

    if condition.use_full_model:
        transition_found, trigger_time, confirmation_time, final_time, final_n_tot, event_count, invalid_code, final_state = _simulate_full_core_fast(
            initial_state,
            condition.t_max,
            transition_code,
            settings.n_subunits,
            settings.theta,
            condition.k1_value,
            condition.k2_value,
            condition.kh1_uM,
            condition.kh2_uM,
            condition.hill_exponent,
            condition.rest_ca_uM,
            settings.k_cat,
            settings.effective_km_count,
            condition.delta_holo,
            numerical.propensity_overflow_limit,
            numerical.state_tolerance,
            use_kahan,
            seed,
        )
        warning_code = 0
    else:
        transition_found, trigger_time, confirmation_time, final_time, final_n_tot, event_count, invalid_code, warning_code, final_state = _simulate_simple_core_fast(
            initial_state,
            condition.t_max,
            transition_code,
            settings.n_subunits,
            settings.theta,
            condition.r1_value,
            condition.epsilon_value,
            settings.k_cat,
            settings.effective_km_count,
            condition.gamma,
            condition.exchange_code,
            condition.delta_subunit,
            condition.delta_holo,
            int(condition.ca_noise_enabled),
            settings.ca_noise_low_uM,
            settings.ca_noise_high_uM,
            settings.ca_noise_refresh_seconds,
            settings.kh1_uM,
            settings.hill_exponent,
            numerical.propensity_overflow_limit,
            numerical.max_consecutive_exchange_events,
            numerical.state_tolerance,
            use_kahan,
            seed,
        )

    invalid = invalid_code != 0
    observed_time = condition.t_max
    censored = 1
    if transition_found == 1:
        observed_time = trigger_time
        censored = 0

    return {
        "run_id": condition.run_id,
        "model": condition.model_label,
        "transition": condition.transition_label,
        "M": condition.M,
        "gamma": condition.gamma,
        "delta_subunit": condition.delta_subunit,
        "delta_holo": condition.delta_holo,
        "r1_value": condition.r1_value,
        "epsilon_value": condition.epsilon_value,
        "k1_value": condition.k1_value,
        "k2_value": condition.k2_value,
        "kh1_uM": condition.kh1_uM,
        "kh2_uM": condition.kh2_uM,
        "ca_noise_enabled": int(condition.ca_noise_enabled),
        "trajectory_index": trajectory_index,
        "seed": seed,
        "transition_found": int(transition_found),
        "censored": censored,
        "observed_time": observed_time,
        "trigger_time": trigger_time,
        "confirmation_time": confirmation_time,
        "final_time": final_time,
        "final_n_tot": final_n_tot,
        "event_count": event_count,
        "invalid": int(invalid),
        "invalid_code": invalid_code,
        "invalid_reason": INVALID_REASON_MAP.get(invalid_code, "unknown_invalid_code"),
        "warning_code": warning_code,
        "final_state": ";".join(str(int(value)) for value in final_state.tolist()),
        "notes": condition.notes,
    }


def simulate_notebook_trajectory_worker(
    condition: NotebookConditionSpec,
    numerical: NumericalConfig,
    seed: int,
    trajectory_index: int,
) -> Dict[str, Any]:
    transition_code = transition_code_from_label(condition.transition_label)
    initial_state = notebook_initial_state(condition)

    transition_found, transition_time, final_time, final_n_tot, final_up_count, event_count, invalid_code, final_state = _simulate_notebook_core(
        initial_state,
        condition.t_max,
        transition_code,
        condition.N,
        condition.theta,
        condition.r1,
        condition.Vmax,
        condition.Km,
        condition.gamma,
        condition.delta,
        condition.autophosphorylation_step,
        int(condition.calcium_spike_enabled),
        condition.calcium_spike_start_seconds,
        condition.calcium_spike_duration_seconds,
        condition.calcium_spike_period_seconds,
        condition.calcium_spike_count,
        condition.calcium_spike_r1_multiplier,
        numerical.propensity_overflow_limit,
        numerical.state_tolerance,
        seed,
    )

    observed_time = condition.t_max
    censored = 1
    if transition_found == 1:
        observed_time = transition_time
        censored = 0

    return {
        "run_id": condition.run_id,
        "model": condition.model_label,
        "transition": condition.transition_label,
        "M": condition.M,
        "N": condition.N,
        "theta": condition.theta,
        "r1": condition.r1,
        "Vmax": condition.Vmax,
        "Km": condition.Km,
        "gamma": condition.gamma,
        "delta": condition.delta,
        "autophosphorylation_step": condition.autophosphorylation_step,
        "calcium_spike_enabled": int(condition.calcium_spike_enabled),
        "calcium_spike_start_seconds": condition.calcium_spike_start_seconds,
        "calcium_spike_duration_seconds": condition.calcium_spike_duration_seconds,
        "calcium_spike_period_seconds": condition.calcium_spike_period_seconds,
        "calcium_spike_count": condition.calcium_spike_count,
        "calcium_spike_r1_multiplier": condition.calcium_spike_r1_multiplier,
        "initial_mode": condition.initial_mode,
        "sweep_parameter": condition.sweep_parameter,
        "sweep_value": condition.sweep_value,
        "trajectory_index": trajectory_index,
        "seed": seed,
        "transition_found": int(transition_found),
        "censored": censored,
        "observed_time": observed_time,
        "trigger_time": transition_time,
        "confirmation_time": transition_time,
        "final_time": final_time,
        "final_n_tot": final_n_tot,
        "final_up_count": final_up_count,
        "final_f_up": final_up_count / condition.M,
        "event_count": event_count,
        "invalid": int(invalid_code != 0),
        "invalid_code": invalid_code,
        "invalid_reason": INVALID_REASON_MAP.get(invalid_code, "unknown_invalid_code"),
        "final_state": ";".join(str(int(value)) for value in final_state.tolist()),
        "notes": condition.notes,
    }


def simulate_notebook_trace(
    condition: NotebookConditionSpec,
    numerical: NumericalConfig,
    seed: int,
    duration_seconds: float,
    max_events: int,
    record_every_events: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    state = notebook_initial_state(condition)
    t_now = 0.0
    event_count = 0
    records = [
        {
            "time": 0.0,
            "n_tot": int(state.sum()),
            "f_up": float(np.mean(state >= condition.theta)),
            "event_count": 0,
        }
    ]

    while t_now < duration_seconds and event_count < max_events:
        n_tot = int(state.sum())
        r1_rates = condition.r1 * (condition.N - state) * state / condition.N
        total_r1 = float(np.sum(r1_rates))
        total_r2 = condition.Vmax * n_tot / (condition.Km + n_tot) if n_tot > 0 else 0.0
        diagonal_exchange = float(np.sum(state * (condition.N - state)))
        total_exchange = (condition.gamma / condition.M) * (
            n_tot * (condition.M * condition.N - n_tot) - diagonal_exchange
        )
        total_exchange = max(total_exchange, 0.0)
        total_turnover = condition.delta * n_tot * n_tot / (condition.M * condition.N) if n_tot > 0 else 0.0
        a0 = total_r1 + total_r2 + total_exchange + total_turnover
        if a0 <= 0.0 or a0 >= numerical.propensity_overflow_limit:
            break

        tau_step = -math.log(max(rng.random(), 1.0e-15)) / a0
        if t_now + tau_step > duration_seconds:
            t_now = duration_seconds
            records.append(
                {
                    "time": t_now,
                    "n_tot": int(state.sum()),
                    "f_up": float(np.mean(state >= condition.theta)),
                    "event_count": event_count,
                }
            )
            break

        threshold = rng.random() * a0
        t_now += tau_step

        if threshold < total_r1:
            cumulative = np.cumsum(r1_rates)
            alpha = int(np.searchsorted(cumulative, threshold, side="left"))
            alpha = min(alpha, condition.M - 1)
            state[alpha] = min(state[alpha] + condition.autophosphorylation_step, condition.N)
        elif threshold < total_r1 + total_r2:
            local_threshold = threshold - total_r1
            weights = condition.Vmax * state / (condition.Km + n_tot) if n_tot > 0 else np.zeros(condition.M)
            cumulative = np.cumsum(weights)
            alpha = int(np.searchsorted(cumulative, local_threshold, side="left"))
            alpha = min(alpha, condition.M - 1)
            state[alpha] = max(state[alpha] - 1, 0)
        elif threshold < total_r1 + total_r2 + total_exchange:
            local_threshold = threshold - total_r1 - total_r2
            accept_weights = condition.N - state
            donor_rates = (condition.gamma / condition.M) * state * (np.sum(accept_weights) - accept_weights)
            cumulative = np.cumsum(donor_rates)
            alpha = int(np.searchsorted(cumulative, local_threshold, side="left"))
            alpha = min(alpha, condition.M - 1)
            beta_weights = np.zeros(condition.M, dtype=float)
            for beta in range(condition.M):
                if beta != alpha:
                    beta_weights[beta] = (condition.gamma / condition.M) * state[alpha] * (condition.N - state[beta])
            beta_cumulative = np.cumsum(beta_weights)
            donor_start = cumulative[alpha] - donor_rates[alpha]
            beta = int(np.searchsorted(beta_cumulative, local_threshold - donor_start, side="left"))
            beta = min(beta, condition.M - 1)
            if beta != alpha and state[alpha] > 0 and state[beta] < condition.N:
                state[alpha] -= 1
                state[beta] += 1
        else:
            local_threshold = threshold - total_r1 - total_r2 - total_exchange
            weights = total_turnover * state / n_tot if n_tot > 0 else np.zeros(condition.M)
            cumulative = np.cumsum(weights)
            alpha = int(np.searchsorted(cumulative, local_threshold, side="left"))
            alpha = min(alpha, condition.M - 1)
            state[alpha] = max(state[alpha] - 1, 0)

        event_count += 1
        if event_count % max(record_every_events, 1) == 0:
            records.append(
                {
                    "time": t_now,
                    "n_tot": int(state.sum()),
                    "f_up": float(np.mean(state >= condition.theta)),
                    "event_count": event_count,
                }
            )

    return pd.DataFrame(records)


def build_results_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def simulate_simple_trace(
    condition: ConditionSpec,
    settings: SimulationSettings,
    numerical: NumericalConfig,
    seed: int,
    duration_seconds: float,
    max_events: int,
    downsample_every_events: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_state = initialize_simple_up_state(condition, settings, numerical, seed + 500).astype(np.int64)
    m_count = condition.M
    n_subunits = settings.n_subunits
    t_now = 0.0
    current_r1 = condition.r1_value
    event_count = 0
    records = [{"time": 0.0, "n_tot": int(np.sum(n_state)), "event_count": 0}]

    while t_now < duration_seconds and event_count < max_events:
        n_tot = int(np.sum(n_state))
        dephospho_denominator = settings.effective_km_count + n_tot
        r1_rates = current_r1 * (n_subunits - n_state) * (n_state / n_subunits + condition.epsilon_value)
        r2_rates = np.zeros(m_count, dtype=float)
        if dephospho_denominator > 0.0:
            r2_rates = settings.k_cat * n_state / dephospho_denominator

        if condition.exchange_code == EXCHANGE_B_MONOMER:
            scale = condition.gamma / m_count
            accept_weights = n_subunits - n_state
            donor_rates = scale * n_state * (np.sum(accept_weights) - accept_weights)
            total_exchange = float(np.sum(donor_rates))
        elif condition.exchange_code == EXCHANGE_A_MONOMER:
            scale = condition.gamma
            accept_weights = n_subunits - n_state
            donor_rates = scale * n_state * (np.sum(accept_weights) - accept_weights)
            total_exchange = float(np.sum(donor_rates))
        elif condition.exchange_code == EXCHANGE_B_DIMER:
            scale = condition.gamma / m_count
            donor_units = n_state // 2
            accept_units = (n_subunits - n_state) // 2
            donor_rates = scale * donor_units * (np.sum(accept_units) - accept_units)
            total_exchange = float(np.sum(donor_rates))
        else:
            scale = 0.0
            accept_weights = np.zeros_like(n_state)
            donor_rates = np.zeros(m_count, dtype=float)
            total_exchange = 0.0

        total_r1 = float(np.sum(r1_rates))
        total_r2 = float(np.sum(r2_rates))
        a0 = total_r1 + total_r2 + total_exchange
        if a0 <= 0.0 or a0 >= numerical.propensity_overflow_limit:
            break

        tau_step = -math.log(max(rng.random(), 1.0e-15)) / a0
        if t_now + tau_step > duration_seconds:
            t_now = duration_seconds
            records.append({"time": t_now, "n_tot": int(np.sum(n_state)), "event_count": event_count})
            break

        t_now += tau_step
        threshold = rng.random() * a0
        previous_total = int(np.sum(n_state))

        if threshold < total_r1:
            cumulative = np.cumsum(r1_rates)
            alpha = int(np.searchsorted(cumulative, threshold, side="left"))
            if alpha >= m_count:
                alpha = m_count - 1
            n_state[alpha] += 1
        elif threshold < total_r1 + total_r2:
            local_threshold = threshold - total_r1
            cumulative = np.cumsum(r2_rates)
            alpha = int(np.searchsorted(cumulative, local_threshold, side="left"))
            if alpha >= m_count:
                alpha = m_count - 1
            n_state[alpha] -= 1
        else:
            local_threshold = threshold - total_r1 - total_r2
            cumulative = np.cumsum(donor_rates)
            alpha = int(np.searchsorted(cumulative, local_threshold, side="left"))
            if alpha >= m_count:
                alpha = m_count - 1
            donor_before = int(n_state[alpha])
            beta_weights = np.zeros(m_count, dtype=float)
            for beta in range(m_count):
                if beta == alpha:
                    continue
                if condition.exchange_code == EXCHANGE_B_DIMER:
                    beta_weights[beta] = scale * (donor_before // 2) * ((n_subunits - int(n_state[beta])) // 2)
                else:
                    beta_weights[beta] = scale * donor_before * (n_subunits - int(n_state[beta]))
            beta_cumulative = np.cumsum(beta_weights)
            donor_start = cumulative[alpha] - donor_rates[alpha]
            beta_threshold = local_threshold - donor_start
            beta = int(np.searchsorted(beta_cumulative, beta_threshold, side="left"))
            if beta >= m_count:
                beta = m_count - 1
            if beta == alpha:
                for fallback_beta in range(m_count - 1, -1, -1):
                    if fallback_beta != alpha and beta_weights[fallback_beta] > 0.0:
                        beta = fallback_beta
                        break
            if condition.exchange_code == EXCHANGE_B_DIMER:
                n_state[alpha] -= 2
                n_state[beta] += 2
            else:
                n_state[alpha] -= 1
                n_state[beta] += 1

        event_count += 1
        current_total = int(np.sum(n_state))
        if current_total != previous_total or event_count % max(downsample_every_events, 1) == 0:
            records.append({"time": t_now, "n_tot": current_total, "event_count": event_count})

    return pd.DataFrame(records)
