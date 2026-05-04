from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


RUN_EXECUTION_ORDER: List[str] = [
    "simple_sanity_check",
    "zabotinsky_reproduction",
    "miller_reproduction",
    "my_model",
    "caliberation_checks",
    "modified_autophosphrylation_propensity_check",
    "calcium_spike_condition_analysis",
]


RUN_DISPLAY_NAMES: Dict[str, str] = {
    "simple_sanity_check": "Simple sanity check",
    "zabotinsky_reproduction": "Zabotinsky reproduction",
    "miller_reproduction": "Miller reproduction",
    "my_model": "My model",
    "caliberation_checks": "Caliberation checks",
    "modified_autophosphrylation_propensity_check": "Modified_autophosphrylation_propensity check",
    "calcium_spike_condition_analysis": "Calcium_spike condition analysis",
}


RUN_ALIASES: Dict[str, str] = {
    "all": "all",
    "sanity_check": "simple_sanity_check",
    "simple sanity check": "simple_sanity_check",
    "zabotinsky reproduction": "zabotinsky_reproduction",
    "zhabotinsky reproduction": "zabotinsky_reproduction",
    "miller reproduction": "miller_reproduction",
    "my model": "my_model",
    "caliberation checks": "caliberation_checks",
    "caliberation_checks": "caliberation_checks",
    "calibration checks": "caliberation_checks",
    "calibration_checks": "caliberation_checks",
    "modified_autophosphrylation_propensity check": "modified_autophosphrylation_propensity_check",
    "modified_autophosphrylation_propensity_check": "modified_autophosphrylation_propensity_check",
    "modified_autophosphorylation_propensity_check": "modified_autophosphrylation_propensity_check",
    "calcium_spike condition analysis": "calcium_spike_condition_analysis",
}


RUN_DEPENDENCIES: Dict[str, List[str]] = {
    "simple_sanity_check": [],
    "zabotinsky_reproduction": [],
    "miller_reproduction": [],
    "my_model": ["simple_sanity_check"],
    "caliberation_checks": ["my_model"],
    "modified_autophosphrylation_propensity_check": ["my_model"],
    "calcium_spike_condition_analysis": ["my_model"],
}


MODEL_STYLES: Dict[str, Dict[str, str]] = {
    "MZ-full": {"color": "black", "linestyle": "-", "label": "MZ-full"},
    "MZ-simple": {"color": "gray", "linestyle": "-", "label": "MZ-simple"},
    "Exchange-A": {"color": "red", "linestyle": "-", "label": "Exchange-A"},
    "Exchange-B": {"color": "blue", "linestyle": "-", "label": "Exchange-B"},
    "Exchange-B+T": {"color": "blue", "linestyle": "--", "label": "Exchange-B+T"},
    "Exchange-B+Dimer": {"color": "green", "linestyle": "-", "label": "Exchange-B+Dimer"},
    "Exchange-B+Ca": {"color": "orange", "linestyle": "-", "label": "Exchange-B+Ca"},
}


GAMMA_LINESTYLES: Dict[float, str] = {
    0.001: "-",
    0.01: "-",
    0.05: "-.",
    0.1: "--",
    0.5: ":",
    1.0: ":",
}


@dataclass
class OutputConfig:
    project_root: Path
    results_dir_name: str = "results"
    checkpoints_dir_name: str = "checkpoints"
    figure_dpi: int = 300
    csv_float_format: str = "%.10e"
    json_indent: int = 2
    generate_plots: bool = False
    generate_survival_plots: bool = False


@dataclass
class ParallelConfig:
    n_jobs: int = 20
    backend: str = "loky"
    batch_size: int = 8
    verbose: int = 0
    parallelize_conditions: bool = True
    condition_n_jobs: int = 6
    trajectory_n_jobs_per_condition: int = 1


@dataclass
class NumericalConfig:
    use_numba: bool = True
    kahan_required_m_threshold: int = 20
    propensity_overflow_limit: float = 1.0e12
    max_consecutive_exchange_events: int = 1_000_000
    state_tolerance: float = 1.0e-9


@dataclass
class SimulationSettings:
    n_subunits: int = 12
    theta: int = 6
    k_cat: float = 2.0
    effective_km_count: float = 2.0
    basal_first_step_target_rate: float = 7.94e-5
    delta_holo_miller: float = 9.26e-6
    rest_ca_uM: float = 0.1
    ltp_ca_uM: float = 1.0
    ca_noise_low_uM: float = 0.06
    ca_noise_high_uM: float = 0.14
    ca_noise_refresh_seconds: float = 2.0
    m_scan_main: List[int] = field(default_factory=lambda: [8, 12, 16, 20, 25])
    m_scan_turnover: List[int] = field(default_factory=lambda: [12, 16, 25])
    gamma_scan_main: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0])
    gamma_scan_phase: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.05, 0.1, 0.5, 1.0])
    delta_scan_subunit: List[float] = field(default_factory=lambda: [0.0, 1.0e-5, 1.0e-4, 5.0e-4, 1.0e-3])
    vmax_scan: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    my_model_m_scan: List[int] = field(default_factory=lambda: [2, 4, 6, 8])
    my_model_gamma_scan: List[float] = field(default_factory=lambda: [0.0, 0.001, 0.01, 0.1])
    my_model_delta: float = 2.0e-3
    my_model_t_max: float = 1.0e5
    my_model_record_every_events: int = 200
    calcium_spike_start_seconds: float = 500.0
    calcium_spike_duration_seconds: float = 20.0
    calcium_spike_period_seconds: float = 1_000.0
    calcium_spike_count: int = 20
    calcium_spike_r1_multiplier: float = 8.0
    r1_candidate_values: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])
    initial_trial_r1: float = 1.5
    k1_full: float = 6.0
    k2_full: float = 7.0
    kh1_uM: float = 0.7
    kh1_adjusted_uM: float = 4.0
    kh2_uM: float = 0.7
    hill_exponent: float = 2.0
    use_adjusted_kh1_for_full_model: bool = False
    map_simple_model_ltp_to_hill_scaled_r1: bool = True
    tmax_by_m: Dict[int, float] = field(
        default_factory=lambda: {
            1: 1.0e7,
            8: 1.0e7,
            12: 1.0e8,
            16: 5.0e8,
            20: 1.0e9,
            25: 1.0e9,
        }
    )

    def resting_hill_first_step(self) -> float:
        return self.rest_ca_uM ** self.hill_exponent / (
            self.active_kh1_uM() ** self.hill_exponent + self.rest_ca_uM ** self.hill_exponent
        )

    def ltp_hill_first_step(self) -> float:
        return self.ltp_ca_uM ** self.hill_exponent / (
            self.active_kh1_uM() ** self.hill_exponent + self.ltp_ca_uM ** self.hill_exponent
        )

    def active_kh1_uM(self) -> float:
        if self.use_adjusted_kh1_for_full_model:
            return self.kh1_adjusted_uM
        return self.kh1_uM


@dataclass
class RunSettings:
    selected_runs: List[str] = field(default_factory=lambda: ["all"])
    skip_completed_runs: bool = True
    force_recompute_selected_runs: bool = False
    simple_sanity_trajectories: int = 1
    miller_reproduction_trajectories: int = 300
    my_model_trajectories: int = 3
    caliberation_check_trajectories: int = 3
    modified_autophosphrylation_trajectories: int = 3
    calcium_spike_trajectories: int = 3
    bootstrap_replicates: int = 1000
    survival_min_uncensored: int = 50
    survival_linearity_r2_threshold: float = 0.97
    show_progress: bool = True


@dataclass
class LockedParameterOverrides:
    epsilon: Optional[float] = None
    k_miller: Optional[float] = None
    r1_star: Optional[float] = None


@dataclass
class PipelineConfig:
    output: OutputConfig
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    runs: RunSettings = field(default_factory=RunSettings)
    overrides: LockedParameterOverrides = field(default_factory=LockedParameterOverrides)
    base_random_seed: int = 123456

    def to_serializable_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["output"]["project_root"] = str(self.output.project_root)
        return payload


def build_default_config(project_root: Path) -> PipelineConfig:
    return PipelineConfig(output=OutputConfig(project_root=project_root))


def selected_runs_with_dependencies(selected_runs: List[str]) -> List[str]:
    requested = []
    for run_id in selected_runs:
        normalized = run_id.strip().lower()
        requested.append(RUN_ALIASES.get(normalized, run_id))
    if not requested or "all" in requested:
        return RUN_EXECUTION_ORDER.copy()

    resolved: List[str] = []
    seen: set[str] = set()

    def visit(run_id: str) -> None:
        if run_id in seen:
            return
        if run_id not in RUN_DEPENDENCIES:
            raise ValueError(f"Unknown run id '{run_id}'.")
        for dependency in RUN_DEPENDENCIES[run_id]:
            visit(dependency)
        seen.add(run_id)
        resolved.append(run_id)

    for run_id in requested:
        visit(run_id)

    order_index = {name: index for index, name in enumerate(RUN_EXECUTION_ORDER)}
    resolved.sort(key=lambda name: order_index[name])
    return resolved
