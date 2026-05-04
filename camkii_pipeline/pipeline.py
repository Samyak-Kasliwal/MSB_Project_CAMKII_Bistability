from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import PipelineConfig, RUN_DISPLAY_NAMES, RUN_EXECUTION_ORDER, selected_runs_with_dependencies
from .io_utils import (
    append_dataframe_csv,
    build_condition_id,
    create_project_layout,
    ensure_run_directories,
    load_dataframe_csv,
    load_pipeline_state,
    save_dataframe_csv,
    save_pipeline_state,
    stable_int_from_text,
    write_json,
)
from .plotting import (
    plot_delta_fixed_points,
    plot_figure_1,
    plot_gamma_m_coupling,
    plot_parameter_sweep,
    plot_survival_curve,
)
from .simulation import (
    ConditionSpec,
    NotebookConditionSpec,
    build_results_dataframe,
    notebook_fixed_points_single,
    notebook_propensity_sanity,
    simulate_notebook_trace,
    simulate_notebook_trajectory_worker,
    simulate_trajectory_worker,
)
from .statistics import (
    bootstrap_lifetime_ci,
    censored_exponential_mle,
    fit_log_linear_relationship,
    kaplan_meier_median,
    slope_uncertainty_envelope,
    survival_linearity_r_squared,
)


def _render_progress_bar(completed: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    ratio = min(max(completed / total, 0.0), 1.0)
    filled = min(width, int(round(ratio * width)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _print_progress(config: PipelineConfig, label: str, completed: int, total: int) -> None:
    if not config.runs.show_progress:
        return
    bar = _render_progress_bar(completed, total)
    print(f"{label} {bar} {completed}/{total}", flush=True)


def _save_state(layout: Any, state: Dict[str, Any], config: PipelineConfig) -> None:
    save_pipeline_state(layout, state, config)


def _set_locked_value(state: Dict[str, Any], key: str, value: Any) -> None:
    state.setdefault("locked_parameters", {})[key] = value


def _mark_completed(state: Dict[str, Any], run_id: str, artifacts: Dict[str, Any]) -> None:
    state.setdefault("completed_runs", {})[run_id] = True
    state.setdefault("artifacts", {})[run_id] = artifacts


def _trajectory_seed(config: PipelineConfig, condition_id: str, trajectory_index: int) -> int:
    base = stable_int_from_text(condition_id, modulus=2_000_000_000)
    return int((config.base_random_seed + base + 97 * trajectory_index) % 2_000_000_000)


def _condition_output_paths(run_paths: Dict[str, Path], condition_id: str) -> Dict[str, Path]:
    return {
        "final_csv": run_paths["raw"] / f"{condition_id}.csv",
        "checkpoint_csv": run_paths["checkpoints"] / f"{condition_id}__partial.csv",
        "metadata_json": run_paths["checkpoints"] / f"{condition_id}__meta.json",
    }


def _condition_id(condition: ConditionSpec) -> str:
    parts = {
        "model": condition.model_label,
        "transition": condition.transition_label,
        "M": condition.M,
        "gamma": condition.gamma,
        "deltaSub": condition.delta_subunit,
        "deltaHolo": condition.delta_holo,
        "r1": condition.r1_value,
        "eps": condition.epsilon_value,
    }
    if condition.use_full_model:
        parts["k1"] = condition.k1_value
        parts["k2"] = condition.k2_value
        parts["kh1"] = condition.kh1_uM
        parts["kh2"] = condition.kh2_uM
    return build_condition_id(parts)


def _notebook_condition_id(condition: NotebookConditionSpec) -> str:
    parts = {
        "model": condition.model_label,
        "transition": condition.transition_label,
        "M": condition.M,
        "gamma": condition.gamma,
        "delta": condition.delta,
        "r1": condition.r1,
        "Vmax": condition.Vmax,
        "step": condition.autophosphorylation_step,
        "spike": int(condition.calcium_spike_enabled),
    }
    if condition.sweep_parameter:
        parts["sweep"] = condition.sweep_parameter
        parts["value"] = condition.sweep_value
    return build_condition_id(parts)


def _load_existing_condition_results(paths: Dict[str, Path]) -> pd.DataFrame:
    partial_df = load_dataframe_csv(paths["checkpoint_csv"])
    if partial_df.empty:
        partial_df = load_dataframe_csv(paths["final_csv"])
    if partial_df.empty:
        return partial_df
    return partial_df.drop_duplicates(subset=["trajectory_index"], keep="last").sort_values("trajectory_index")


def _run_single_condition(
    condition: ConditionSpec,
    config: PipelineConfig,
    run_paths: Dict[str, Path],
) -> pd.DataFrame:
    condition_id = _condition_id(condition)
    paths = _condition_output_paths(run_paths, condition_id)

    if config.runs.force_recompute_selected_runs:
        for path in paths.values():
            if path.exists():
                path.unlink()
        existing_df = pd.DataFrame()
    else:
        existing_df = _load_existing_condition_results(paths)

    completed = int(existing_df["trajectory_index"].nunique()) if not existing_df.empty else 0
    if config.runs.skip_completed_runs and completed >= condition.n_trajectories and paths["final_csv"].exists():
        _print_progress(config, f"{condition.run_id}:{condition_id}", completed, condition.n_trajectories)
        return existing_df

    _print_progress(config, f"{condition.run_id}:{condition_id}", completed, condition.n_trajectories)

    while completed < condition.n_trajectories:
        batch_size = min(config.parallel.batch_size, condition.n_trajectories - completed)
        trajectory_indices = list(range(completed, completed + batch_size))
        seeds = [_trajectory_seed(config, condition_id, index) for index in trajectory_indices]
        inner_n_jobs = min(max(config.parallel.trajectory_n_jobs_per_condition, 1), batch_size)
        if inner_n_jobs == 1:
            worker_rows = [
                simulate_trajectory_worker(
                    condition,
                    config.simulation,
                    config.numerical,
                    seed,
                    trajectory_index,
                )
                for seed, trajectory_index in zip(seeds, trajectory_indices)
            ]
        else:
            worker_rows = Parallel(
                n_jobs=inner_n_jobs,
                backend=config.parallel.backend,
                verbose=config.parallel.verbose,
            )(
                delayed(simulate_trajectory_worker)(
                    condition,
                    config.simulation,
                    config.numerical,
                    seed,
                    trajectory_index,
                )
                for seed, trajectory_index in zip(seeds, trajectory_indices)
            )

        batch_df = build_results_dataframe(worker_rows)
        append_dataframe_csv(paths["checkpoint_csv"], batch_df, config.output.csv_float_format)

        if existing_df.empty:
            existing_df = batch_df
        else:
            existing_df = pd.concat([existing_df, batch_df], ignore_index=True)
            existing_df = existing_df.drop_duplicates(subset=["trajectory_index"], keep="last").sort_values("trajectory_index")

        completed = int(existing_df["trajectory_index"].nunique())
        _print_progress(config, f"{condition.run_id}:{condition_id}", completed, condition.n_trajectories)
        write_json(
            paths["metadata_json"],
            {
                "condition": condition.as_dict(),
                "condition_id": condition_id,
                "completed_trajectories": completed,
                "target_trajectories": condition.n_trajectories,
                "final_csv": str(paths["final_csv"]),
                "checkpoint_csv": str(paths["checkpoint_csv"]),
            },
            indent=config.output.json_indent,
        )

    save_dataframe_csv(paths["final_csv"], existing_df.sort_values("trajectory_index"), config.output.csv_float_format)
    return existing_df.sort_values("trajectory_index")


def _run_single_notebook_condition(
    condition: NotebookConditionSpec,
    config: PipelineConfig,
    run_paths: Dict[str, Path],
) -> pd.DataFrame:
    condition_id = _notebook_condition_id(condition)
    paths = _condition_output_paths(run_paths, condition_id)

    if config.runs.force_recompute_selected_runs:
        for path in paths.values():
            if path.exists():
                path.unlink()
        existing_df = pd.DataFrame()
    else:
        existing_df = _load_existing_condition_results(paths)

    completed = int(existing_df["trajectory_index"].nunique()) if not existing_df.empty else 0
    if config.runs.skip_completed_runs and completed >= condition.n_trajectories and paths["final_csv"].exists():
        _print_progress(config, f"{condition.run_id}:{condition_id}", completed, condition.n_trajectories)
        return existing_df

    _print_progress(config, f"{condition.run_id}:{condition_id}", completed, condition.n_trajectories)
    while completed < condition.n_trajectories:
        batch_size = min(config.parallel.batch_size, condition.n_trajectories - completed)
        trajectory_indices = list(range(completed, completed + batch_size))
        seeds = [_trajectory_seed(config, condition_id, index) for index in trajectory_indices]
        inner_n_jobs = min(max(config.parallel.trajectory_n_jobs_per_condition, 1), batch_size)

        if inner_n_jobs == 1:
            worker_rows = [
                simulate_notebook_trajectory_worker(condition, config.numerical, seed, trajectory_index)
                for seed, trajectory_index in zip(seeds, trajectory_indices)
            ]
        else:
            worker_rows = Parallel(
                n_jobs=inner_n_jobs,
                backend=config.parallel.backend,
                verbose=config.parallel.verbose,
            )(
                delayed(simulate_notebook_trajectory_worker)(condition, config.numerical, seed, trajectory_index)
                for seed, trajectory_index in zip(seeds, trajectory_indices)
            )

        batch_df = build_results_dataframe(worker_rows)
        append_dataframe_csv(paths["checkpoint_csv"], batch_df, config.output.csv_float_format)

        if existing_df.empty:
            existing_df = batch_df
        else:
            existing_df = pd.concat([existing_df, batch_df], ignore_index=True)
            existing_df = existing_df.drop_duplicates(subset=["trajectory_index"], keep="last").sort_values("trajectory_index")

        completed = int(existing_df["trajectory_index"].nunique())
        _print_progress(config, f"{condition.run_id}:{condition_id}", completed, condition.n_trajectories)
        write_json(
            paths["metadata_json"],
            {
                "condition": condition.as_dict(),
                "condition_id": condition_id,
                "completed_trajectories": completed,
                "target_trajectories": condition.n_trajectories,
                "final_csv": str(paths["final_csv"]),
                "checkpoint_csv": str(paths["checkpoint_csv"]),
            },
            indent=config.output.json_indent,
        )

    save_dataframe_csv(paths["final_csv"], existing_df.sort_values("trajectory_index"), config.output.csv_float_format)
    return existing_df.sort_values("trajectory_index")


def _summarize_condition_results(
    condition: ConditionSpec,
    trajectories_df: pd.DataFrame,
    config: PipelineConfig,
    run_paths: Dict[str, Path],
) -> Dict[str, Any]:
    valid_df = trajectories_df[trajectories_df["invalid"] == 0].copy()
    invalid_count = int((trajectories_df["invalid"] == 1).sum()) if not trajectories_df.empty else 0

    if valid_df.empty:
        return {
            **condition.as_dict(),
            "condition_id": _condition_id(condition),
            "valid_trajectories": 0,
            "invalid_trajectories": invalid_count,
            "uncensored_events": 0,
            "censored_observations": 0,
            "reported_metric": "none",
            "reported_lifetime": np.nan,
            "reported_lifetime_lower": np.nan,
            "reported_lifetime_upper": np.nan,
            "tau_hat_mle": np.nan,
            "kaplan_meier_median": np.nan,
            "survival_r_squared": np.nan,
        }

    times = valid_df["observed_time"].to_numpy(dtype=float)
    events = valid_df["transition_found"].to_numpy(dtype=int)
    mle = censored_exponential_mle(times, events)
    km_median = kaplan_meier_median(times, events)
    survival_r2 = survival_linearity_r_squared(times, events)
    uncensored = int(np.sum(events))
    reported_metric = "exponential_mle"
    reported_lifetime = mle["tau_hat"]
    bootstrap_metric = "mle"

    if (
        config.output.generate_plots
        and config.output.generate_survival_plots
        and uncensored >= config.runs.survival_min_uncensored
        and np.isfinite(survival_r2)
    ):
        survival_plot_path = run_paths["plots"] / "survival" / f"{_condition_id(condition)}__survival.png"
        plot_survival_curve(times, events, survival_plot_path, config.output.figure_dpi, f"Survival: {_condition_id(condition)}")
        if survival_r2 < config.runs.survival_linearity_r2_threshold:
            reported_metric = "kaplan_meier_median"
            reported_lifetime = km_median
            bootstrap_metric = "kaplan_meier_median"

    bootstrap = bootstrap_lifetime_ci(
        times,
        events,
        n_bootstrap=config.runs.bootstrap_replicates,
        seed=_trajectory_seed(config, _condition_id(condition), 999_999),
        metric=bootstrap_metric,
    )

    return {
        **condition.as_dict(),
        "condition_id": _condition_id(condition),
        "valid_trajectories": int(len(valid_df)),
        "invalid_trajectories": invalid_count,
        "uncensored_events": uncensored,
        "censored_observations": int(len(valid_df) - uncensored),
        "reported_metric": reported_metric,
        "reported_lifetime": reported_lifetime,
        "reported_lifetime_lower": bootstrap["lower"],
        "reported_lifetime_upper": bootstrap["upper"],
        "tau_hat_mle": mle["tau_hat"],
        "kaplan_meier_median": km_median,
        "survival_r_squared": survival_r2,
    }


def _summarize_notebook_condition_results(
    condition: NotebookConditionSpec,
    trajectories_df: pd.DataFrame,
    config: PipelineConfig,
    run_paths: Dict[str, Path],
) -> Dict[str, Any]:
    valid_df = trajectories_df[trajectories_df["invalid"] == 0].copy()
    invalid_count = int((trajectories_df["invalid"] == 1).sum()) if not trajectories_df.empty else 0

    if valid_df.empty:
        return {
            **condition.as_dict(),
            "condition_id": _notebook_condition_id(condition),
            "valid_trajectories": 0,
            "invalid_trajectories": invalid_count,
            "uncensored_events": 0,
            "censored_observations": 0,
            "reported_metric": "none",
            "reported_lifetime": np.nan,
            "reported_lifetime_lower": np.nan,
            "reported_lifetime_upper": np.nan,
            "tau_hat_mle": np.nan,
            "kaplan_meier_median": np.nan,
            "survival_r_squared": np.nan,
            "mean_final_f_up": np.nan,
        }

    times = valid_df["observed_time"].to_numpy(dtype=float)
    events = valid_df["transition_found"].to_numpy(dtype=int)
    mle = censored_exponential_mle(times, events)
    km_median = kaplan_meier_median(times, events)
    survival_r2 = survival_linearity_r_squared(times, events)
    uncensored = int(np.sum(events))
    reported_metric = "exponential_mle"
    reported_lifetime = mle["tau_hat"]
    bootstrap_metric = "mle"

    if (
        config.output.generate_plots
        and config.output.generate_survival_plots
        and uncensored >= config.runs.survival_min_uncensored
        and np.isfinite(survival_r2)
    ):
        survival_plot_path = run_paths["plots"] / "survival" / f"{_notebook_condition_id(condition)}__survival.png"
        plot_survival_curve(times, events, survival_plot_path, config.output.figure_dpi, f"Survival: {_notebook_condition_id(condition)}")
        if survival_r2 < config.runs.survival_linearity_r2_threshold:
            reported_metric = "kaplan_meier_median"
            reported_lifetime = km_median
            bootstrap_metric = "kaplan_meier_median"

    bootstrap = bootstrap_lifetime_ci(
        times,
        events,
        n_bootstrap=config.runs.bootstrap_replicates,
        seed=_trajectory_seed(config, _notebook_condition_id(condition), 999_999),
        metric=bootstrap_metric,
    )

    return {
        **condition.as_dict(),
        "condition_id": _notebook_condition_id(condition),
        "valid_trajectories": int(len(valid_df)),
        "invalid_trajectories": invalid_count,
        "uncensored_events": uncensored,
        "censored_observations": int(len(valid_df) - uncensored),
        "reported_metric": reported_metric,
        "reported_lifetime": reported_lifetime,
        "reported_lifetime_lower": bootstrap["lower"],
        "reported_lifetime_upper": bootstrap["upper"],
        "tau_hat_mle": mle["tau_hat"],
        "kaplan_meier_median": km_median,
        "survival_r_squared": survival_r2,
        "mean_final_f_up": float(valid_df["final_f_up"].mean()) if "final_f_up" in valid_df else np.nan,
    }


def _fit_slopes(summary_df: pd.DataFrame, group_columns: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for group_key, group_df in summary_df.groupby(group_columns):
        ordered = group_df.sort_values("M")
        fit = fit_log_linear_relationship(ordered["M"].to_numpy(dtype=float), ordered["reported_lifetime"].to_numpy(dtype=float))
        envelope = slope_uncertainty_envelope(
            ordered["M"].to_numpy(dtype=float),
            ordered["reported_lifetime_lower"].to_numpy(dtype=float),
            ordered["reported_lifetime_upper"].to_numpy(dtype=float),
        )
        row: Dict[str, Any] = {}
        if len(group_columns) == 1:
            row[group_columns[0]] = group_key
        else:
            for column_name, column_value in zip(group_columns, group_key):
                row[column_name] = column_value
        row.update(fit)
        row.update(envelope)
        row["n_conditions"] = int(len(ordered))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_summary_tables(
    run_paths: Dict[str, Path],
    config: PipelineConfig,
    summary_df: pd.DataFrame,
    summary_name: str,
) -> Path:
    output_path = run_paths["tables"] / summary_name
    save_dataframe_csv(output_path, summary_df, config.output.csv_float_format)
    return output_path


def _run_and_summarize_condition(
    condition: ConditionSpec,
    config: PipelineConfig,
    run_paths: Dict[str, Path],
) -> Dict[str, Any]:
    trajectories_df = _run_single_condition(condition, config, run_paths)
    return _summarize_condition_results(condition, trajectories_df, config, run_paths)


def _run_grid(
    run_id: str,
    conditions: Iterable[ConditionSpec],
    config: PipelineConfig,
    run_paths: Dict[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    condition_list = list(conditions)
    if config.runs.show_progress:
        print(f"{run_id}: starting {len(condition_list)} condition(s)", flush=True)
    if config.parallel.parallelize_conditions and len(condition_list) > 1:
        summary_rows = Parallel(
            n_jobs=min(config.parallel.condition_n_jobs, len(condition_list)),
            backend=config.parallel.backend,
            verbose=config.parallel.verbose,
        )(
            delayed(_run_and_summarize_condition)(condition, config, run_paths)
            for condition in condition_list
        )
    else:
        summary_rows = []
        for index, condition in enumerate(condition_list, start=1):
            _print_progress(config, f"{run_id}:conditions", index - 1, len(condition_list))
            summary_rows.append(_run_and_summarize_condition(condition, config, run_paths))
        _print_progress(config, f"{run_id}:conditions", len(condition_list), len(condition_list))

    trajectories_lookup = {_condition_id(condition): True for condition in condition_list}
    summary_df = pd.DataFrame(summary_rows)
    summary_path = _write_summary_tables(run_paths, config, summary_df, f"{run_id}__condition_summary.csv")
    trajectories_index_path = run_paths["tables"] / f"{run_id}__trajectory_index.json"
    write_json(
        trajectories_index_path,
        {key: str((run_paths["raw"] / f"{key}.csv").resolve()) for key in trajectories_lookup},
        indent=config.output.json_indent,
    )
    return summary_df, pd.DataFrame([{"summary_csv": str(summary_path), "trajectory_index_json": str(trajectories_index_path)}])


def _run_and_summarize_notebook_condition(
    condition: NotebookConditionSpec,
    config: PipelineConfig,
    run_paths: Dict[str, Path],
) -> Dict[str, Any]:
    trajectories_df = _run_single_notebook_condition(condition, config, run_paths)
    return _summarize_notebook_condition_results(condition, trajectories_df, config, run_paths)


def _run_notebook_grid(
    run_id: str,
    conditions: Iterable[NotebookConditionSpec],
    config: PipelineConfig,
    run_paths: Dict[str, Path],
    summary_name: str | None = None,
) -> pd.DataFrame:
    condition_list = list(conditions)
    if config.runs.show_progress:
        display = RUN_DISPLAY_NAMES.get(run_id, run_id)
        print(f"{display}: starting {len(condition_list)} condition(s)", flush=True)

    if config.parallel.parallelize_conditions and len(condition_list) > 1:
        summary_rows = Parallel(
            n_jobs=min(config.parallel.condition_n_jobs, len(condition_list)),
            backend=config.parallel.backend,
            verbose=config.parallel.verbose,
        )(
            delayed(_run_and_summarize_notebook_condition)(condition, config, run_paths)
            for condition in condition_list
        )
    else:
        summary_rows = []
        for index, condition in enumerate(condition_list, start=1):
            _print_progress(config, f"{run_id}:conditions", index - 1, len(condition_list))
            summary_rows.append(_run_and_summarize_notebook_condition(condition, config, run_paths))
        _print_progress(config, f"{run_id}:conditions", len(condition_list), len(condition_list))

    summary_df = pd.DataFrame(summary_rows)
    file_name = summary_name if summary_name is not None else f"{run_id}__condition_summary.csv"
    summary_path = _write_summary_tables(run_paths, config, summary_df, file_name)
    trajectories_index_path = run_paths["tables"] / f"{run_id}__trajectory_index.json"
    write_json(
        trajectories_index_path,
        { _notebook_condition_id(condition): str((run_paths["raw"] / f"{_notebook_condition_id(condition)}.csv").resolve()) for condition in condition_list },
        indent=config.output.json_indent,
    )
    return summary_df


def _condition_mz_full(
    run_id: str,
    transition_label: str,
    m_value: int,
    n_trajectories: int,
    t_max: float,
    config: PipelineConfig,
) -> ConditionSpec:
    return ConditionSpec(
        run_id=run_id,
        model_label="MZ-full",
        transition_label=transition_label,
        M=m_value,
        n_trajectories=n_trajectories,
        t_max=t_max,
        delta_holo=config.simulation.delta_holo_miller,
        k1_value=config.simulation.k1_full,
        k2_value=config.simulation.k2_full,
        kh1_uM=config.simulation.active_kh1_uM(),
        kh2_uM=config.simulation.kh2_uM,
        hill_exponent=config.simulation.hill_exponent,
        rest_ca_uM=config.simulation.rest_ca_uM,
        ltp_ca_uM=config.simulation.ltp_ca_uM,
        use_full_model=True,
        notes="Verify kh2_uM against Miller et al. Table 1 before final publication runs.",
    )


def _notebook_condition(
    config: PipelineConfig,
    run_id: str,
    model_label: str,
    transition_label: str,
    m_value: int,
    n_trajectories: int,
    t_max: float,
    gamma: float | None = None,
    delta: float | None = None,
    r1: float | None = None,
    vmax: float | None = None,
    autophosphorylation_step: int = 1,
    calcium_spike_enabled: bool = False,
    initial_mode: str = "up",
    sweep_parameter: str = "",
    sweep_value: float = np.nan,
    notes: str = "",
) -> NotebookConditionSpec:
    return NotebookConditionSpec(
        run_id=run_id,
        model_label=model_label,
        transition_label=transition_label,
        M=m_value,
        n_trajectories=n_trajectories,
        t_max=t_max,
        N=config.simulation.n_subunits,
        theta=config.simulation.theta,
        r1=config.simulation.initial_trial_r1 if r1 is None else r1,
        Vmax=config.simulation.k_cat if vmax is None else vmax,
        Km=config.simulation.effective_km_count,
        gamma=0.01 if gamma is None else gamma,
        delta=config.simulation.my_model_delta if delta is None else delta,
        autophosphorylation_step=autophosphorylation_step,
        calcium_spike_enabled=calcium_spike_enabled,
        calcium_spike_start_seconds=config.simulation.calcium_spike_start_seconds,
        calcium_spike_duration_seconds=config.simulation.calcium_spike_duration_seconds,
        calcium_spike_period_seconds=config.simulation.calcium_spike_period_seconds,
        calcium_spike_count=config.simulation.calcium_spike_count,
        calcium_spike_r1_multiplier=config.simulation.calcium_spike_r1_multiplier,
        initial_mode=initial_mode,
        sweep_parameter=sweep_parameter,
        sweep_value=sweep_value,
        notes=notes,
    )


def _simple_sanity_check(config: PipelineConfig, layout: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    run_id = "simple_sanity_check"
    run_paths = ensure_run_directories(layout, run_id)
    condition = _notebook_condition(
        config,
        run_id=run_id,
        model_label="Notebook-exact",
        transition_label="up_to_down",
        m_value=16,
        n_trajectories=config.runs.simple_sanity_trajectories,
        t_max=1.0e4,
        gamma=0.01,
        delta=1.0e-4,
        notes="Notebook DEFAULTS sanity check: half-phosphorylated propensity table and one short UP trace.",
    )

    sanity_df = pd.DataFrame([notebook_propensity_sanity(condition)])
    sanity_path = _write_summary_tables(run_paths, config, sanity_df, "simple_sanity_check__propensities.csv")

    trace_df = simulate_notebook_trace(
        condition,
        config.numerical,
        _trajectory_seed(config, "simple_sanity_check_trace", 0),
        duration_seconds=1.0e4,
        max_events=50_000,
        record_every_events=config.simulation.my_model_record_every_events,
    )
    trace_path = run_paths["raw"] / "simple_sanity_check__trace.csv"
    save_dataframe_csv(trace_path, trace_df, config.output.csv_float_format)

    artifacts = {
        "propensity_csv": str(sanity_path),
        "trace_csv": str(trace_path),
        "channels": int(sanity_df.iloc[0]["channels"]),
        "total_propensity": float(sanity_df.iloc[0]["total_propensity"]),
    }
    _mark_completed(state, run_id, artifacts)
    return artifacts


def _zabotinsky_reproduction(config: PipelineConfig, layout: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    run_id = "zabotinsky_reproduction"
    run_paths = ensure_run_directories(layout, run_id)
    base = notebook_fixed_points_single(
        config.simulation.n_subunits,
        config.simulation.initial_trial_r1,
        config.simulation.k_cat,
        config.simulation.effective_km_count,
        1.0e-4,
    )
    base_df = pd.DataFrame(
        [
            {
                **{key: value for key, value in base.items() if key != "fixed_points"},
                "lower_fixed_point": base["fixed_points"][0] if len(base["fixed_points"]) > 0 else np.nan,
                "upper_fixed_point": base["fixed_points"][1] if len(base["fixed_points"]) > 1 else np.nan,
            }
        ]
    )
    base_path = _write_summary_tables(run_paths, config, base_df, "zabotinsky_reproduction__fixed_points.csv")

    scan_rows: List[Dict[str, Any]] = []
    for delta_value in np.logspace(-6, -1, 300):
        result = notebook_fixed_points_single(
            config.simulation.n_subunits,
            config.simulation.initial_trial_r1,
            config.simulation.k_cat,
            config.simulation.effective_km_count,
            float(delta_value),
        )
        fixed_points = result["fixed_points"]
        scan_rows.append(
            {
                "delta": float(delta_value),
                "discriminant": result["discriminant"],
                "is_bistable": int(result["is_bistable"]),
                "lower_fixed_point": fixed_points[0] if len(fixed_points) > 0 else np.nan,
                "upper_fixed_point": fixed_points[1] if len(fixed_points) > 1 else np.nan,
                "delta_crit": result["delta_crit"],
            }
        )
    scan_df = pd.DataFrame(scan_rows)
    scan_path = _write_summary_tables(run_paths, config, scan_df, "zabotinsky_reproduction__delta_scan.csv")
    figure_path = run_paths["plots"] / "zabotinsky_reproduction__delta_scan.png"
    if config.output.generate_plots:
        plot_delta_fixed_points(scan_df, figure_path, config.output.figure_dpi)

    artifacts = {
        "fixed_points_csv": str(base_path),
        "delta_scan_csv": str(scan_path),
        "figure": str(figure_path),
        "delta_crit": float(base["delta_crit"]),
        "is_bistable": bool(base["is_bistable"]),
    }
    _mark_completed(state, run_id, artifacts)
    return artifacts


def _miller_reproduction(config: PipelineConfig, layout: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    run_id = "miller_reproduction"
    run_paths = ensure_run_directories(layout, run_id)
    conditions = [
        _condition_mz_full(
            run_id=run_id,
            transition_label="up_to_down",
            m_value=m_value,
            n_trajectories=config.runs.miller_reproduction_trajectories,
            t_max=config.simulation.tmax_by_m[m_value],
            config=config,
        )
        for m_value in config.simulation.m_scan_main
    ]
    summary_df, _ = _run_grid(run_id, conditions, config, run_paths)
    fit_df = _fit_slopes(summary_df, ["model"])
    fit_path = _write_summary_tables(run_paths, config, fit_df, "miller_reproduction__slope_fit.csv")
    summary_path = run_paths["tables"] / f"{run_id}__condition_summary.csv"
    figure_path = run_paths["plots"] / "miller_reproduction__lifetime_vs_M.png"
    if config.output.generate_plots and not fit_df.empty:
        plot_figure_1(summary_df.sort_values("M"), fit_df.iloc[0], None, figure_path, config.output.figure_dpi)
    if not fit_df.empty:
        _set_locked_value(state, "k_miller", float(fit_df.iloc[0]["slope"]))
    artifacts = {
        "summary_csv": str(summary_path),
        "slope_csv": str(fit_path),
        "figure": str(figure_path),
        "k_miller": float(fit_df.iloc[0]["slope"]) if not fit_df.empty else np.nan,
    }
    _mark_completed(state, run_id, artifacts)
    return artifacts


def _my_model(config: PipelineConfig, layout: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    run_id = "my_model"
    run_paths = ensure_run_directories(layout, run_id)
    conditions = [
        _notebook_condition(
            config,
            run_id=run_id,
            model_label="My model gamma/M coupling",
            transition_label="up_to_down",
            m_value=m_value,
            n_trajectories=config.runs.my_model_trajectories,
            t_max=config.simulation.my_model_t_max,
            gamma=gamma_value,
            delta=config.simulation.my_model_delta,
            notes="Notebook-exact R1/R2/R3/R4 with R3 normalized as gamma/M and fair-sampling turnover.",
        )
        for gamma_value in config.simulation.my_model_gamma_scan
        for m_value in config.simulation.my_model_m_scan
    ]
    summary_df = _run_notebook_grid(run_id, conditions, config, run_paths, "my_model__gamma_M_coupling_summary.csv")
    slope_df = _fit_slopes(summary_df, ["gamma"]).sort_values("gamma")
    slope_path = _write_summary_tables(run_paths, config, slope_df, "my_model__gamma_M_coupling_slopes.csv")
    figure_path = run_paths["plots"] / "my_model__gamma_M_coupling.png"
    if config.output.generate_plots:
        plot_gamma_m_coupling(summary_df, figure_path, config.output.figure_dpi, "My Model: Gamma/M Coupling")
    artifacts = {
        "summary_csv": str(run_paths["tables"] / "my_model__gamma_M_coupling_summary.csv"),
        "slope_csv": str(slope_path),
        "figure": str(figure_path),
    }
    _mark_completed(state, run_id, artifacts)
    return artifacts


def _caliberation_checks(config: PipelineConfig, layout: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    run_id = "caliberation_checks"
    run_paths = ensure_run_directories(layout, run_id)
    base_m = config.simulation.my_model_m_scan[-1]
    base_gamma = 0.01
    base_delta = config.simulation.my_model_delta
    conditions: List[NotebookConditionSpec] = []

    for value in config.simulation.r1_candidate_values:
        conditions.append(
            _notebook_condition(
                config,
                run_id,
                "Calibration r1 sweep",
                "up_to_down",
                base_m,
                config.runs.caliberation_check_trajectories,
                config.simulation.my_model_t_max,
                gamma=base_gamma,
                delta=base_delta,
                r1=float(value),
                sweep_parameter="r1",
                sweep_value=float(value),
            )
        )
    for value in config.simulation.my_model_gamma_scan:
        conditions.append(
            _notebook_condition(
                config,
                run_id,
                "Calibration gamma sweep",
                "up_to_down",
                base_m,
                config.runs.caliberation_check_trajectories,
                config.simulation.my_model_t_max,
                gamma=float(value),
                delta=base_delta,
                sweep_parameter="gamma",
                sweep_value=float(value),
            )
        )
    for value in config.simulation.delta_scan_subunit:
        conditions.append(
            _notebook_condition(
                config,
                run_id,
                "Calibration delta sweep",
                "up_to_down",
                base_m,
                config.runs.caliberation_check_trajectories,
                config.simulation.my_model_t_max,
                gamma=base_gamma,
                delta=float(value),
                sweep_parameter="delta",
                sweep_value=float(value),
            )
        )
    for value in config.simulation.vmax_scan:
        conditions.append(
            _notebook_condition(
                config,
                run_id,
                "Calibration Vmax sweep",
                "up_to_down",
                base_m,
                config.runs.caliberation_check_trajectories,
                config.simulation.my_model_t_max,
                gamma=base_gamma,
                delta=base_delta,
                vmax=float(value),
                sweep_parameter="Vmax",
                sweep_value=float(value),
            )
        )

    summary_df = _run_notebook_grid(run_id, conditions, config, run_paths, "caliberation_checks__summary.csv")
    figure_path = run_paths["plots"] / "caliberation_checks__sweeps.png"
    if config.output.generate_plots:
        plot_parameter_sweep(summary_df, figure_path, config.output.figure_dpi, "Caliberation Checks")
    artifacts = {
        "summary_csv": str(run_paths["tables"] / "caliberation_checks__summary.csv"),
        "figure": str(figure_path),
    }
    _mark_completed(state, run_id, artifacts)
    return artifacts


def _modified_autophosphrylation_propensity_check(
    config: PipelineConfig,
    layout: Any,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    run_id = "modified_autophosphrylation_propensity_check"
    run_paths = ensure_run_directories(layout, run_id)
    conditions = [
        _notebook_condition(
            config,
            run_id=run_id,
            model_label=f"My model autophosphorylation step {step}",
            transition_label="up_to_down",
            m_value=m_value,
            n_trajectories=config.runs.modified_autophosphrylation_trajectories,
            t_max=config.simulation.my_model_t_max,
            gamma=gamma_value,
            delta=config.simulation.my_model_delta,
            autophosphorylation_step=step,
            notes="Modified my-model check: an R1 event flips two subunits ON at once when step=2.",
        )
        for step in (1, 2)
        for gamma_value in config.simulation.my_model_gamma_scan
        for m_value in config.simulation.my_model_m_scan
    ]
    summary_df = _run_notebook_grid(run_id, conditions, config, run_paths, "modified_autophosphrylation_propensity_check__summary.csv")
    slope_df = _fit_slopes(summary_df, ["model", "gamma"]).sort_values(["model", "gamma"])
    slope_path = _write_summary_tables(run_paths, config, slope_df, "modified_autophosphrylation_propensity_check__slopes.csv")
    figure_path = run_paths["plots"] / "modified_autophosphrylation_propensity_check__gamma_M.png"
    if config.output.generate_plots:
        plot_gamma_m_coupling(summary_df, figure_path, config.output.figure_dpi, "Modified_autophosphrylation_propensity Check")
    artifacts = {
        "summary_csv": str(run_paths["tables"] / "modified_autophosphrylation_propensity_check__summary.csv"),
        "slope_csv": str(slope_path),
        "figure": str(figure_path),
    }
    _mark_completed(state, run_id, artifacts)
    return artifacts


def _calcium_spike_condition_analysis(config: PipelineConfig, layout: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    run_id = "calcium_spike_condition_analysis"
    run_paths = ensure_run_directories(layout, run_id)
    conditions = [
        _notebook_condition(
            config,
            run_id=run_id,
            model_label="My model no calcium spikes" if not spike_enabled else "My model calcium spikes",
            transition_label="up_to_down",
            m_value=m_value,
            n_trajectories=config.runs.calcium_spike_trajectories,
            t_max=config.simulation.my_model_t_max,
            gamma=gamma_value,
            delta=config.simulation.my_model_delta,
            calcium_spike_enabled=spike_enabled,
            notes="Calcium spike addition: r1 is multiplied during scheduled spike windows.",
        )
        for spike_enabled in (False, True)
        for gamma_value in config.simulation.my_model_gamma_scan
        for m_value in config.simulation.my_model_m_scan
    ]
    summary_df = _run_notebook_grid(run_id, conditions, config, run_paths, "calcium_spike_condition_analysis__summary.csv")
    slope_df = _fit_slopes(summary_df, ["model", "gamma"]).sort_values(["model", "gamma"])
    slope_path = _write_summary_tables(run_paths, config, slope_df, "calcium_spike_condition_analysis__slopes.csv")
    figure_path = run_paths["plots"] / "calcium_spike_condition_analysis__gamma_M.png"
    if config.output.generate_plots:
        plot_gamma_m_coupling(summary_df, figure_path, config.output.figure_dpi, "Calcium Spike Condition Analysis")
    artifacts = {
        "summary_csv": str(run_paths["tables"] / "calcium_spike_condition_analysis__summary.csv"),
        "slope_csv": str(slope_path),
        "figure": str(figure_path),
    }
    _mark_completed(state, run_id, artifacts)
    return artifacts


RUN_FUNCTIONS: Dict[str, Callable[[PipelineConfig, Any, Dict[str, Any]], Dict[str, Any]]] = {
    "simple_sanity_check": _simple_sanity_check,
    "zabotinsky_reproduction": _zabotinsky_reproduction,
    "miller_reproduction": _miller_reproduction,
    "my_model": _my_model,
    "caliberation_checks": _caliberation_checks,
    "modified_autophosphrylation_propensity_check": _modified_autophosphrylation_propensity_check,
    "calcium_spike_condition_analysis": _calcium_spike_condition_analysis,
}


def execute_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    layout = create_project_layout(config)
    initial_payload = load_pipeline_state(layout)
    state = initial_payload.get("state", {"completed_runs": {}, "locked_parameters": {}, "artifacts": {}})

    requested_runs = selected_runs_with_dependencies(config.runs.selected_runs)
    resolved_order = [run_id for run_id in RUN_EXECUTION_ORDER if run_id in requested_runs]

    for run_id in resolved_order:
        already_completed = bool(state.get("completed_runs", {}).get(run_id, False))
        if already_completed and config.runs.skip_completed_runs and not config.runs.force_recompute_selected_runs:
            continue
        artifacts = RUN_FUNCTIONS[run_id](config, layout, state)
        state.setdefault("artifacts", {})[run_id] = artifacts
        _save_state(layout, state, config)

    return state
