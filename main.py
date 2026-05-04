"""CaMKII gamma/M coupling analysis entrypoint.

Edit `build_user_config()` to choose runs, sample sizes, and scan values.
The default selection keeps the most important analysis first: the notebook-
exact my-model gamma/M coupling scan.
"""

from __future__ import annotations

from pathlib import Path

from camkii_pipeline import build_default_config, execute_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent


def build_user_config():
    config = build_default_config(PROJECT_ROOT)

    # Requested run names:
    #   simple_sanity_check
    #   zabotinsky_reproduction
    #   miller_reproduction
    #   my_model
    #   caliberation_checks
    #   modified_autophosphrylation_propensity_check
    #   calcium_spike_condition_analysis
    #
    # Use ["all"] to execute the full ordered pipeline. The default runs the
    # central gamma/M coupling analysis and its sanity-check dependency.
    config.runs.selected_runs = ["my_model"]

    config.runs.skip_completed_runs = True
    config.runs.force_recompute_selected_runs = False
    config.runs.show_progress = True

    config.parallel.n_jobs = 20
    config.parallel.backend = "loky"
    config.parallel.batch_size = 4
    config.parallel.parallelize_conditions = True
    config.parallel.condition_n_jobs = 6
    config.parallel.trajectory_n_jobs_per_condition = 1

    config.output.generate_plots = False
    config.output.generate_survival_plots = False
    config.output.figure_dpi = 300

    # Notebook DEFAULTS and gamma/M-focused scan.
    config.simulation.n_subunits = 12
    config.simulation.theta = 6
    config.simulation.initial_trial_r1 = 1.5
    config.simulation.k_cat = 2.0
    config.simulation.effective_km_count = 2.0
    config.simulation.my_model_m_scan = [2, 4, 6, 8]
    config.simulation.my_model_gamma_scan = [0.0, 0.001, 0.01, 0.1]
    config.simulation.my_model_delta = 2.0e-3
    config.simulation.my_model_t_max = 1.0e5
    config.simulation.m_scan_main = [8, 12, 16]

    # Sweeps used by caliberation_checks.
    config.simulation.r1_candidate_values = [1.0, 1.5, 2.0]
    config.simulation.delta_scan_subunit = [0.0, 1.0e-5, 1.0e-4, 1.0e-3, 2.0e-3]
    config.simulation.vmax_scan = [1.0, 2.0, 4.0]

    # Calcium spike addition on my_model: r1 is multiplied during each spike.
    config.simulation.calcium_spike_start_seconds = 500.0
    config.simulation.calcium_spike_duration_seconds = 20.0
    config.simulation.calcium_spike_period_seconds = 1_000.0
    config.simulation.calcium_spike_count = 20
    config.simulation.calcium_spike_r1_multiplier = 8.0

    # Small defaults keep the pipeline runnable; increase for final estimates.
    config.runs.simple_sanity_trajectories = 1
    config.runs.miller_reproduction_trajectories = 3
    config.runs.my_model_trajectories = 3
    config.runs.caliberation_check_trajectories = 3
    config.runs.modified_autophosphrylation_trajectories = 3
    config.runs.calcium_spike_trajectories = 3
    config.runs.bootstrap_replicates = 100

    config.base_random_seed = 123456
    return config


def main():
    execute_pipeline(build_user_config())


if __name__ == "__main__":
    main()
