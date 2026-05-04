from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import GAMMA_LINESTYLES, MODEL_STYLES
from .statistics import fixed_slope_intercept, kaplan_meier_curve


def _style_for_model(model_label: str) -> Dict[str, str]:
    return MODEL_STYLES.get(model_label, {"color": "black", "linestyle": "-", "label": model_label})


def _gamma_linestyle(gamma_value: float) -> str:
    for known_gamma, style in GAMMA_LINESTYLES.items():
        if abs(gamma_value - known_gamma) < 1.0e-12:
            return style
    return "-"


def _prepare_axes(title: str, xlabel: str, ylabel: str) -> tuple[plt.Figure, plt.Axes]:
    figure, axis = plt.subplots(figsize=(8.5, 6.0))
    axis.set_title(title, fontsize=12)
    axis.set_xlabel(xlabel, fontsize=11)
    axis.set_ylabel(ylabel, fontsize=11)
    axis.tick_params(labelsize=11)
    axis.grid(True, alpha=0.25)
    return figure, axis


def _save_figure(figure: plt.Figure, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_figure_1(
    summary_df: pd.DataFrame,
    fit_row: pd.Series,
    analytical_slope: float | None,
    output_path: Path,
    dpi: int,
) -> None:
    figure, axis = _prepare_axes("Figure 1: MZ-full Replication", "M", "ln lifetime")
    finite = (
        np.isfinite(summary_df["reported_lifetime"])
        & (summary_df["reported_lifetime"] > 0.0)
        & np.isfinite(summary_df["reported_lifetime_lower"])
        & np.isfinite(summary_df["reported_lifetime_upper"])
        & (summary_df["reported_lifetime_lower"] > 0.0)
        & (summary_df["reported_lifetime_upper"] > 0.0)
    )
    filtered = summary_df.loc[finite].copy()
    if filtered.empty:
        axis.text(0.5, 0.5, "No finite lifetime estimates available.", ha="center", va="center", transform=axis.transAxes)
        _save_figure(figure, output_path, dpi)
        return
    y_values = np.log(filtered["reported_lifetime"].to_numpy())
    y_lower = np.log(filtered["reported_lifetime_lower"].to_numpy())
    y_upper = np.log(filtered["reported_lifetime_upper"].to_numpy())
    axis.errorbar(
        filtered["M"],
        y_values,
        yerr=[y_values - y_lower, y_upper - y_values],
        fmt="o",
        color="black",
        label="MZ-full simulation",
    )

    x_line = np.linspace(filtered["M"].min(), filtered["M"].max(), 100)
    fit_y = fit_row["slope"] * x_line + fit_row["intercept"]
    axis.plot(x_line, fit_y, linestyle="--", color="black", label="Linear fit")

    if analytical_slope is not None and np.isfinite(analytical_slope):
        anchor_intercept = fit_row.get("analytic_overlay_intercept", np.nan)
        if np.isfinite(anchor_intercept):
            axis.plot(
                x_line,
                analytical_slope * x_line + anchor_intercept,
                linestyle=":",
                color="black",
                label="Analytical slope",
            )

    axis.legend(fontsize=11)
    _save_figure(figure, output_path, dpi)


def plot_log_lifetime_vs_m(
    summary_df: pd.DataFrame,
    output_path: Path,
    dpi: int,
    title: str,
    filter_gamma: float | None = None,
    reference_df: pd.DataFrame | None = None,
    reference_label: str = "Reference",
    analytic_slope: float | None = None,
    analytic_label: str = "Analytical slope",
) -> None:
    figure, axis = _prepare_axes(title, "M", "ln lifetime")

    plot_df = summary_df.copy()
    if filter_gamma is not None and "gamma" in plot_df.columns:
        plot_df = plot_df[np.isclose(plot_df["gamma"], filter_gamma)]
    if plot_df.empty:
        axis.text(0.5, 0.5, "No matching conditions found.", ha="center", va="center", transform=axis.transAxes)
        _save_figure(figure, output_path, dpi)
        return

    for model_label, model_group in plot_df.groupby("model"):
        style = _style_for_model(model_label)
        ordered = model_group.sort_values("M")
        ordered = ordered[
            np.isfinite(ordered["reported_lifetime"])
            & (ordered["reported_lifetime"] > 0.0)
            & np.isfinite(ordered["reported_lifetime_lower"])
            & np.isfinite(ordered["reported_lifetime_upper"])
            & (ordered["reported_lifetime_lower"] > 0.0)
            & (ordered["reported_lifetime_upper"] > 0.0)
            & (ordered["delta_subunit"] > 0.0)
        ]
        if ordered.empty:
            continue
        y_values = np.log(ordered["reported_lifetime"].to_numpy())
        y_lower = np.log(ordered["reported_lifetime_lower"].to_numpy())
        y_upper = np.log(ordered["reported_lifetime_upper"].to_numpy())
        linestyle = style["linestyle"]
        if filter_gamma is None and "gamma" in ordered.columns and len(np.unique(ordered["gamma"])) == 1:
            linestyle = _gamma_linestyle(float(ordered["gamma"].iloc[0]))
        axis.errorbar(
            ordered["M"],
            y_values,
            yerr=[y_values - y_lower, y_upper - y_values],
            fmt="o-",
            color=style["color"],
            linestyle=linestyle,
            label=style["label"],
        )

    if reference_df is not None and not reference_df.empty:
        ordered = reference_df.sort_values("M")
        ordered = ordered[
            np.isfinite(ordered["reported_lifetime"])
            & (ordered["reported_lifetime"] > 0.0)
        ]
        if not ordered.empty:
            axis.plot(
                ordered["M"],
                np.log(ordered["reported_lifetime"]),
                linestyle="--",
                color="black",
                linewidth=1.8,
                label=reference_label,
            )
            if analytic_slope is not None and np.isfinite(analytic_slope):
                analytic_intercept = fixed_slope_intercept(
                    ordered["M"].to_numpy(dtype=float),
                    ordered["reported_lifetime"].to_numpy(dtype=float),
                    analytic_slope,
                )
                x_line = np.linspace(ordered["M"].min(), ordered["M"].max(), 100)
                axis.plot(
                    x_line,
                    analytic_slope * x_line + analytic_intercept,
                    linestyle=":",
                    color="black",
                    linewidth=1.5,
                    label=analytic_label,
                )

    axis.legend(fontsize=11)
    _save_figure(figure, output_path, dpi)


def plot_slope_vs_gamma(
    slope_df: pd.DataFrame,
    output_path: Path,
    dpi: int,
    title: str,
    reference_slope: float | None = None,
    model_order: Iterable[str] | None = None,
) -> None:
    figure, axis = _prepare_axes(title, "gamma (s^-1)", "slope k")
    axis.set_xscale("log")

    order = list(model_order) if model_order is not None else sorted(slope_df["model"].unique().tolist())
    for model_label in order:
        model_group = slope_df[slope_df["model"] == model_label]
        if model_group.empty:
            continue
        style = _style_for_model(model_label)
        ordered = model_group.sort_values("gamma")
        ordered = ordered[
            np.isfinite(ordered["gamma"])
            & np.isfinite(ordered["slope"])
            & np.isfinite(ordered["slope_lower"])
            & np.isfinite(ordered["slope_upper"])
        ]
        if ordered.empty:
            continue
        lower = np.clip(ordered["slope"] - ordered["slope_lower"], a_min=0.0, a_max=None)
        upper = np.clip(ordered["slope_upper"] - ordered["slope"], a_min=0.0, a_max=None)
        axis.errorbar(
            ordered["gamma"],
            ordered["slope"],
            yerr=[lower, upper],
            fmt="o-",
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
        )

    if reference_slope is not None and np.isfinite(reference_slope):
        axis.axhline(reference_slope, linestyle="--", color="black", label="k_Miller")

    axis.legend(fontsize=11)
    _save_figure(figure, output_path, dpi)


def plot_turnover_erosion(summary_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    figure, axis = _prepare_axes("Figure 3: Turnover Erosion", "delta (s^-1)", "ln lifetime")
    axis.set_xscale("log")
    for m_value, group in summary_df.groupby("M"):
        ordered = group.sort_values("delta_subunit")
        ordered = ordered[
            np.isfinite(ordered["reported_lifetime"])
            & (ordered["reported_lifetime"] > 0.0)
            & np.isfinite(ordered["reported_lifetime_lower"])
            & np.isfinite(ordered["reported_lifetime_upper"])
            & (ordered["reported_lifetime_lower"] > 0.0)
            & (ordered["reported_lifetime_upper"] > 0.0)
        ]
        if ordered.empty:
            continue
        y = np.log(ordered["reported_lifetime"])
        y_lower = np.log(ordered["reported_lifetime_lower"])
        y_upper = np.log(ordered["reported_lifetime_upper"])
        axis.plot(ordered["delta_subunit"], y, marker="o", label=f"M={m_value}")
        axis.fill_between(ordered["delta_subunit"], y_lower, y_upper, alpha=0.15)

    axis.axvline(1.0e-4, color="black", linestyle="--", linewidth=1.0, label="2 h half-life")
    axis.axvline(9.26e-6, color="gray", linestyle=":", linewidth=1.0, label="Miller holo rate")
    axis.axhline(np.log(3600.0), color="black", linestyle="-.", linewidth=1.0, label="1 hour lifetime")
    axis.legend(fontsize=10)
    _save_figure(figure, output_path, dpi)


def plot_phase_diagram(summary_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    figure, axis = _prepare_axes("Figure 4: Phase Diagram", "gamma (s^-1)", "M")
    pivot = summary_df.pivot(index="M", columns="gamma", values="reported_lifetime").sort_index().sort_index(axis=1)
    x_values = pivot.columns.to_numpy(dtype=float)
    y_values = pivot.index.to_numpy(dtype=float)
    lifetime_grid = pivot.to_numpy(dtype=float)
    z_values = np.where(np.isfinite(lifetime_grid) & (lifetime_grid > 0.0), np.log10(lifetime_grid), np.nan)
    masked = np.ma.masked_invalid(z_values)

    image = axis.imshow(
        masked,
        aspect="auto",
        origin="lower",
        extent=[x_values.min(), x_values.max(), y_values.min(), y_values.max()],
        cmap="viridis",
    )
    contour_levels = np.log10([3600.0, 86400.0, 3.15e7])
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    if np.any(np.isfinite(z_values)):
        contour = axis.contour(grid_x, grid_y, masked, levels=contour_levels, colors="white", linewidths=1.0)
        axis.clabel(contour, inline=True, fontsize=9, fmt=lambda level: f"{10 ** level:.2e} s")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("log10 lifetime", fontsize=11)
    axis.set_xscale("linear")
    axis.set_xticks(x_values)
    axis.set_yticks(y_values)
    _save_figure(figure, output_path, dpi)


def plot_quasi_potential(quasi_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    figure, axis = _prepare_axes("Figure S1: Quasi-potential", "n", "phi(n)")
    axis.plot(quasi_df["n"], quasi_df["phi"], marker="o", color="black")
    barrier_row = quasi_df.loc[quasi_df["is_barrier"] == 1]
    up_row = quasi_df.loc[quasi_df["is_up_attractor"] == 1]
    if not barrier_row.empty:
        axis.scatter(barrier_row["n"], barrier_row["phi"], color="red", label="Barrier")
    if not up_row.empty:
        axis.scatter(up_row["n"], up_row["phi"], color="blue", label="UP attractor")
    axis.legend(fontsize=11)
    _save_figure(figure, output_path, dpi)


def plot_survival_curve(times: np.ndarray, events: np.ndarray, output_path: Path, dpi: int, title: str) -> None:
    figure, axis = _prepare_axes(title, "time (s)", "Survival S(t)")
    curve_times, curve_survival = kaplan_meier_curve(times, events)
    axis.step(curve_times, curve_survival, where="post", color="black")
    axis.set_yscale("log")
    _save_figure(figure, output_path, dpi)


def plot_sanity_traces(trace_frames: List[pd.DataFrame], output_path: Path, dpi: int) -> None:
    figure, axis = _prepare_axes("Sanity Check: Exchange-B Trajectories", "time (s)", "n_tot")
    for index, frame in enumerate(trace_frames, start=1):
        axis.plot(frame["time"], frame["n_tot"], linewidth=1.0, label=f"Trajectory {index}")
    axis.legend(fontsize=9, ncol=1)
    _save_figure(figure, output_path, dpi)


def plot_ca_comparison_panel_c(
    comparison_df: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    figure, axis = _prepare_axes("Figure 6C: Lifetime Ratio", "M", "tau_CaNoise / tau_FixedCa")
    for gamma_value, group in comparison_df.groupby("gamma"):
        ordered = group.sort_values("M")
        ordered = ordered[np.isfinite(ordered["ratio"])]
        if ordered.empty:
            continue
        axis.plot(
            ordered["M"],
            ordered["ratio"],
            marker="o",
            linestyle=_gamma_linestyle(float(gamma_value)),
            label=f"gamma={gamma_value}",
        )
    axis.legend(fontsize=11)
    _save_figure(figure, output_path, dpi)


def plot_gamma_m_coupling(
    summary_df: pd.DataFrame,
    output_path: Path,
    dpi: int,
    title: str,
) -> None:
    figure, axis = _prepare_axes(title, "M", "ln lifetime")
    for gamma_value, group in summary_df.groupby("gamma"):
        ordered = group.sort_values("M")
        ordered = ordered[np.isfinite(ordered["reported_lifetime"]) & (ordered["reported_lifetime"] > 0.0)]
        if ordered.empty:
            continue
        axis.plot(
            ordered["M"],
            np.log(ordered["reported_lifetime"]),
            marker="o",
            linestyle=_gamma_linestyle(float(gamma_value)),
            label=f"gamma={gamma_value:g}",
        )
    axis.legend(fontsize=10)
    _save_figure(figure, output_path, dpi)


def plot_delta_fixed_points(
    delta_scan_df: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    figure, axis = _prepare_axes("Zabotinsky Reproduction: Bistability vs Delta", "delta (s^-1)", "fixed point n")
    axis.set_xscale("log")
    if "upper_fixed_point" in delta_scan_df.columns:
        axis.plot(delta_scan_df["delta"], delta_scan_df["upper_fixed_point"], color="#2563EB", label="UP fixed point")
    if "lower_fixed_point" in delta_scan_df.columns:
        axis.plot(delta_scan_df["delta"], delta_scan_df["lower_fixed_point"], color="#DC2626", label="DOWN fixed point")
    axis.legend(fontsize=10)
    _save_figure(figure, output_path, dpi)


def plot_parameter_sweep(
    summary_df: pd.DataFrame,
    output_path: Path,
    dpi: int,
    title: str,
) -> None:
    figure, axis = _prepare_axes(title, "parameter value", "reported lifetime (s)")
    finite_x = summary_df["sweep_value"].to_numpy(dtype=float)
    finite_x = finite_x[np.isfinite(finite_x)]
    if finite_x.size > 0 and np.all(finite_x > 0.0):
        axis.set_xscale("log")
    axis.set_yscale("log")
    for parameter_name, group in summary_df.groupby("sweep_parameter"):
        ordered = group.sort_values("sweep_value")
        ordered = ordered[np.isfinite(ordered["reported_lifetime"]) & (ordered["reported_lifetime"] > 0.0)]
        if ordered.empty:
            continue
        axis.plot(ordered["sweep_value"], ordered["reported_lifetime"], marker="o", label=str(parameter_name))
    axis.legend(fontsize=10)
    _save_figure(figure, output_path, dpi)
