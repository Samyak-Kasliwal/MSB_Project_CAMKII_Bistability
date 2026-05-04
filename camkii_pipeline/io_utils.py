from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .config import PipelineConfig


@dataclass
class ProjectLayout:
    project_root: Path
    results_root: Path
    checkpoints_root: Path


def create_project_layout(config: PipelineConfig) -> ProjectLayout:
    project_root = config.output.project_root.resolve()
    results_root = project_root / config.output.results_dir_name
    checkpoints_root = project_root / config.output.checkpoints_dir_name
    results_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    return ProjectLayout(project_root=project_root, results_root=results_root, checkpoints_root=checkpoints_root)


def ensure_run_directories(layout: ProjectLayout, run_id: str) -> Dict[str, Path]:
    run_results = layout.results_root / run_id
    run_checkpoints = layout.checkpoints_root / run_id
    plots = run_results / "plots"
    raw = run_results / "raw_trajectories"
    tables = run_results / "tables"
    debug = run_results / "debug"
    for path in (run_results, run_checkpoints, plots, raw, tables, debug):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "results": run_results,
        "checkpoints": run_checkpoints,
        "plots": plots,
        "raw": raw,
        "tables": tables,
        "debug": debug,
    }


def write_json(path: Path, payload: Dict[str, Any], indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_dataframe_csv(path: Path, dataframe: pd.DataFrame, float_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False, float_format=float_format)


def append_dataframe_csv(path: Path, dataframe: pd.DataFrame, float_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    dataframe.to_csv(path, mode="a", index=False, header=write_header, float_format=float_format)


def load_dataframe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_pipeline_state(layout: ProjectLayout) -> Dict[str, Any]:
    state_path = layout.checkpoints_root / "pipeline_state.json"
    state = read_json(state_path)
    if state is None:
        return {"state": {"completed_runs": {}, "locked_parameters": {}, "artifacts": {}}}
    if "state" not in state:
        return {"state": state}
    return state


def save_pipeline_state(layout: ProjectLayout, state: Dict[str, Any], config: PipelineConfig) -> None:
    state_path = layout.checkpoints_root / "pipeline_state.json"
    payload = {
        "config_snapshot": config.to_serializable_dict(),
        "state": state,
    }
    write_json(state_path, payload, indent=config.output.json_indent)


def unpack_pipeline_state(layout: ProjectLayout) -> Dict[str, Any]:
    payload = read_json(layout.checkpoints_root / "pipeline_state.json")
    if payload is None:
        return {"completed_runs": {}, "locked_parameters": {}, "artifacts": {}}
    return payload.get("state", {"completed_runs": {}, "locked_parameters": {}, "artifacts": {}})


def format_number_for_name(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.8g}"
        text = text.replace("-", "m").replace(".", "p")
        return text
    return str(value).replace(" ", "_")


def build_condition_id(parts: Dict[str, Any]) -> str:
    labels = []
    for key, value in parts.items():
        labels.append(f"{key}-{format_number_for_name(value)}")
    return "__".join(labels)


def stable_int_from_text(text: str, modulus: int = 2_000_000_000) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulus
