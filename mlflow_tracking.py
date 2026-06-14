"""Shared MLflow helpers for notebook-based fine-tuning experiments."""

from __future__ import annotations

import csv
import json
import math
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

try:
    import mlflow
except ImportError:  # pragma: no cover - exercised indirectly in tests
    mlflow = None


DEFAULT_EXPERIMENT_NAME = "rft-learning"
DEFAULT_TRACKING_DB_FILENAME = "mlflow.db"
DEFAULT_ARTIFACT_DIRNAME = "mlartifacts"


@dataclass(frozen=True, slots=True)
class TrackingSetup:
    """Resolved MLflow tracking configuration."""

    tracking_uri: str
    experiment_name: str
    tracking_path: Path
    artifact_path: Path | None


def _require_mlflow() -> Any:
    """Return the imported mlflow module or raise a helpful error."""
    if mlflow is None:
        raise RuntimeError(
            "mlflow is not installed. Run `uv sync` to install project dependencies before "
            "using notebook tracking."
        )
    return mlflow


def project_root() -> Path:
    """Return the repository root inferred from this module location."""
    return Path(__file__).resolve().parent


def default_tracking_path() -> Path:
    """Return the default local MLflow tracking database path."""
    return project_root() / DEFAULT_TRACKING_DB_FILENAME


def default_artifact_path() -> Path:
    """Return the default local MLflow artifact directory."""
    return project_root() / DEFAULT_ARTIFACT_DIRNAME


def _sqlite_uri(path: Path) -> str:
    """Return an absolute SQLite tracking URI for a local database path."""
    return f"sqlite:///{path.resolve().as_posix()}"


def _ensure_experiment(
    mlflow_module: Any,
    experiment_name: str,
    artifact_location: str | None = None,
) -> None:
    """Create the experiment with an artifact location if needed, then activate it."""
    client = mlflow_module.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )
    mlflow_module.set_experiment(experiment_name)


def initialize_tracking(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
) -> TrackingSetup:
    """Configure MLflow to use a local SQLite store unless overridden."""
    mlflow_module = _require_mlflow()
    resolved_path = default_tracking_path()
    resolved_artifact_path: Path | None = None

    if tracking_uri is None:
        resolved_uri = _sqlite_uri(resolved_path)
        resolved_artifact_path = default_artifact_path()
        resolved_artifact_path.mkdir(parents=True, exist_ok=True)
        artifact_location = artifact_location or resolved_artifact_path.as_uri()
    else:
        resolved_uri = tracking_uri
        if artifact_location is not None and artifact_location.startswith("file://"):
            resolved_artifact_path = Path(artifact_location.removeprefix("file://"))

    mlflow_module.set_tracking_uri(resolved_uri)
    _ensure_experiment(
        mlflow_module,
        experiment_name=experiment_name,
        artifact_location=artifact_location,
    )
    return TrackingSetup(
        tracking_uri=resolved_uri,
        experiment_name=experiment_name,
        tracking_path=resolved_path,
        artifact_path=resolved_artifact_path,
    )


def serialize_config(value: Any) -> Any:
    """Convert configs, paths, and containers into JSON-serializable values."""
    if isinstance(value, Enum):
        return serialize_config(value.value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if is_dataclass(value):
        return serialize_config(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serialize_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_config(item) for item in value]
    if isinstance(value, set):
        return sorted(serialize_config(item) for item in value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return serialize_config(value.to_dict())
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return serialize_config(vars(value))
    return value


def _stringify_param(value: Any) -> str:
    """Convert a config value into a compact MLflow param string."""
    serialized = serialize_config(value)
    if serialized is None:
        return "null"
    if isinstance(serialized, bool):
        return "true" if serialized else "false"
    if isinstance(serialized, (int, float, str)):
        return str(serialized)
    return json.dumps(serialized, sort_keys=True)


def flatten_params(values: dict[str, Any], prefix: str | None = None) -> dict[str, str]:
    """Flatten nested config values into MLflow-friendly param strings."""
    flattened: dict[str, str] = {}

    def visit(current_prefix: str, current_value: Any) -> None:
        serialized = serialize_config(current_value)
        if isinstance(serialized, dict):
            for key, item in serialized.items():
                next_prefix = f"{current_prefix}.{key}" if current_prefix else str(key)
                visit(next_prefix, item)
            return
        flattened[current_prefix] = _stringify_param(serialized)

    for key, value in values.items():
        start_prefix = f"{prefix}.{key}" if prefix else str(key)
        visit(start_prefix, value)

    return flattened


def _is_metric_value(value: Any) -> bool:
    """Return whether a value can be logged as a numeric MLflow metric."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def metrics_only(values: dict[str, Any], prefix: str | None = None) -> dict[str, float]:
    """Filter a dict down to finite numeric MLflow metrics."""
    metrics: dict[str, float] = {}
    for key, value in values.items():
        if _is_metric_value(value):
            metric_key = f"{prefix}.{key}" if prefix else key
            metrics[metric_key] = float(value)
    return metrics


@contextmanager
def start_parent_run(
    notebook_name: str,
    notebook_type: str,
    run_name: str | None = None,
    tags: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Start a parent MLflow run for one notebook execution."""
    mlflow_module = _require_mlflow()
    combined_tags = {
        "run_level": "parent",
        "notebook_name": notebook_name,
        "notebook_type": notebook_type,
    }
    if tags:
        combined_tags.update({key: str(value) for key, value in tags.items()})
    with mlflow_module.start_run(run_name=run_name or notebook_name, tags=combined_tags) as run:
        yield run


@contextmanager
def start_child_run(phase: str, tags: dict[str, Any] | None = None) -> Iterator[Any]:
    """Start a nested child run for one notebook phase."""
    mlflow_module = _require_mlflow()
    combined_tags = {"run_level": "child", "phase": phase}
    if tags:
        combined_tags.update({key: str(value) for key, value in tags.items()})
    with mlflow_module.start_run(
        run_name=phase,
        nested=True,
        tags=combined_tags,
    ) as run:
        yield run


def log_params(values: dict[str, Any], prefix: str | None = None) -> dict[str, str]:
    """Flatten and log a config dict as MLflow params."""
    mlflow_module = _require_mlflow()
    flattened = flatten_params(values, prefix=prefix)
    if flattened:
        mlflow_module.log_params(flattened)
    return flattened


def log_metrics(values: dict[str, Any], prefix: str | None = None, step: int | None = None) -> dict[str, float]:
    """Filter and log numeric metrics."""
    mlflow_module = _require_mlflow()
    metrics = metrics_only(values, prefix=prefix)
    for key, value in metrics.items():
        if step is None:
            mlflow_module.log_metric(key, value)
        else:
            mlflow_module.log_metric(key, value, step=step)
    return metrics


def log_json_artifact(filename: str, payload: Any, artifact_path: str | None = None) -> None:
    """Write a JSON artifact to a temporary file and log it to MLflow."""
    mlflow_module = _require_mlflow()
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_file = Path(temp_dir) / filename
        artifact_file.write_text(
            json.dumps(serialize_config(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        mlflow_module.log_artifact(str(artifact_file), artifact_path=artifact_path)


def log_rows_artifact(filename: str, rows: list[dict[str, Any]], artifact_path: str | None = None) -> None:
    """Log row-based data as both JSON and CSV when rows are available."""
    if not rows:
        return

    base_name = filename.rsplit(".", 1)[0]
    log_json_artifact(f"{base_name}.json", rows, artifact_path=artifact_path)

    mlflow_module = _require_mlflow()
    serialized_rows = [serialize_config(row) for row in rows]
    fieldnames = sorted({key for row in serialized_rows for key in row.keys()})
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_file = Path(temp_dir) / f"{base_name}.csv"
        with artifact_file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in serialized_rows:
                writer.writerow(row)
        mlflow_module.log_artifact(str(artifact_file), artifact_path=artifact_path)


def log_eval_result(
    phase: str,
    result: dict[str, Any],
    artifact_path: str = "evaluations",
    parent_metric_prefix: str | None = None,
) -> None:
    """Log one evaluation result into the active run and optionally to the parent run."""
    metrics = result.get("metrics", {})
    settings = result.get("settings", {})
    samples = result.get("samples", [])
    rows = result.get("rows", [])
    label = result.get("label", phase)

    log_params({"label": label, "settings": settings}, prefix=f"eval.{phase}")
    log_metrics(metrics, prefix=phase)
    log_json_artifact(f"{phase}_summary.json", result, artifact_path=artifact_path)
    log_rows_artifact(f"{phase}_samples.json", samples, artifact_path=artifact_path)
    log_rows_artifact(f"{phase}_rows.json", rows, artifact_path=artifact_path)

    if parent_metric_prefix:
        log_metrics(metrics, prefix=parent_metric_prefix)


def log_history(
    history: list[dict[str, Any]],
    artifact_stem: str,
    metric_prefix: str | None = None,
) -> None:
    """Log trainer history rows and any numeric values they contain."""
    if not history:
        return

    log_rows_artifact(f"{artifact_stem}.json", history, artifact_path="training_history")

    if metric_prefix is None:
        return

    for row in history:
        step_value = row.get("step")
        step = int(step_value) if isinstance(step_value, (int, float)) else None
        numeric_values = {
            key: value
            for key, value in row.items()
            if key != "step" and _is_metric_value(value)
        }
        if numeric_values:
            log_metrics(numeric_values, prefix=metric_prefix, step=step)
