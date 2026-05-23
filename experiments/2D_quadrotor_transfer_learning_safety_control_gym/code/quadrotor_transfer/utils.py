
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, TextIO, Tuple, Union

import numpy as np

def safe_float_name(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")

def lambda_to_name(lambda_penalty: float) -> str:
    return f"lambda_{safe_float_name(lambda_penalty)}"

def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def make_run_id(
    *,
    lambda_penalty: float,
    seed: int,
    std_mode: str,
    timestamp: Optional[str] = None,
    prefix: str = "reduced_quadrotor",
    extra: Optional[str] = None,
) -> str:
    pieces = [
        prefix,
        lambda_to_name(lambda_penalty),
        f"std_{std_mode}",
        f"seed_{int(seed)}",
        timestamp or now_timestamp(),
    ]
    if extra:
        pieces.append(str(extra).replace(" ", "_"))
    return "_".join(pieces)

def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

def read_json(path: Union[Path, str]) -> dict:
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def write_json(path: Union[Path, str], payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=json_default)

def append_jsonl(path: Union[Path, str], payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=json_default) + "\n")

class TeeStream:

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

def setup_tee_logging(log_path: Union[Path, str]):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)
    return log_file

def unique_floats(values: Sequence[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        value = float(value)
        if not any(abs(value - old) < 1e-12 for old in out):
            out.append(value)
    return out

def discount_sum(rewards: Iterable[float], gamma: float) -> float:
    rewards_arr = np.asarray(list(rewards), dtype=float).reshape(-1)
    if rewards_arr.size == 0:
        return float("nan")
    powers = float(gamma) ** np.arange(rewards_arr.size, dtype=float)
    return float(np.dot(powers, rewards_arr))

def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr))

def scalar(value) -> float:
    return float(np.asarray(value, dtype=float).reshape(-1)[0])

def as_1d_float_array(values: Union[Sequence[float], float], *, length: int = 2, name: str = "value") -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr[0], length)
    if arr.size != length:
        raise ValueError(f"{name} must be one scalar or {length} values; got {arr}.")
    return arr.astype(float)

def std_to_log_std(std_values: Union[Sequence[float], float], *, length: int = 2, name: str = "std") -> np.ndarray:
    arr = as_1d_float_array(std_values, length=length, name=name)
    if np.any(arr <= 0):
        raise ValueError(f"{name} entries must be positive; got {arr}.")
    return np.log(arr)

def log_std_to_std(log_std_values: Union[Sequence[float], float], *, length: int = 2, name: str = "log_std") -> np.ndarray:
    arr = as_1d_float_array(log_std_values, length=length, name=name)
    return np.exp(arr)

def parse_optional_float_list(text: Optional[str]) -> Optional[list[float]]:
    if text is None or text.strip() == "":
        return None
    return [float(piece.strip()) for piece in text.split(",") if piece.strip()]

def _lambda_name_variants(lambda_penalty: float) -> list[str]:
    lam = float(lambda_penalty)
    variants = {
        lambda_to_name(lam),
        f"lambda_{lam:g}",
        f"lambda_{safe_float_name(lam)}",
    }
    if abs(lam - int(lam)) < 1e-12:
        variants.add(f"lambda_{int(lam)}")
    return sorted(v for v in variants if v)

def _config_matches_lambda(config_path: Path, lambda_penalty: float) -> bool:
    cfg = read_json(config_path)
    if not cfg:
        return False
    for key in ("lambda_penalty", "lambda_pen", "lam", "lambda"):
        if key in cfg:
            try:
                return abs(float(cfg[key]) - float(lambda_penalty)) < 1e-12
            except Exception:
                return False
    nested = cfg.get("config", {})
    if isinstance(nested, dict) and "lambda_penalty" in nested:
        try:
            return abs(float(nested["lambda_penalty"]) - float(lambda_penalty)) < 1e-12
        except Exception:
            return False
    return False

def update_run_index(
    *,
    model_root: Union[Path, str],
    lambda_penalty: float,
    run_id: str,
    run_dir: Union[Path, str],
    final_model_path: Union[Path, str],
    best_model_path: Optional[Union[Path, str]],
    final_vecnormalize_path: Optional[Union[Path, str]],
    best_vecnormalize_path: Optional[Union[Path, str]],
    config_path: Union[Path, str],
    best_metric: Optional[float],
    best_metric_name: str,
) -> dict:
    model_root = Path(model_root)
    lam_name = lambda_to_name(lambda_penalty)
    index_dir = model_root / "by_lambda" / lam_name
    record = {
        "lambda_penalty": float(lambda_penalty),
        "run_id": run_id,
        "run_dir": str(Path(run_dir)),
        "final_model_path": str(Path(final_model_path)),
        "best_model_path": str(Path(best_model_path)) if best_model_path else None,
        "final_vecnormalize_path": str(Path(final_vecnormalize_path)) if final_vecnormalize_path else None,
        "best_vecnormalize_path": str(Path(best_vecnormalize_path)) if best_vecnormalize_path else None,
        "config_path": str(Path(config_path)),
        "best_metric": None if best_metric is None else float(best_metric),
        "best_metric_name": best_metric_name,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(index_dir / "latest.json", record)
    append_jsonl(index_dir / "history.jsonl", record)
    return record

def _vecnormalize_for_model(model_path: Path, selector: str = "best") -> Optional[Path]:
    names: list[str]
    if selector == "best":
        names = ["best_vecnormalize.pkl", "vecnormalize.pkl", "final_vecnormalize.pkl"]
    elif selector == "final":
        names = ["vecnormalize.pkl", "final_vecnormalize.pkl", "best_vecnormalize.pkl"]
    else:
        names = ["vecnormalize.pkl", "final_vecnormalize.pkl", "best_vecnormalize.pkl"]
    for name in names:
        path = model_path.parent / name
        if path.exists():
            return path
    return None

def _candidate_model_from_record(record: dict, selector: str) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    if not record:
        return None, None, None
    selector = selector.lower()
    if selector == "best":
        model_keys = ["best_model_path", "final_model_path"]
        vec_keys = ["best_vecnormalize_path", "final_vecnormalize_path"]
    elif selector == "final":
        model_keys = ["final_model_path", "best_model_path"]
        vec_keys = ["final_vecnormalize_path", "best_vecnormalize_path"]
    else:
        model_keys = ["final_model_path", "best_model_path"]
        vec_keys = ["final_vecnormalize_path", "best_vecnormalize_path"]

    model_path = None
    for key in model_keys:
        raw = record.get(key)
        if raw and Path(raw).exists():
            model_path = Path(raw)
            break
    if model_path is None:
        return None, None, None

    vec_path = None
    for key in vec_keys:
        raw = record.get(key)
        if raw and Path(raw).exists():
            vec_path = Path(raw)
            break
    config_path = Path(record["config_path"]) if record.get("config_path") and Path(record["config_path"]).exists() else None
    return model_path, vec_path, config_path

def find_policy_artifacts(
    model_root: Union[Path, str],
    lambda_penalty: float,
    explicit_model: Optional[Union[Path, str]] = None,
    explicit_vecnormalize: Optional[Union[Path, str]] = None,
    selector: str = "best",
) -> tuple[Path, Optional[Path], Optional[Path]]:
    selector = selector.lower()
    if selector not in {"best", "final", "latest"}:
        raise ValueError("selector must be one of: best, final, latest")

    model_root = Path(model_root)
    if explicit_model is not None:
        model_path = Path(explicit_model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        vec_path = Path(explicit_vecnormalize) if explicit_vecnormalize else _vecnormalize_for_model(model_path, selector)
        if vec_path is not None and not vec_path.exists():
            raise FileNotFoundError(f"VecNormalize path does not exist: {vec_path}")
        config_path = model_path.parent / "config.json"
        return model_path, vec_path, config_path if config_path.exists() else None

    if not model_root.exists():
        raise FileNotFoundError(f"Model root does not exist: {model_root}")

    lam_name = lambda_to_name(lambda_penalty)
    latest_record_path = model_root / "by_lambda" / lam_name / "latest.json"
    model_path, vec_path, config_path = _candidate_model_from_record(read_json(latest_record_path), selector)
    if model_path is not None:
        return model_path, vec_path, config_path

    variants = _lambda_name_variants(lambda_penalty)
    candidates: list[Path] = []

    run_roots = [model_root / "runs", model_root]
    file_preferences = {
        "best": ["best_model.zip", "final_model.zip", "model.zip", "policy.zip"],
        "final": ["final_model.zip", "model.zip", "policy.zip", "best_model.zip"],
        "latest": ["final_model.zip", "model.zip", "policy.zip", "best_model.zip"],
    }[selector]
    for run_root in run_roots:
        if not run_root.exists():
            continue
        for run_dir in run_root.iterdir():
            if not run_dir.is_dir():
                continue
            name_hit = any(v in run_dir.name for v in variants)
            cfg_hit = _config_matches_lambda(run_dir / "config.json", lambda_penalty)
            if not (name_hit or cfg_hit):
                continue
            for filename in file_preferences:
                path = run_dir / filename
                if path.exists():
                    candidates.append(path)
                    break

    for variant in variants:
        for filename in file_preferences:
            path = model_root / variant / filename
            if path.exists():
                candidates.append(path)

    for variant in variants:
        for filename in (f"{variant}_policy.zip", f"{variant}_model.zip", f"{variant}.zip"):
            path = model_root / filename
            if path.exists():
                candidates.append(path)

    if not candidates:
        for path in model_root.rglob("*.zip"):
            if "checkpoint" in {p.lower() for p in path.parts}:
                continue
            parent = path.parent
            name_hit = any(v in parent.name or v in path.name for v in variants)
            cfg_hit = _config_matches_lambda(parent / "config.json", lambda_penalty)
            if name_hit or cfg_hit:
                if path.name in set(file_preferences) or "policy" in path.name:
                    candidates.append(path)

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(candidate)

    if not unique_candidates:
        raise FileNotFoundError(
            f"Could not find a {selector} PPO model for lambda={float(lambda_penalty):g} under {model_root}. "
            "Train it first or pass --model explicitly."
        )

    model_path = max(unique_candidates, key=lambda p: p.stat().st_mtime)
    vec_path = Path(explicit_vecnormalize) if explicit_vecnormalize else _vecnormalize_for_model(model_path, selector)
    config_candidates = [model_path.parent / "config.json"] + list(model_path.parent.glob("*_config.json"))
    config_path = next((p for p in config_candidates if p.exists()), None)
    return model_path, vec_path, config_path

def describe_action_std(std_normalized: Union[Sequence[float], float], theta_max_rad: float, delta_thrust_max: float) -> dict:
    std = as_1d_float_array(std_normalized, length=2, name="std_normalized")
    return {
        "std_norm_delta_thrust_action": float(std[0]),
        "std_norm_theta_action": float(std[1]),
        "std_delta_thrust_N": float(std[0] * float(delta_thrust_max)),
        "std_theta_star_rad": float(std[1] * float(theta_max_rad)),
        "std_theta_star_deg": float(np.rad2deg(std[1] * float(theta_max_rad))),
    }
