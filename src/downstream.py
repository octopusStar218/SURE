#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn import linear_model
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dataset import CityDataset, infer_feature_dims, load_county_ids_file, parse_county_ids
from model import UrbanMotifModel
from train_utils import autocast_context, load_checkpoint
from utils import load_yaml, setup_torch_performance


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH_TASK_ROOT = str(PROJECT_ROOT / "data/raw/downstream/task_npy_466county")
DEFAULT_LANDUSE_15CLASS_ROOT = str(PROJECT_ROOT / "data/raw/downstream/landuse_clustering_npy")
DEFAULT_BENCH_OUT_DIR = "outputs/sure_downstream"
PROBE_NAME = "linear"
ENCODER_ARG_DEFAULTS = {
    "num_motifs": 64,
    "dim": 256,
    "dropout": 0.05,
    "smooth_steps": 2,
    "temperature": 0.15,
    "view_smooth_type": "learnable_diffuse",
    "tract_context_type": "gcn",
    "tract_context_position": "post_residual",
    "tract_context_graph": "spatial",
    "tract_context_layers": 2,
    "residual_fusion": "attn_gated",
    "motif_consensus_type": "shared",
    "profile_residual_decomp": "on",
    "enabled_views": "all",
}


def build_encoder(
    *,
    processed_root: str,
    checkpoint: str,
    dim: int,
    num_motifs: int,
    smooth_steps: int,
    temperature: float,
    view_smooth_type: str,
    tract_context_type: str,
    tract_context_position: str,
    tract_context_graph: str,
    tract_context_layers: int,
    dropout: float,
    device: torch.device | str,
    residual_fusion: str = "attn_gated",
    motif_consensus_type: str = "shared",
    profile_residual_decomp: str = "on",
    enabled_views: str = "all",
) -> UrbanMotifModel:
    poi_dim, lu_dim, source_dim, destination_dim = infer_feature_dims(processed_root)
    model = UrbanMotifModel(
        poi_dim=poi_dim,
        lu_dim=lu_dim,
        source_dim=source_dim,
        destination_dim=destination_dim,
        dim=dim,
        num_motifs=num_motifs,
        smooth_steps=smooth_steps,
        temperature=temperature,
        view_smooth_type=view_smooth_type,
        tract_context_type=tract_context_type,
        tract_context_position=tract_context_position,
        tract_context_graph=tract_context_graph,
        tract_context_layers=tract_context_layers,
        residual_fusion=residual_fusion,
        motif_consensus_type=motif_consensus_type,
        profile_residual_decomp=profile_residual_decomp,
        enabled_views=enabled_views,
        dropout=dropout,
    ).to(device)
    load_checkpoint(checkpoint, model)
    return model


def resolve_encoder_args(args: argparse.Namespace) -> Dict[str, object]:
    cfg = {}
    if args.checkpoint:
        cfg_path = Path(args.checkpoint).resolve().parent / "args.yaml"
        if cfg_path.exists():
            cfg = load_yaml(str(cfg_path))
    return {
        key: type(default)(cfg.get(key, default) if getattr(args, key, None) is None else getattr(args, key))
        for key, default in ENCODER_ARG_DEFAULTS.items()
    }


@dataclass(frozen=True)
class TaskSpec:
    kind: str
    n_classes: Optional[int] = None
    n_labels: Optional[int] = None
    y_transform: str = "identity"
    clip: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class RegProbeConfig:
    small_n: int
    small_alpha: float
    medium_alpha: float


TASK_SPECS: Dict[str, TaskSpec] = {
    "population": TaskSpec(kind="reg", y_transform="log1p", clip=(0.0, float("inf"))),
    "employment": TaskSpec(kind="bin"),
    "safety": TaskSpec(kind="mc", n_classes=13),
    "income": TaskSpec(kind="mc", n_classes=7),
    "co2": TaskSpec(kind="ml", n_labels=5),
    "landcover": TaskSpec(kind="mc", n_classes=6),
}

TASK_ORDER: List[str] = [
    "population",
    "employment",
    "safety",
    "income",
    "co2",
    "landcover",
]

SUMMARY_COLUMNS = [
    "embedding_model",
    "task",
    "probe",
    "metric",
    "county_count",
    "total_samples",
    "macro_mean",
    "macro_std",
    "weighted_mean_by_samples",
]


def normalize_geoid11(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values).astype(str)
    return np.char.zfill(np.char.strip(arr), 11).astype(str)


def safe_f1(y_true: np.ndarray, y_pred: np.ndarray, *, average: str) -> float:
    try:
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))
    except TypeError:
        return float(f1_score(y_true, y_pred, average=average))


def nanstd(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.std(arr))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)))


def metric_stats(
    overall: Dict[str, float],
    fold: Dict[str, List[float]],
    keys: Sequence[str],
) -> Dict[str, Tuple[float, float]]:
    return {
        key: (float(overall[key]), nanstd(np.asarray(fold.get(key, []), dtype=float)))
        for key in keys
        if key in overall
    }


def clean_metric_dict(
    metrics: Dict[str, Tuple[float, float]],
    *,
    skip_non_finite_metrics: bool,
) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for key, (mean, std) in metrics.items():
        mean = float(mean)
        std = float(std)
        if skip_non_finite_metrics and (not np.isfinite(mean) or not np.isfinite(std)):
            continue
        out[key] = (mean, std)
    return out


def reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def bin_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": balanced_accuracy(y_true, y_pred),
        "f1": safe_f1(y_true, y_pred, average="binary"),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["auroc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            pass
        try:
            out["auprc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            pass
    return out


def mc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": balanced_accuracy(y_true, y_pred),
        "macro_f1": safe_f1(y_true, y_pred, average="macro"),
    }


def ml_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {
        "micro_f1": safe_f1(y_true, y_pred, average="micro"),
        "macro_f1": safe_f1(y_true, y_pred, average="macro"),
    }
    if y_score is not None:
        aps, aucs = [], []
        for j in range(y_true.shape[1]):
            yt = y_true[:, j]
            ys = y_score[:, j]
            if len(np.unique(yt)) < 2:
                continue
            try:
                aps.append(average_precision_score(yt, ys))
            except Exception:
                pass
            try:
                aucs.append(roc_auc_score(yt, ys))
            except Exception:
                pass
        if aps:
            out["mAP"] = float(np.mean(aps))
        if aucs:
            out["auroc_macro"] = float(np.mean(aucs))
    return out


def format_y_value(value: object) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        v = arr.item()
        if isinstance(v, (np.floating, float)):
            vf = float(v)
            if np.isnan(vf):
                return "nan"
            if np.isposinf(vf):
                return "inf"
            if np.isneginf(vf):
                return "-inf"
            return f"{vf:.10g}"
        return str(v)
    return json.dumps(arr.tolist(), ensure_ascii=False, separators=(",", ":"))


def serialize_task_infos(task_infos: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for task_name, info in task_infos.items():
        item: Dict[str, object] = {}
        for key, value in info.items():
            if isinstance(value, np.generic):
                item[key] = value.item()
            elif isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                item[key] = value
            else:
                item[key] = str(value)
        out[task_name] = item
    return out


def _effective_n_splits(n_samples: int, requested: int) -> int:
    return max(2, min(int(requested), int(n_samples)))


def _log1p_forward(y: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(y, 0.0))


def _log1p_inverse(y_t: np.ndarray) -> np.ndarray:
    return np.expm1(y_t)


def _logit_forward(y: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    y = np.clip(y, eps, 1.0 - eps)
    return np.log(y / (1.0 - y))


def _logit_inverse(y_t: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-y_t))


def transform_y(y: np.ndarray, mode: str):
    if mode == "identity":
        return y.astype(np.float64), (lambda z: z)
    if mode == "log1p":
        return _log1p_forward(y.astype(np.float64)), _log1p_inverse
    if mode == "logit01":
        return _logit_forward(y.astype(np.float64)), _logit_inverse
    raise ValueError(f"Unknown transform: {mode}")


def clip_pred(y_pred: np.ndarray, clip: Optional[Tuple[float, float]]) -> np.ndarray:
    if clip is None:
        return y_pred
    lo, hi = clip
    return np.clip(y_pred, lo, hi)


def make_splitter(task: TaskSpec, y: np.ndarray, n_splits: int, seed: int):
    n_splits = _effective_n_splits(len(y), n_splits)
    if task.kind == "reg":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    strat = np.asarray(y).astype(int)
    if task.kind == "ml":
        strat = strat.sum(axis=1)
    uniq, cnt = np.unique(strat, return_counts=True)
    if len(uniq) >= 2 and cnt.min() >= n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _scale_X(X_train: np.ndarray, X_test: np.ndarray):
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def _make_fast_logreg(max_iter: int):
    return linear_model.LogisticRegression(
        solver="lbfgs",
        max_iter=max_iter,
        tol=1e-3,
        class_weight="balanced",
    )


def _positive_class_score(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    pos_idx_arr = np.where(np.asarray(classes, dtype=int) == 1)[0]
    pos_idx = int(pos_idx_arr[0]) if len(pos_idx_arr) else min(1, proba.shape[1] - 1)
    return proba[:, pos_idx]


def _robust_clip_with_train_distribution(
    pred: np.ndarray,
    train_values: np.ndarray,
    *,
    q_low: float = 0.01,
    q_high: float = 0.99,
    expand: float = 3.0,
) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    train_values = np.asarray(train_values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(train_values)
    if not finite.any():
        return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    tv = train_values[finite]
    if tv.size == 1:
        lo = hi = float(tv[0])
    else:
        ql, qh = np.quantile(tv, [q_low, q_high])
        span = max(float(qh - ql), 1e-6)
        lo = float(ql - expand * span)
        hi = float(qh + expand * span)
        if lo > hi:
            lo, hi = hi, lo
    fallback = float(np.median(tv))
    pred = np.nan_to_num(pred, nan=fallback, posinf=hi, neginf=lo)
    return np.clip(pred, lo, hi)


def fit_predict_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    task: TaskSpec,
    reg_probe_cfg: RegProbeConfig,
) -> np.ndarray:
    X_train_s, X_test_s = _scale_X(X_train, X_test)
    y_t, inv = transform_y(y_train, task.y_transform)
    n = X_train_s.shape[0]
    alpha = float(reg_probe_cfg.small_alpha) if n < int(reg_probe_cfg.small_n) else float(reg_probe_cfg.medium_alpha)
    try:
        model = linear_model.Ridge(alpha=alpha, solver="lsqr")
        model.fit(X_train_s, y_t)
        y_pred_t = model.predict(X_test_s)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        backoff = linear_model.Ridge(alpha=max(10.0, alpha), solver="lsqr")
        backoff.fit(X_train_s, y_t)
        y_pred_t = backoff.predict(X_test_s)
    y_pred_t = _robust_clip_with_train_distribution(y_pred_t, y_t)
    y_pred = inv(np.asarray(y_pred_t, dtype=np.float64))
    y_pred = _robust_clip_with_train_distribution(y_pred, y_train)
    if task.y_transform == "log1p":
        y_pred = np.maximum(y_pred, 0.0)
    return clip_pred(y_pred, task.clip).astype(np.float64)


def fit_predict_binary(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
    y_train = np.asarray(y_train).astype(int)
    X_train_s, X_test_s = _scale_X(X_train, X_test)
    uniq = np.unique(y_train)
    if len(uniq) < 2:
        const = int(uniq[0])
        pred = np.full(X_test_s.shape[0], const, dtype=int)
        score = np.full(X_test_s.shape[0], float(const), dtype=float)
        return pred, score
    clf = _make_fast_logreg(max_iter=100)
    clf.fit(X_train_s, y_train)
    proba = clf.predict_proba(X_test_s)
    score = _positive_class_score(proba, clf.classes_)
    pred = (score >= 0.5).astype(int)
    return pred, score


def fit_predict_multiclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
):
    y_train = np.asarray(y_train).astype(int)
    X_train_s, X_test_s = _scale_X(X_train, X_test)
    uniq = np.unique(y_train)
    if len(uniq) < 2:
        const = int(uniq[0])
        pred = np.full(X_test_s.shape[0], const, dtype=int)
        proba = np.zeros((X_test_s.shape[0], n_classes), dtype=float)
        if 0 <= const < n_classes:
            proba[:, const] = 1.0
        return pred, proba
    clf = _make_fast_logreg(max_iter=120)
    clf.fit(X_train_s, y_train)
    proba_small = clf.predict_proba(X_test_s)
    classes = np.asarray(clf.classes_, dtype=int)
    proba = np.zeros((proba_small.shape[0], n_classes), dtype=float)
    keep = (classes >= 0) & (classes < n_classes)
    proba[:, classes[keep]] = proba_small[:, keep]
    pred = np.argmax(proba, axis=1).astype(int)
    return pred, proba


def fit_predict_multilabel(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_labels: int,
):
    y_train = np.asarray(y_train).astype(int)
    X_train_s, X_test_s = _scale_X(X_train, X_test)
    n_test = X_test_s.shape[0]
    score = np.zeros((n_test, n_labels), dtype=float)
    for j in range(n_labels):
        yt = y_train[:, j]
        uniq = np.unique(yt)
        if len(uniq) < 2:
            score[:, j] = float(uniq[0])
            continue
        clf = _make_fast_logreg(max_iter=100)
        clf.fit(X_train_s, yt)
        proba = clf.predict_proba(X_test_s)
        score[:, j] = _positive_class_score(proba, clf.classes_)
    pred = (score >= 0.5).astype(int)
    return pred, score


def cv_eval_reg(
    X: np.ndarray,
    y: np.ndarray,
    task: TaskSpec,
    n_splits: int,
    seed: int,
    reg_probe_cfg: RegProbeConfig,
):
    splitter = make_splitter(task, y, n_splits=n_splits, seed=seed)
    fold_mae, fold_rmse, fold_r2 = [], [], []
    oof_pred = np.zeros_like(y, dtype=np.float64)
    for tr, te in splitter.split(X, y):
        y_pred = fit_predict_regression(X[tr], y[tr], X[te], task, reg_probe_cfg)
        oof_pred[te] = y_pred
        metrics = reg_metrics(y[te], y_pred)
        fold_mae.append(metrics["mae"])
        fold_rmse.append(metrics["rmse"])
        fold_r2.append(metrics["r2"])
    overall = reg_metrics(y, oof_pred)
    return {
        "mae": (overall["mae"], float(np.std(fold_mae))),
        "rmse": (overall["rmse"], float(np.std(fold_rmse))),
        "r2": (overall["r2"], float(np.std(np.clip(fold_r2, -1.0, 1.0)))),
    }


def cv_eval_bin(
    X: np.ndarray,
    y: np.ndarray,
    task: TaskSpec,
    n_splits: int,
    seed: int,
):
    splitter = make_splitter(task, y, n_splits=n_splits, seed=seed)
    y = np.asarray(y).astype(int)
    oof_pred = np.zeros_like(y, dtype=int)
    oof_score = np.zeros_like(y, dtype=float)
    fold: Dict[str, List[float]] = {}
    for tr, te in splitter.split(X, y):
        pred, score = fit_predict_binary(X[tr], y[tr], X[te])
        oof_pred[te] = pred
        oof_score[te] = score
        metrics = bin_metrics(y[te], pred, score)
        for key, val in metrics.items():
            fold.setdefault(key, []).append(float(val))
    overall = bin_metrics(y, oof_pred, oof_score)
    return metric_stats(overall, fold, ["acc", "bacc", "f1", "auroc", "auprc"])


def cv_eval_mc(
    X: np.ndarray,
    y: np.ndarray,
    task: TaskSpec,
    n_splits: int,
    seed: int,
):
    splitter = make_splitter(task, y, n_splits=n_splits, seed=seed)
    y = np.asarray(y).astype(int)
    oof_pred = np.zeros_like(y, dtype=int)
    fold = {"acc": [], "bacc": [], "macro_f1": []}
    for tr, te in splitter.split(X, y):
        pred, _ = fit_predict_multiclass(X[tr], y[tr], X[te], n_classes=int(task.n_classes))
        oof_pred[te] = pred
        metrics = mc_metrics(y[te], pred)
        for key in fold:
            fold[key].append(metrics[key])
    overall = mc_metrics(y, oof_pred)
    return {key: (float(overall[key]), float(np.std(fold[key]))) for key in fold}


def cv_eval_ml(
    X: np.ndarray,
    y: np.ndarray,
    task: TaskSpec,
    n_splits: int,
    seed: int,
):
    splitter = make_splitter(task, y, n_splits=n_splits, seed=seed)
    y = np.asarray(y).astype(int)
    n, label_count = y.shape
    oof_pred = np.zeros((n, label_count), dtype=int)
    oof_score = np.zeros((n, label_count), dtype=float)
    fold: Dict[str, List[float]] = {}
    strat = y.sum(axis=1)
    for tr, te in splitter.split(X, strat):
        pred, score = fit_predict_multilabel(X[tr], y[tr], X[te], n_labels=label_count)
        oof_pred[te] = pred
        oof_score[te] = score
        metrics = ml_metrics(y[te], pred, score)
        for key, val in metrics.items():
            fold.setdefault(key, []).append(float(val))
    overall = ml_metrics(y, oof_pred, oof_score)
    return metric_stats(overall, fold, ["micro_f1", "macro_f1", "mAP", "auroc_macro"])


def evaluate_task_with_probe(
    X: np.ndarray,
    y: np.ndarray,
    task: TaskSpec,
    n_splits: int,
    seed: int,
    reg_probe_cfg: RegProbeConfig,
    skip_non_finite_metrics: bool = True,
) -> Dict[str, Tuple[float, float]]:
    if task.kind == "reg":
        metrics = cv_eval_reg(X, y.astype(np.float64), task, n_splits=n_splits, seed=seed, reg_probe_cfg=reg_probe_cfg)
    elif task.kind == "bin":
        metrics = cv_eval_bin(X, y.astype(int), task, n_splits=n_splits, seed=seed)
    elif task.kind == "mc":
        metrics = cv_eval_mc(X, y.astype(int), task, n_splits=n_splits, seed=seed)
    else:
        metrics = cv_eval_ml(X, y.astype(int), task, n_splits=n_splits, seed=seed)
    return clean_metric_dict(metrics, skip_non_finite_metrics=skip_non_finite_metrics)


def landuse_15class_roots(task_root: str) -> List[Path]:
    candidates: List[Path] = []
    env_root = os.environ.get("URBANSENSE_LANDUSE_15CLASS_ROOT", "").strip()
    if not env_root:
        env_root = os.environ.get("URBANMOTIF_LANDUSE_15CLASS_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))
    task_root_path = Path(task_root)
    candidates.append(task_root_path / "landuse")
    candidates.append(task_root_path.parent / "landuse_clustering_npy")
    candidates.append(Path(DEFAULT_LANDUSE_15CLASS_ROOT))

    roots: List[Path] = []
    seen: set[str] = set()
    for root in candidates:
        root = root.expanduser()
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    return roots


def landuse_15class_label_path(task_root: str, county_fip: str) -> Optional[Path]:
    filename = f"{str(county_fip).zfill(5)}.npy"
    for root in landuse_15class_roots(task_root):
        for folder in (root, root / "counties"):
            path = folder / filename
            if path.exists():
                return path
    return None


STATE_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


def safety_mapping_paths(task_root: str) -> List[Path]:
    root = Path(task_root)
    candidates = [
        root.parent / "us_cities_fips.csv",
        PROJECT_ROOT / "data/raw/downstream/us_cities_fips.csv",
    ]
    out: List[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def city_file_stem_candidates(city: str, state: str) -> List[str]:
    text = str(city).strip()
    text = re.sub(r"\s+\(balance\)$", "", text, flags=re.IGNORECASE)
    base = re.sub(
        r"\s+(city|town|village|borough|municipality|county|urban county|metropolitan government|cdp)$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    def slug(value: str) -> str:
        value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
        return re.sub(r"_+", "_", value).strip("_")

    stems = [slug(base), slug(text)]
    abbr = STATE_ABBR.get(str(state).strip())
    if abbr:
        stems.extend([f"{slug(base)}_{abbr}", f"{slug(text)}_{abbr}"])
    deduped: List[str] = []
    for stem in stems:
        if stem and stem not in deduped:
            deduped.append(stem)
    return deduped


def safety_label_path(task_root: str, county_fip: str) -> Optional[Path]:
    safety_root = Path(task_root) / "safety"
    direct = safety_root / f"{str(county_fip).zfill(5)}.npy"
    if direct.exists():
        return direct
    if not safety_root.is_dir():
        return None
    existing = {p.stem: p for p in safety_root.glob("*.npy")}
    county_fip = str(county_fip).zfill(5)
    for mapping_path in safety_mapping_paths(task_root):
        if not mapping_path.exists():
            continue
        mapping = pd.read_csv(mapping_path, dtype={"FIPS": str})
        if not {"City", "State", "FIPS"}.issubset(mapping.columns):
            continue
        sub = mapping[mapping["FIPS"].astype(str).str.zfill(5) == county_fip]
        for _, row in sub.iterrows():
            for stem in city_file_stem_candidates(row["City"], row["State"]):
                if stem in existing:
                    return existing[stem]
    return None


def load_task_labels(task_root: str, task_name: str, county_fip: str) -> Tuple[np.ndarray, np.ndarray]:
    if task_name == "landuse":
        path = landuse_15class_label_path(task_root, county_fip)
        if path is None:
            filename = f"{str(county_fip).zfill(5)}.npy"
            searched = ", ".join(
                str(candidate)
                for root in landuse_15class_roots(task_root)
                for candidate in (root / filename, root / "counties" / filename)
            )
            raise FileNotFoundError(
                "landuse is configured as a 15-class task and requires dominant_group_id labels. "
                f"Searched: {searched}"
            )
    elif task_name == "safety":
        path = safety_label_path(task_root, county_fip)
        if path is None:
            searched = ", ".join(str(p) for p in safety_mapping_paths(task_root))
            raise FileNotFoundError(
                "safety is configured as a city-file task and requires a city-to-FIPS mapping. "
                f"Searched mapping files: {searched}; label root: {Path(task_root) / 'safety'}"
            )
    else:
        path = Path(task_root) / task_name / f"{county_fip}.npy"
    arr = np.load(path, allow_pickle=True)
    if not getattr(arr.dtype, "fields", None):
        expected = "GEOID20/dominant_group_id" if task_name == "landuse" else "GEOID20/y"
        raise ValueError(f"Expected structured array with {expected} fields: {path}")
    fields_lower = {k.lower(): k for k in arr.dtype.fields.keys()}
    geoid_key = fields_lower.get("geoid20")
    y_key = fields_lower.get("dominant_group_id") if task_name == "landuse" else fields_lower.get("y")
    if geoid_key is None or y_key is None:
        expected_y = "dominant_group_id" if task_name == "landuse" else "y"
        raise ValueError(f"Structured array missing GEOID20/{expected_y} fields: {path}")
    y = np.asarray(arr[y_key])
    geoids_raw = np.asarray(arr[geoid_key])
    if geoids_raw.size == 0:
        return np.asarray([], dtype=str), y[:0]
    geoids = normalize_geoid11(geoids_raw)
    if task_name == "safety":
        keep = np.char.startswith(geoids, str(county_fip).zfill(5))
        geoids = geoids[keep]
        y = y[keep]
    return geoids.astype(str), y


def filter_and_cast_task_labels(y_candidate: np.ndarray, task: TaskSpec) -> Tuple[np.ndarray, np.ndarray]:
    def cast_integer_labels(y_raw: np.ndarray, lo: int, hi: int) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y_raw, dtype=np.float64).reshape(-1)
        finite = np.isfinite(y)
        intlike = np.zeros_like(finite, dtype=bool)
        intlike[finite] = np.isclose(y[finite], np.round(y[finite]))
        y_int = np.zeros_like(y, dtype=np.int64)
        y_int[finite] = np.round(y[finite]).astype(np.int64)
        in_range = np.zeros_like(finite, dtype=bool)
        in_range[finite] = (y_int[finite] >= lo) & (y_int[finite] <= hi)
        valid = finite & intlike & in_range
        return y_int, valid

    if task.kind == "reg":
        y = np.asarray(y_candidate, dtype=np.float64).reshape(-1)
        valid = np.isfinite(y)
        return y, valid
    if task.kind == "bin":
        return cast_integer_labels(y_candidate, 0, 1)
    if task.kind == "mc":
        y_int, valid = cast_integer_labels(y_candidate, 0, int(task.n_classes) - 1)
        return y_int, valid
    y = np.asarray(y_candidate)
    if y.ndim == 1:
        y = y.reshape(-1, int(task.n_labels))
    yf = y.astype(np.float64)
    finite = np.all(np.isfinite(yf), axis=1)
    y_round = np.round(yf)
    intlike = np.all(np.isclose(yf, y_round), axis=1)
    y_int = y_round.astype(np.int64)
    valid = finite & intlike & np.all((y_int >= 0) & (y_int <= 1), axis=1)
    return y_int, valid


def check_label_diversity(task: TaskSpec, y: np.ndarray) -> Optional[str]:
    if task.kind == "reg":
        return None
    if task.kind in {"bin", "mc"}:
        return None if np.unique(np.asarray(y, dtype=np.int64)).size >= 2 else "insufficient_label_diversity"
    y_ml = np.asarray(y, dtype=np.int64)
    if y_ml.ndim == 1:
        y_ml = y_ml.reshape(-1, int(task.n_labels))
    for j in range(y_ml.shape[1]):
        if np.unique(y_ml[:, j]).size >= 2:
            return None
    return "insufficient_label_diversity"


def prepare_task_data(
    county_fip: str,
    task_name: str,
    task_root: str,
    X_county: np.ndarray,
    emb_geoids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], List[Dict[str, object]]]:
    spec = TASK_SPECS[task_name]
    label_geoids, y_raw = load_task_labels(task_root, task_name, county_fip)
    n_raw = int(len(label_geoids))

    index: Dict[str, int] = {}
    for i, geoid in enumerate(emb_geoids):
        key = str(geoid).strip()
        if key not in index:
            index[key] = i

    has_embedding = np.zeros((n_raw,), dtype=bool)
    emb_row_index = np.full((n_raw,), -1, dtype=int)
    for i, geoid in enumerate(label_geoids):
        idx = index.get(str(geoid).strip())
        if idx is not None:
            has_embedding[i] = True
            emb_row_index[i] = int(idx)

    candidate_idx = np.flatnonzero(has_embedding)
    y_candidate = y_raw[candidate_idx]
    row_ids = emb_row_index[candidate_idx]
    X_candidate = X_county[row_ids] if len(row_ids) else np.zeros((0, X_county.shape[1]), dtype=X_county.dtype)
    n_after_align = int(len(candidate_idx))
    y_cast, value_mask = filter_and_cast_task_labels(y_candidate, spec)
    X_out = X_candidate[value_mask]
    y_out = y_cast[value_mask]
    used_final = np.zeros((n_raw,), dtype=bool)
    if len(candidate_idx):
        used_final[candidate_idx[value_mask]] = True
    n_final = int(len(y_out))
    county_fip = str(county_fip).zfill(5)
    statuses = np.where(used_final, "kept", "dropped")
    reasons = np.where(~has_embedding, "no_embedding", np.where(~used_final, "invalid_label_value", ""))
    tract_rows = [
        {
            "county_fip": county_fip,
            "task": task_name,
            "geoid20": geoid,
            "tract_status": status,
            "drop_reason": reason,
            "kept_by_value_filter": True,
            "has_embedding": bool(has_emb),
            "used_in_eval": bool(used),
            "embedding_row_index": int(row_idx),
            "y_raw": format_y_value(y_val),
        }
        for geoid, status, reason, has_emb, used, row_idx, y_val in zip(
            label_geoids,
            statuses,
            reasons,
            has_embedding,
            used_final,
            emb_row_index,
            y_raw,
        )
    ]
    stats = {
        "num_labels_raw": n_raw,
        "num_labels_after_value_filter": n_final,
        "num_labels_dropped_value_filter": int(n_after_align - n_final),
        "num_labels_after_geoid_overlap": n_after_align,
        "num_labels_dropped_no_embedding": int(n_raw - n_after_align),
        "num_labels_after_final_nonfinite_filter": n_final,
        "num_labels_dropped_final_nonfinite": 0,
    }
    return X_out, y_out, stats, tract_rows


def result_row(
    model_name: str,
    county_fip: str,
    task_name: str,
    coverage: Dict[str, object],
    *,
    status: str,
    reason: str = "",
    metric: str = "",
    mean: float = np.nan,
    std: float = np.nan,
    n_samples: int = 0,
) -> Dict[str, object]:
    return {
        "embedding_model": model_name,
        "county_fip": county_fip,
        "task": task_name,
        "probe": PROBE_NAME,
        "status": status,
        "reason": reason,
        "metric": metric,
        "mean": float(mean),
        "std": float(std),
        "n_samples": int(n_samples),
        "num_tracts_embedded": int(coverage.get("num_tracts_embedded", 0)),
        "embedding_dim": int(coverage.get("embedding_dim", 0)),
        "embedding_path": str(coverage.get("embedding_path", "")),
    }


def evaluate_one_county(
    model_name: str,
    county_fip: str,
    *,
    task_root: str,
    tasks: List[str],
    n_splits: int,
    seed: int,
    reg_probe_cfg: RegProbeConfig,
    skip_non_finite_metrics: bool,
    emb: np.ndarray,
    emb_geoids: np.ndarray,
    emb_path: str,
    county_rows: np.ndarray | None = None,
) -> Tuple[
    str,
    List[Dict[str, object]],
    List[Dict[str, object]],
    Dict[str, object],
    Dict[str, Dict[str, Tuple[float, float]]],
    Dict[str, Dict[str, object]],
]:
    rows: List[Dict[str, object]] = []
    tract_rows: List[Dict[str, object]] = []
    county_fip = str(county_fip).strip().zfill(5)
    county_results: Dict[str, Dict[str, Tuple[float, float]]] = {}
    county_task_infos: Dict[str, Dict[str, object]] = {}
    if county_rows is None:
        county_rows = np.flatnonzero(np.char.startswith(emb_geoids, county_fip))
    X_county = emb[county_rows]
    geoids_county = emb_geoids[county_rows]
    coverage: Dict[str, object] = {
        "county_fip": county_fip,
        "embedding_path": emb_path,
        "num_tracts_embedded": int(X_county.shape[0]),
        "embedding_dim": int(emb.shape[1]),
    }
    for task_name in tasks:
        spec = TASK_SPECS[task_name]
        task_info: Dict[str, object] = {
            "num_samples": 0,
            "status": "skipped",
            "reason": "",
            "num_tracts_embedded": int(coverage["num_tracts_embedded"]),
        }

        def skip_task(reason: str) -> None:
            task_info["reason"] = reason
            county_task_infos[task_name] = task_info
            rows.append(result_row(model_name, county_fip, task_name, coverage, status="skipped", reason=reason))

        try:
            X, y, label_stats, task_tract_rows = prepare_task_data(county_fip, task_name, task_root, X_county, geoids_county)
            for tr in task_tract_rows:
                tr["embedding_model"] = model_name
            task_info.update(label_stats)
            tract_rows.extend(task_tract_rows)
        except FileNotFoundError:
            skip_task("label_file_missing")
            continue
        except Exception as exc:
            skip_task(f"prepare_failed:{exc}")
            continue
        task_info["num_samples"] = int(len(y))
        if X.shape[0] == 0:
            skip_task("no_geoid_overlap")
            continue
        if X.shape[0] < 2:
            skip_task("insufficient_samples")
            continue
        diversity_reason = check_label_diversity(spec, y)
        if diversity_reason is not None:
            skip_task(diversity_reason)
            continue
        try:
            metrics = evaluate_task_with_probe(
                X,
                y,
                spec,
                n_splits=n_splits,
                seed=seed,
                reg_probe_cfg=reg_probe_cfg,
                skip_non_finite_metrics=skip_non_finite_metrics,
            )
            if not metrics:
                skip_task("no_finite_metric_after_filter")
                continue
            county_results[task_name] = metrics
            for metric_name, (mean, std) in metrics.items():
                rows.append(
                    result_row(
                        model_name,
                        county_fip,
                        task_name,
                        coverage,
                        status="ok",
                        metric=metric_name,
                        mean=mean,
                        std=std,
                        n_samples=int(len(y)),
                    )
                )
            task_info["status"] = "ok"
            task_info["reason"] = ""
        except Exception as exc:
            skip_task(f"eval_failed:{exc}")
            continue
        county_task_infos[task_name] = task_info
    return county_fip, rows, tract_rows, coverage, county_results, county_task_infos


def normalize_county_fip(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"\.0$", "", text)
    return text.zfill(5)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "status" not in df.columns:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows: List[Dict[str, object]] = []
    for keys, g in ok.groupby(["embedding_model", "task", "probe", "metric"]):
        embedding_model, task, probe, metric = keys
        m = pd.to_numeric(g["mean"], errors="coerce").to_numpy(dtype=float)
        n = pd.to_numeric(g["n_samples"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        valid = np.isfinite(m) & np.isfinite(n) & (n > 0)
        wmean = float(np.average(m[valid], weights=n[valid])) if valid.any() else float(np.nanmean(m))
        rows.append(
            {
                "embedding_model": embedding_model,
                "task": task,
                "probe": probe,
                "metric": metric,
                "county_count": int(g["county_fip"].nunique()),
                "total_samples": int(n.sum()),
                "macro_mean": float(np.nanmean(m)),
                "macro_std": float(np.nanstd(m)),
                "weighted_mean_by_samples": wmean,
            }
        )
    return pd.DataFrame(rows).sort_values(["embedding_model", "task", "probe", "metric"]).reset_index(drop=True)


def discover_counties(task_root: str, task_name: str = "population") -> List[str]:
    if task_name == "landuse":
        for root in landuse_15class_roots(task_root):
            for folder in (root, root / "counties"):
                if folder.is_dir():
                    files = [
                        p
                        for p in sorted(folder.glob("*.npy"))
                        if len(p.stem) == 5 and p.stem.isdigit()
                    ]
                    if files:
                        return [p.stem.zfill(5) for p in files]
        searched = ", ".join(
            str(candidate)
            for root in landuse_15class_roots(task_root)
            for candidate in (root, root / "counties")
        )
        raise FileNotFoundError(
            "landuse is configured as a 15-class task and requires dominant_group_id county labels. "
            f"Searched: {searched}"
        )
    if task_name == "safety":
        safety_root = Path(task_root) / "safety"
        if not safety_root.is_dir():
            raise FileNotFoundError(str(safety_root))
        existing = {p.stem for p in safety_root.glob("*.npy")}
        counties: set[str] = set()
        for mapping_path in safety_mapping_paths(task_root):
            if not mapping_path.exists():
                continue
            mapping = pd.read_csv(mapping_path, dtype={"FIPS": str})
            if not {"City", "State", "FIPS"}.issubset(mapping.columns):
                continue
            for _, row in mapping.iterrows():
                if any(stem in existing for stem in city_file_stem_candidates(row["City"], row["State"])):
                    counties.add(str(row["FIPS"]).zfill(5))
            if counties:
                break
        if not counties:
            raise FileNotFoundError(f"No safety counties resolved from {safety_mapping_paths(task_root)}")
        return sorted(counties)
    folder = Path(task_root) / task_name
    if not folder.is_dir():
        raise FileNotFoundError(str(folder))
    return [p.stem.zfill(5) for p in sorted(folder.glob("*.npy"))]


def extract_global_embeddings(
    *,
    processed_root: str,
    checkpoint: str,
    device: str,
    amp: str,
    embedding_type: str,
    dim: int,
    num_motifs: int,
    smooth_steps: int,
    temperature: float,
    view_smooth_type: str,
    tract_context_type: str,
    tract_context_position: str,
    tract_context_graph: str,
    tract_context_layers: int,
    residual_fusion: str,
    motif_consensus_type: str,
    profile_residual_decomp: str,
    enabled_views: str,
    dropout: float,
    counties: Sequence[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    setup_torch_performance()
    model = build_encoder(
        processed_root=processed_root,
        checkpoint=checkpoint,
        dim=dim,
        num_motifs=num_motifs,
        smooth_steps=smooth_steps,
        temperature=temperature,
        view_smooth_type=view_smooth_type,
        tract_context_type=tract_context_type,
        tract_context_position=tract_context_position,
        tract_context_graph=tract_context_graph,
        tract_context_layers=tract_context_layers,
        residual_fusion=residual_fusion,
        motif_consensus_type=motif_consensus_type,
        profile_residual_decomp=profile_residual_decomp,
        enabled_views=enabled_views,
        dropout=dropout,
        device=device,
    )
    model.eval()
    ds = CityDataset(processed_root, county_ids=counties)
    emb_chunks: List[np.ndarray] = []
    tract_chunks: List[np.ndarray] = []
    embedding_type = str(embedding_type).lower()
    with torch.no_grad():
        for city in tqdm(ds, desc="extract urbanmotif embeddings"):
            city = city.to(device)
            with autocast_context(torch.device(device), amp):
                out = model(city)
            if embedding_type == "z":
                emb_tensor = out.z
            elif embedding_type in {"common", "c"}:
                emb_tensor = out.common
            elif embedding_type in {"concat_z_common", "z_common", "concat"}:
                emb_tensor = torch.cat([out.z, out.common], dim=-1)
            elif embedding_type == "q":
                emb_tensor = out.Q
            else:
                raise ValueError(f"Unknown --embedding-type: {embedding_type}")
            emb_chunks.append(emb_tensor.detach().float().cpu().numpy())
            tract_chunks.append(normalize_geoid11(np.asarray(city.tract_ids, dtype=str)))
    emb = np.concatenate(emb_chunks, axis=0).astype(np.float32, copy=False)
    tract_ids = np.concatenate(tract_chunks, axis=0).astype(str, copy=False)
    return emb, tract_ids


def load_embedding_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as loaded:
        if "embeddings" not in loaded.files or "tract_ids" not in loaded.files:
            raise ValueError(f"Embedding cache must contain embeddings and tract_ids: {path}")
        emb = np.asarray(loaded["embeddings"], dtype=np.float32)
        tract_ids = normalize_geoid11(np.asarray(loaded["tract_ids"]))
    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape={emb.shape} from {path}")
    if emb.shape[0] != tract_ids.shape[0]:
        raise ValueError(f"Embedding rows {emb.shape[0]} != tract_ids rows {tract_ids.shape[0]} for {path}")
    return emb, tract_ids


def filter_embeddings_to_counties(
    emb: np.ndarray,
    tract_ids: np.ndarray,
    counties: Sequence[str] | None,
) -> Tuple[np.ndarray, np.ndarray]:
    requested = parse_county_ids(counties)
    if not requested:
        return emb, tract_ids
    wanted = set(requested)
    county_ids = np.asarray([str(geoid)[:5].zfill(5) for geoid in tract_ids])
    mask = np.asarray([county in wanted for county in county_ids], dtype=bool)
    covered = set(county_ids[mask].tolist())
    missing = [county for county in requested if county not in covered]
    if missing:
        raise ValueError(f"Embedding cache is missing {len(missing)} requested counties, first={missing[:10]}")
    return emb[mask], tract_ids[mask]


def maybe_compute_or_load_embeddings(
    args: argparse.Namespace,
    *,
    counties: Sequence[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    emb_path = str(args.embedding_path).strip()
    if emb_path and Path(emb_path).exists():
        emb, tract_ids = load_embedding_npz(emb_path)
        emb, tract_ids = filter_embeddings_to_counties(emb, tract_ids, counties)
        return emb, tract_ids, str(Path(emb_path).resolve())
    if not args.processed_root or not args.checkpoint:
        raise ValueError("Need either an existing --embedding-path or both --processed-root and --checkpoint.")
    encoder_args = resolve_encoder_args(args)
    emb, tract_ids = extract_global_embeddings(
        processed_root=args.processed_root,
        checkpoint=args.checkpoint,
        device=args.device,
        amp=args.amp,
        embedding_type=args.embedding_type,
        dim=int(encoder_args["dim"]),
        num_motifs=int(encoder_args["num_motifs"]),
        smooth_steps=int(encoder_args["smooth_steps"]),
        temperature=float(encoder_args["temperature"]),
        view_smooth_type=str(encoder_args["view_smooth_type"]),
        tract_context_type=str(encoder_args["tract_context_type"]),
        tract_context_position=str(encoder_args["tract_context_position"]),
        tract_context_graph=str(encoder_args["tract_context_graph"]),
        tract_context_layers=int(encoder_args["tract_context_layers"]),
        residual_fusion=str(encoder_args["residual_fusion"]),
        motif_consensus_type=str(encoder_args["motif_consensus_type"]),
        profile_residual_decomp=str(encoder_args["profile_residual_decomp"]),
        enabled_views=str(encoder_args["enabled_views"]),
        dropout=float(encoder_args["dropout"]),
        counties=counties,
    )
    if emb_path:
        out_path = Path(emb_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, embeddings=emb.astype(np.float32), tract_ids=tract_ids.astype(str))
        return emb, tract_ids, str(out_path.resolve())
    return emb, tract_ids, f"checkpoint:{Path(args.checkpoint).resolve()}"


def evaluate_counties(
    model_name: str,
    emb: np.ndarray,
    emb_geoids: np.ndarray,
    embedding_ref: str,
    *,
    task_root: str,
    tasks: List[str],
    counties: List[str],
    n_splits: int,
    seed: int,
    n_jobs: int,
    reg_probe_cfg: RegProbeConfig,
    skip_non_finite_metrics: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    emb_counties = np.asarray([geoid[:5] for geoid in emb_geoids], dtype="<U5")
    order = np.argsort(emb_counties, kind="stable")
    sorted_counties = emb_counties[order]
    unique_counties, start_idx = np.unique(sorted_counties, return_index=True)
    end_idx = np.r_[start_idx[1:], len(order)]
    county_row_index = {
        county: order[start:end]
        for county, start, end in zip(unique_counties.tolist(), start_idx.tolist(), end_idx.tolist())
    }

    def eval_county(county: str):
        return evaluate_one_county(
            model_name,
            county,
            task_root=task_root,
            tasks=tasks,
            n_splits=n_splits,
            seed=seed,
            reg_probe_cfg=reg_probe_cfg,
            skip_non_finite_metrics=skip_non_finite_metrics,
            emb=emb,
            emb_geoids=emb_geoids,
            emb_path=embedding_ref,
            county_rows=county_row_index.get(normalize_county_fip(county)),
        )

    all_rows, all_tract_rows, coverage_rows = [], [], []
    results_by_county, task_infos_by_county = {}, {}
    if counties:
        if n_jobs <= 1:
            iterator = ((county, eval_county(county), None) for county in counties)
        else:
            pool = cf.ThreadPoolExecutor(max_workers=min(n_jobs, len(counties)))
            futures = {pool.submit(eval_county, county): county for county in counties}
            iterator = ((futures[fut], None, fut) for fut in cf.as_completed(futures))
        done = 0
        try:
            for county, output, future in iterator:
                county_key = normalize_county_fip(county)
                if future is not None:
                    try:
                        output = future.result()
                    except Exception as exc:
                        cov = {
                            "county_fip": county_key,
                            "error": str(exc),
                            "num_tracts_embedded": 0,
                            "embedding_dim": int(emb.shape[1]),
                            "embedding_path": embedding_ref,
                        }
                        rows = [
                            result_row(
                                model_name,
                                county_key,
                                task_name,
                                cov,
                                status="skipped",
                                reason=f"county_failed:{exc}",
                            )
                            for task_name in tasks
                        ]
                        task_infos_by_county[county_key] = {
                            task_name: {
                                "num_samples": 0,
                                "status": "skipped",
                                "reason": f"county_failed:{exc}",
                                "num_tracts_embedded": 0,
                            }
                            for task_name in tasks
                        }
                        coverage_rows.append({"embedding_model": model_name, **cov})
                        all_rows.extend(rows)
                        results_by_county[county_key] = {}
                        done += 1
                        if done % 20 == 0 or done == len(counties):
                            print(f"[MODEL:{model_name}] progress {done}/{len(counties)} counties", flush=False)
                        continue
                county_key, rows, tract_rows, cov, county_results, county_task_infos = output
                all_rows.extend(rows)
                all_tract_rows.extend(tract_rows)
                coverage_rows.append({"embedding_model": model_name, **cov})
                results_by_county[county_key] = county_results
                task_infos_by_county[county_key] = county_task_infos
                done += 1
                if done % 20 == 0 or done == len(counties):
                    print(f"[MODEL:{model_name}] progress {done}/{len(counties)} counties", flush=False)
        finally:
            if n_jobs > 1:
                pool.shutdown(wait=True)

    result_df = pd.DataFrame(all_rows)
    per_tract_df = pd.DataFrame(all_tract_rows)
    cov_df = pd.DataFrame(coverage_rows)
    summary_df = summarize_results(result_df)
    payload = {
        "embedding_model": model_name,
        "embedding_path": embedding_ref,
        "results": {
            county: {
                PROBE_NAME: {
                    task: {metric: [float(mean), float(std)] for metric, (mean, std) in metrics.items()}
                    for task, metrics in county_results.items()
                }
            }
            for county, county_results in sorted(results_by_county.items())
        },
        "task_infos": {k: serialize_task_infos(v) for k, v in sorted(task_infos_by_county.items())},
    }
    return result_df, per_tract_df, cov_df, {"summary_df": summary_df, "payload": payload}


def build_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate SURE tract embeddings on the 466-county benchmark with county-wise linear probes."
    )
    parser.add_argument("--processed-root", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--embedding-path", default="")
    parser.add_argument("--task-root", default=DEFAULT_BENCH_TASK_ROOT)
    parser.add_argument("--tasks", default=",".join(TASK_ORDER))
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--counties", default="")
    parser.add_argument("--county-fips-file", default="", help="Optional CSV/list with a county_fips column.")
    parser.add_argument(
        "--allow-non-finite-metrics",
        action="store_true",
        help="If set, keep NaN/Inf metrics instead of skipping them.",
    )
    parser.add_argument("--linear-ridge-small-n", type=int, default=50)
    parser.add_argument("--linear-ridge-small-alpha", type=float, default=1000.0)
    parser.add_argument("--linear-ridge-medium-alpha", type=float, default=100.0)
    parser.add_argument("--out-dir", default=DEFAULT_BENCH_OUT_DIR)
    parser.add_argument("--output-prefix", default="sure_466")
    parser.add_argument("--model-name", default="SURE")
    parser.add_argument(
        "--embedding-type",
        choices=["z", "common", "concat_z_common", "q"],
        default="z",
        help="Which SURE representation to evaluate.",
    )
    parser.add_argument("--num-motifs", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--smooth-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--view-smooth-type", default=None)
    parser.add_argument("--tract-context-type", default=None)
    parser.add_argument("--tract-context-position", default=None)
    parser.add_argument("--tract-context-graph", default=None)
    parser.add_argument("--tract-context-layers", type=int, default=None)
    parser.add_argument("--residual-fusion", default=None)
    parser.add_argument("--motif-consensus-type", default=None)
    parser.add_argument("--profile-residual-decomp", default=None)
    parser.add_argument("--enabled-views", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="bf16")
    return parser


def run_benchmark466(args: argparse.Namespace) -> int:
    tasks = [t.strip().lower() for t in str(args.tasks).split(",") if t.strip()]
    if not tasks:
        raise ValueError("No tasks specified. Please pass --tasks.")
    invalid_tasks = [t for t in tasks if t not in TASK_SPECS]
    if invalid_tasks:
        raise ValueError(f"Unknown tasks: {invalid_tasks}. Allowed: {sorted(TASK_SPECS)}")
    if args.counties.strip():
        counties = parse_county_ids(args.counties)
    elif str(args.county_fips_file).strip():
        counties = load_county_ids_file(args.county_fips_file)
    else:
        discover_task = "population" if "population" in tasks else tasks[0]
        counties = discover_counties(args.task_root, task_name=discover_task)
    encoder_args = resolve_encoder_args(args) if args.checkpoint else {}
    emb, emb_geoids, embedding_ref = maybe_compute_or_load_embeddings(args, counties=counties)
    print(
        f"[START] model={args.model_name} counties={len(counties)} tasks={tasks} "
        f"embedding_rows={emb.shape[0]} embedding_dim={emb.shape[1]}"
    )
    reg_probe_cfg = RegProbeConfig(
        small_n=int(args.linear_ridge_small_n),
        small_alpha=float(args.linear_ridge_small_alpha),
        medium_alpha=float(args.linear_ridge_medium_alpha),
    )
    skip_non_finite_metrics = not bool(args.allow_non_finite_metrics)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.output_prefix).strip()
    result_df, per_tract_df, cov_df, pack = evaluate_counties(
        args.model_name,
        emb,
        emb_geoids,
        embedding_ref,
        task_root=args.task_root,
        tasks=tasks,
        counties=counties,
        n_splits=args.n_splits,
        seed=args.seed,
        n_jobs=args.n_jobs,
        reg_probe_cfg=reg_probe_cfg,
        skip_non_finite_metrics=skip_non_finite_metrics,
    )
    summary_df = pack["summary_df"]
    per_county_csv = out_dir / f"{prefix}_per_county.csv"
    per_tract_csv = out_dir / f"{prefix}_per_tract.csv"
    summary_csv = out_dir / f"{prefix}_summary_task_probe.csv"
    coverage_csv = out_dir / f"{prefix}_coverage.csv"
    json_path = out_dir / f"{prefix}_results.json"
    result_df.to_csv(per_county_csv, index=False)
    per_tract_df.to_csv(per_tract_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    cov_df.to_csv(coverage_csv, index=False)
    payload = {
        "config": {
            "task_root": str(Path(args.task_root).resolve()),
            "tasks": tasks,
            "model_name": args.model_name,
            "embedding_path": embedding_ref,
            "embedding_type": args.embedding_type,
            "n_splits": int(args.n_splits),
            "seed": int(args.seed),
            "n_jobs": int(args.n_jobs),
            "encoder": encoder_args,
            "reg_probe": {
                "small_n": int(reg_probe_cfg.small_n),
                "small_alpha": float(reg_probe_cfg.small_alpha),
                "medium_alpha": float(reg_probe_cfg.medium_alpha),
            },
            "skip_non_finite_metrics": bool(skip_non_finite_metrics),
        },
        "counts": {
            "num_counties_requested": int(len(counties)),
            "num_rows_total": int(len(result_df)),
            "num_ok_rows": int((result_df["status"] == "ok").sum()) if not result_df.empty else 0,
            "num_skipped_rows": int((result_df["status"] == "skipped").sum()) if not result_df.empty else 0,
        },
        "results_by_model": {args.model_name: pack["payload"]},
        "output_files": {
            "per_county_csv": str(per_county_csv),
            "per_tract_csv": str(per_tract_csv),
            "summary_csv": str(summary_csv),
            "coverage_csv": str(coverage_csv),
            "results_json": str(json_path),
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("[DONE] outputs:")
    print(f"  per_county_csv: {per_county_csv}")
    print(f"  per_tract_csv:  {per_tract_csv}")
    print(f"  summary_csv:    {summary_csv}")
    print(f"  coverage_csv:   {coverage_csv}")
    print(f"  results_json:   {json_path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_benchmark_parser().parse_args(argv)
    return run_benchmark466(args)


if __name__ == "__main__":
    raise SystemExit(main())
