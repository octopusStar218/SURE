from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LOWER_IS_BETTER = {"mae", "rmse", "mse"}
SELECTED_METRICS = [
    ("population", "mae", "Population MAE↓"),
    ("population", "rmse", "Population RMSE↓"),
    ("income", "acc", "Income ACC↑"),
    ("income", "macro_f1", "Income Macro-F1↑"),
    ("landcover", "acc", "Land cover ACC↑"),
    ("landcover", "macro_f1", "Land cover Macro-F1↑"),
    ("safety", "acc", "Safety ACC↑"),
    ("safety", "macro_f1", "Safety Macro-F1↑"),
    ("co2", "auroc_macro", "Carbon AUROC↑"),
    ("co2", "mAP", "Carbon mAP↑"),
    ("employment", "acc", "Employment ACC↑"),
    ("employment", "f1", "Employment Macro-F1↑"),
]


def direction(metric: str) -> str:
    return "lower" if str(metric).lower() in LOWER_IS_BETTER else "higher"


def collect_summaries(run_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_root.glob("seed_*/downstream/*_summary_task_probe.csv")):
        seed = path.parent.parent.name.removeprefix("seed_")
        df = pd.read_csv(path)
        df["seed"] = seed
        df["summary_csv"] = str(path)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No seed_*/downstream/*_summary_task_probe.csv files found under {run_root}")
    return pd.concat(rows, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["task", "probe", "metric"]
    out = (
        df.groupby(group_cols, sort=True)
        .agg(
            mean=("macro_mean", "mean"),
            seed_std=("macro_mean", "std"),
            seed_count=("seed", "nunique"),
            downstream_std=("macro_std", "mean"),
            county_count=("county_count", "mean"),
            total_samples=("total_samples", "mean"),
            weighted_mean_by_samples=("weighted_mean_by_samples", "mean"),
        )
        .reset_index()
    )
    out["direction"] = out["metric"].map(direction)
    selected = {(task, metric) for task, metric, _ in SELECTED_METRICS}
    out["selected_metric"] = [(task, metric) in selected for task, metric in zip(out["task"], out["metric"])]
    out["seed_std"] = out["seed_std"].fillna(0.0)
    return out


def format_value(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return ""
    if not np.isfinite(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} (± {std:.3f})"


def write_main_row(summary: pd.DataFrame, out_dir: Path) -> None:
    row: dict[str, str] = {"Method": "SURE (Ours)"}
    long_rows = []
    for task, metric, label in SELECTED_METRICS:
        hit = summary[(summary["task"].eq(task)) & (summary["probe"].eq("linear")) & (summary["metric"].eq(metric))]
        if hit.empty:
            row[label] = ""
            continue
        item = hit.iloc[0]
        value = float(item["mean"])
        std = float(item["seed_std"])
        row[label] = format_value(value, std)
        long_rows.append(
            {
                "task": task,
                "probe": "linear",
                "metric": metric,
                "direction": direction(metric),
                "mean": value,
                "seed_std": std,
                "seed_count": int(item["seed_count"]),
                "downstream_std": float(item["downstream_std"]),
                "county_count": float(item["county_count"]),
                "total_samples": float(item["total_samples"]),
                "formatted": row[label],
            }
        )

    pd.DataFrame([row]).to_csv(out_dir / "sure_main_table_row.csv", index=False)
    pd.DataFrame(long_rows).to_csv(out_dir / "sure_main_selected_metrics.csv", index=False)
    try:
        pd.DataFrame([row]).to_markdown(out_dir / "sure_main_table_row.md", index=False)
        pd.DataFrame(long_rows).to_markdown(out_dir / "sure_main_selected_metrics.md", index=False)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize SURE-only multi-seed main experiment results.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = collect_summaries(run_root)
    raw.to_csv(out_dir / "sure_main_per_seed_task_probe.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(out_dir / "sure_main_all_metrics.csv", index=False)
    write_main_row(summary, out_dir)

    print(f"[sure-main-summary] {out_dir / 'sure_main_table_row.csv'}")
    print(f"[sure-main-summary] {out_dir / 'sure_main_selected_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
