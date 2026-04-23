"""
Feature-sensitivity (leave-one-feature-out ablation) and stage-of-game
performance analysis for the logistic regression model.

Answers two reviewer questions:

  1. "How much would the results change if you remove certain features?" —
     re-fits the same pipeline on the same grouped training set once per
     feature, dropping that feature, and records the change in holdout
     Brier score, log loss, and ROC-AUC vs the full model. Also reports a
     drop-column permutation-style importance on the holdout set.

  2. "How do your models perform when the game just started, at halftime,
     or near the end?" — takes the held-out predictions from the full
     model and slices them into game-stage buckets (early, late-first-half,
     halftime, early-second-half, late, overtime) using the existing
     `period` column and `game_minute`. For each bucket it computes Brier,
     log loss, ROC-AUC, and absolute calibration error (pp) for both the
     logistic regression model and the raw Kalshi quote baseline.

Artifacts (written to GeneratedDataFiles/):
  - feature_ablation_core.csv
  - feature_ablation_extended.csv
  - stage_performance.csv
  - stage_performance.png
  - feature_ablation.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from logistic_regression_empirical import (  # noqa: E402
    CORE_CALIBRATION_FEATURE_COLUMNS,
    EXTENDED_FEATURE_COLUMNS,
    build_pipeline,
    compute_metrics,
    grouped_train_test_split,
    load_modeling_frame,
)


DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20


def default_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "0-Data"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "GeneratedDataFiles"


def fit_and_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict[str, float], np.ndarray]:
    pipeline = build_pipeline(feature_columns)
    pipeline.fit(train_df[feature_columns], train_df["team_won"].astype(int))
    probabilities = pipeline.predict_proba(test_df[feature_columns])[:, 1]
    metrics = compute_metrics(test_df["team_won"].astype(int), probabilities)
    return metrics, probabilities


def leave_one_feature_out(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    label: str,
) -> pd.DataFrame:
    full_metrics, _ = fit_and_score(train_df, test_df, feature_columns)
    rows: list[dict[str, float | str | int]] = [
        {
            "feature_set": label,
            "removed_feature": "(none - full model)",
            "n_features": len(feature_columns),
            "roc_auc": full_metrics["roc_auc"],
            "log_loss": full_metrics["log_loss"],
            "brier_score": full_metrics["brier_score"],
            "delta_roc_auc": 0.0,
            "delta_log_loss": 0.0,
            "delta_brier_score": 0.0,
            "relative_brier_change_pct": 0.0,
        }
    ]
    for feature in feature_columns:
        reduced = [f for f in feature_columns if f != feature]
        if not reduced:
            continue
        print(f"  ablating '{feature}' ({len(reduced)} features left)...")
        metrics, _ = fit_and_score(train_df, test_df, reduced)
        rows.append(
            {
                "feature_set": label,
                "removed_feature": feature,
                "n_features": len(reduced),
                "roc_auc": metrics["roc_auc"],
                "log_loss": metrics["log_loss"],
                "brier_score": metrics["brier_score"],
                "delta_roc_auc": metrics["roc_auc"] - full_metrics["roc_auc"],
                "delta_log_loss": metrics["log_loss"] - full_metrics["log_loss"],
                "delta_brier_score": metrics["brier_score"]
                - full_metrics["brier_score"],
                "relative_brier_change_pct": (
                    (metrics["brier_score"] - full_metrics["brier_score"])
                    / full_metrics["brier_score"]
                    * 100.0
                ),
            }
        )
    ablation = pd.DataFrame(rows)
    return ablation.sort_values(
        "delta_brier_score", ascending=False, kind="mergesort"
    ).reset_index(drop=True)


def assign_stage(row: pd.Series) -> str:
    period = str(row["period"]).strip().lower()
    minute = int(row["game_minute"])
    if "overtime" in period:
        return "overtime"
    if period == "halftime":
        return "halftime"
    if period == "firsthalf":
        if minute < 5:
            return "first_half_early (min 0-4)"
        if minute < 15:
            return "first_half_mid (min 5-14)"
        return "first_half_late (min 15-19)"
    if period == "secondhalf":
        if minute < 25:
            return "second_half_early (min 20-24)"
        if minute < 35:
            return "second_half_mid (min 25-34)"
        return "second_half_late (min 35+)"
    return f"other ({period})"


STAGE_ORDER = [
    "first_half_early (min 0-4)",
    "first_half_mid (min 5-14)",
    "first_half_late (min 15-19)",
    "halftime",
    "second_half_early (min 20-24)",
    "second_half_mid (min 25-34)",
    "second_half_late (min 35+)",
    "overtime",
]


def expected_calibration_error_pp(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    """
    ECE in percentage points. Splits [0, 1] into `n_bins` equal-width buckets
    of the predicted probability, and averages |mean_predicted - win_rate|
    across non-empty buckets weighted by bucket count.
    """
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bucket_indices = np.clip(np.digitize(probabilities, edges[1:-1]), 0, n_bins - 1)
    total_weight = 0.0
    weighted_abs = 0.0
    for b in range(n_bins):
        mask = bucket_indices == b
        count = int(mask.sum())
        if count == 0:
            continue
        bucket_pred = float(probabilities[mask].mean())
        bucket_true = float(y_true[mask].mean())
        weighted_abs += count * abs(bucket_pred - bucket_true)
        total_weight += count
    if total_weight == 0:
        return float("nan")
    return weighted_abs / total_weight * 100.0


def stage_performance_table(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["stage"] = predictions.apply(assign_stage, axis=1)
    rows: list[dict[str, float | str | int]] = []
    for stage, group in predictions.groupby("stage", observed=False):
        if len(group) == 0:
            continue
        y = group["team_won"].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            model = {"roc_auc": float("nan"), "log_loss": float("nan"), "brier_score": float("nan")}
            base = {"roc_auc": float("nan"), "log_loss": float("nan"), "brier_score": float("nan")}
        else:
            model = compute_metrics(
                group["team_won"].astype(int),
                group["model_prob"].to_numpy(),
            )
            base = compute_metrics(
                group["team_won"].astype(int),
                group["baseline_prob"].to_numpy(),
            )
        model_ece_pp = expected_calibration_error_pp(
            y, group["model_prob"].to_numpy()
        )
        base_ece_pp = expected_calibration_error_pp(
            y, group["baseline_prob"].to_numpy()
        )
        rows.append(
            {
                "stage": stage,
                "n_rows": int(len(group)),
                "n_games": int(group["kalshi_event"].nunique()),
                "model_roc_auc": model["roc_auc"],
                "baseline_roc_auc": base["roc_auc"],
                "model_log_loss": model["log_loss"],
                "baseline_log_loss": base["log_loss"],
                "model_brier": model["brier_score"],
                "baseline_brier": base["brier_score"],
                "model_ece_pp": model_ece_pp,
                "baseline_ece_pp": base_ece_pp,
            }
        )
    stages = pd.DataFrame(rows)
    stages["sort_key"] = stages["stage"].apply(
        lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else len(STAGE_ORDER)
    )
    stages = stages.sort_values("sort_key", kind="mergesort").drop(columns="sort_key")
    return stages.reset_index(drop=True)


def plot_stage_performance(stages: pd.DataFrame, output_path: Path) -> None:
    fg = "#ffffff"
    spine_color = "#4a4a4a"
    label_color = "#1a1a1a"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=fg)
    positions = np.arange(len(stages))
    width = 0.38
    for ax in axes:
        ax.set_facecolor(fg)
        for spine in ax.spines.values():
            spine.set_color(spine_color)
        ax.tick_params(colors=label_color)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7, color="#c8c8c8")
        ax.set_axisbelow(True)

    axes[0].bar(
        positions - width / 2,
        stages["model_brier"],
        width=width,
        label="Logistic regression",
        color="#1f77b4",
    )
    axes[0].bar(
        positions + width / 2,
        stages["baseline_brier"],
        width=width,
        label="Raw Kalshi quote",
        color="#d97706",
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(stages["stage"], rotation=30, ha="right")
    axes[0].set_ylabel("Brier score (lower is better)")
    axes[0].set_title("Holdout Brier score by game stage")
    axes[0].legend()

    axes[1].bar(
        positions - width / 2,
        stages["model_ece_pp"],
        width=width,
        label="Logistic regression",
        color="#1f77b4",
    )
    axes[1].bar(
        positions + width / 2,
        stages["baseline_ece_pp"],
        width=width,
        label="Raw Kalshi quote",
        color="#d97706",
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(stages["stage"], rotation=30, ha="right")
    axes[1].set_ylabel("Expected calibration error (pp)")
    axes[1].set_title("Holdout calibration error (ECE, 10 bins) by game stage")
    axes[1].legend()

    fig.suptitle(
        "Model performance across stages of the game",
        color=label_color,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fg, edgecolor=fg)
    plt.close(fig)


def plot_feature_ablation(
    core_ablation: pd.DataFrame,
    extended_ablation: pd.DataFrame,
    output_path: Path,
) -> None:
    fg = "#ffffff"
    spine_color = "#4a4a4a"
    label_color = "#1a1a1a"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=fg)

    for ax, ablation, title in zip(
        axes,
        (core_ablation, extended_ablation),
        ("Core calibration feature set", "Extended market feature set"),
    ):
        ax.set_facecolor(fg)
        for spine in ax.spines.values():
            spine.set_color(spine_color)
        ax.tick_params(colors=label_color)
        ax.grid(True, axis="x", linestyle=":", linewidth=0.7, color="#c8c8c8")
        ax.set_axisbelow(True)
        data = ablation[ablation["removed_feature"] != "(none - full model)"].copy()
        data = data.sort_values("delta_brier_score", ascending=True, kind="mergesort")
        ax.barh(
            data["removed_feature"],
            data["delta_brier_score"],
            color=np.where(data["delta_brier_score"] >= 0, "#c0392b", "#2e8b57"),
        )
        ax.axvline(0, color=spine_color, linewidth=0.8)
        ax.set_xlabel(
            "Δ Brier score when feature is removed\n"
            "(positive = feature helped, negative = feature hurt)"
        )
        ax.set_title(title)

    fig.suptitle(
        "Leave-one-feature-out ablation on the holdout set",
        color=label_color,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fg, edgecolor=fg)
    plt.close(fig)


def main() -> None:
    data_dir = default_data_dir()
    output_dir = default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_dir}")
    modeling_df = load_modeling_frame(data_dir)
    print(
        f"Prepared modeling frame with {len(modeling_df):,} team-minute rows across "
        f"{modeling_df['kalshi_event'].nunique():,} games."
    )

    train_df, test_df = grouped_train_test_split(
        modeling_df,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
    )
    print(
        f"Train: {train_df['kalshi_event'].nunique():,} games / "
        f"{len(train_df):,} rows. "
        f"Holdout: {test_df['kalshi_event'].nunique():,} games / "
        f"{len(test_df):,} rows."
    )

    print("\n=== Leave-one-feature-out ablation (core_calibration) ===")
    core_ablation = leave_one_feature_out(
        train_df, test_df, CORE_CALIBRATION_FEATURE_COLUMNS, "core_calibration"
    )
    print(core_ablation.to_string(index=False))
    core_ablation.to_csv(output_dir / "feature_ablation_core.csv", index=False)

    print("\n=== Leave-one-feature-out ablation (extended_market) ===")
    extended_ablation = leave_one_feature_out(
        train_df, test_df, EXTENDED_FEATURE_COLUMNS, "extended_market"
    )
    print(extended_ablation.to_string(index=False))
    extended_ablation.to_csv(output_dir / "feature_ablation_extended.csv", index=False)

    plot_feature_ablation(
        core_ablation, extended_ablation, output_dir / "feature_ablation.png"
    )

    print("\n=== Stage-of-game performance (core_calibration full model) ===")
    _full_metrics, full_prob = fit_and_score(
        train_df, test_df, CORE_CALIBRATION_FEATURE_COLUMNS
    )
    predictions = test_df[["kalshi_event", "game_minute", "period", "team_won"]].copy()
    predictions["model_prob"] = full_prob
    predictions["baseline_prob"] = (
        test_df["kalshi_win_prob_pct"].astype(float).to_numpy() / 100.0
    )
    stages = stage_performance_table(predictions)
    print(stages.to_string(index=False))
    stages.to_csv(output_dir / "stage_performance.csv", index=False)
    plot_stage_performance(stages, output_dir / "stage_performance.png")

    print(f"\nArtifacts written to {output_dir}")


if __name__ == "__main__":
    main()
