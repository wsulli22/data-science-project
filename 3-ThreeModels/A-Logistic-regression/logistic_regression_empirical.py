from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20
DEFAULT_N_SPLITS = 5

BASE_COLUMNS = [
    "kalshi_event",
    "realworld_timestamp",
    "game_elapsed_seconds",
    "period",
    "team_1",
    "team_2",
    "team_1_win_prob_pct",
    "team_2_win_prob_pct",
    "team_1_volume",
    "team_2_volume",
    "winning_team",
]

EXTENDED_FEATURE_COLUMNS = [
    "kalshi_win_prob_pct",
    "opponent_win_prob_pct",
    "game_minute",
    "elapsed_fraction",
    "quote_gap_pct",
    "abs_quote_gap_pct",
    "log_team_volume",
    "log_opponent_volume",
    "log_total_volume",
    "volume_share",
    "kalshi_prob_x_minute",
    "is_first_half",
    "is_second_half",
    "is_halftime",
    "is_overtime",
]

CORE_CALIBRATION_FEATURE_COLUMNS = [
    "kalshi_logit",
    "game_minute",
    "elapsed_fraction",
    "kalshi_logit_x_minute",
    "is_first_half",
    "is_second_half",
    "is_halftime",
    "is_overtime",
]

FEATURE_SETS = {
    "core_calibration": CORE_CALIBRATION_FEATURE_COLUMNS,
    "extended_market": EXTENDED_FEATURE_COLUMNS,
}


def default_data_dir() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "Data"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "GeneratedDataFiles" / "logistic_regression"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a grouped logistic regression model that maps Kalshi in-game quotes "
            "and game minute features to an empirical win probability."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Directory containing week_1_games.csv through week_19_games.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory to write metrics, predictions, and calibration artifacts.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of games reserved for the final holdout set.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help="Number of GroupKFold splits on the training games.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for the grouped 80/20 split.",
    )
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SETS),
        default="core_calibration",
        help="Choose the feature family used by logistic regression.",
    )
    return parser.parse_args()


def week_file_paths(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.glob("week_*_games.csv"))
    if not paths:
        raise FileNotFoundError(f"No week files found in {data_dir}")
    return paths


def normalize_period(period_series: pd.Series) -> pd.Series:
    return (
        period_series.astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": "unknown"})
    )


def explode_team_rows(week_df: pd.DataFrame) -> pd.DataFrame:
    left = week_df.rename(
        columns={
            "team_1": "team",
            "team_2": "opponent_team",
            "team_1_win_prob_pct": "kalshi_win_prob_pct",
            "team_2_win_prob_pct": "opponent_win_prob_pct",
            "team_1_volume": "team_volume",
            "team_2_volume": "opponent_volume",
        }
    )[
        [
            "kalshi_event",
            "realworld_timestamp",
            "game_elapsed_seconds",
            "period",
            "team",
            "opponent_team",
            "kalshi_win_prob_pct",
            "opponent_win_prob_pct",
            "team_volume",
            "opponent_volume",
            "winning_team",
        ]
    ]
    right = week_df.rename(
        columns={
            "team_2": "team",
            "team_1": "opponent_team",
            "team_2_win_prob_pct": "kalshi_win_prob_pct",
            "team_1_win_prob_pct": "opponent_win_prob_pct",
            "team_2_volume": "team_volume",
            "team_1_volume": "opponent_volume",
        }
    )[
        [
            "kalshi_event",
            "realworld_timestamp",
            "game_elapsed_seconds",
            "period",
            "team",
            "opponent_team",
            "kalshi_win_prob_pct",
            "opponent_win_prob_pct",
            "team_volume",
            "opponent_volume",
            "winning_team",
        ]
    ]
    out = pd.concat([left, right], ignore_index=True)
    out["team_won"] = (out["team"] == out["winning_team"]).astype(int)
    return out


def downsample_week_to_team_minutes(path: Path) -> pd.DataFrame:
    week_df = pd.read_csv(path, usecols=BASE_COLUMNS)
    week_df["realworld_timestamp"] = pd.to_datetime(week_df["realworld_timestamp"])
    team_rows = explode_team_rows(week_df)
    team_rows["period"] = normalize_period(team_rows["period"])
    team_rows["game_minute"] = (
        np.floor(team_rows["game_elapsed_seconds"].astype(float) / 60.0)
        .astype(int)
        .clip(lower=0)
    )
    team_rows = team_rows.sort_values(
        ["kalshi_event", "team", "game_minute", "realworld_timestamp"],
        kind="mergesort",
    )

    # Keep the latest quote seen in each game-minute for each team.
    minute_rows = (
        team_rows.groupby(["kalshi_event", "team", "game_minute"], sort=False)
        .last()
        .reset_index()
    )
    minute_rows["source_week_file"] = path.name
    return minute_rows


def load_modeling_frame(data_dir: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in week_file_paths(data_dir):
        print(f"Loading and downsampling {path.name}...")
        pieces.append(downsample_week_to_team_minutes(path))

    df = pd.concat(pieces, ignore_index=True)
    df["elapsed_fraction"] = (
        df["game_elapsed_seconds"].astype(float) / 2400.0
    ).clip(lower=0.0, upper=2.0)
    df["quote_gap_pct"] = (
        df["kalshi_win_prob_pct"].astype(float) - df["opponent_win_prob_pct"].astype(float)
    )
    df["abs_quote_gap_pct"] = df["quote_gap_pct"].abs()
    df["log_team_volume"] = np.log1p(df["team_volume"].astype(float))
    df["log_opponent_volume"] = np.log1p(df["opponent_volume"].astype(float))
    total_volume = df["team_volume"].astype(float) + df["opponent_volume"].astype(float)
    df["log_total_volume"] = np.log1p(total_volume)
    df["volume_share"] = np.divide(
        df["team_volume"].astype(float),
        total_volume,
        out=np.full(len(df), 0.5, dtype=float),
        where=total_volume.to_numpy() > 0,
    )
    df["kalshi_prob_x_minute"] = (
        df["kalshi_win_prob_pct"].astype(float) * df["game_minute"].astype(float)
    )
    clipped_prob = np.clip(df["kalshi_win_prob_pct"].astype(float).to_numpy() / 100.0, 1e-4, 1 - 1e-4)
    df["kalshi_logit"] = np.log(clipped_prob / (1.0 - clipped_prob))
    df["kalshi_logit_x_minute"] = df["kalshi_logit"] * df["game_minute"].astype(float)
    df["is_first_half"] = df["period"].eq("firsthalf").astype(int)
    df["is_second_half"] = df["period"].eq("secondhalf").astype(int)
    df["is_halftime"] = df["period"].eq("halftime").astype(int)
    df["is_overtime"] = df["period"].str.contains("overtime", regex=False).astype(int)
    return df


def build_pipeline(feature_columns: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_columns,
            )
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def compute_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def add_baseline_metrics(metrics: dict[str, float], y_true: pd.Series, baseline_prob: np.ndarray) -> dict[str, float]:
    updated = dict(metrics)
    baseline = compute_metrics(y_true, baseline_prob)
    for name, value in baseline.items():
        updated[f"baseline_kalshi_{name}"] = value
    return updated


def calibration_table(y_true: pd.Series, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    bucket = pd.cut(
        probabilities,
        bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True,
        duplicates="drop",
    )
    table = (
        pd.DataFrame({"predicted_prob": probabilities, "actual": y_true, "bucket": bucket})
        .groupby("bucket", observed=False)
        .agg(
            sample_count=("actual", "size"),
            mean_predicted_prob=("predicted_prob", "mean"),
            empirical_win_rate=("actual", "mean"),
        )
        .reset_index()
    )
    table["bucket"] = table["bucket"].astype(str)
    return table


def save_calibration_plot(
    model_calibration: pd.DataFrame,
    baseline_calibration: pd.DataFrame,
    output_path: Path,
) -> None:
    fg = "#ffffff"
    label_color = "#1a1a1a"
    grid_color = "#c8c8c8"
    spine_color = "#4a4a4a"
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=fg)
    ax.set_facecolor(fg)
    for spine in ax.spines.values():
        spine.set_color(spine_color)
    ax.tick_params(colors=label_color)
    ax.xaxis.label.set_color(label_color)
    ax.yaxis.label.set_color(label_color)
    ax.title.set_color(label_color)
    ax.grid(True, linestyle=":", linewidth=0.7, color=grid_color, alpha=0.95)
    ax.set_axisbelow(True)

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#888888",
        linewidth=1,
        label="Perfect calibration",
    )
    ax.plot(
        model_calibration["mean_predicted_prob"],
        model_calibration["empirical_win_rate"],
        marker="o",
        color="#1f77b4",
        markerfacecolor="#1f77b4",
        markeredgecolor="#0d3d5c",
        markeredgewidth=0.6,
        label="Logistic regression",
    )
    ax.plot(
        baseline_calibration["mean_predicted_prob"],
        baseline_calibration["empirical_win_rate"],
        marker="o",
        color="#d97706",
        markerfacecolor="#d97706",
        markeredgecolor="#7c3d00",
        markeredgewidth=0.6,
        label="Raw Kalshi quote",
    )
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Empirical win rate")
    ax.set_title("Holdout calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    legend = ax.legend(facecolor="#f6f6f6", edgecolor=spine_color)
    for text in legend.get_texts():
        text.set_color(label_color)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fg, edgecolor=fg)
    plt.close(fig)


def grouped_train_test_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(df, groups=df["kalshi_event"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def cross_validate_training_set(
    train_df: pd.DataFrame,
    n_splits: int,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_groups = train_df["kalshi_event"].nunique()
    actual_splits = min(n_splits, n_groups)
    if actual_splits < 2:
        raise ValueError("Need at least two games in the training set for GroupKFold.")

    X = train_df[feature_columns]
    y = train_df["team_won"].astype(int)
    groups = train_df["kalshi_event"]
    baseline_prob = train_df["kalshi_win_prob_pct"].astype(float).to_numpy() / 100.0

    oof_pred = np.zeros(len(train_df), dtype=float)
    fold_rows: list[dict[str, float | int]] = []
    splitter = GroupKFold(n_splits=actual_splits)
    for fold_number, (fit_idx, val_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        model = build_pipeline(feature_columns)
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        val_prob = model.predict_proba(X.iloc[val_idx])[:, 1]
        oof_pred[val_idx] = val_prob

        fold_metrics = compute_metrics(y.iloc[val_idx], val_prob)
        baseline_metrics = compute_metrics(y.iloc[val_idx], baseline_prob[val_idx])
        fold_rows.append(
            {
                "fold": fold_number,
                "n_train_rows": int(len(fit_idx)),
                "n_val_rows": int(len(val_idx)),
                "n_val_games": int(train_df.iloc[val_idx]["kalshi_event"].nunique()),
                **fold_metrics,
                **{f"baseline_kalshi_{k}": v for k, v in baseline_metrics.items()},
            }
        )

    summary = add_baseline_metrics(
        compute_metrics(y, oof_pred),
        y,
        baseline_prob,
    )
    summary["n_training_rows"] = int(len(train_df))
    summary["n_training_games"] = int(n_groups)
    return pd.DataFrame(fold_rows), pd.DataFrame([summary])


def fit_final_model(train_df: pd.DataFrame, feature_columns: list[str]) -> Pipeline:
    model = build_pipeline(feature_columns)
    model.fit(train_df[feature_columns], train_df["team_won"].astype(int))
    return model


def coefficient_table(model: Pipeline, feature_columns: list[str]) -> pd.DataFrame:
    logistic = model.named_steps["model"]
    coefficients = logistic.coef_[0]
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
        }
    ).sort_values("coefficient", ascending=False, kind="mergesort")


def evaluate_holdout(
    model: Pipeline,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_test = test_df[feature_columns]
    y_test = test_df["team_won"].astype(int)
    test_prob = model.predict_proba(X_test)[:, 1]
    baseline_prob = test_df["kalshi_win_prob_pct"].astype(float).to_numpy() / 100.0

    metrics = add_baseline_metrics(compute_metrics(y_test, test_prob), y_test, baseline_prob)
    metrics["n_test_rows"] = int(len(test_df))
    metrics["n_test_games"] = int(test_df["kalshi_event"].nunique())

    predictions = test_df[
        [
            "kalshi_event",
            "team",
            "opponent_team",
            "game_minute",
            "period",
            "kalshi_win_prob_pct",
            "opponent_win_prob_pct",
            "team_won",
        ]
    ].copy()
    predictions["predicted_empirical_win_prob"] = test_prob
    predictions["baseline_kalshi_prob"] = baseline_prob
    return pd.DataFrame([metrics]), predictions


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = FEATURE_SETS[args.feature_set]

    modeling_df = load_modeling_frame(args.data_dir)
    print(
        "Prepared modeling frame with "
        f"{len(modeling_df):,} team-minute rows across "
        f"{modeling_df['kalshi_event'].nunique():,} games."
    )

    train_df, test_df = grouped_train_test_split(
        modeling_df,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(
        "Grouped split complete: "
        f"{train_df['kalshi_event'].nunique():,} train games, "
        f"{test_df['kalshi_event'].nunique():,} test games."
    )

    cv_fold_metrics, cv_summary = cross_validate_training_set(
        train_df,
        n_splits=args.n_splits,
        feature_columns=feature_columns,
    )
    final_model = fit_final_model(train_df, feature_columns=feature_columns)
    test_metrics, test_predictions = evaluate_holdout(
        final_model,
        test_df,
        feature_columns=feature_columns,
    )
    coefficients = coefficient_table(final_model, feature_columns=feature_columns)

    model_calibration = calibration_table(
        test_predictions["team_won"].astype(int),
        test_predictions["predicted_empirical_win_prob"].to_numpy(),
    )
    baseline_calibration = calibration_table(
        test_predictions["team_won"].astype(int),
        test_predictions["baseline_kalshi_prob"].to_numpy(),
    )

    cv_fold_metrics.to_csv(args.output_dir / "cv_fold_metrics.csv", index=False)
    cv_summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    test_metrics.to_csv(args.output_dir / "test_metrics.csv", index=False)
    test_predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    coefficients.to_csv(args.output_dir / "model_coefficients.csv", index=False)
    model_calibration.to_csv(args.output_dir / "model_calibration.csv", index=False)
    baseline_calibration.to_csv(args.output_dir / "baseline_kalshi_calibration.csv", index=False)
    save_calibration_plot(
        model_calibration,
        baseline_calibration,
        args.output_dir / "holdout_calibration.png",
    )

    metrics_payload = {
        "feature_set": args.feature_set,
        "cross_validation": cv_summary.iloc[0].to_dict(),
        "holdout_test": test_metrics.iloc[0].to_dict(),
    }
    with open(args.output_dir / "metrics_summary.json", "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)

    print("\nCross-validation summary:")
    print(cv_summary.to_string(index=False))
    print("\nHoldout test summary:")
    print(test_metrics.to_string(index=False))
    print(f"\nArtifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
