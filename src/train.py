from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["Mileage", "EngineV", "Year"]
CATEGORICAL_FEATURES = ["Brand", "Body", "Engine Type", "Registration"]
TARGET = "Price"
DROP_COLUMNS = ["Model"]


@dataclass
class TrainArtifacts:
    best_model_name: str
    best_pipeline: Pipeline
    metrics: pd.DataFrame


def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.replace("\ufeff", "", regex=False)

    for col in DROP_COLUMNS:
        if col in data.columns:
            data = data.drop(columns=col)

    data = data.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]).copy()

    # Keep rows compatible with log target and reasonable domain values.
    data = data[(data[TARGET] > 0) & (data["EngineV"] > 0) & (data["Year"] >= 1950)]

    return data


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipe = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def build_model_candidates() -> Dict[str, object]:
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }


def evaluate_models(x: pd.DataFrame, y_log: pd.Series) -> pd.DataFrame:
    preprocessor = build_preprocessor()
    candidates = build_model_candidates()

    rows = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in candidates.items():
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        scores = cross_validate(
            pipeline,
            x,
            y_log,
            cv=cv,
            scoring={
                "r2": "r2",
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
            },
            n_jobs=-1,
        )
        rows.append(
            {
                "model": name,
                "cv_r2_mean": np.mean(scores["test_r2"]),
                "cv_mae_mean": -np.mean(scores["test_mae"]),
                "cv_rmse_mean": -np.mean(scores["test_rmse"]),
            }
        )

    return pd.DataFrame(rows).sort_values("cv_rmse_mean", ascending=True).reset_index(drop=True)


def fit_best_model(data: pd.DataFrame) -> TrainArtifacts:
    x = data.drop(columns=[TARGET])
    y = data[TARGET]
    y_log = np.log(y)

    metrics = evaluate_models(x, y_log)
    best_model_name = metrics.iloc[0]["model"]

    candidates = build_model_candidates()
    best_model = candidates[best_model_name]
    pipeline = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", best_model)])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_log, test_size=0.2, random_state=42
    )

    pipeline.fit(x_train, y_train)
    y_pred_log = pipeline.predict(x_test)

    y_true = np.exp(y_test)
    y_pred = np.exp(y_pred_log)

    holdout = {
        "model": f"{best_model_name} (holdout)",
        "cv_r2_mean": r2_score(y_true, y_pred),
        "cv_mae_mean": mean_absolute_error(y_true, y_pred),
        "cv_rmse_mean": mean_squared_error(y_true, y_pred) ** 0.5,
    }
    metrics = pd.concat([metrics, pd.DataFrame([holdout])], ignore_index=True)

    return TrainArtifacts(best_model_name=best_model_name, best_pipeline=pipeline, metrics=metrics)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "carPrice.csv"
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_clean_data(csv_path)
    artifacts = fit_best_model(data)

    model_path = models_dir / "car_price_pipeline.joblib"
    metrics_path = models_dir / "metrics.csv"

    joblib.dump(artifacts.best_pipeline, model_path)
    artifacts.metrics.to_csv(metrics_path, index=False)

    print("Training complete.")
    print(f"Best model: {artifacts.best_model_name}")
    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print("\nMetrics:")
    print(artifacts.metrics.to_string(index=False))


if __name__ == "__main__":
    main()
