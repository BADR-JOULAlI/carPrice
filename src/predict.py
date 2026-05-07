from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_path = root / "models" / "car_price_pipeline.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "Model not found. Run `python src/train.py` first to create it."
        )

    pipeline = joblib.load(model_path)

    sample = pd.DataFrame(
        [
            {
                "Brand": "Toyota",
                "Body": "sedan",
                "Mileage": 110,
                "EngineV": 2.0,
                "Engine Type": "Petrol",
                "Registration": "yes",
                "Year": 2015,
            }
        ]
    )

    pred_log = pipeline.predict(sample)[0]
    pred_price = float(np.exp(pred_log))

    print(f"Predicted price: {pred_price:.2f}")


if __name__ == "__main__":
    main()
