# Car Price Prediction (v2)

Ce projet prédit le prix de voitures d'occasion avec un pipeline scikit-learn plus robuste que la version notebook initiale.

## Améliorations implémentées

- `Pipeline` + `ColumnTransformer` pour éviter le data leakage
- `OneHotEncoder(handle_unknown="ignore")` pour les variables catégorielles
- Comparaison de plusieurs modèles:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Validation croisée (5 folds) avec métriques:
  - R2
  - MAE
  - RMSE
- Évaluation holdout test
- Sauvegarde du meilleur modèle (`joblib`)

## Structure

- `src/train.py`: entraînement, comparaison, évaluation, export
- `src/predict.py`: exemple de prédiction avec le modèle sauvegardé
- `models/car_price_pipeline.joblib`: modèle final (généré)
- `models/metrics.csv`: résultats CV + holdout (généré)
- `carPricePrediction.ipynb`: notebook original

## Installation

```bash
python -m pip install -r requirements.txt
```

## Entraîner le modèle

```bash
python src/train.py
```

## Faire une prédiction exemple

```bash
python src/predict.py
```

## Notes

- La cible `Price` est modélisée en `log(Price)` puis reconvertie avec `exp` pour les métriques/prédictions en prix réel.
- Le script nettoie les valeurs manquantes et filtre quelques valeurs invalides (`Price <= 0`, `EngineV <= 0`, `Year < 1950`).
