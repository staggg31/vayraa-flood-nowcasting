"""
core/model.py
=============
Machine-learning inference layer for the SIH Flood Nowcasting system.

``FloodRiskEngine`` wraps an XGBoost (or Random Forest fallback) classifier
behind a clean ``fit_or_load`` / ``predict_proba`` interface, keeping the
rest of the pipeline decoupled from any specific ML library.
"""

from __future__ import annotations

import os
import warnings
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# Feature columns expected by the model (order matters for prediction)
FEATURE_COLS: List[str] = [
    "elevation_m",
    "slope_deg",
    "dist_to_river_m",
    "drainage_density",
    "impervious_surface_ratio",
    "rainfall_1h_mm",
]
TARGET_COL: str = "flood_occurred"


# ---------------------------------------------------------------------------
# Helper: try importing XGBoost, fall back gracefully
# ---------------------------------------------------------------------------
def _build_classifier(class_weights: dict) -> object:
    """Return an XGBClassifier if xgboost is installed, else RandomForest."""
    try:
        from xgboost import XGBClassifier  # type: ignore

        scale_pos_weight = (
            class_weights.get(0, 1.0) / class_weights.get(1, 1.0)
            if 1 in class_weights
            else 1.0
        )
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        print("  [model] Backend: XGBClassifier")
        return clf
    except ImportError:
        warnings.warn(
            "xgboost not installed — falling back to RandomForestClassifier.",
            UserWarning,
            stacklevel=3,
        )
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        print("  [model] Backend: RandomForestClassifier (xgboost not found)")
        return clf


# ---------------------------------------------------------------------------
# FloodRiskEngine
# ---------------------------------------------------------------------------

class FloodRiskEngine:
    """Flood probability predictor built on a gradient-boosted tree classifier.

    Usage
    -----
    >>> engine = FloodRiskEngine()
    >>> engine.fit_or_load()
    >>> probs = engine.predict_proba(df_with_features)

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline or None
        The fitted ``StandardScaler`` + classifier pipeline.  ``None`` until
        :meth:`fit_or_load` is called.
    model_path : str
        The path where the serialised model is saved / loaded from.
    """

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.model_path: str = ""

    # ------------------------------------------------------------------
    # fit_or_load
    # ------------------------------------------------------------------
    def fit_or_load(
        self,
        train_csv: str = "data/synthetic_train.csv",
        model_path: str = "core/flood_model.pkl",
    ) -> "FloodRiskEngine":
        """Load an existing model or train a new one from the training CSV.

        Parameters
        ----------
        train_csv:
            Path to the CSV produced by ``scripts/bootstrap_grid.py``.
            Required only when no pre-trained model exists at ``model_path``.
        model_path:
            Joblib serialisation path for the trained pipeline.

        Returns
        -------
        FloodRiskEngine
            Self (enables method chaining).

        Raises
        ------
        FileNotFoundError
            If ``train_csv`` does not exist and no model is saved.
        ValueError
            If the CSV is missing expected feature or target columns.
        """
        self.model_path = model_path

        if os.path.exists(model_path):
            print(f"  [model] Loading cached model from '{model_path}' ...")
            self.pipeline = joblib.load(model_path)
            print("  [model] Model loaded OK.")
            return self

        # --- Train from scratch ---
        print(f"  [model] No cached model found. Training from '{train_csv}' ...")

        if not os.path.exists(train_csv):
            raise FileNotFoundError(
                f"Training CSV not found at '{train_csv}'. "
                "Run scripts/bootstrap_grid.py first."
            )

        df = pd.read_csv(train_csv)

        missing_feats = [c for c in FEATURE_COLS if c not in df.columns]
        if missing_feats:
            raise ValueError(
                f"Training CSV missing feature columns: {missing_feats}"
            )
        if TARGET_COL not in df.columns:
            raise ValueError(
                f"Training CSV missing target column '{TARGET_COL}'."
            )

        X = df[FEATURE_COLS].values
        y = df[TARGET_COL].values

        # Compute class weights to handle imbalance
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        class_weight_map = dict(zip(classes.tolist(), weights.tolist()))

        # Optional train/val split for a quick sanity-check log
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

        clf = _build_classifier(class_weight_map)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

        pipeline.fit(X_tr, y_tr)

        # Validation accuracy (informational)
        val_acc = pipeline.score(X_val, y_val)
        val_preds = pipeline.predict(X_val)
        n_pos_pred = int(val_preds.sum())
        print(
            f"  [model] Validation accuracy: {val_acc:.4f} "
            f"| Positive predictions on val set: {n_pos_pred}/{len(y_val)}"
        )

        # Persist
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        joblib.dump(pipeline, model_path)
        self.pipeline = pipeline
        print(f"  [model] Model saved to '{model_path}'.")

        return self

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Return flood occurrence probability for each row in ``df``.

        Parameters
        ----------
        df:
            DataFrame containing at least the columns in ``FEATURE_COLS``.
            Extra columns are silently ignored.

        Returns
        -------
        pd.Series
            Float values in [0.0, 1.0], sharing the same index as ``df``.

        Raises
        ------
        RuntimeError
            If :meth:`fit_or_load` has not been called.
        ValueError
            If ``df`` is missing any required feature columns.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "Model not initialised. Call fit_or_load() before predict_proba()."
            )

        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame missing feature columns required for inference: {missing}"
            )

        X = df[FEATURE_COLS].values
        proba_matrix = self.pipeline.predict_proba(X)

        # Robustly pick the positive-class column
        clf_step = self.pipeline.named_steps["clf"]
        classes = list(getattr(clf_step, "classes_", [0, 1]))
        pos_idx = classes.index(1) if 1 in classes else 1

        return pd.Series(proba_matrix[:, pos_idx], index=df.index, name="flood_prob")
