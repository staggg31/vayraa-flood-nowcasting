"""
core/alert_engine.py
====================
Multi-factor risk fusion and alert classification layer for the
SIH Flood Nowcasting system.

``DynamicRiskFusion`` combines ML flood probabilities with physics-based
drainage stress and historical flood exposure into a single actionable
risk score, then classifies each grid cell into HIGH / MEDIUM / LOW alerts
with an estimated time-to-flood.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class DynamicRiskFusion:
    """Fuse ML predictions and physical signals into ranked flood alerts.

    Risk score formula
    ------------------
    ::

        drainage_stress  = (ISR * rainfall_6h) / (drainage_density + 0.1)
        norm_stress      = (stress - min) / (max - min + eps)   # [0, 1]
        historical_norm  = clip(historical_flood_count / 5, 0, 1)

        risk_score = (
            ml_prob        * 0.45 +
            norm_stress    * 0.35 +
            hist_norm      * 0.20
        ) * 100                     # scaled to [0, 100]

    Alert thresholds
    ----------------
    * **HIGH**   – ``risk_score >= 70``
    * **MEDIUM** – ``40 <= risk_score < 70``
    * **LOW**    – ``risk_score < 40``

    Time-to-flood estimates
    -----------------------
    * HIGH   → ``clip(100  / (rainfall_1h + 1), 0.5,  4.0)`` hours
    * MEDIUM → ``clip(200  / (rainfall_1h + 1), 4.0, 12.0)`` hours
    * LOW    → ``None``
    """

    # Blend weights (must sum to 1.0)
    _W_ML   = 0.45
    _W_DRAIN = 0.35
    _W_HIST  = 0.20

    # Alert thresholds
    _HIGH_THRESH   = 70.0
    _MEDIUM_THRESH = 40.0

    def evaluate(
        self,
        df: pd.DataFrame,
        ml_probs: pd.Series,
    ) -> pd.DataFrame:
        """Evaluate risk for all grid cells and return a ranked alert table.

        Parameters
        ----------
        df:
            Grid DataFrame enriched with rainfall columns (output of
            :func:`core.feature_engine.fuse_rainfall_scenario`).  Must contain:

            * ``impervious_surface_ratio``
            * ``rainfall_forecast_6h_mm``
            * ``drainage_density``
            * ``historical_flood_count``
            * ``rainfall_1h_mm``

        ml_probs:
            Per-cell flood probability in [0, 1] (output of
            :meth:`core.model.FloodRiskEngine.predict_proba`).
            Must share the same index as ``df``.

        Returns
        -------
        pd.DataFrame
            Copy of ``df`` with five additional columns, sorted by
            ``risk_score`` descending:

            * ``drainage_stress``   – raw drainage stress value
            * ``norm_drainage_stress`` – normalised to [0, 1]
            * ``risk_score``        – blended score in [0, 100]
            * ``alert_level``       – 'HIGH' | 'MEDIUM' | 'LOW'
            * ``time_to_flood_hrs`` – float or None

        Raises
        ------
        TypeError
            If ``df`` is not a :class:`pandas.DataFrame` or ``ml_probs`` is
            not a :class:`pandas.Series`.
        ValueError
            If required columns are missing from ``df``, or if ``df`` and
            ``ml_probs`` do not share the same index.
        """
        # --- Input validation ---
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame for df, got {type(df).__name__}.")
        if not isinstance(ml_probs, pd.Series):
            raise TypeError(
                f"Expected pd.Series for ml_probs, got {type(ml_probs).__name__}."
            )

        required_cols = [
            "impervious_surface_ratio",
            "rainfall_forecast_6h_mm",
            "drainage_density",
            "historical_flood_count",
            "rainfall_1h_mm",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"df is missing required columns: {missing}")

        if not df.index.equals(ml_probs.index):
            raise ValueError(
                "df and ml_probs must share the same index. "
                f"df index size={len(df)}, ml_probs index size={len(ml_probs)}."
            )

        out = df.copy()

        # ------------------------------------------------------------------
        # 1. Drainage stress
        # ------------------------------------------------------------------
        out["drainage_stress"] = (
            out["impervious_surface_ratio"] * out["rainfall_forecast_6h_mm"]
        ) / (out["drainage_density"] + 0.1)

        # Normalise to [0, 1]
        stress_min = out["drainage_stress"].min()
        stress_max = out["drainage_stress"].max()
        eps = 1e-9
        out["norm_drainage_stress"] = (
            (out["drainage_stress"] - stress_min) / (stress_max - stress_min + eps)
        ).clip(0.0, 1.0)

        # ------------------------------------------------------------------
        # 2. Historical flood component (normalised to [0, 1])
        # ------------------------------------------------------------------
        hist_norm = (out["historical_flood_count"] / 5.0).clip(0.0, 1.0)

        # ------------------------------------------------------------------
        # 3. Blended risk score [0, 100]
        # ------------------------------------------------------------------
        out["risk_score"] = (
            (ml_probs.values          * self._W_ML)
            + (out["norm_drainage_stress"].values * self._W_DRAIN)
            + (hist_norm.values        * self._W_HIST)
        ) * 100

        out["risk_score"] = out["risk_score"].round(1)

        # ------------------------------------------------------------------
        # 4. Alert level classification
        # ------------------------------------------------------------------
        def _classify(score: float) -> str:
            if score >= self._HIGH_THRESH:
                return "HIGH"
            elif score >= self._MEDIUM_THRESH:
                return "MEDIUM"
            return "LOW"

        out["alert_level"] = out["risk_score"].apply(_classify)

        # ------------------------------------------------------------------
        # 5. Time-to-flood estimate
        # ------------------------------------------------------------------
        def _time_to_flood(row: pd.Series) -> Optional[float]:
            r1h = row["rainfall_1h_mm"]
            if row["alert_level"] == "HIGH":
                return round(float(np.clip(100.0 / (r1h + 1), 0.5, 4.0)), 1)
            elif row["alert_level"] == "MEDIUM":
                return round(float(np.clip(200.0 / (r1h + 1), 4.0, 12.0)), 1)
            return None

        out["time_to_flood_hrs"] = out.apply(_time_to_flood, axis=1)

        # ------------------------------------------------------------------
        # 6. Sort by risk descending
        # ------------------------------------------------------------------
        return out.sort_values("risk_score", ascending=False)
