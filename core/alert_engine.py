"""
core/alert_engine.py
====================
Multi-factor risk fusion and alert classification layer for the
SIH Flood Nowcasting system (SIH26085 — Hyderabad deployment).

UPGRADE (SIH26085)
------------------
Risk score now decomposes into five additive, deterministic factors:

    rainfall_factor          (0–35)
    elevation_factor         (0–25)
    drainage_surcharge_factor(0–20)
    impervious_factor        (0–12)
    historical_factor        (0–8)
    ──────────────────────────────
    risk_score               (0–100)

The five factors are derived from actual grid data and hydraulic state.
ML flood probability is used to modulate the rainfall and elevation factors.

ROUNDING POLICY
---------------
Each factor is rounded to 1 decimal place.
risk_score = round(sum(factors), 1).
The 5 factors are then individually scaled so their sum equals risk_score exactly.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from core.hydraulic_config import (
    ALERT_HIGH_THRESHOLD,
    ALERT_MEDIUM_THRESHOLD,
    ETA_MAX_MINUTES,
    ETA_MIN_MINUTES,
    RISK_WEIGHT_DRAINAGE,
    RISK_WEIGHT_ELEVATION,
    RISK_WEIGHT_HISTORICAL,
    RISK_WEIGHT_IMPERVIOUS,
    RISK_WEIGHT_RAINFALL,
)

log = logging.getLogger("alert-engine")


class DynamicRiskFusion:
    """Fuse ML predictions, hydraulic state, and geospatial features into
    ranked flood alerts with explainable five-factor decomposition.

    Factors
    -------
    rainfall_factor (0–35):
        (rainfall_1h_mm / 200) × RISK_WEIGHT_RAINFALL
        Modulated by ml_prob to reflect model confidence.

    elevation_factor (0–25):
        Normalised inverse elevation: lower cells score higher.
        elevation_factor = (1 - norm_elev) × RISK_WEIGHT_ELEVATION

    drainage_surcharge_factor (0–20):
        Derived from hydraulic utilisation ratio and surcharge status.
        If surcharged: utilisation contribution × RISK_WEIGHT_DRAINAGE
        Otherwise: proportional to utilisation_ratio.

    impervious_factor (0–12):
        impervious_surface_ratio × RISK_WEIGHT_IMPERVIOUS

    historical_factor (0–8):
        clip(historical_flood_count / 5, 0, 1) × RISK_WEIGHT_HISTORICAL
    """

    def evaluate(
        self,
        df: pd.DataFrame,
        ml_probs: pd.Series,
    ) -> pd.DataFrame:
        """Evaluate risk for all grid cells and return enriched alert table.

        Parameters
        ----------
        df:
            Grid DataFrame (output of HydraulicEngine.run or fuse_rainfall_scenario).
            Must contain: ``rainfall_1h_mm``, ``elevation_m``,
            ``impervious_surface_ratio``, ``historical_flood_count``.
            Optional hydraulic columns (added by HydraulicEngine):
            ``utilization_ratio``, ``surcharged``, ``surcharge_volume_m3``,
            ``depth_cm``, ``eta_minutes``, ``drainage_node_id``,
            ``drainage_flow_m3_s``, ``drainage_capacity_m3_s``.

        ml_probs:
            Per-cell flood probability [0, 1], same index as df.

        Returns
        -------
        pd.DataFrame
            Copy of df with additional columns:
            ``risk_score``, ``alert_level``, ``time_to_flood_hrs``,
            ``depth_cm``, ``eta_minutes``,
            ``rf_rainfall``, ``rf_elevation``, ``rf_drainage``,
            ``rf_impervious``, ``rf_historical``,
            ``dominant_risk_factor``.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}.")
        if not isinstance(ml_probs, pd.Series):
            raise TypeError(f"Expected pd.Series for ml_probs, got {type(ml_probs).__name__}.")
        if not df.index.equals(ml_probs.index):
            raise ValueError(
                "df and ml_probs must share the same index. "
                f"df={len(df)}, ml_probs={len(ml_probs)}."
            )

        required = ["rainfall_1h_mm", "elevation_m", "impervious_surface_ratio", "historical_flood_count"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"df missing required columns: {missing}")

        out = df.copy()
        prob = ml_probs.values.clip(0.0, 1.0)

        # ------------------------------------------------------------------
        # 1. Rainfall factor (0–35)
        # ------------------------------------------------------------------
        rain = out["rainfall_1h_mm"].values
        # Normalise to 0–1 with 200 mm/h as reference maximum
        norm_rain = np.clip(rain / 200.0, 0.0, 1.0)
        # Modulate by ML probability (0.5 base + 0.5 × ml_prob weighting)
        rain_factor = norm_rain * (0.6 + 0.4 * prob) * RISK_WEIGHT_RAINFALL
        rain_factor = np.clip(rain_factor, 0.0, RISK_WEIGHT_RAINFALL)

        # ------------------------------------------------------------------
        # 2. Elevation factor (0–25) — lower is higher risk
        # ------------------------------------------------------------------
        elev = out["elevation_m"].values
        elev_min = elev.min()
        elev_max = elev.max()
        elev_range = elev_max - elev_min
        if elev_range < 1e-6:
            norm_elev = np.zeros_like(elev)
        else:
            norm_elev = (elev - elev_min) / elev_range
        # Lower elevation → higher factor
        elev_factor = (1.0 - norm_elev) * RISK_WEIGHT_ELEVATION
        elev_factor = np.clip(elev_factor, 0.0, RISK_WEIGHT_ELEVATION)

        # ------------------------------------------------------------------
        # 3. Drainage surcharge factor (0–20)
        # ------------------------------------------------------------------
        util = out.get("utilization_ratio", pd.Series(np.zeros(len(out)), index=out.index)).values
        surcharged = out.get("surcharged", pd.Series([False] * len(out), index=out.index)).values

        # Utilisation ratio contribution (0–1 → 0–RISK_WEIGHT_DRAINAGE)
        norm_util = np.clip(util / 2.0, 0.0, 1.0)  # cap at 2× capacity for scoring
        drain_factor = norm_util * RISK_WEIGHT_DRAINAGE
        # Bonus if currently surcharged
        drain_factor = np.where(surcharged, drain_factor * 1.3, drain_factor)
        drain_factor = np.clip(drain_factor, 0.0, RISK_WEIGHT_DRAINAGE)

        # ------------------------------------------------------------------
        # 4. Impervious factor (0–12)
        # ------------------------------------------------------------------
        isr = np.clip(out["impervious_surface_ratio"].values, 0.0, 1.0)
        imp_factor = isr * RISK_WEIGHT_IMPERVIOUS

        # ------------------------------------------------------------------
        # 5. Historical factor (0–8)
        # ------------------------------------------------------------------
        hist = out["historical_flood_count"].values
        norm_hist = np.clip(hist / 5.0, 0.0, 1.0)
        hist_factor = norm_hist * RISK_WEIGHT_HISTORICAL

        # ------------------------------------------------------------------
        # 6. Total risk score
        # ------------------------------------------------------------------
        raw_score = rain_factor + elev_factor + drain_factor + imp_factor + hist_factor
        raw_score = np.clip(raw_score, 0.0, 100.0)
        risk_score = np.round(raw_score, 1)

        # Normalise factors so they sum exactly to risk_score
        # Avoids floating-point drift causing sum ≠ risk_score
        factor_sum = rain_factor + elev_factor + drain_factor + imp_factor + hist_factor
        scale = np.where(factor_sum > 1e-9, risk_score / factor_sum, 1.0)

        out["rf_rainfall"]   = np.round(rain_factor  * scale, 1)
        out["rf_elevation"]  = np.round(elev_factor  * scale, 1)
        out["rf_drainage"]   = np.round(drain_factor * scale, 1)
        out["rf_impervious"] = np.round(imp_factor   * scale, 1)
        out["rf_historical"] = np.round(hist_factor  * scale, 1)
        out["risk_score"]    = risk_score

        # ------------------------------------------------------------------
        # 7. Alert level
        # ------------------------------------------------------------------
        def _classify(score: float) -> str:
            if score >= ALERT_HIGH_THRESHOLD:
                return "HIGH"
            elif score >= ALERT_MEDIUM_THRESHOLD:
                return "MEDIUM"
            return "LOW"

        out["alert_level"] = out["risk_score"].apply(_classify)

        # ------------------------------------------------------------------
        # 8. Time to flood (hours) — backward-compatible with old API
        # ------------------------------------------------------------------
        # If HydraulicEngine already computed eta_minutes, derive hrs from it
        if "eta_minutes" in out.columns:
            out["time_to_flood_hrs"] = out.apply(
                lambda row: round(row["eta_minutes"] / 60.0, 2)
                if row["alert_level"] in ("HIGH", "MEDIUM") else None,
                axis=1,
            )
        else:
            def _ttf(row: pd.Series) -> Optional[float]:
                r1h = row["rainfall_1h_mm"]
                if row["alert_level"] == "HIGH":
                    return round(float(np.clip(100.0 / (r1h + 1), 0.5, 4.0)), 1)
                elif row["alert_level"] == "MEDIUM":
                    return round(float(np.clip(200.0 / (r1h + 1), 4.0, 12.0)), 1)
                return None
            out["time_to_flood_hrs"] = out.apply(_ttf, axis=1)

        # Ensure depth_cm and eta_minutes columns exist (may be absent if
        # HydraulicEngine was not used for this call)
        if "depth_cm" not in out.columns:
            out["depth_cm"] = 0.0
        if "eta_minutes" not in out.columns:
            out["eta_minutes"] = None

        # ------------------------------------------------------------------
        # 9. Dominant risk factor
        # ------------------------------------------------------------------
        factor_names = ["rainfall", "elevation", "drainage", "impervious", "historical"]
        factor_cols  = ["rf_rainfall", "rf_elevation", "rf_drainage", "rf_impervious", "rf_historical"]
        out["dominant_risk_factor"] = out[factor_cols].idxmax(axis=1).map(
            dict(zip(factor_cols, factor_names))
        )

        # ------------------------------------------------------------------
        # 10. Backward-compatible columns (drainage_stress for old API callers)
        # ------------------------------------------------------------------
        if "drainage_density" in out.columns:
            out["drainage_stress"] = (
                out["impervious_surface_ratio"] * out.get("rainfall_forecast_6h_mm", out["rainfall_1h_mm"] * 4.5)
            ) / (out["drainage_density"] + 0.1)
            stress_min = out["drainage_stress"].min()
            stress_max = out["drainage_stress"].max()
            out["norm_drainage_stress"] = (
                (out["drainage_stress"] - stress_min) / (stress_max - stress_min + 1e-9)
            ).clip(0.0, 1.0)
        else:
            out["drainage_stress"] = 0.0
            out["norm_drainage_stress"] = 0.0

        if "rainfall_forecast_6h_mm" not in out.columns:
            out["rainfall_forecast_6h_mm"] = out["rainfall_1h_mm"] * 4.5
        if "rainfall_1h_mm" not in out.columns:
            out["rainfall_1h_mm"] = 0.0

        return out.sort_values("risk_score", ascending=False)
