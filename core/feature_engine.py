"""
core/feature_engine.py
======================
Feature engineering layer for the SIH Flood Nowcasting system.

Responsibilities
----------------
- Load static geospatial grid properties from GeoJSON into a pandas DataFrame.
- Fuse dynamic rainfall scenarios (live or simulated) into the grid DataFrame,
  adding per-cell perturbed rainfall columns ready for model inference.
"""

from __future__ import annotations

import json
import os
import random
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_static_grid(
    geojson_path: str = "data/grid_cells.geojson",
) -> pd.DataFrame:
    """Load grid cell properties from a GeoJSON FeatureCollection.

    Reads every Feature's ``properties`` dict and returns them as a
    :class:`pandas.DataFrame` indexed by ``cell_id``.

    Parameters
    ----------
    geojson_path:
        Path to the GeoJSON file produced by ``scripts/bootstrap_grid.py``.

    Returns
    -------
    pd.DataFrame
        One row per grid cell, indexed by ``cell_id``.  Columns:

        * ``dist_to_river_m``       – float, metres from northern river border
        * ``elevation_m``           – float, metres above sea level
        * ``slope_deg``             – float, terrain slope in degrees
        * ``drainage_density``      – float, km/km²
        * ``impervious_surface_ratio`` – float [0, 1]
        * ``historical_flood_count``   – int [0, 5]

    Raises
    ------
    FileNotFoundError
        If ``geojson_path`` does not exist.
    ValueError
        If the file is not a valid GeoJSON FeatureCollection.
    """
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(
            f"Grid GeoJSON not found at '{geojson_path}'. "
            "Run scripts/bootstrap_grid.py first."
        )

    with open(geojson_path, "r", encoding="utf-8") as fh:
        fc = json.load(fh)

    if fc.get("type") != "FeatureCollection":
        raise ValueError(
            f"Expected GeoJSON type 'FeatureCollection', got '{fc.get('type')}'."
        )

    features = fc.get("features", [])
    if not features:
        raise ValueError("GeoJSON FeatureCollection contains no features.")

    records = []
    for feat in features:
        props = feat.get("properties", {})
        if "cell_id" not in props:
            raise ValueError("Feature missing required property 'cell_id'.")
        records.append(props)

    df = pd.DataFrame(records)
    df = df.set_index("cell_id")

    # Enforce dtypes for downstream safety
    float_cols = [
        "dist_to_river_m", "elevation_m", "slope_deg",
        "drainage_density", "impervious_surface_ratio",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    if "historical_flood_count" in df.columns:
        df["historical_flood_count"] = df["historical_flood_count"].astype(int)

    return df


def fuse_rainfall_scenario(
    df: pd.DataFrame,
    scenario_mm: float,
    noise_std_1h: float = 0.8,
    noise_std_6h: float = 2.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Fuse a rainfall scenario into the grid DataFrame.

    Perturbs ``scenario_mm`` with per-cell Gaussian noise to simulate
    spatial variability, then appends two rainfall columns to a *copy* of
    the input DataFrame.

    Parameters
    ----------
    df:
        Grid DataFrame (typically from :func:`load_static_grid`).
    scenario_mm:
        Baseline 1-hour rainfall accumulation in millimetres.
    noise_std_1h:
        Standard deviation (mm) of the per-cell 1-hour noise.  Default 0.8.
    noise_std_6h:
        Standard deviation (mm) of the per-cell 6-hour noise.  Default 2.0.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with two new columns:

        * ``rainfall_1h_mm``         – per-cell 1-hour rainfall (>= 0).
        * ``rainfall_forecast_6h_mm``– per-cell 6-hour forecast (>= 0).

    Raises
    ------
    ValueError
        If ``scenario_mm`` is negative.
    TypeError
        If ``df`` is not a :class:`pandas.DataFrame`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}.")
    if scenario_mm < 0:
        raise ValueError(f"scenario_mm must be >= 0, got {scenario_mm}.")

    rng = np.random.default_rng(seed)
    n = len(df)

    out = df.copy()

    # 1-hour: scenario + Gaussian noise, clipped to >= 0
    noise_1h = rng.normal(loc=0.0, scale=noise_std_1h, size=n)
    out["rainfall_1h_mm"] = np.clip(scenario_mm + noise_1h, 0.0, None)

    # 6-hour forecast: scenario * 4.5 + larger noise, clipped to >= 0
    base_6h = scenario_mm * 4.5
    noise_6h = rng.normal(loc=0.0, scale=noise_std_6h, size=n)
    out["rainfall_forecast_6h_mm"] = np.clip(base_6h + noise_6h, 0.0, None)

    return out
