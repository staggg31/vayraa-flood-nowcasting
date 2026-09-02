"""
server/api.py
=============
Production-grade FastAPI server for the SIH Flood Nowcasting system.

Endpoints
---------
GET  /api/v1/health     — Liveness probe + model/grid readiness check.
GET  /api/v1/grid       — Return the raw GeoJSON FeatureCollection.
POST /api/v1/simulate   — Run a full rainfall scenario through the ML pipeline
                          and return ranked risk alerts + enriched GeoJSON.

Architecture
------------
- Single-load pattern: the static grid, trained model, and fusion engine are
  loaded once at startup via the lifespan context and stored in `app.state`.
- CORS is fully open so Streamlit (or any local client) can query the API
  without network-level blocks.
- Pydantic v2 models are used for request/response validation.
- All prediction logic is wrapped in try/except blocks returning HTTP 500 with
  descriptive messages on failure.
"""

from __future__ import annotations

import json
import logging
import traceback
from contextlib import asynccontextmanager
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.alert_engine import DynamicRiskFusion
from core.feature_engine import fuse_rainfall_scenario, load_static_grid
from core.model import FloodRiskEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("flood-api")

# ---------------------------------------------------------------------------
# Paths (relative to project root; server is launched from project root)
# ---------------------------------------------------------------------------
GRID_PATH  = "data/grid_cells.geojson"
TRAIN_CSV  = "data/synthetic_train.csv"
MODEL_PATH = "core/flood_model.pkl"

# ---------------------------------------------------------------------------
# Lifespan: single-load initialisation
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once at startup; release on shutdown."""
    log.info("=== SIH Flood API — Startup ===")

    # 1. Static grid
    log.info("Loading static grid from '%s' …", GRID_PATH)
    app.state.grid_df = load_static_grid(GRID_PATH)
    log.info("Grid loaded: %d cells.", len(app.state.grid_df))

    # 2. Raw GeoJSON (cached for /api/v1/grid endpoint)
    with open(GRID_PATH, "r", encoding="utf-8") as fh:
        app.state.raw_geojson = json.load(fh)

    # 3. ML model
    log.info("Initialising FloodRiskEngine …")
    engine = FloodRiskEngine()
    engine.fit_or_load(train_csv=TRAIN_CSV, model_path=MODEL_PATH)
    app.state.engine = engine
    log.info("Model ready.")

    # 4. Risk fusion engine
    app.state.fusion = DynamicRiskFusion()
    log.info("DynamicRiskFusion initialised.")

    log.info("=== Startup complete — API is serving ===")
    yield
    # --- Shutdown ---
    log.info("=== SIH Flood API — Shutdown ===")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SIH Flood Nowcasting API",
    description=(
        "Real-time flood risk nowcasting for the Patna Ganges sector. "
        "Combines XGBoost ML predictions with physics-based drainage stress "
        "fusion to generate ranked flood alerts and enriched GeoJSON outputs."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — open for Streamlit / local browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class SimulateRequest(BaseModel):
    """Rainfall scenario to simulate."""

    rainfall_scenario_mm: float = Field(
        default=65.0,
        ge=0.0,
        le=250.0,
        description="1-hour rainfall accumulation in millimetres (0 – 250 mm).",
    )


class AlertRecord(BaseModel):
    """A single HIGH-risk grid cell alert."""

    cell_id: str
    risk_score: float
    alert_level: str
    time_to_flood_hrs: Optional[float]
    dist_to_river_m: float
    elevation_m: float
    rainfall_1h_mm: float
    rainfall_forecast_6h_mm: float
    drainage_stress: float


class SimulateResponse(BaseModel):
    """Full simulate endpoint response."""

    status: str
    scenario_intensity_mm: float
    total_cells: int
    critical_cells_count: int
    medium_cells_count: int
    alerts: list[AlertRecord]
    grid_geojson: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_enriched_geojson(
    raw_geojson: dict,
    result_df: pd.DataFrame,
) -> dict:
    """Return a GeoJSON FeatureCollection with risk metrics merged into each
    feature's ``properties`` dict.

    Parameters
    ----------
    raw_geojson:
        The original FeatureCollection loaded from disk.
    result_df:
        Output of :meth:`DynamicRiskFusion.evaluate`, indexed by ``cell_id``.

    Returns
    -------
    dict
        A new FeatureCollection with ``risk_score``, ``alert_level``,
        ``time_to_flood_hrs``, and ``rainfall_1h_mm`` injected per feature.
    """
    # Build a fast lookup: cell_id -> risk row
    risk_lookup: dict[str, dict] = result_df[[
        "risk_score", "alert_level", "time_to_flood_hrs",
        "rainfall_1h_mm", "rainfall_forecast_6h_mm",
        "norm_drainage_stress", "drainage_stress",
    ]].to_dict(orient="index")

    enriched_features = []
    for feat in raw_geojson.get("features", []):
        cell_id = feat["properties"].get("cell_id")
        new_feat = {
            "type": "Feature",
            "geometry": feat["geometry"],
            "properties": {
                **feat["properties"],
                **{
                    k: (None if pd.isna(v) else v)
                    for k, v in risk_lookup.get(cell_id, {}).items()
                },
            },
        }
        enriched_features.append(new_feat)

    return {
        "type": "FeatureCollection",
        "name": raw_geojson.get("name", "patna_flood_grid"),
        "crs": raw_geojson.get("crs", {}),
        "features": enriched_features,
    }


def _series_to_alert_records(
    result_df: pd.DataFrame,
) -> list[AlertRecord]:
    """Extract HIGH-risk rows from the result DataFrame as AlertRecord objects."""
    high_df = result_df[result_df["alert_level"].isin(["HIGH", "MEDIUM"])]
    records: list[AlertRecord] = []
    for cell_id, row in high_df.iterrows():
        records.append(AlertRecord(
            cell_id=str(cell_id),
            risk_score=round(float(row["risk_score"]), 1),
            alert_level=str(row["alert_level"]),
            time_to_flood_hrs=(
                None if pd.isna(row["time_to_flood_hrs"])
                else float(row["time_to_flood_hrs"])
            ),
            dist_to_river_m=float(row["dist_to_river_m"]),
            elevation_m=float(row["elevation_m"]),
            rainfall_1h_mm=round(float(row["rainfall_1h_mm"]), 3),
            rainfall_forecast_6h_mm=round(float(row["rainfall_forecast_6h_mm"]), 3),
            drainage_stress=round(float(row["drainage_stress"]), 4),
        ))
    return records


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/health",
    summary="Liveness and readiness probe",
    tags=["System"],
)
async def health() -> dict:
    """Return API status, model readiness, and number of loaded grid cells.

    Returns
    -------
    dict
        ``{"status": "online", "model_ready": bool, "cells_loaded": int}``
    """
    model_ready = (
        hasattr(app.state, "engine")
        and app.state.engine.pipeline is not None
    )
    cells_loaded = (
        len(app.state.grid_df)
        if hasattr(app.state, "grid_df")
        else 0
    )
    return {
        "status": "online",
        "model_ready": model_ready,
        "cells_loaded": cells_loaded,
    }


@app.get(
    "/api/v1/grid",
    summary="Raw GeoJSON grid",
    tags=["Data"],
)
async def get_grid() -> JSONResponse:
    """Return the cached GeoJSON FeatureCollection for the Patna flood grid.

    The GeoJSON is loaded from disk once at startup and served from memory
    for every subsequent request.
    """
    if not hasattr(app.state, "raw_geojson"):
        raise HTTPException(
            status_code=503,
            detail="Grid data not yet available. Server may still be starting up.",
        )
    return JSONResponse(content=app.state.raw_geojson)


@app.post(
    "/api/v1/simulate",
    response_model=SimulateResponse,
    summary="Run a rainfall scenario through the ML pipeline",
    tags=["Simulation"],
)
async def simulate(request: SimulateRequest) -> SimulateResponse:
    """Execute a full flood nowcasting pipeline for a given rainfall scenario.

    Steps
    -----
    1. Fuse ``rainfall_scenario_mm`` into the in-memory grid.
    2. Predict flood probability per cell (XGBoost / RF).
    3. Compute dynamic risk fusion scores and alert levels.
    4. Extract HIGH-risk alerts.
    5. Build and return an enriched GeoJSON FeatureCollection.

    Parameters
    ----------
    request:
        JSON body with ``rainfall_scenario_mm`` (0 – 250 mm).

    Returns
    -------
    SimulateResponse
        Full response including alert list and enriched GeoJSON.

    Raises
    ------
    HTTPException(503):
        If the model or grid has not been initialised yet.
    HTTPException(500):
        If any computation step fails unexpectedly.
    """
    # Guard: ensure startup completed
    if not hasattr(app.state, "engine") or not hasattr(app.state, "grid_df"):
        raise HTTPException(
            status_code=503,
            detail="Server is still initialising. Try again in a moment.",
        )

    scenario_mm = request.rainfall_scenario_mm
    log.info("Simulate request: rainfall=%.1f mm", scenario_mm)

    try:
        # Step 1: Feature fusion
        df_fused = fuse_rainfall_scenario(
            app.state.grid_df, scenario_mm=scenario_mm
        )
    except Exception as exc:
        log.error("Feature fusion failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Feature fusion error: {exc}",
        ) from exc

    try:
        # Step 2: ML prediction
        ml_probs = app.state.engine.predict_proba(df_fused)
    except Exception as exc:
        log.error("Model prediction failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction error: {exc}",
        ) from exc

    try:
        # Step 3: Dynamic risk fusion
        result_df = app.state.fusion.evaluate(df_fused, ml_probs)
        
        import numpy as np
        
        # 1. Overwrite rainfall
        result_df["rainfall_1h_mm"] = scenario_mm
        
        # Ensure ml_probs is aligned (assuming 1D or 2nd col of 2D)
        probs = np.array(ml_probs)
        if probs.ndim == 2:
            probs = probs[:, 1]
            
        # 2. Dynamic Risk Formula
        base_prob = probs * 50.0
        rain_factor = min(50.0, (scenario_mm / 200.0) * 50.0)
        river_factor = np.maximum(0.0, (5000.0 - result_df["dist_to_river_m"]) / 5000.0) * 20.0
        elev_factor = np.maximum(0.0, (60.0 - result_df["elevation_m"]) / 60.0) * 15.0
        
        risk_score = base_prob + rain_factor + river_factor + elev_factor
        result_df["risk_score"] = np.clip(risk_score, 0, 100)
        
        # 3. Categorize cells
        conditions = [
            (result_df["risk_score"] >= 70),
            (result_df["risk_score"] >= 40) & (result_df["risk_score"] < 70)
        ]
        choices = ["HIGH", "MEDIUM"]
        result_df["alert_level"] = np.select(conditions, choices, default="LOW")
        
        # 4. Estimate time_to_flood_hrs
        high_ttf = round(max(0.5, 6.0 - (scenario_mm / 50.0)), 1)
        med_ttf = round(max(2.0, 10.0 - (scenario_mm / 40.0)), 1)
        
        ttf_conditions = [
            result_df["alert_level"] == "HIGH",
            result_df["alert_level"] == "MEDIUM"
        ]
        ttf_choices = [high_ttf, med_ttf]
        result_df["time_to_flood_hrs"] = np.select(ttf_conditions, ttf_choices, default=np.nan)

    except Exception as exc:
        log.error("Risk fusion failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Risk fusion error: {exc}",
        ) from exc

    try:
        # Step 4: Extract alert records
        alert_records = _series_to_alert_records(result_df)

        # Step 5: Enrich GeoJSON
        enriched_geojson = _build_enriched_geojson(
            app.state.raw_geojson, result_df
        )

        critical_count = int((result_df["alert_level"] == "HIGH").sum())
        medium_count   = int((result_df["alert_level"] == "MEDIUM").sum())

        log.info(
            "Simulate complete: HIGH=%d, MEDIUM=%d, LOW=%d",
            critical_count,
            medium_count,
            int((result_df["alert_level"] == "LOW").sum()),
        )

        return SimulateResponse(
            status="success",
            scenario_intensity_mm=scenario_mm,
            total_cells=len(result_df),
            critical_cells_count=critical_count,
            medium_cells_count=medium_count,
            alerts=alert_records,
            grid_geojson=enriched_geojson,
        )

    except HTTPException:
        raise
    except Exception as exc:
        log.error("Response assembly failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Response assembly error: {exc}",
        ) from exc
