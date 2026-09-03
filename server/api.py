"""
server/api.py
=============
Production-grade FastAPI server for the SIH Flood Nowcasting system.
SIH26085 — Hyderabad Metropolitan & Musi River Basin deployment.

Endpoints
---------
GET  /api/v1/health       — Liveness probe + model/grid readiness check.
GET  /api/v1/grid         — Return the raw GeoJSON FeatureCollection.
POST /api/v1/simulate     — Run a full rainfall scenario through the coupled
                            hydraulic+ML pipeline and return enriched GeoJSON
                            with 5-factor explainable risk decomposition.
POST /api/v1/safe-route   — Compute standard and flood-safe dispatch routes.

Architecture
------------
- Single-load pattern: static grid, drainage graph, trained model, and
  fusion engine are loaded once at startup via the lifespan context.
- CORS fully open for Streamlit / local clients.
- Pydantic v2 models for request/response validation.
- All prediction logic wrapped in try/except blocks returning HTTP 500.
"""

from __future__ import annotations

import json
import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from core.alert_engine import DynamicRiskFusion
from core.drainage_graph import DrainageGraph
from core.feature_engine import fuse_rainfall_scenario, load_static_grid
from core.hydraulics import HydraulicEngine
from core.model import FloodRiskEngine
from core.routing import FloodSafeRouter

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
# Paths
# ---------------------------------------------------------------------------
GRID_PATH  = "data/grid_cells.geojson"
TRAIN_CSV  = "data/train_features.csv"
MODEL_PATH = "core/flood_model.pkl"

# ---------------------------------------------------------------------------
# Lifespan: single-load initialisation
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy resources once at startup."""
    log.info("=== SIH26085 Flood API — Startup ===")

    # 1. Static grid DataFrame
    log.info("Loading static grid from '%s' …", GRID_PATH)
    app.state.grid_df = load_static_grid(GRID_PATH)
    log.info("Grid loaded: %d cells.", len(app.state.grid_df))

    # 2. Raw GeoJSON (cached for /api/v1/grid endpoint)
    with open(GRID_PATH, "r", encoding="utf-8") as fh:
        app.state.raw_geojson = json.load(fh)

    # 3. Grid GeoDataFrame for spatial operations
    try:
        app.state.grid_gdf = gpd.read_file(GRID_PATH)
    except Exception as e:
        log.warning("Could not load grid as GeoDataFrame: %s", e)
        app.state.grid_gdf = None

    # 4. Drainage graph
    log.info("Building drainage graph …")
    dg = DrainageGraph()
    try:
        dg.build(
            waterways_path="data/raw/hyderabad_waterways.geojson",
            grid_geojson_path=GRID_PATH,
        )
        log.info("Drainage graph: %d nodes, %d edges", dg.G.number_of_nodes(), dg.G.number_of_edges())
    except Exception as e:
        log.error("Drainage graph build failed: %s", e)
    app.state.drainage_graph = dg

    # 5. Hydraulic engine
    app.state.hydraulic_engine = HydraulicEngine(dg)

    # 6. ML model
    log.info("Initialising FloodRiskEngine …")
    engine = FloodRiskEngine()
    engine.fit_or_load(train_csv=TRAIN_CSV, model_path=MODEL_PATH)
    app.state.engine = engine
    log.info("Model ready.")

    # 7. Risk fusion engine
    app.state.fusion = DynamicRiskFusion()

    # 8. Routing engine
    app.state.router = FloodSafeRouter()
    log.info("All engines initialised. API is serving.")

    # Store last simulation result for routing endpoint use
    app.state.last_risk_df = None

    yield
    log.info("=== SIH26085 Flood API — Shutdown ===")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SIH26085 Flood Nowcasting API — Hyderabad",
    description=(
        "Urban Flood Nowcasting System coupling drainage network, DEM, and "
        "rainfall nowcasting for Hyderabad Metropolitan & Musi River Basin. "
        "Combines XGBoost ML predictions with Manning/Rational Method hydraulics."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

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
    rainfall_scenario_mm: float = Field(
        default=65.0, ge=0.0, le=250.0,
        description="1-hour rainfall accumulation in mm (0–250).",
    )


class RiskExplainability(BaseModel):
    rainfall:    float
    elevation:   float
    drain_surcharge: float
    impervious:  float
    historical:  float


class AlertRecord(BaseModel):
    cell_id:              str
    risk_score:           float
    alert_level:          str
    depth_cm:             float
    eta_minutes:          Optional[float]
    time_to_flood_hrs:    Optional[float]
    dist_to_river_m:      float
    elevation_m:          float
    rainfall_1h_mm:       float
    rainfall_forecast_6h_mm: float
    drainage_stress:      float
    drainage_node_id:     Optional[str]
    drainage_flow_m3_s:   Optional[float]
    drainage_capacity_m3_s: Optional[float]
    surcharge_volume_m3:  Optional[float]
    surcharged:           bool
    dominant_risk_factor: Optional[str]
    explainability:       RiskExplainability


class SimulateResponse(BaseModel):
    status:                str
    scenario_intensity_mm: float
    total_cells:           int
    critical_cells_count:  int
    medium_cells_count:    int
    alerts:                List[AlertRecord]
    grid_geojson:          Dict[str, Any]


class SafeRouteRequest(BaseModel):
    origin_lat:         float = Field(..., ge=17.0, le=18.0)
    origin_lon:         float = Field(..., ge=78.0, le=79.0)
    destination_lat:    float = Field(..., ge=17.0, le=18.0)
    destination_lon:    float = Field(..., ge=78.0, le=79.0)
    current_rainfall_mm: float = Field(default=0.0, ge=0.0, le=250.0)

    @field_validator("origin_lat", "destination_lat")
    @classmethod
    def validate_lat(cls, v: float) -> float:
        if not (17.0 <= v <= 18.0):
            raise ValueError(f"Latitude {v} outside Hyderabad domain [17.0, 18.0]")
        return v

    @field_validator("origin_lon", "destination_lon")
    @classmethod
    def validate_lon(cls, v: float) -> float:
        if not (78.0 <= v <= 79.0):
            raise ValueError(f"Longitude {v} outside Hyderabad domain [78.0, 79.0]")
        return v


class SafeRouteResponse(BaseModel):
    standard_route_geometry:   Optional[Dict[str, Any]]
    safe_route_geometry:       Optional[Dict[str, Any]]
    blocked_segments:          List[Dict[str, Any]]
    route_distance_m:          float
    safe_route_distance_m:     float
    estimated_difference_pct:  float
    routing_explanation:       str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert a potentially NaN/None value to a safe float."""
    try:
        f = float(val)
        return f if (f == f) else default  # NaN check
    except (TypeError, ValueError):
        return default


def _build_enriched_geojson(raw_geojson: dict, result_df: pd.DataFrame) -> dict:
    """Merge risk metrics from result_df into each GeoJSON feature's properties."""
    enriched_cols = [
        "risk_score", "alert_level", "time_to_flood_hrs",
        "rainfall_1h_mm", "rainfall_forecast_6h_mm",
        "drainage_stress", "norm_drainage_stress",
        "depth_cm", "eta_minutes",
        "rf_rainfall", "rf_elevation", "rf_drainage", "rf_impervious", "rf_historical",
        "dominant_risk_factor", "surcharged",
        "drainage_node_id", "drainage_flow_m3_s", "drainage_capacity_m3_s",
        "surcharge_volume_m3", "utilization_ratio",
    ]
    existing_cols = [c for c in enriched_cols if c in result_df.columns]
    risk_lookup: Dict[str, dict] = result_df[existing_cols].to_dict(orient="index")

    enriched_features = []
    for feat in raw_geojson.get("features", []):
        cell_id = feat["properties"].get("cell_id")
        risk_row = risk_lookup.get(cell_id, {})
        new_props = {
            **feat["properties"],
            **{k: (None if (isinstance(v, float) and v != v) else v)
               for k, v in risk_row.items()},
        }
        enriched_features.append({
            "type": "Feature",
            "geometry": feat["geometry"],
            "properties": new_props,
        })

    return {
        "type": "FeatureCollection",
        "name": "hyderabad_flood_grid",
        "features": enriched_features,
    }


def _series_to_alert_records(result_df: pd.DataFrame) -> List[AlertRecord]:
    """Extract HIGH/MEDIUM rows from result_df as AlertRecord objects."""
    high_df = result_df[result_df["alert_level"].isin(["HIGH", "MEDIUM"])]
    records: List[AlertRecord] = []

    for cell_id, row in high_df.iterrows():
        records.append(AlertRecord(
            cell_id=str(cell_id),
            risk_score=round(_safe_float(row["risk_score"]), 1),
            alert_level=str(row.get("alert_level", "LOW")),
            depth_cm=round(_safe_float(row.get("depth_cm", 0)), 1),
            eta_minutes=_safe_float(row.get("eta_minutes"), None) if row.get("eta_minutes") is not None else None,
            time_to_flood_hrs=(
                None if pd.isna(row.get("time_to_flood_hrs", float("nan")))
                else float(row["time_to_flood_hrs"])
            ),
            dist_to_river_m=_safe_float(row.get("dist_to_river_m", 0)),
            elevation_m=_safe_float(row.get("elevation_m", 0)),
            rainfall_1h_mm=round(_safe_float(row.get("rainfall_1h_mm", 0)), 3),
            rainfall_forecast_6h_mm=round(_safe_float(row.get("rainfall_forecast_6h_mm", 0)), 3),
            drainage_stress=round(_safe_float(row.get("drainage_stress", 0)), 4),
            drainage_node_id=str(row.get("drainage_node_id", "")) or None,
            drainage_flow_m3_s=_safe_float(row.get("drainage_flow_m3_s"), None) if "drainage_flow_m3_s" in row.index else None,
            drainage_capacity_m3_s=_safe_float(row.get("drainage_capacity_m3_s"), None) if "drainage_capacity_m3_s" in row.index else None,
            surcharge_volume_m3=_safe_float(row.get("surcharge_volume_m3"), None) if "surcharge_volume_m3" in row.index else None,
            surcharged=bool(row.get("surcharged", False)),
            dominant_risk_factor=str(row.get("dominant_risk_factor", "")) or None,
            explainability=RiskExplainability(
                rainfall=round(_safe_float(row.get("rf_rainfall", 0)), 1),
                elevation=round(_safe_float(row.get("rf_elevation", 0)), 1),
                drain_surcharge=round(_safe_float(row.get("rf_drainage", 0)), 1),
                impervious=round(_safe_float(row.get("rf_impervious", 0)), 1),
                historical=round(_safe_float(row.get("rf_historical", 0)), 1),
            ),
        ))
    return records


# ---------------------------------------------------------------------------
# Action recommendation engine
# ---------------------------------------------------------------------------
_ACTION_MATRIX = [
    # (condition_fn, action_text)
    (lambda r: r.get("surcharged") and r.get("drainage_node_id"),
     "⚠️ Inspect drainage node {drainage_node_id} — SURCHARGED. Deploy dewatering equipment immediately."),
    (lambda r: r.get("depth_cm", 0) >= 30,
     "🔴 INITIATE EVACUATION — Severe inundation ({depth_cm:.0f} cm). Deploy flood barriers."),
    (lambda r: r.get("depth_cm", 0) >= 15,
     "🟡 Divert traffic. Pre-position mobile pump units. Clear storm drains in cell {cell_id}."),
    (lambda r: r.get("dominant_risk_factor") == "drainage",
     "🔧 Inspect and clear drainage infrastructure. Utilisation ratio elevated."),
    (lambda r: r.get("dominant_risk_factor") == "elevation",
     "🏞️ Low-lying cell — activate flood bunds and sandbag vulnerable access routes."),
    (lambda r: r.get("dominant_risk_factor") == "rainfall",
     "🌧️ High rainfall intensity. Alert downstream communities and pre-position response teams."),
    (lambda r: True,
     "📋 Monitor cell {cell_id}. Keep response units on standby."),
]


def _recommend_action(props: dict) -> str:
    for condition_fn, template in _ACTION_MATRIX:
        try:
            if condition_fn(props):
                return template.format(**{k: (v if v is not None else "N/A") for k, v in props.items()})
        except Exception:
            continue
    return "📋 Monitor cell. Keep response units on standby."


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", summary="Serve Cockpit UI", tags=["UI"])
async def serve_cockpit():
    """Serve the C2 Floating Cockpit standalone HTML template."""
    from fastapi.responses import FileResponse
    from pathlib import Path
    cockpit_path = Path(__file__).parent / "templates" / "cockpit.html"
    if not cockpit_path.exists():
        raise HTTPException(status_code=404, detail="Cockpit template not found.")
    return FileResponse(cockpit_path)


@app.get("/api/v1/health", summary="Liveness and readiness probe", tags=["System"])
async def health() -> dict:
    model_ready = (
        hasattr(app.state, "engine") and app.state.engine.pipeline is not None
    )
    cells_loaded = len(app.state.grid_df) if hasattr(app.state, "grid_df") else 0
    graph_nodes = (
        app.state.drainage_graph.G.number_of_nodes()
        if hasattr(app.state, "drainage_graph") else 0
    )
    return {
        "status": "online",
        "model_ready": model_ready,
        "cells_loaded": cells_loaded,
        "drainage_nodes": graph_nodes,
        "version": "2.0.0",
    }


@app.get("/api/v1/grid", summary="Raw GeoJSON grid", tags=["Data"])
async def get_grid() -> JSONResponse:
    if not hasattr(app.state, "raw_geojson"):
        raise HTTPException(status_code=503, detail="Grid not yet loaded.")
    return JSONResponse(content=app.state.raw_geojson)


@app.post(
    "/api/v1/simulate",
    response_model=SimulateResponse,
    summary="Run a rainfall scenario through the coupled hydraulic+ML pipeline",
    tags=["Simulation"],
)
async def simulate(request: SimulateRequest) -> SimulateResponse:
    """Execute the full SIH26085 flood nowcasting pipeline.

    Steps
    -----
    1. Fuse rainfall scenario into grid features.
    2. Inject rainfall_1h_mm before ML inference.
    3. Run hydraulic engine: Rational Method + drainage accumulation + surcharge.
    4. Predict flood probability (XGBoost).
    5. Compute 5-factor explainable risk decomposition.
    6. Return enriched GeoJSON + alert records.
    """
    if not hasattr(app.state, "engine") or not hasattr(app.state, "grid_df"):
        raise HTTPException(status_code=503, detail="Server still initialising.")

    scenario_mm = request.rainfall_scenario_mm
    log.info("Simulate request: rainfall=%.1f mm", scenario_mm)

    try:
        # Step 1: Feature fusion
        df_fused = fuse_rainfall_scenario(app.state.grid_df, scenario_mm=scenario_mm, seed=42)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feature fusion error: {exc}") from exc

    try:
        # Step 2: Inject rainfall before ML inference
        df_fused["rainfall_1h_mm"] = scenario_mm
        ml_probs = app.state.engine.predict_proba(df_fused)
    except Exception as exc:
        log.error("Model prediction failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Model prediction error: {exc}") from exc

    try:
        # Step 3: Hydraulic engine (Rational Method + surcharge + depth + ETA)
        grid_gdf = getattr(app.state, "grid_gdf", None)
        df_hydraulic = app.state.hydraulic_engine.run(
            grid_df=df_fused,
            rainfall_1h_mm=scenario_mm,
            ml_probs=ml_probs,
            grid_gdf=grid_gdf,
        )
    except Exception as exc:
        log.warning("Hydraulic engine failed: %s — falling back to feature-only mode", exc)
        df_hydraulic = df_fused.copy()
        df_hydraulic["depth_cm"] = 0.0
        df_hydraulic["eta_minutes"] = None
        df_hydraulic["drainage_node_id"] = ""
        df_hydraulic["drainage_flow_m3_s"] = 0.0
        df_hydraulic["drainage_capacity_m3_s"] = 0.0
        df_hydraulic["utilization_ratio"] = 0.0
        df_hydraulic["surcharged"] = False
        df_hydraulic["surcharge_volume_m3"] = 0.0

    try:
        # Step 4: Dynamic risk fusion + 5-factor decomposition
        result_df = app.state.fusion.evaluate(df_hydraulic, ml_probs)
    except Exception as exc:
        log.error("Risk fusion failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Risk fusion error: {exc}") from exc

    # Cache for routing endpoint
    app.state.last_risk_df = result_df

    try:
        alert_records = _series_to_alert_records(result_df)
        enriched_geojson = _build_enriched_geojson(app.state.raw_geojson, result_df)
        critical_count = int((result_df["alert_level"] == "HIGH").sum())
        medium_count   = int((result_df["alert_level"] == "MEDIUM").sum())

        log.info("Simulate: HIGH=%d MEDIUM=%d LOW=%d",
                 critical_count, medium_count,
                 int((result_df["alert_level"] == "LOW").sum()))

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
        raise HTTPException(status_code=500, detail=f"Response assembly error: {exc}") from exc


@app.post(
    "/api/v1/safe-route",
    response_model=SafeRouteResponse,
    summary="Compute standard and flood-safe dispatch routes",
    tags=["Routing"],
)
async def safe_route(request: SafeRouteRequest) -> SafeRouteResponse:
    """Generate a flood-aware safe dispatch route over the Hyderabad road network.

    Uses the most recent simulation result for flood zone information.
    If no simulation has been run, routes without flood avoidance.
    """
    if not hasattr(app.state, "router"):
        raise HTTPException(status_code=503, detail="Router not initialised.")

    risk_df = getattr(app.state, "last_risk_df", None)
    grid_gdf = getattr(app.state, "grid_gdf", None)

    try:
        result = app.state.router.route(
            origin_coords=(request.origin_lat, request.origin_lon),
            destination_coords=(request.destination_lat, request.destination_lon),
            current_rainfall_mm=request.current_rainfall_mm,
            risk_df=risk_df,
            grid_gdf=grid_gdf,
        )
    except Exception as exc:
        log.error("Routing failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Routing error: {exc}") from exc

    return SafeRouteResponse(
        standard_route_geometry=result.get("standard_route_geometry"),
        safe_route_geometry=result.get("safe_route_geometry"),
        blocked_segments=result.get("blocked_segments", []),
        route_distance_m=result.get("route_distance_m", 0.0),
        safe_route_distance_m=result.get("safe_route_distance_m", 0.0),
        estimated_difference_pct=result.get("estimated_difference_pct", 0.0),
        routing_explanation=result.get("routing_explanation", ""),
    )
