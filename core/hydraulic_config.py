"""
core/hydraulic_config.py
========================
Centralised engineering assumptions and configurable parameters for the
SIH Flood Nowcasting drainage hydraulics layer.

SEPARATION OF CONCERNS
-----------------------
Every value in this file is an **engineering assumption** derived from
published norms for Hyderabad's drainage network, NOT a measurement of
actual infrastructure.  When real measurements become available (e.g. from
GHMC drainage survey data), replace the relevant constant here and the rest
of the pipeline will automatically use the updated value.

Sources / rationale
--------------------
* Manning's n  — AS/NZS 3725 / IS:7784; concrete-lined drain ≈ 0.013–0.015.
* Width/depth   — Typical GHMC Category-B stormwater drain dimensions
  (HMDA Stormwater Drainage Master Plan, 2016; assumed typical urban nala).
* Rational C    — IS 1892 / standard urban hydrology for Hyderabad soils
  and cover types.
* Min slope     — Minimum hydraulic grade to prevent silting (0.05 %).
* Velocity      — Overland sheet flow ≈ 0.3–0.5 m/s per HEC-HMS guidance.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Manning's equation parameters  [ASSUMED — NOT MEASURED]
# ---------------------------------------------------------------------------

#: Manning roughness coefficient (n) — concrete-lined open channel
#: Assumption: mixed concrete/earthen drains typical of Hyderabad nalas.
MANNING_N: float = 0.020

#: Default drain cross-section width (m)  [ASSUMED — NOT MEASURED]
#: Representative of a Category-B GHMC stormwater drain (≈2 m width).
DEFAULT_DRAIN_WIDTH_M: float = 2.0

#: Default drain cross-section depth (m)  [ASSUMED — NOT MEASURED]
#: Half-width rule of thumb for trapezoidal nala cross-section.
DEFAULT_DRAIN_DEPTH_M: float = 1.0

#: Default drain bank slope (H:V) for trapezoidal sections  [ASSUMED]
DEFAULT_SIDE_SLOPE_HV: float = 1.0  # 1:1

#: Minimum allowable hydraulic slope to prevent instability  [ENGINEERING LIMIT]
MIN_SLOPE: float = 0.0005  # 0.05 %

# ---------------------------------------------------------------------------
# Rational Method parameters  [ASSUMED / CALIBRATED]
# ---------------------------------------------------------------------------

#: Runoff coefficient (C) lookup by waterway type  [ASSUMED]
#: River/nala carry flow efficiently; ditches have higher losses.
RATIONAL_C_BY_TYPE: dict = {
    "river":  0.85,
    "stream": 0.75,
    "canal":  0.80,
    "drain":  0.70,
    "ditch":  0.60,
    "default": 0.65,
}

#: Runoff coefficient for grid cells (accounts for mixed urban coverage).
#: This is overridden by impervious_surface_ratio in the hydraulic pipeline.
CELL_BASE_RUNOFF_C: float = 0.65

# ---------------------------------------------------------------------------
# Surcharge propagation  [CONFIGURABLE]
# ---------------------------------------------------------------------------

#: Maximum radius (m) within which surcharge affects surface cells.
SURCHARGE_PROPAGATION_RADIUS_M: float = 500.0

#: Fraction of surcharge volume that reaches the surface in the propagation zone.
SURCHARGE_SURFACE_FRACTION: float = 0.4

#: Minimum ponding area (m²) to avoid division-by-zero in depth calculation.
MIN_PONDING_AREA_M2: float = 1000.0

# ---------------------------------------------------------------------------
# ETA calculation  [CONFIGURABLE]
# ---------------------------------------------------------------------------

#: Overland flow velocity (m/s) — sheet flow on impervious urban surfaces.
#: Assumption: 0.5 m/s for moderate slopes in urban Hyderabad.
OVERLAND_VELOCITY_M_S: float = 0.5

#: Drainage pipe/nala flow velocity (m/s) — used for transit time.
#: Assumption: 0.8 m/s average for half-full conditions.
DRAIN_VELOCITY_M_S: float = 0.8

#: Minimum ETA (minutes) — avoids returning 0 for co-located cells.
ETA_MIN_MINUTES: float = 5.0

#: Maximum ETA (minutes) — caps physically unrealistic long times.
ETA_MAX_MINUTES: float = 240.0

# ---------------------------------------------------------------------------
# Risk factor weights  [CONFIGURABLE — must sum to 100]
# ---------------------------------------------------------------------------

RISK_WEIGHT_RAINFALL:   float = 35.0
RISK_WEIGHT_ELEVATION:  float = 25.0
RISK_WEIGHT_DRAINAGE:   float = 20.0
RISK_WEIGHT_IMPERVIOUS: float = 12.0
RISK_WEIGHT_HISTORICAL: float = 8.0

assert abs(
    RISK_WEIGHT_RAINFALL + RISK_WEIGHT_ELEVATION + RISK_WEIGHT_DRAINAGE +
    RISK_WEIGHT_IMPERVIOUS + RISK_WEIGHT_HISTORICAL - 100.0
) < 1e-6, "Risk weights must sum to 100"

# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------

ALERT_HIGH_THRESHOLD:   float = 70.0
ALERT_MEDIUM_THRESHOLD: float = 40.0

# ---------------------------------------------------------------------------
# Water depth thresholds (cm)
# ---------------------------------------------------------------------------

DEPTH_MODERATE_CM:  float = 5.0   # >= 5 cm: moderate waterlogging
DEPTH_SEVERE_CM:    float = 15.0  # >= 15 cm: severe inundation

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

#: depth_cm threshold for severe penalty / block
ROUTE_BLOCK_DEPTH_CM:   float = 15.0
ROUTE_HIGH_RISK_SCORE:  float = 70.0
ROUTE_PENALTY_SEVERE:   float = 10.0  # multiplier on segment travel time
ROUTE_PENALTY_HIGH:     float = 3.0
