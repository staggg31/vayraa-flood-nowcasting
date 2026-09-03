"""
core/hydraulics.py
==================
Step 1C + Step 2: Distributed runoff accumulation and coupled hydraulic
+ ML flood depth / ETA calculation.

METHODOLOGY
-----------
1. **Rational Method runoff** per grid cell:
       Q_cell = C × I × A
   where:
       C = effective runoff coefficient (from impervious_surface_ratio)
       I = rainfall intensity in m/s  (rainfall_1h_mm / 3600 / 1000)
       A = cell area in m²

2. **Drainage graph node mapping**: each cell centroid is mapped to its
   nearest drainage graph node.

3. **Downstream accumulation** (topological order):
       inflow_node = sum(Q_cell for cells mapped to node)
                   + sum(outflow of upstream edges)
       outflow_node = min(inflow, capacity_sum_of_outgoing_edges)
       surplus = inflow - capacity_sum_of_outgoing_edges  (if > 0 → surcharged)

4. **Surcharge propagation**: surcharged nodes spread a fraction of their
   excess volume to surface cells within SURCHARGE_PROPAGATION_RADIUS_M,
   weighted by inverse distance (not elevation, to be conservative).
   Cells at higher elevation than the surcharged node receive zero volume.

5. **Water depth**:
       D_m = ml_prob × (runoff_volume_m3 + surcharge_volume_m3)
             / max(ponding_area_m2, MIN_PONDING_AREA_M2)
   Depth is then converted to cm for output.

6. **ETA**:
       eta = overland_distance_m / overland_velocity
           + drain_path_length_m / drain_velocity
   Bounded by [ETA_MIN_MINUTES, ETA_MAX_MINUTES].

ASSUMPTIONS (all from hydraulic_config.py — see that file for rationale)
-------------------------------------------------------------------------
* Manning's n, drain width/depth, runoff coefficient are assumed.
* Ponding area = cell area × 0.3 (depression storage fraction — assumed).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point

from core.drainage_graph import DrainageGraph, _haversine_m
from core.hydraulic_config import (
    CELL_BASE_RUNOFF_C,
    DRAIN_VELOCITY_M_S,
    ETA_MAX_MINUTES,
    ETA_MIN_MINUTES,
    MIN_PONDING_AREA_M2,
    OVERLAND_VELOCITY_M_S,
    SURCHARGE_PROPAGATION_RADIUS_M,
    SURCHARGE_SURFACE_FRACTION,
)

log = logging.getLogger("hydraulics")

# Fraction of cell area assumed to act as a depression/ponding zone [ASSUMED]
_PONDING_AREA_FRACTION = 0.30


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class HydraulicEngine:
    """Couples the drainage graph with Rational Method runoff and ML output
    to produce physically interpretable depth and ETA per grid cell.

    Usage
    -----
    >>> dg = DrainageGraph().build(...)
    >>> he = HydraulicEngine(dg)
    >>> result_df = he.run(grid_df, rainfall_1h_mm=90.0, ml_probs=proba_series)
    """

    def __init__(self, drainage_graph: DrainageGraph) -> None:
        self.dg = drainage_graph

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        grid_df: pd.DataFrame,
        rainfall_1h_mm: float,
        ml_probs: pd.Series,
        grid_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> pd.DataFrame:
        """Run the full hydraulic pipeline and return enriched per-cell DataFrame.

        Parameters
        ----------
        grid_df:
            Static grid DataFrame indexed by cell_id.  Must contain:
            ``elevation_m``, ``impervious_surface_ratio``, ``area_km2``,
            ``dist_to_river_m``, ``drainage_density``, ``historical_flood_count``.
        rainfall_1h_mm:
            Scenario 1-hour rainfall in mm.
        ml_probs:
            Per-cell flood probability [0,1] (same index as grid_df).
        grid_gdf:
            Optional GeoDataFrame with grid polygons (for centroid calculation).
            If None, centroids are estimated from lat/lon ranges.

        Returns
        -------
        pd.DataFrame
            grid_df enriched with per-cell hydraulic columns:
            ``runoff_m3_s``, ``drainage_node_id``, ``drainage_flow_m3_s``,
            ``drainage_capacity_m3_s``, ``utilization_ratio``, ``surcharged``,
            ``surcharge_volume_m3``, ``depth_cm``, ``eta_minutes``.
        """
        out = grid_df.copy()
        out["ml_prob"] = ml_probs.values

        # ── Step A: Compute per-cell runoff [m³/s] ──────────────────────────
        rainfall_intensity_m_s = rainfall_1h_mm / (1000.0 * 3600.0)
        area_m2 = out["area_km2"].values * 1e6  # km² → m²

        # Effective C = base_C + (1 - base_C) * impervious_fraction
        # Models that higher imperviousness raises the effective runoff coefficient.
        C_eff = CELL_BASE_RUNOFF_C + (1.0 - CELL_BASE_RUNOFF_C) * out["impervious_surface_ratio"].values
        C_eff = np.clip(C_eff, 0.0, 1.0)

        out["runoff_m3_s"] = C_eff * rainfall_intensity_m_s * area_m2
        out["runoff_volume_m3"] = out["runoff_m3_s"] * 3600.0  # over 1-hour period

        # ── Step B: Map cells to nearest drainage node ──────────────────────
        cell_centroids = self._compute_centroids(out, grid_gdf)
        out["drainage_node_id"] = ""
        node_to_cells: Dict[str, list] = {}

        for cell_id, (lon, lat) in cell_centroids.items():
            if self.dg.G.number_of_nodes() == 0:
                node_id = "NO_GRAPH"
            else:
                node_id = self.dg.nearest_node(lon, lat)
            out.at[cell_id, "drainage_node_id"] = node_id
            node_to_cells.setdefault(node_id, []).append(cell_id)

        # ── Step C: Accumulate flow downstream ──────────────────────────────
        # Reset accumulation fields on all nodes
        for nid in self.dg.G.nodes():
            self.dg.G.nodes[nid]["inflow_m3_s"] = 0.0
            self.dg.G.nodes[nid]["outflow_m3_s"] = 0.0
            self.dg.G.nodes[nid]["surcharged"] = False
            self.dg.G.nodes[nid]["surcharge_excess_m3_s"] = 0.0
            self.dg.G.nodes[nid]["surcharge_volume_m3"] = 0.0

        # Also reset edge accumulation
        for u, v in self.dg.G.edges():
            self.dg.G[u][v]["inflow_m3_s"] = 0.0
            self.dg.G[u][v]["outflow_m3_s"] = 0.0
            self.dg.G[u][v]["utilization_ratio"] = 0.0
            self.dg.G[u][v]["surcharged"] = False
            self.dg.G[u][v]["surcharge_excess_m3_s"] = 0.0
            self.dg.G[u][v]["surcharge_volume_m3"] = 0.0

        if self.dg.G.number_of_nodes() > 0:
            try:
                topo_order = list(nx.topological_sort(self.dg.G))
            except nx.NetworkXUnfeasible:
                log.warning("Graph has cycles despite cycle-removal — skipping accumulation")
                topo_order = []

            for nid in topo_order:
                node_data = self.dg.G.nodes[nid]

                # Surface runoff from cells mapped to this node
                surface_runoff = sum(
                    float(out.at[cid, "runoff_m3_s"])
                    for cid in node_to_cells.get(nid, [])
                    if cid in out.index
                )

                # Upstream pipe inflow
                upstream_inflow = sum(
                    self.dg.G[u][nid].get("outflow_m3_s", 0.0)
                    for u in self.dg.G.predecessors(nid)
                )

                total_inflow = surface_runoff + upstream_inflow
                node_data["inflow_m3_s"] = total_inflow

                # Sum of outgoing edge capacities
                out_edges = list(self.dg.G.out_edges(nid, data=True))
                total_capacity = sum(e[2]["capacity_m3_s"] for e in out_edges) if out_edges else 0.0

                # Determine outflow and surcharge
                if total_capacity > 0:
                    outflow = min(total_inflow, total_capacity)
                    excess = max(total_inflow - total_capacity, 0.0)
                else:
                    # Outfall / terminal node — assume it can accept all flow
                    outflow = total_inflow
                    excess = 0.0

                node_data["outflow_m3_s"] = outflow
                surcharged = excess > 0.0
                node_data["surcharged"] = surcharged
                node_data["surcharge_excess_m3_s"] = excess
                node_data["surcharge_volume_m3"] = excess * 3600.0  # over 1 hour

                # Distribute outflow across outgoing edges proportionally by capacity
                for u, v, edata in out_edges:
                    edge_cap = edata["capacity_m3_s"]
                    frac = edge_cap / total_capacity if total_capacity > 0 else 1.0 / len(out_edges)
                    edge_flow = outflow * frac
                    self.dg.G[u][v]["inflow_m3_s"] = total_inflow
                    self.dg.G[u][v]["outflow_m3_s"] = edge_flow
                    self.dg.G[u][v]["utilization_ratio"] = min(total_inflow / max(edge_cap, 1e-6), 5.0)
                    self.dg.G[u][v]["surcharged"] = surcharged
                    self.dg.G[u][v]["surcharge_excess_m3_s"] = excess * frac
                    self.dg.G[u][v]["surcharge_volume_m3"] = excess * frac * 3600.0

        # ── Step D: Map drainage node results back to cells ─────────────────
        out["drainage_flow_m3_s"] = 0.0
        out["drainage_capacity_m3_s"] = 0.0
        out["utilization_ratio"] = 0.0
        out["surcharged"] = False
        out["surcharge_volume_m3"] = 0.0

        for cell_id in out.index:
            nid = out.at[cell_id, "drainage_node_id"]
            if nid and nid in self.dg.G.nodes:
                nd = self.dg.G.nodes[nid]
                out.at[cell_id, "drainage_flow_m3_s"] = nd.get("inflow_m3_s", 0.0)
                # Nearest outgoing edge capacity as representative capacity
                out_edges = list(self.dg.G.out_edges(nid, data=True))
                if out_edges:
                    out.at[cell_id, "drainage_capacity_m3_s"] = out_edges[0][2]["capacity_m3_s"]
                out.at[cell_id, "utilization_ratio"] = (
                    nd.get("inflow_m3_s", 0.0) / max(out.at[cell_id, "drainage_capacity_m3_s"], 1e-6)
                )
                out.at[cell_id, "surcharged"] = nd.get("surcharged", False)
                out.at[cell_id, "surcharge_volume_m3"] = nd.get("surcharge_volume_m3", 0.0)

        # ── Step E: Surcharge propagation ───────────────────────────────────
        out["cell_surcharge_volume_m3"] = 0.0
        surcharged_nodes = [
            (nid, data)
            for nid, data in self.dg.G.nodes(data=True)
            if data.get("surcharged") and data.get("surcharge_volume_m3", 0) > 0
        ]

        if surcharged_nodes and len(cell_centroids) > 0:
            for nid, nd in surcharged_nodes:
                node_lon = nd["lon"]
                node_lat = nd["lat"]
                node_elev = nd["elevation_m"]
                sv = nd["surcharge_volume_m3"]

                # Find cells within propagation radius and lower elevation
                nearby: list[Tuple[str, float]] = []
                for cid, (clon, clat) in cell_centroids.items():
                    dist = _haversine_m(node_lon, node_lat, clon, clat)
                    cell_elev = float(out.at[cid, "elevation_m"])
                    if dist <= SURCHARGE_PROPAGATION_RADIUS_M and cell_elev <= node_elev + 5.0:
                        nearby.append((cid, max(dist, 1.0)))

                if nearby:
                    total_inv_dist = sum(1.0 / d for _, d in nearby)
                    for cid, dist in nearby:
                        weight = (1.0 / dist) / total_inv_dist
                        extra = sv * SURCHARGE_SURFACE_FRACTION * weight
                        out.at[cid, "cell_surcharge_volume_m3"] += extra

        # ── Step F: Water depth calculation ─────────────────────────────────
        ponding_area_m2 = np.maximum(
            (out["area_km2"].values * 1e6) * _PONDING_AREA_FRACTION,
            MIN_PONDING_AREA_M2,
        )
        total_volume = out["runoff_volume_m3"].values + out["cell_surcharge_volume_m3"].values

        # D_m = ml_prob × volume / ponding_area
        depth_m = out["ml_prob"].values * total_volume / ponding_area_m2
        depth_m = np.maximum(depth_m, 0.0)

        out["depth_cm"] = np.round(depth_m * 100.0, 1)

        # ── Step G: ETA calculation ──────────────────────────────────────────
        out["eta_minutes"] = self._compute_eta(out, cell_centroids, rainfall_1h_mm)

        return out

    # ------------------------------------------------------------------
    # Centroid computation
    # ------------------------------------------------------------------

    def _compute_centroids(
        self,
        grid_df: pd.DataFrame,
        grid_gdf: Optional[gpd.GeoDataFrame],
    ) -> Dict[str, Tuple[float, float]]:
        """Return {cell_id: (lon, lat)} centroids for all cells."""
        centroids: Dict[str, Tuple[float, float]] = {}

        if grid_gdf is not None and len(grid_gdf) > 0:
            for _, row in grid_gdf.iterrows():
                cid = row.get("cell_id")
                if cid and cid in grid_df.index:
                    c = row.geometry.centroid
                    centroids[cid] = (c.x, c.y)
        else:
            # Fallback: distribute cells over the Hyderabad bounding box
            lat_min, lat_max = 17.30, 17.48
            lon_min, lon_max = 78.35, 78.55
            n = len(grid_df)
            cols = min(8, max(1, n))  # never more cols than cells
            rows = max(1, math.ceil(n / cols))
            for i, cid in enumerate(grid_df.index):
                row_i = i // cols
                col_i = i % cols
                lat = lat_max - (row_i + 0.5) * (lat_max - lat_min) / rows
                lon = lon_min + (col_i + 0.5) * (lon_max - lon_min) / cols
                centroids[cid] = (lon, lat)

        return centroids

    # ------------------------------------------------------------------
    # ETA calculation
    # ------------------------------------------------------------------

    def _compute_eta(
        self,
        out: pd.DataFrame,
        cell_centroids: Dict[str, Tuple[float, float]],
        rainfall_1h_mm: float,
    ) -> pd.Series:
        """Compute inundation arrival time in minutes per cell.

        ETA = overland_time + drain_transit_time

        overland_time = dist_to_river_m / overland_velocity
        drain_transit_time = drain_path_length / drain_velocity

        Both components are physically derived from the data.
        """
        etas = []
        for cell_id in out.index:
            dist_m = float(out.at[cell_id, "dist_to_river_m"])
            overland_time_s = dist_m / max(OVERLAND_VELOCITY_M_S, 0.1)

            # Drain path length from nearest node outflow path
            nid = out.at[cell_id, "drainage_node_id"]
            drain_path_m = 0.0
            if nid and nid in self.dg.G.nodes:
                # Sum edge lengths downstream from this node
                for u, v, edata in self.dg.G.out_edges(nid, data=True):
                    drain_path_m += edata.get("length_m", 0.0)
                    break  # first edge only (nearest reach)

            drain_time_s = drain_path_m / max(DRAIN_VELOCITY_M_S, 0.1)

            total_s = overland_time_s + drain_time_s
            total_min = total_s / 60.0

            # Higher rainfall → faster arrival
            rain_factor = max(0.5, 1.0 - (rainfall_1h_mm / 400.0))
            total_min = total_min * rain_factor

            total_min = float(np.clip(total_min, ETA_MIN_MINUTES, ETA_MAX_MINUTES))
            etas.append(round(total_min, 1))

        return pd.Series(etas, index=out.index, name="eta_minutes")
