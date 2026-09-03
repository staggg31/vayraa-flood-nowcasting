"""
core/routing.py
===============
Step 4: Flood-safe emergency dispatch routing for Hyderabad.

METHODOLOGY
-----------
* A road-network graph is constructed from a deterministic set of Hyderabad
  arterial road segments covering the grid domain (Lat 17.30–17.48, Lon 78.35–78.55).
* The graph is compatible with NetworkX and supports both A* and Dijkstra search.
* For each road segment, the flood risk of intersecting grid cells is assessed.
* Dynamic edge weight penalties are applied:
    - depth_cm > 15: severe penalty (ROUTE_PENALTY_SEVERE × travel time)
    - risk_score > 70: high penalty (ROUTE_PENALTY_HIGH × travel time)
* Two routes are returned: standard shortest-path and flood-aware safe route.

OFFLINE FALLBACK
----------------
If live OSM data is unavailable, the module uses a built-in deterministic
road graph covering major Hyderabad arterials in the bounding box.
This fallback is always used in the current implementation to ensure
reliable demo/offline operation.

DATA ASSUMPTION
---------------
* The fallback road graph is derived from publicly known Hyderabad road
  geometry (approximate coordinates of major arterials).
* Segment lengths are computed via Haversine.
* No real-time traffic data is used.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point

from core.drainage_graph import _haversine_m
from core.hydraulic_config import (
    ROUTE_BLOCK_DEPTH_CM,
    ROUTE_HIGH_RISK_SCORE,
    ROUTE_PENALTY_HIGH,
    ROUTE_PENALTY_SEVERE,
)

log = logging.getLogger("routing")

# ---------------------------------------------------------------------------
# Fallback road graph: major Hyderabad arterials (approximate, public knowledge)
# ---------------------------------------------------------------------------
# Each road segment: (name, [(lon, lat), ...])
# Covering the grid domain 17.30–17.48 N, 78.35–78.55 E
_FALLBACK_ROADS: List[Tuple[str, List[Tuple[float, float]]]] = [
    # Outer Ring Road (partial western arc)
    ("ORR-W", [(78.355, 17.315), (78.360, 17.340), (78.360, 17.370), (78.358, 17.400),
               (78.362, 17.430), (78.370, 17.455), (78.385, 17.470)]),
    # NH-65 (NH-9) Pune Highway corridor
    ("NH65",  [(78.360, 17.380), (78.380, 17.385), (78.400, 17.392), (78.420, 17.396),
               (78.440, 17.398), (78.460, 17.400), (78.480, 17.402), (78.500, 17.400)]),
    # Jubilee Hills – Banjara Hills corridor
    ("JBH",   [(78.390, 17.420), (78.400, 17.425), (78.415, 17.428), (78.430, 17.432),
               (78.445, 17.430), (78.460, 17.425)]),
    # Inner Ring Road (partial southern)
    ("IRR-S", [(78.435, 17.350), (78.445, 17.360), (78.455, 17.370), (78.465, 17.380),
               (78.470, 17.395), (78.468, 17.410), (78.460, 17.425)]),
    # Mehdipatnam – Tolichowki corridor
    ("MTK",   [(78.410, 17.392), (78.415, 17.400), (78.418, 17.408), (78.422, 17.415),
               (78.428, 17.420)]),
    # Abids – Nampally corridor (north–south city centre)
    ("ABN",   [(78.470, 17.380), (78.470, 17.395), (78.467, 17.410), (78.463, 17.425),
               (78.460, 17.440)]),
    # Secunderabad – Begumpet connector
    ("SCB",   [(78.495, 17.440), (78.492, 17.445), (78.487, 17.450), (78.482, 17.452),
               (78.475, 17.455), (78.465, 17.460)]),
    # LB Nagar – Uppal road (eastern corridor)
    ("LBU",   [(78.520, 17.350), (78.510, 17.360), (78.500, 17.375), (78.490, 17.385),
               (78.480, 17.395)]),
    # North–south connector (western)
    ("NSW",   [(78.370, 17.310), (78.372, 17.335), (78.375, 17.360), (78.375, 17.385),
               (78.374, 17.410), (78.372, 17.435)]),
    # Sanath Nagar – Balanagar (NW corridor)
    ("SNB",   [(78.430, 17.460), (78.420, 17.462), (78.410, 17.465), (78.400, 17.468),
               (78.390, 17.467), (78.375, 17.462)]),
]


def _build_road_graph() -> nx.Graph:
    """Build an undirected road network graph from the fallback road segments.

    All road segments are connected into a single component by adding
    inter-connector edges between component centroids.
    """
    G = nx.Graph()

    for road_name, coords in _FALLBACK_ROADS:
        for i in range(len(coords) - 1):
            lon0, lat0 = coords[i]
            lon1, lat1 = coords[i + 1]
            u = f"{lon0:.4f},{lat0:.4f}"
            v = f"{lon1:.4f},{lat1:.4f}"

            if not G.has_node(u):
                G.add_node(u, lon=lon0, lat=lat0)
            if not G.has_node(v):
                G.add_node(v, lon=lon1, lat=lat1)

            dist = _haversine_m(lon0, lat0, lon1, lat1)
            travel_time_s = dist / 13.9  # 50 km/h urban speed

            G.add_edge(u, v,
                name=road_name,
                length_m=dist,
                travel_time_s=travel_time_s,
                weight=travel_time_s,
                geometry=LineString([(lon0, lat0), (lon1, lat1)]),
                depth_cm=0.0,
                risk_score=0.0,
                blocked=False,
            )

    # Connect disconnected components by linking component centroids
    components = list(nx.connected_components(G))
    if len(components) > 1:
        # Find a representative node (first node) from each component
        rep_nodes = [next(iter(comp)) for comp in components]
        # Chain them together in sequence
        for i in range(len(rep_nodes) - 1):
            u_rep = rep_nodes[i]
            v_rep = rep_nodes[i + 1]
            if not G.has_edge(u_rep, v_rep):
                n1_data = G.nodes[u_rep]
                n2_data = G.nodes[v_rep]
                conn_dist = _haversine_m(
                    n1_data["lon"], n1_data["lat"],
                    n2_data["lon"], n2_data["lat"],
                )
                conn_time = conn_dist / 13.9
                G.add_edge(u_rep, v_rep,
                    name="CONNECTOR",
                    length_m=conn_dist,
                    travel_time_s=conn_time,
                    weight=conn_time,
                    geometry=LineString([
                        (n1_data["lon"], n1_data["lat"]),
                        (n2_data["lon"], n2_data["lat"]),
                    ]),
                    depth_cm=0.0,
                    risk_score=0.0,
                    blocked=False,
                )
    return G


def _heuristic(G: nx.Graph, u: str, v: str) -> float:
    """A* heuristic: straight-line travel time estimate."""
    n1, n2 = G.nodes[u], G.nodes[v]
    dist = _haversine_m(n1["lon"], n1["lat"], n2["lon"], n2["lat"])
    return dist / 13.9  # 50 km/h


def _nearest_road_node(G: nx.Graph, lon: float, lat: float) -> str:
    """Return the road graph node closest to (lon, lat)."""
    best, best_d = None, float("inf")
    for nid, data in G.nodes(data=True):
        d = _haversine_m(lon, lat, data["lon"], data["lat"])
        if d < best_d:
            best_d, best = d, nid
    return best


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class FloodSafeRouter:
    """Generate standard and flood-aware safe routes over the Hyderabad road network.

    Usage
    -----
    >>> router = FloodSafeRouter()
    >>> result = router.route(origin, destination, rainfall_mm, cell_risk_df)
    """

    def __init__(self) -> None:
        self._road_graph = _build_road_graph()

    # ------------------------------------------------------------------
    # Apply flood penalties
    # ------------------------------------------------------------------

    def _apply_flood_penalties(
        self,
        risk_df: pd.DataFrame,
        grid_gdf: Optional[Any] = None,
    ) -> nx.Graph:
        """Return a copy of the road graph with flood-adjusted edge weights.

        Each road segment is checked against cells with high risk or depth.
        The flood-aware graph uses penalised travel times as edge weights.
        """
        import copy
        G_flood = copy.deepcopy(self._road_graph)

        if risk_df is None or risk_df.empty:
            return G_flood

        # Build a simple lookup: cells with risk
        high_risk_cells = risk_df[
            (risk_df.get("risk_score", pd.Series(dtype=float)) >= ROUTE_HIGH_RISK_SCORE) |
            (risk_df.get("depth_cm", pd.Series(dtype=float)) >= ROUTE_BLOCK_DEPTH_CM)
        ] if "risk_score" in risk_df.columns else pd.DataFrame()

        if high_risk_cells.empty:
            return G_flood

        # Extract centroid approximations from risk_df if available
        flood_zones: List[Tuple[float, float, float, float]] = []  # lon, lat, depth, risk
        if "drainage_node_id" in risk_df.columns:
            pass  # node positions not directly available here

        # Use grid_gdf centroids if available
        if grid_gdf is not None:
            for _, row in grid_gdf.iterrows():
                cid = row.get("cell_id")
                if cid in risk_df.index:
                    c = row.geometry.centroid
                    depth = float(risk_df.at[cid, "depth_cm"]) if "depth_cm" in risk_df.columns else 0.0
                    rscore = float(risk_df.at[cid, "risk_score"]) if "risk_score" in risk_df.columns else 0.0
                    flood_zones.append((c.x, c.y, depth, rscore))

        if not flood_zones:
            return G_flood

        # Penalise edges whose midpoint is near a flooded zone
        for u, v, data in G_flood.edges(data=True):
            geom: LineString = data.get("geometry")
            if geom is None:
                continue
            mid = geom.interpolate(0.5, normalized=True)
            mid_lon, mid_lat = mid.x, mid.y

            max_depth, max_risk = 0.0, 0.0
            for flon, flat, fdepth, frisk in flood_zones:
                d = _haversine_m(mid_lon, mid_lat, flon, flat)
                if d < 1000.0:  # within 1 km of flooded zone
                    max_depth = max(max_depth, fdepth)
                    max_risk = max(max_risk, frisk)

            base_time = data["travel_time_s"]

            if max_depth >= ROUTE_BLOCK_DEPTH_CM:
                G_flood[u][v]["weight"] = base_time * ROUTE_PENALTY_SEVERE
                G_flood[u][v]["blocked"] = True
                G_flood[u][v]["depth_cm"] = max_depth
                G_flood[u][v]["risk_score"] = max_risk
            elif max_risk >= ROUTE_HIGH_RISK_SCORE:
                G_flood[u][v]["weight"] = base_time * ROUTE_PENALTY_HIGH
                G_flood[u][v]["risk_score"] = max_risk
            else:
                G_flood[u][v]["depth_cm"] = max_depth
                G_flood[u][v]["risk_score"] = max_risk

        return G_flood

    # ------------------------------------------------------------------
    # Public route method
    # ------------------------------------------------------------------

    def route(
        self,
        origin_coords: Tuple[float, float],
        destination_coords: Tuple[float, float],
        current_rainfall_mm: float = 0.0,
        risk_df: Optional[pd.DataFrame] = None,
        grid_gdf: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Compute standard and flood-safe routes.

        Parameters
        ----------
        origin_coords:
            (lat, lon) of origin.
        destination_coords:
            (lat, lon) of destination.
        current_rainfall_mm:
            Current 1h rainfall scenario.
        risk_df:
            Enriched risk DataFrame (output of HydraulicEngine.run + risk decomp).
        grid_gdf:
            Optional grid GeoDataFrame for zone intersection.

        Returns
        -------
        dict with keys:
            standard_route_geometry, safe_route_geometry,
            blocked_segments, route_distance_m,
            safe_route_distance_m, estimated_difference_pct,
            routing_explanation.
        """
        o_lat, o_lon = origin_coords
        d_lat, d_lon = destination_coords

        G_standard = self._road_graph
        G_flood = self._apply_flood_penalties(risk_df, grid_gdf)

        o_node = _nearest_road_node(G_standard, o_lon, o_lat)
        d_node = _nearest_road_node(G_standard, d_lon, d_lat)

        def _compute_path(G: nx.Graph) -> Tuple[Optional[List[str]], float]:
            try:
                path = nx.astar_path(G, o_node, d_node,
                    heuristic=lambda u, v: _heuristic(G, u, v),
                    weight="weight")
                dist = sum(
                    G[path[i]][path[i + 1]]["length_m"]
                    for i in range(len(path) - 1)
                )
                return path, dist
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                try:
                    path = nx.dijkstra_path(G, o_node, d_node, weight="weight")
                    dist = sum(
                        G[path[i]][path[i + 1]]["length_m"]
                        for i in range(len(path) - 1)
                    )
                    return path, dist
                except Exception:
                    # Return a direct 2-node path (straight line fallback)
                    return [o_node, d_node] if o_node != d_node else [o_node], 0.0

        std_path, std_dist = _compute_path(G_standard)
        safe_path, safe_dist = _compute_path(G_flood)

        def _path_to_geojson(G: nx.Graph, path: Optional[List[str]]) -> Dict:
            """Convert path to GeoJSON LineString; returns empty geometry if None."""
            if not path or len(path) < 2:
                return {"type": "LineString", "coordinates": []}
            coords = []
            for nid in path:
                nd = G.nodes[nid]
                coords.append([nd["lon"], nd["lat"]])
            return {"type": "LineString", "coordinates": coords}

        # Find blocked segments in the standard route
        blocked = []
        if std_path and len(std_path) > 1:
            for i in range(len(std_path) - 1):
                u, v = std_path[i], std_path[i + 1]
                if G_flood.has_edge(u, v) and G_flood[u][v].get("blocked"):
                    blocked.append({
                        "from_node": u,
                        "to_node": v,
                        "depth_cm": G_flood[u][v].get("depth_cm", 0),
                        "risk_score": G_flood[u][v].get("risk_score", 0),
                        "road_name": G_standard[u][v].get("name", ""),
                    })

        diff_pct = 0.0
        if std_dist > 0 and safe_dist > 0:
            diff_pct = round((safe_dist - std_dist) / std_dist * 100.0, 1)

        explanation_parts = [
            f"Rainfall scenario: {current_rainfall_mm:.0f} mm/hr.",
            f"Standard route: {std_dist/1000:.2f} km.",
            f"Safe route: {safe_dist/1000:.2f} km (+{diff_pct}%).",
        ]
        if blocked:
            explanation_parts.append(
                f"{len(blocked)} segment(s) blocked due to depth >= {ROUTE_BLOCK_DEPTH_CM} cm."
            )
        else:
            explanation_parts.append("No segments blocked at current flood conditions.")

        return {
            "standard_route_geometry": _path_to_geojson(G_standard, std_path),
            "safe_route_geometry":    _path_to_geojson(G_flood, safe_path),
            "blocked_segments":        blocked,
            "route_distance_m":        round(std_dist, 1),
            "safe_route_distance_m":   round(safe_dist, 1),
            "estimated_difference_pct": diff_pct,
            "routing_explanation":     " ".join(explanation_parts),
            "origin_node":             o_node,
            "destination_node":        d_node,
        }
