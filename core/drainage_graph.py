"""
core/drainage_graph.py
======================
Step 1: Construct a directed drainage network graph from OSM waterway data.

DATA SOURCES (ACTUAL)
---------------------
* hyderabad_waterways.geojson — 371 LineString features from OSM containing
  waterway type tags: river, stream, canal, drain, ditch.
  Properties per feature: @id, name, waterway.

WHAT IS MISSING (ENGINEERING ASSUMPTIONS DOCUMENTED IN hydraulic_config.py)
---------------------------------------------------------------------------
* No cross-section width/depth measurements from any survey.
* No Manning roughness measurements — default applied by waterway type.
* No pipe inverts / manhole data.
* Elevation at nodes is sampled from the DEM raster (SRTM data).

GRAPH CONSTRUCTION METHODOLOGY
-------------------------------
1. Each LineString is treated as a drainage segment (edge).
2. Endpoints of adjacent LineStrings that share coordinates (within tolerance)
   are merged into shared nodes.
3. Flow direction is determined by elevation: the upstream node is the end
   with higher elevation (from DEM), the downstream node is lower.
4. Where both endpoints have equal elevation (flat), the existing coordinate
   order is preserved.
5. After graph construction:
   a. Cycle detection via DFS — any back-edge in the strongly-connected
      component search is logged and removed.
   b. The resulting graph is validated as a DAG before hydraulic use.
6. Capacity is computed via Manning's equation using assumed cross-section
   geometry (see hydraulic_config.py).
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point

from core.hydraulic_config import (
    DEFAULT_DRAIN_DEPTH_M,
    DEFAULT_DRAIN_WIDTH_M,
    DEFAULT_SIDE_SLOPE_HV,
    MANNING_N,
    MIN_SLOPE,
)

log = logging.getLogger("drainage-graph")

# ---------------------------------------------------------------------------
# Coordinate equality tolerance (degrees ≈ 1 m at these latitudes)
# ---------------------------------------------------------------------------
_NODE_SNAP_TOLERANCE = 1e-5  # ~1 m at Hyderabad latitude

# Capacity multipliers by waterway type (relative to standard drain section)
_CAPACITY_MULTIPLIER = {
    "river":  8.0,
    "stream": 3.0,
    "canal":  5.0,
    "drain":  1.0,
    "ditch":  0.6,
    "default": 1.0,
}

# Width multipliers by waterway type relative to DEFAULT_DRAIN_WIDTH_M
_WIDTH_MULTIPLIER = {
    "river":  10.0,
    "stream": 4.0,
    "canal":  6.0,
    "drain":  1.0,
    "ditch":  0.7,
    "default": 1.0,
}

# Depth multipliers by waterway type relative to DEFAULT_DRAIN_DEPTH_M
_DEPTH_MULTIPLIER = {
    "river":  5.0,
    "stream": 2.5,
    "canal":  3.0,
    "drain":  1.0,
    "ditch":  0.6,
    "default": 1.0,
}

# Manning n by waterway type (overrides global default for rougher channels)
_MANNING_N_BY_TYPE = {
    "river":  0.030,  # natural channel
    "stream": 0.035,
    "canal":  0.020,
    "drain":  MANNING_N,
    "ditch":  0.040,
    "default": MANNING_N,
}


# ---------------------------------------------------------------------------
# Helper: coordinate → node-id string (snapped)
# ---------------------------------------------------------------------------

def _node_id(lon: float, lat: float) -> str:
    """Round coordinate to snap-tolerance and return a string node key."""
    factor = 1.0 / _NODE_SNAP_TOLERANCE
    lon_s = round(lon * factor) / factor
    lat_s = round(lat * factor) / factor
    return f"{lon_s:.5f},{lat_s:.5f}"


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return great-circle distance in metres between two WGS-84 points."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Elevation sampling from grid GeoJSON
# ---------------------------------------------------------------------------

def _build_elevation_lookup(grid_geojson_path: str) -> Dict[str, float]:
    """Return a cell_id → elevation_m dict from the grid GeoJSON."""
    import json
    with open(grid_geojson_path, "r", encoding="utf-8") as fh:
        fc = json.load(fh)
    lookup: Dict[str, float] = {}
    for feat in fc.get("features", []):
        props = feat.get("properties", {})
        cid = props.get("cell_id")
        elev = props.get("elevation_m")
        if cid and elev is not None:
            lookup[cid] = float(elev)
    return lookup


def _sample_elevation_at_point(
    lon: float, lat: float, grid_gdf: gpd.GeoDataFrame,
) -> float:
    """Return the DEM elevation of the grid cell that contains (lon, lat).

    Falls back to a regional average if no cell contains the point.
    """
    pt = Point(lon, lat)
    hits = grid_gdf[grid_gdf.geometry.contains(pt)]
    if not hits.empty:
        return float(hits.iloc[0]["elevation_m"])
    # Nearest centroid fallback
    centroids = grid_gdf.geometry.centroid
    dists = centroids.distance(pt)
    nearest_idx = dists.idxmin()
    return float(grid_gdf.loc[nearest_idx, "elevation_m"])


# ---------------------------------------------------------------------------
# Manning capacity calculation
# ---------------------------------------------------------------------------

def _manning_capacity(
    waterway_type: str,
    slope: float,
) -> float:
    """Estimate channel capacity (m³/s) using Manning's equation.

    ASSUMPTION: Trapezoidal cross-section with configurable width/depth.
    All geometry is assumed — not measured.

    Q = (1/n) * A * R^(2/3) * S^(1/2)

    Parameters
    ----------
    waterway_type : str
        OSM waterway tag (river, stream, canal, drain, ditch).
    slope : float
        Hydraulic slope (dimensionless, > MIN_SLOPE).

    Returns
    -------
    float
        Estimated channel capacity in m³/s.
        Marked as an assumption — not measured infrastructure capacity.
    """
    wm = _WIDTH_MULTIPLIER.get(waterway_type, _WIDTH_MULTIPLIER["default"])
    dm = _DEPTH_MULTIPLIER.get(waterway_type, _DEPTH_MULTIPLIER["default"])
    n  = _MANNING_N_BY_TYPE.get(waterway_type, _MANNING_N_BY_TYPE["default"])

    width  = DEFAULT_DRAIN_WIDTH_M * wm
    depth  = DEFAULT_DRAIN_DEPTH_M * dm
    z      = DEFAULT_SIDE_SLOPE_HV  # side slope H:V

    # Trapezoidal cross-section at full flow (assumed)
    # Area:      A = (b + z*d) * d
    # Wetted perimeter: P = b + 2*d*sqrt(1 + z^2)
    A = (width + z * depth) * depth
    P = width + 2.0 * depth * math.sqrt(1.0 + z ** 2)
    R = A / P if P > 0 else 0.01  # hydraulic radius

    s_eff = max(slope, MIN_SLOPE)
    Q = (1.0 / n) * A * (R ** (2.0 / 3.0)) * math.sqrt(s_eff)
    return max(Q, 0.0)


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

class DrainageGraph:
    """Directed graph of the Hyderabad stormwater drainage network.

    Attributes
    ----------
    G : nx.DiGraph
        The directed drainage graph.  Each node has attributes:
        ``lon``, ``lat``, ``elevation_m``, ``node_id``.
        Each edge has attributes:
        ``length_m``, ``slope``, ``capacity_m3_s``,
        ``waterway_type``, ``osm_id``, ``estimated_capacity`` (always True),
        ``upstream_node``, ``downstream_node``, ``geometry``.
    nodes_by_id : dict
        Mapping node_id_str → node attributes.
    """

    def __init__(self) -> None:
        self.G: nx.DiGraph = nx.DiGraph()
        self._grid_gdf: Optional[gpd.GeoDataFrame] = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        waterways_path: str = "data/raw/hyderabad_waterways.geojson",
        grid_geojson_path: str = "data/grid_cells.geojson",
    ) -> "DrainageGraph":
        """Construct the drainage graph from raw GeoJSON files.

        Parameters
        ----------
        waterways_path:
            Path to hyderabad_waterways.geojson.
        grid_geojson_path:
            Path to data/grid_cells.geojson (used for DEM elevation lookup).

        Returns
        -------
        DrainageGraph
            Self (enables method chaining).
        """
        log.info("Loading waterways from '%s'", waterways_path)
        waterways = gpd.read_file(waterways_path)
        log.info("Loaded %d waterway features", len(waterways))

        # Load grid for DEM elevation sampling
        log.info("Loading grid GeoJSON for elevation sampling")
        grid_gdf = gpd.read_file(grid_geojson_path)
        self._grid_gdf = grid_gdf

        # Pre-collect all nodes and edges
        edges_to_add: List[Tuple[str, str, Dict[str, Any]]] = []
        node_attrs: Dict[str, Dict[str, Any]] = {}

        for _, row in waterways.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # Normalise to a list of coordinate pairs
            if geom.geom_type == "LineString":
                coords = list(geom.coords)
            elif geom.geom_type == "MultiLineString":
                coords = list(geom.geoms[0].coords)
            else:
                continue

            if len(coords) < 2:
                continue

            waterway_type = row.get("waterway", "drain") or "drain"
            osm_id = str(row.get("@id", ""))

            # Start and end points
            lon0, lat0 = coords[0][0], coords[0][1]
            lon1, lat1 = coords[-1][0], coords[-1][1]

            n0 = _node_id(lon0, lat0)
            n1 = _node_id(lon1, lat1)

            # Sample elevation at both endpoints
            elev0 = _sample_elevation_at_point(lon0, lat0, grid_gdf)
            elev1 = _sample_elevation_at_point(lon1, lat1, grid_gdf)

            # Total length of segment
            length_m = 0.0
            for i in range(len(coords) - 1):
                length_m += _haversine_m(
                    coords[i][0], coords[i][1],
                    coords[i + 1][0], coords[i + 1][1],
                )
            length_m = max(length_m, 1.0)  # prevent zero-length edges

            # Determine flow direction (higher elevation → lower elevation)
            if elev0 >= elev1:
                u_node, v_node = n0, n1
                elev_u, elev_v = elev0, elev1
                lon_u, lat_u = lon0, lat0
                lon_v, lat_v = lon1, lat1
            else:
                u_node, v_node = n1, n0
                elev_u, elev_v = elev1, elev0
                lon_u, lat_u = lon1, lat1
                lon_v, lat_v = lon0, lat0

            slope = max((elev_u - elev_v) / length_m, MIN_SLOPE)
            capacity = _manning_capacity(waterway_type, slope)

            # Register node attributes
            for nid, lon_, lat_, elev_ in [
                (u_node, lon_u, lat_u, elev_u),
                (v_node, lon_v, lat_v, elev_v),
            ]:
                if nid not in node_attrs:
                    node_attrs[nid] = {
                        "lon": lon_, "lat": lat_,
                        "elevation_m": elev_,
                        "node_id": nid,
                        # Accumulation fields (filled during flow calc)
                        "inflow_m3_s": 0.0,
                        "outflow_m3_s": 0.0,
                        "surcharged": False,
                    }

            edges_to_add.append((
                u_node, v_node, {
                    "length_m":          length_m,
                    "slope":             slope,
                    "capacity_m3_s":     capacity,
                    "waterway_type":     waterway_type,
                    "osm_id":            osm_id,
                    "estimated_capacity": True,  # always True: no survey data
                    "upstream_node":     u_node,
                    "downstream_node":   v_node,
                    "geometry":          geom,
                    "inflow_m3_s":       0.0,
                    "outflow_m3_s":      0.0,
                    "utilization_ratio": 0.0,
                    "surcharged":        False,
                    "surcharge_excess_m3_s": 0.0,
                    "surcharge_volume_m3":   0.0,
                }
            ))

        # Add all nodes and edges to the graph
        for nid, attrs in node_attrs.items():
            self.G.add_node(nid, **attrs)

        for u, v, attrs in edges_to_add:
            self.G.add_edge(u, v, **attrs)

        log.info(
            "Initial graph: %d nodes, %d edges",
            self.G.number_of_nodes(), self.G.number_of_edges(),
        )

        # --- Cycle detection and removal ---
        self._resolve_cycles()

        log.info(
            "Final DAG: %d nodes, %d edges",
            self.G.number_of_nodes(), self.G.number_of_edges(),
        )
        return self

    # ------------------------------------------------------------------
    # Cycle resolution
    # ------------------------------------------------------------------

    def _resolve_cycles(self) -> None:
        """Detect and remove back-edges that create cycles in the graph.

        Uses DFS to identify strongly-connected components.  Any back-edge
        discovered during traversal (i.e., edge pointing to an ancestor node
        in the DFS tree) is removed.  This produces a DAG suitable for
        downstream hydraulic accumulation.
        """
        removed = 0
        # find_cycle raises nx.NetworkXNoCycle if none found
        while True:
            try:
                cycle = nx.find_cycle(self.G, orientation="original")
                # Remove the last edge in the cycle (the back-edge)
                u, v, _ = cycle[-1]
                self.G.remove_edge(u, v)
                removed += 1
            except nx.NetworkXNoCycle:
                break
        if removed:
            log.warning("Removed %d back-edge(s) to resolve cycles", removed)
        else:
            log.info("No cycles detected — graph is already a valid DAG")

    # ------------------------------------------------------------------
    # Node nearest to a given point
    # ------------------------------------------------------------------

    def nearest_node(self, lon: float, lat: float) -> str:
        """Return the node ID closest to (lon, lat)."""
        best_nid, best_dist = None, float("inf")
        for nid, data in self.G.nodes(data=True):
            d = _haversine_m(lon, lat, data["lon"], data["lat"])
            if d < best_dist:
                best_dist = d
                best_nid = nid
        return best_nid

    # ------------------------------------------------------------------
    # Topological sort (upstream → downstream)
    # ------------------------------------------------------------------

    def topological_order(self) -> List[str]:
        """Return nodes in topological order (upstream first)."""
        return list(nx.topological_sort(self.G))
