"""
bootstrap_grid.py
=================
Senior Geospatial Data Engineer -- SIH Flood Nowcasting
Target: Patna sector along the Ganges (lat: 25.5941, lon: 85.1376)

Outputs
-------
  data/grid_cells.geojson   -- 8x8 bounding-box grid (64 polygon cells)
  data/synthetic_train.csv  -- 2,500 synthetic training records
"""

import json
import math
import os
import sys
import random
import csv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CENTER_LAT = 25.5941
CENTER_LON = 85.1376
STEP       = 0.01
GRID_N     = 8
N_TRAIN    = 2500
SEED       = 42

random.seed(SEED)

ORIGIN_LAT = CENTER_LAT - (GRID_N / 2) * STEP
ORIGIN_LON = CENTER_LON - (GRID_N / 2) * STEP

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)
GEOJSON_PATH = os.path.join(OUT_DIR, "grid_cells.geojson")
CSV_PATH     = os.path.join(OUT_DIR, "synthetic_train.csv")

_M_PER_DEG_LAT = 111320.0
_M_PER_DEG_LON = 111320.0 * math.cos(math.radians(CENTER_LAT))


def build_grid():
    features = []
    cell_index = 1
    river_lat = ORIGIN_LAT + GRID_N * STEP

    for row in range(GRID_N):
        for col in range(GRID_N):
            lon_min = ORIGIN_LON + col * STEP
            lat_min = ORIGIN_LAT + row * STEP
            lon_max = lon_min + STEP
            lat_max = lat_min + STEP

            ring = [
                [lon_min, lat_min],
                [lon_max, lat_min],
                [lon_max, lat_max],
                [lon_min, lat_max],
                [lon_min, lat_min],
            ]

            cell_id = "CELL_{:03d}".format(cell_index)
            cell_center_lat = lat_min + STEP / 2
            dist_to_river_m = round(max(100.0, min(8500.0,
                abs(river_lat - cell_center_lat) * _M_PER_DEG_LAT)), 2)

            elev_base = 42.0 + (dist_to_river_m / 100) * 0.35
            elevation_m = round(max(38.0, elev_base + random.gauss(0, 0.4)), 2)

            slope_base = 0.2 + (dist_to_river_m / 8500) * 2.2
            slope_deg = round(max(0.2, min(2.8, slope_base + random.gauss(0, 0.1))), 3)

            drainage_density = round(max(0.2, min(2.2,
                1.8 - (dist_to_river_m / 8500) * 1.4 + random.uniform(-0.2, 0.2))), 3)

            urban_gradient = 0.5 + 0.3 * math.sin(math.pi * col / (GRID_N - 1))
            impervious_surface_ratio = round(
                max(0.15, min(0.85, urban_gradient + random.gauss(0, 0.04))), 4)

            flood_prob = max(0.0, 1.0 - dist_to_river_m / 6000)
            historical_flood_count = min(5, max(0,
                int(round(flood_prob * 5 * random.uniform(0.6, 1.0)))))

            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "cell_id":                  cell_id,
                    "dist_to_river_m":          dist_to_river_m,
                    "elevation_m":              elevation_m,
                    "slope_deg":                slope_deg,
                    "drainage_density":         drainage_density,
                    "impervious_surface_ratio": impervious_surface_ratio,
                    "historical_flood_count":   historical_flood_count,
                },
            })
            cell_index += 1

    return features


def write_geojson(features, path):
    fc = {
        "type": "FeatureCollection",
        "name": "patna_flood_grid",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fc, fh, indent=2)
    print("  [OK] GeoJSON saved  -> {}".format(os.path.relpath(path)))
    print("       {} polygon features".format(len(features)))


FLOOD_COLS = [
    "elevation_m", "slope_deg", "dist_to_river_m",
    "drainage_density", "impervious_surface_ratio",
    "rainfall_1h_mm", "flood_occurred",
]


def _flood_label(elevation_m, impervious_surface_ratio, dist_to_river_m, rainfall_1h_mm):
    score = (
        rainfall_1h_mm             * 0.45
        + impervious_surface_ratio * 25
        - elevation_m              * 0.3
        - dist_to_river_m          * 0.015
    )
    return int(score > 12)


def build_training_csv(n, path):
    rows = []
    for _ in range(n):
        dist_to_river_m          = random.uniform(100, 8500)
        elevation_m              = max(38.0, 42.0 + (dist_to_river_m / 100) * 0.35
                                       + random.gauss(0, 1.2))
        slope_deg                = round(max(0.2, min(2.8, random.uniform(0.2, 2.8))), 3)
        drainage_density         = round(max(0.2, min(2.2, random.uniform(0.2, 2.2))), 3)
        impervious_surface_ratio = round(max(0.15, min(0.85, random.uniform(0.15, 0.85))), 4)
        rainfall_1h_mm           = max(0.0, random.expovariate(1 / 18))

        rows.append({
            "elevation_m":              round(elevation_m, 2),
            "slope_deg":                slope_deg,
            "dist_to_river_m":          round(dist_to_river_m, 2),
            "drainage_density":         drainage_density,
            "impervious_surface_ratio": impervious_surface_ratio,
            "rainfall_1h_mm":           round(rainfall_1h_mm, 3),
            "flood_occurred":           _flood_label(elevation_m,
                                            impervious_surface_ratio,
                                            dist_to_river_m, rainfall_1h_mm),
        })

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FLOOD_COLS)
        writer.writeheader()
        writer.writerows(rows)

    flood_count = sum(r["flood_occurred"] for r in rows)
    print("  [OK] Training CSV saved -> {}".format(os.path.relpath(path)))
    print("       {} records | flood_occurred=1: {} ({:.1f}%)".format(
        n, flood_count, flood_count / n * 100))


def print_summary(features):
    props = [f["properties"] for f in features]
    fields = ["dist_to_river_m", "elevation_m", "slope_deg",
              "drainage_density", "impervious_surface_ratio", "historical_flood_count"]
    print("\n  Grid cell property ranges:")
    print("  {:<30} {:>10} {:>10}".format("Field", "Min", "Max"))
    print("  " + "-" * 52)
    for field in fields:
        vals = [p[field] for p in props]
        print("  {:<30} {:>10.3f} {:>10.3f}".format(field, min(vals), max(vals)))


def main():
    print("\n" + "=" * 52)
    print("  SIH Flood Nowcasting -- Bootstrap Grid")
    print("  Target: Patna Ganges sector")
    print("  Centre: {}N, {}E".format(CENTER_LAT, CENTER_LON))
    print("  Grid:   {}x{} = {} cells  (step={}deg)".format(
        GRID_N, GRID_N, GRID_N ** 2, STEP))
    print("=" * 52 + "\n")

    print("[1/2] Building grid ...")
    features = build_grid()
    write_geojson(features, GEOJSON_PATH)
    print_summary(features)

    print("\n[2/2] Generating synthetic training data ...")
    build_training_csv(N_TRAIN, CSV_PATH)

    print("\n" + "=" * 52)
    print("  Bootstrap complete. Data foundation ready.")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    main()
    sys.exit(0)
