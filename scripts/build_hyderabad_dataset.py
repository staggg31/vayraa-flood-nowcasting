import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point
import rasterio
from rasterio.mask import mask

# Grid dimensions
LAT_MIN, LAT_MAX = 17.30, 17.48
LON_MIN, LON_MAX = 78.35, 78.55
ROWS = 8
COLS = 8

def generate_grid(lat_min, lat_max, lon_min, lon_max, rows, cols):
    lat_step = (lat_max - lat_min) / rows
    lon_step = (lon_max - lon_min) / cols
    
    polygons = []
    ids = []
    
    count = 1
    for r in range(rows):
        for c in range(cols):
            cell_lat_min = lat_max - (r + 1) * lat_step
            cell_lat_max = lat_max - r * lat_step
            cell_lon_min = lon_min + c * lon_step
            cell_lon_max = lon_min + (c + 1) * lon_step
            
            p = box(cell_lon_min, cell_lat_min, cell_lon_max, cell_lat_max)
            polygons.append(p)
            ids.append(f"HYD-{count:03d}")
            count += 1
            
    gdf = gpd.GeoDataFrame({'cell_id': ids, 'geometry': polygons}, crs="EPSG:4326")
    return gdf

def main():
    root_dir = os.path.abspath(os.getcwd())
    raw_dir = os.path.join(root_dir, 'data', 'raw')
    
    print("Generating Grid...")
    grid_gdf = generate_grid(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, ROWS, COLS)
    
    # Calculate cell areas in km2 using an equal area projection (e.g. EPSG:6933)
    grid_gdf_proj = grid_gdf.to_crs("EPSG:6933")
    grid_gdf['area_km2'] = grid_gdf_proj.geometry.area / 10**6
    
    print("Loading Waterways...")
    waterways = gpd.read_file(os.path.join(raw_dir, 'hyderabad_waterways.geojson'))
    waterways_proj = waterways.to_crs("EPSG:6933")
    
    # Compute dist to nearest river and drainage density
    dist_to_river_m = []
    drainage_density = []
    for idx, row in grid_gdf.iterrows():
        cell_centroid_proj = grid_gdf_proj.loc[idx, 'geometry'].centroid
        cell_geom_proj = grid_gdf_proj.loc[idx, 'geometry']
        
        # min distance to any waterway in meters
        dists = waterways_proj.distance(cell_centroid_proj)
        if not dists.empty:
            dist_to_river_m.append(dists.min())
        else:
            dist_to_river_m.append(0.0)
            
        # total waterway length in cell
        clipped_waterways = gpd.clip(waterways_proj, cell_geom_proj)
        if not clipped_waterways.empty:
            total_length = clipped_waterways.length.sum() # in meters
            density = (total_length / 1000) / row['area_km2'] # km / km2
        else:
            density = 0.0
        drainage_density.append(density)
        
    grid_gdf['dist_to_river_m'] = dist_to_river_m
    grid_gdf['drainage_density'] = drainage_density
    
    print("Processing Inundation Points...")
    inun_path = os.path.join(raw_dir, 'hyderabad_inundation.json')
    with open(inun_path, 'r', encoding='utf-8-sig') as f:
        inundation_data = json.load(f)
        
    inundation_points = [Point(feat['geometry']['x'], feat['geometry']['y']) for feat in inundation_data['features']]
    inundation_gdf = gpd.GeoDataFrame(geometry=inundation_points, crs="EPSG:4326")
    
    flood_counts = []
    for idx, row in grid_gdf.iterrows():
        # count points inside cell
        pts_in_cell = gpd.clip(inundation_gdf, row['geometry'])
        flood_counts.append(len(pts_in_cell))
        
    grid_gdf['historical_flood_count'] = flood_counts
    
    print("Processing DEM...")
    dem_tar_path = os.path.join(raw_dir, 'rasters_SRTMGL1.tar.gz').replace('\\', '/')
    dem_path = f'/vsitar/{dem_tar_path}/output_SRTMGL1.tif'
    
    mean_elevs = []
    mean_slopes = []
    
    try:
        with rasterio.open(dem_path) as src:
            for idx, row in grid_gdf.iterrows():
                try:
                    out_image, out_transform = mask(src, [row['geometry']], crop=True)
                    out_image = out_image[0]
                    # Exclude nodata values
                    valid_mask = (out_image != src.nodata)
                    if np.any(valid_mask):
                        valid_pixels = out_image[valid_mask]
                        mean_elev = np.mean(valid_pixels)
                        
                        # Calculate slope
                        dy, dx = np.gradient(out_image)
                        slope = np.sqrt(dx**2 + dy**2)
                        valid_slope = slope[valid_mask]
                        mean_slope = np.mean(valid_slope)
                    else:
                        mean_elev = 0
                        mean_slope = 0
                except ValueError: # crop out of bounds
                    mean_elev = 0
                    mean_slope = 0
                
                mean_elevs.append(mean_elev)
                mean_slopes.append(mean_slope)
    except Exception as e:
        print(f"Warning: Could not process DEM via vsitar. Error: {e}")
        mean_elevs = [500] * len(grid_gdf)
        mean_slopes = [2.0] * len(grid_gdf)

    grid_gdf['elevation_m'] = mean_elevs
    grid_gdf['slope_deg'] = mean_slopes

    print("Processing Land Cover...")
    lc_zip_path = os.path.join(raw_dir, 'terrascope_download_20260902_225023.zip').replace('\\', '/')
    lc_path = f'/vsizip/{lc_zip_path}/WORLDCOVER/ESA_WORLDCOVER_10M_2021_V200/MAP/ESA_WorldCover_10m_2021_v200_N15E078_Map/ESA_WorldCover_10m_2021_v200_N15E078_Map.tif'
    
    impervious_ratios = []
    try:
        with rasterio.open(lc_path) as src:
            for idx, row in grid_gdf.iterrows():
                try:
                    out_image, out_transform = mask(src, [row['geometry']], crop=True)
                    out_image = out_image[0]
                    
                    valid_mask = (out_image != src.nodata)
                    if np.any(valid_mask):
                        valid_pixels = out_image[valid_mask]
                        # class 50 is built-up / impervious
                        impervious_count = np.sum(valid_pixels == 50)
                        ratio = impervious_count / len(valid_pixels)
                    else:
                        ratio = 0.0
                except ValueError:
                    ratio = 0.0
                impervious_ratios.append(ratio)
    except Exception as e:
        print(f"Warning: Could not process Land Cover via vsizip. Error: {e}")
        impervious_ratios = [0.5] * len(grid_gdf)
        
    grid_gdf['impervious_surface_ratio'] = impervious_ratios
    
    # Export Grid
    out_dir = os.path.join(root_dir, 'data')
    os.makedirs(out_dir, exist_ok=True)
    grid_gdf.to_file(os.path.join(out_dir, 'grid_cells.geojson'), driver='GeoJSON')
    print("Exported data/grid_cells.geojson")
    
    print("Generating Training Dataset...")
    
    np.random.seed(42)
    records = []
    
    # Safe medians to avoid NaNs
    median_elev = np.median(mean_elevs) if mean_elevs else 0
    median_drainage = np.median(drainage_density) if drainage_density else 0

    for idx, row in grid_gdf.iterrows():
        # generate 30 random rainfall events
        for day in range(30):
            # simulate rainfall from monsoon distribution [20, 220]
            rainfall = np.random.uniform(0, 220) if np.random.rand() > 0.5 else np.random.uniform(0, 20)
            
            # Simple heuristic for flood labels
            flood_prob = (rainfall / 220.0) * 0.4
            flood_prob += (row['impervious_surface_ratio']) * 0.3
            if row['elevation_m'] < median_elev:
                flood_prob += 0.2
            if row['drainage_density'] < median_drainage:
                flood_prob += 0.1
                
            flood_occurred = 1 if (flood_prob + np.random.normal(0, 0.1)) > 0.6 else 0
            
            records.append({
                'cell_id': row['cell_id'],
                'day': day,
                'rainfall_mm': rainfall,
                'elevation_m': row['elevation_m'],
                'slope_deg': row['slope_deg'],
                'dist_to_river_m': row['dist_to_river_m'],
                'drainage_density': row['drainage_density'],
                'impervious_surface_ratio': row['impervious_surface_ratio'],
                'historical_flood_count': row['historical_flood_count'],
                'flood_occurred': flood_occurred
            })
            
    train_df = pd.DataFrame(records)
    train_df.to_csv(os.path.join(out_dir, 'train_features.csv'), index=False)
    print("Exported data/train_features.csv")

if __name__ == '__main__':
    if not os.path.exists('data/raw'):
        print("Please run this script from the project root directory where data/raw exists.")
        exit(1)
    main()
