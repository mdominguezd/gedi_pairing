import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import ee
import geemap
import pandas as pd
import numpy as np
from shapely.geometry import box
from tqdm import tqdm


def terrain_correction(image):
    """
    Applies volumetric terrain correction to Sentinel-1 GRD imagery
    """
    img_geom = image.geometry()
    srtm = ee.Image('USGS/SRTMGL1_003').clip(img_geom)  # 30m srtm
    
    sigma0_pow = ee.Image.constant(10).pow(image.divide(10.0))
    
    # Article (numbers relate to chapters)
    # 2.1.1 Radar geometry
    theta_i = image.select('angle')
    phi_i = ee.Terrain.aspect(theta_i) \
        .reduceRegion(ee.Reducer.mean(), theta_i.get('system:footprint'), 1000) \
        .get('aspect')
    
    # 2.1.2 Terrain geometry
    alpha_s = ee.Terrain.slope(srtm).select('slope')
    phi_s = ee.Terrain.aspect(srtm).select('aspect')
    
    # 2.1.3 Model geometry
    # Reduce to 3 angles
    phi_r = ee.Image.constant(phi_i).subtract(phi_s)
    
    # Convert all to radians
    phi_r_rad = phi_r.multiply(math.pi / 180)
    alpha_s_rad = alpha_s.multiply(math.pi / 180)
    theta_i_rad = theta_i.multiply(math.pi / 180)
    ninety_rad = ee.Image.constant(90).multiply(math.pi / 180)
    
    # Slope steepness in range (eq. 2)
    alpha_r = (alpha_s_rad.tan().multiply(phi_r_rad.cos())).atan()
    
    # Slope steepness in azimuth (eq. 3)
    alpha_az = (alpha_s_rad.tan().multiply(phi_r_rad.sin())).atan()
    
    # Local incidence angle (eq. 4)
    theta_lia = (alpha_az.cos().multiply((theta_i_rad.subtract(alpha_r)).cos())).acos()
    theta_lia_deg = theta_lia.multiply(180 / math.pi)
    
    # 2.2
    # Gamma_nought_flat
    gamma0 = sigma0_pow.divide(theta_i_rad.cos())
    gamma0_db = ee.Image.constant(10).multiply(gamma0.log10())
    ratio_1 = gamma0_db.select('VV').subtract(gamma0_db.select('VH'))
    
    # Volumetric Model
    nominator = (ninety_rad.subtract(theta_i_rad).add(alpha_r)).tan()
    denominator = (ninety_rad.subtract(theta_i_rad)).tan()
    vol_model = (nominator.divide(denominator)).abs()
    
    # Apply model
    gamma0_volume = gamma0.divide(vol_model)
    gamma0_volume_db = ee.Image.constant(10).multiply(gamma0_volume.log10())
    
    # We add a layover/shadow mask to the original implementation
    # Layover: where slope > radar viewing angle
    alpha_r_deg = alpha_r.multiply(180 / math.pi)
    layover = alpha_r_deg.lt(theta_i)
    
    # Shadow: where LIA > 90
    shadow = theta_lia_deg.lt(85)
    
    # Calculate the ratio for RGB visualization
    ratio = gamma0_volume_db.select('VV').subtract(gamma0_volume_db.select('VH'))
    
    output = gamma0_volume_db.addBands(ratio).addBands(alpha_r).addBands(phi_s) \
        .addBands(theta_i_rad).addBands(layover).addBands(shadow) \
        .addBands(gamma0_db).addBands(ratio_1)
    
    return image.addBands(
        output.select(['VV', 'VH'], ['VV', 'VH']),
        None,
        True
    )


def power_to_db(img):
    """Convert from linear power to dB"""
    return ee.Image(10).multiply(img.log10())


def db_to_power(img):
    """Convert from dB to linear power"""
    return ee.Image(10).pow(img.divide(10))


def create_spatial_chunks(gdf, max_chunk_size=4500):
    """
    Split a GeoDataFrame into spatial chunks using a grid-based approach.
    Chunks are created so that no chunk extent exceeds max_chunk_size x max_chunk_size meters.
    
    Parameters:
    gdf (GeoDataFrame): Input GeoDataFrame with point geometries
    max_chunk_size (float): Maximum size (in meters) of chunk extent in either dimension
    
    Returns:
    list: List of GeoDataFrame chunks
    """
    # Work with projected CRS for accurate distances (EPSG:25830 is in meters)
    gdf_proj = gdf.to_crs(25830)
    
    # Get bounds in meters
    minx, miny, maxx, maxy = gdf_proj.total_bounds
    
    # Calculate total extent
    total_width = maxx - minx
    total_height = maxy - miny
    
    # Calculate number of columns and rows needed
    n_cols = int(np.ceil(total_width / max_chunk_size))
    n_rows = int(np.ceil(total_height / max_chunk_size))
    

    # Calculate actual cell dimensions (will be <= max_chunk_size)
    cell_width = total_width / n_cols
    cell_height = total_height / n_rows
    
    chunks = []
    chunk_info = []
    
    # Create grid cells and extract points
    for i in range(n_rows):
        for j in range(n_cols):
            # Define cell bounds with small buffer to capture boundary points
            cell_minx = minx + j * cell_width
            cell_miny = miny + i * cell_height
            cell_maxx = minx + (j + 1) * cell_width
            cell_maxy = miny + (i + 1) * cell_height
            
            # Add small epsilon to max bounds for last row/column to ensure all points are captured
            if i == n_rows - 1:
                cell_maxy += 1e-6  # Small epsilon
            if j == n_cols - 1:
                cell_maxx += 1e-6
            
            # Create bounding box
            cell_box = box(cell_minx, cell_miny, cell_maxx, cell_maxy)
            
            # Find points within or on the boundary of this cell
            # Use intersects instead of within to capture boundary points
            mask = gdf_proj.geometry.intersects(cell_box)
            
            # For all cells except the last row/column, exclude points on the max boundaries
            # to avoid duplicates (each point should only appear in one chunk)
            if i < n_rows - 1 or j < n_cols - 1:
                # Get coordinates
                coords = gdf_proj.geometry.get_coordinates()
                x_coords = coords['x']
                y_coords = coords['y']
                
                # Exclude points exactly on the max boundaries (they'll be in the next cell)
                if i < n_rows - 1:
                    mask = mask & (y_coords < cell_maxy)
                if j < n_cols - 1:
                    mask = mask & (x_coords < cell_maxx)
            
            # Use .copy() to ensure we have a proper copy with preserved index
            chunk_gdf = gdf[mask].copy()
            
            if len(chunk_gdf) > 0:
                chunks.append(chunk_gdf)
                
                # Calculate actual chunk extent
                chunk_bounds = gdf_proj[mask].total_bounds
                chunk_width = chunk_bounds[2] - chunk_bounds[0]
                chunk_height = chunk_bounds[3] - chunk_bounds[1]
                chunk_area = chunk_width * chunk_height
                
                chunk_info.append({
                    'chunk_id': len(chunks) - 1,
                    'n_points': len(chunk_gdf),
                    'extent_width': chunk_width,
                    'extent_height': chunk_height,
                    'extent_area': chunk_area
                })
    
    return chunks


def _ee_extraction_chunk(gdf_chunk, img_col, band_name, agg_fun, scale, dates, 
                         slope, aspect, cloud_free, s1_corr):
    """
    Internal function to extract EE values for a chunk of the GeoDataFrame.
    This is called by each thread in the concurrent processing.
    """
    if dates is None or len(dates) != 2:
        if agg_fun == 'mean':
            img = ee.ImageCollection(img_col).select(band_name).mean()
        elif agg_fun == 'median':
            img = ee.ImageCollection(img_col).select(band_name).median()
        elif agg_fun == 'mode':
            img = ee.ImageCollection(img_col).select(band_name).mode()
        else:
            raise ValueError(f"Unsupported aggregation function: {agg_fun}")
        
        if slope:
            img = ee.Terrain.slope(img.reproject(crs='EPSG:4326', scale=scale)).select('slope')
        
        if aspect:
            img = ee.Terrain.aspect(img.reproject(crs='EPSG:4326', scale=scale)).select('aspect')
    else:
        # Get EE Image and apply the aggregation function
        col = ee.ImageCollection(img_col).filterDate(dates[0], dates[1])

        if cloud_free:
            # Ensure both the target band and MSK_CLDPRB are present
            def mask_clouds(img):
                cloud_mask = img.select('MSK_CLDPRB').lt(5)
                return img.updateMask(cloud_mask)

            col = col.map(mask_clouds)

        if s1_corr:
            col = col.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            col = col.map(terrain_correction)
            col = col.map(db_to_power)

        if band_name != None:
            col = col.select(band_name)
    
        if agg_fun == 'mean':
            img = col.mean()
        elif agg_fun == 'median':
            img = col.median()
        elif agg_fun == 'mode':
            img = col.mode()
        else:
            raise ValueError(f"Unsupported aggregation function: {agg_fun}")

    # Extract values at points
    gdf_chunk = gdf_chunk.copy()
    gdf_chunk['ee_id'] = gdf_chunk.index.astype(str)
    points = geemap.gdf_to_ee(gdf_chunk[['ee_id', 'geometry']])
    values = img.sampleRegions(collection=points, scale=scale).getInfo()

    # Map results back using the 'ee_id'
    if slope:
        results = {f['properties']['ee_id']: f['properties'].get('slope', None) for f in values['features']}
        for col_name in gdf_chunk.columns:
            if col_name.startswith('_extracted_slope_'):
                gdf_chunk[col_name] = gdf_chunk['ee_id'].map(results)
    elif aspect:
        results = {f['properties']['ee_id']: f['properties'].get('aspect', None) for f in values['features']}
        for col_name in gdf_chunk.columns:
            if col_name.startswith('_extracted_aspect_'):
                gdf_chunk[col_name] = gdf_chunk['ee_id'].map(results)
    else:
        if band_name != None:
            for band in values['features'][0]['properties'].keys():
                if band != 'ee_id':
                    results = {f['properties']['ee_id']: f['properties'].get(band, None) for f in values['features']}
                    gdf_chunk[band] = gdf_chunk['ee_id'].map(results)

    return gdf_chunk.drop(columns=['ee_id'])


def extract_ee_values(gdf, img_col, dates, band_name=None, agg_fun='mean', 
                      column_name='extracted_value', scale=10, slope=False, 
                      aspect=False, cloud_free=False, s1_corr=False, 
                      max_size=4700*4700, n_workers=10, max_chunk_size=4500):
    """
    Extracts aggregated Earth Engine image values at points in a GeoDataFrame using concurrent requests
    with spatial chunking for optimal performance.

    Parameters:
    gdf (GeoDataFrame): A GeoDataFrame containing point geometries.
    img_col (str): The name of the Earth Engine image collection.
    dates (list): A list of two strings representing the start and end dates (e.g., ['2020-01-01', '2020-12-31']).
    band_name (str or list): Band name(s) to extract.
    agg_fun (str): Aggregation function to apply (e.g., 'mean', 'median', etc.).
    column_name (str): Name of the new column to store extracted values.
    scale (int): Scale for data extraction.
    slope (bool): Whether to extract slope values.
    aspect (bool): Whether to extract aspect values.
    cloud_free (bool): Whether to apply cloud masking (for Sentinel-2).
    s1_corr (bool): Whether to apply terrain correction (for Sentinel-1).
    max_size (float): Maximum area of extent of points over which values will be extracted (warning threshold).
    n_workers (int): Number of concurrent workers (threads) to use.
    max_chunk_size (float): Maximum size (in meters) of chunk extent in either dimension (default: 4500m).

    Returns:
    GeoDataFrame: The input GeoDataFrame with additional columns containing the extracted values.
    """
    
    if len(gdf) == 1:
        if slope:
            temp_col = f'_extracted_slope_{column_name}'
            gdf[temp_col] = None
        elif aspect:
            temp_col = f'_extracted_aspect_{column_name}'
            gdf[temp_col] = None

        result_gdf = _ee_extraction_chunk(gdf, img_col, band_name, agg_fun, scale, dates, slope, aspect, cloud_free, s1_corr)
    else:
        minx, miny, maxx, maxy = gdf.to_crs(25830).total_bounds
        size = (maxx - minx) * (maxy - miny)

        # Check if the area exceeds max_size and warn the user
        if size > max_size:
            warnings.warn(
                f"The extent area ({size:.0f}m²) exceeds max_size ({max_size:.0f}m²). "
                "Using spatial chunking to process in smaller batches.",
                UserWarning
            )

        # Create spatial chunks based on maximum extent size
        gdf_chunks = create_spatial_chunks(gdf, max_chunk_size=max_chunk_size)

        # Add temporary column names for slope/aspect if needed
        if slope:
            temp_col = f'_extracted_slope_{column_name}'
            for chunk in gdf_chunks:
                chunk[temp_col] = None
        elif aspect:
            temp_col = f'_extracted_aspect_{column_name}'
            for chunk in gdf_chunks:
                chunk[temp_col] = None

        # Process chunks concurrently with progress bar
        results = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(
                    _ee_extraction_chunk,
                    chunk,
                    img_col,
                    band_name,
                    agg_fun,
                    scale,
                    dates,
                    slope,
                    aspect,
                    cloud_free,
                    s1_corr
                ): i for i, chunk in enumerate(gdf_chunks)
            }
            
            # Collect results as they complete with progress bar
            # with tqdm(total=len(gdf_chunks), desc="Processing chunks") as pbar:
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append((chunk_idx, result))
                except Exception as exc:
                    raise exc

        # Sort results by chunk index to maintain consistency
        results.sort(key=lambda x: x[0])
        processed_chunks = [r[1] for r in results]

        try:
            # Concatenate all chunks back together
            result_gdf = pd.concat(processed_chunks, ignore_index=False)
        except Exception as e:
            print('Error at concatenation: ', len(processed_chunks), len(gdf), e)

    
    # Sort by the original index to preserve order
    result_gdf = result_gdf.sort_index()
    
    # Verify all indices are present
    missing_indices = set(gdf.index) - set(result_gdf.index)
    if missing_indices:
        raise ValueError(f"Missing {len(missing_indices)} indices after processing: {list(missing_indices)[:10]}...")
    
    # Reindex to match original order exactly
    result_gdf = result_gdf.reindex(gdf.index)
    
    # Rename temporary columns if slope/aspect was used
    if slope and temp_col in result_gdf.columns:
        result_gdf = result_gdf.rename(columns={temp_col: column_name})
    elif aspect and temp_col in result_gdf.columns:
        result_gdf = result_gdf.rename(columns={temp_col: column_name})

    return result_gdf

import ee

def build_feature_stack(region, dates, scale=10):
    """
    15-band feature image matching the extract_ee_values pairing:
      S2  -> B1,B2,B3,B4,B5,B6,B7,B8,B9,B11,B12  (cloud-masked, MEDIAN)
      S1  -> VV, VH                              (terrain-corrected -> power, MEDIAN)
      DEM -> slope, aspect  (COPERNICUS/DEM/GLO30, reprojected EPSG:4326 @ 25 m)
    region : ee.Geometry
    dates  : [start, end]  (used for S2 & S1; DEM ignores dates)
    """

    # ---------- Sentinel-2: cloud-masked median ----------
    def mask_s2(img):
        return img.updateMask(img.select('MSK_CLDPRB').lt(5))

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(region)
          .filterDate(dates[0], dates[1])
          .map(mask_s2)
          .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6',
                   'B7', 'B8', 'B9', 'B11', 'B12'])
          .median())

    # ---------- Sentinel-1: same filtering as extraction ----------
    # extraction only applies: filterDate + listContains('VV'), then correction + db_to_power
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(region)
          .filterDate(dates[0], dates[1])
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .map(terrain_correction)
          .map(db_to_power)
          .select(['VV', 'VH'])
          .median())

    # ---------- DEM slope & aspect: replicate extract_ee_values exactly ----------
    # your call: img_col='COPERNICUS/DEM/GLO30', band_name='DEM',
    #            agg_fun='mean', scale=25, slope/aspect=True
    dem    = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mean()
    dem_rp = dem.reproject(crs='EPSG:25830', scale=25)
    
    slope  = ee.Terrain.slope(dem_rp).select('slope').reproject(crs='EPSG:4326', scale=25)
    aspect = ee.Terrain.aspect(dem_rp).select('aspect').reproject(crs='EPSG:4326', scale=25)

    stack = (s2.addBands(s1)
               .addBands(slope)
               .addBands(aspect)
               .toFloat())

    return stack.clip(region)