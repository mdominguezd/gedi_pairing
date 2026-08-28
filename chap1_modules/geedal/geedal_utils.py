"""
    Module with code used for interaction with google earth engine (Image download, filtering, etc.)
"""

import glob
import os
import shutil
import warnings
import json
import math

from osgeo import gdal, ogr
import rasterio as rio
import ee
import numpy as np
import geemap
import geopandas as gpd
from shapely.geometry import box
import pandas as pd

def earthengine_init():
    """
        Function to initialize earth engine API (Assumes authentication has already been sorted out)
    """

    ee.Initialize()

def sample_raster(raster_path, gdf_, new_column_name = "value"):
    """
    Samples raster values at the locations specified by a GeoDataFrame's point geometries and adds the sampled values as a new column.
    Parameters:
        raster_path (str): Path to the raster file to sample from.
        gdf (geopandas.GeoDataFrame): GeoDataFrame containing point geometries to sample raster values at.
        new_column_name (str, optional): Name of the new column to store sampled raster values. Defaults to "value".
    Returns:
        geopandas.GeoDataFrame: The input GeoDataFrame with an additional column containing the sampled raster values. If a point falls on a raster nodata value, None is assigned.
    """
    # Mask out points outside burned areas
    with rio.open(raster_path) as src:
        # Get values at the point coordinates
        coords = [(x,y) for x, y in zip(gdf_.geometry.x, gdf_.geometry.y)]
        values = list(src.sample(coords))

    gdf_[new_column_name] = [v[0] if v[0] != src.nodata else None for v in values]
    
    return gdf_

def get_image_aoi(img_col :str, aoi_fn: str, dates : list, agg_fun : str = 'min'):
    """
        Function that gets an image from GEE and clips it based on an area of interest.

        Input:
            - img_col: Image collection as described on https://developers.google.com/earth-engine/datasets/catalog/
            - aoi_fn: Filename of the area of interest with which the image will be clipped.
            - dates: List with a start date and end date of images from the collection to be considered. (Format: YYYY-MM-DD)
            - agg_fun: Function to be used for aggregating images. Min by default
    """

    # Map of aggregation functions
    agg_methods = {
        'min': lambda col: col.min(),
        'max': lambda col: col.max(),
        'mean': lambda col: col.mean(),
        'mode': lambda col: col.mode(),
        'median': lambda col: col.median(),
        'first': lambda col: col.first()
    }

    # Validate agg_fun input
    if agg_fun not in agg_methods:
        raise ValueError(f"Invalid agg_fun '{agg_fun}'.\
                          Supported options: {list(agg_methods.keys())}")


    aoi = geemap.geojson_to_ee(aoi_fn)

    img_collection = ee.ImageCollection(img_col)\
        .filterBounds(aoi.geometry())\
        .filterDate(*dates)
    image = agg_methods[agg_fun](img_collection)    

    return image.clip(aoi)

def merge_tifs(input_folder, output_file):
    """
        Function to merge multiple tifs into one
    """
    # Find all .tif files in the input folder
    tif_files = glob.glob(os.path.join(input_folder, "*.tif"))
    
    # Open all input .tif files as datasets
    input_files = [gdal.Open(tif) for tif in tif_files]

    # Use gdal.Warp to merge the tifs into one
    gdal.Warp(output_file, input_files, format='GTiff')

def calculate_intervals(aoi, resolution, max_pixels=1e7):
    """
    Calculate fishnet intervals (h_interval and v_interval) based on AOI size,
    resolution, and maximum allowable pixels per tile.

    Input:
        - aoi: The area of interest as an EE Feature or Geometry.
        - resolution: The desired resolution in meters.
        - max_pixels: Maximum number of pixels per tile.

    Output:
        - h_interval: Horizontal interval in degrees.
        - v_interval: Vertical interval in degrees.
    """
    # Get AOI bounds
    bounds = aoi.geometry().bounds().getInfo()['coordinates'][0]
    lon_min, lat_min = bounds[0][0], bounds[0][1]
    lon_max, lat_max = bounds[2][0], bounds[2][1]

    # Calculate AOI dimensions in degrees
    width = lon_max - lon_min
    height = lat_max - lat_min

    # Convert degrees to meters using approximate scaling at the equator
    # (1 degree longitude ≈ 111.32 km, 1 degree latitude ≈ 111.32 km)
    width_meters = width * 111320
    height_meters = height * 111320

    # Calculate optimal tile dimensions in meters
    tile_area = max_pixels * (resolution**2)  # Area of one tile in square meters
    tile_width_meters = tile_height_meters = tile_area**0.5

    # Convert tile dimensions back to degrees
    h_interval = tile_width_meters / 111320
    v_interval = tile_height_meters / 111320

    return min(h_interval, width), min(v_interval, height)

def download_gee_image(
    image, 
    aoi_fn: str, 
    out_dir: str, 
    resolution: int = 250, 
    out_fn: str = 'output.tif', 
    max_pixels: int = int(1e7),
    max_tile_size: int = 32
):
    """
    Function to download an image from GEE to local drive by dividing it using a dynamically calculated grid,
    and then merging it back together using GDAL.

    Input:
        - image: GEE image to download.
        - aoi_fn: Filename of the AOI GeoJSON.
        - out_dir: Output directory for the tiles.
        - resolution: Resolution in meters (default: 250).
        - out_fn: Output filename for the merged image (default: 'output.tif').
        - max_pixels: Maximum allowable pixels per tile (default: 10 million).
        - max_tile_size: Maximum size of each tile in MB (default: 32).
        - max_tile_dim: Maximum width/height in pixels for each tile (default: 256).
    """
    
    # If the output directory exists, remove all its contents
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # Load AOI
    aoi = geemap.geojson_to_ee(aoi_fn)

    # Dynamically calculate fishnet intervals
    h_interval, v_interval = calculate_intervals(aoi, resolution, max_pixels)

    # Generate fishnet grid
    fishnet = geemap.fishnet(aoi, h_interval=h_interval, v_interval=v_interval, delta=0.5)

    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Download image tiles in parallel with stricter tile limits
    geemap.download_ee_image_tiles_parallel(
        image, 
        fishnet, 
        out_dir=out_dir, 
        scale=resolution, 
        crs="EPSG:4326", 
        max_tile_size=max_tile_size,
    )

    # Merge the downloaded tiles
    merge_tifs(out_dir, out_fn)

    print(f"Image successfully downloaded and merged into {out_fn}")

    


def vectorize_raster(raster_file : str, vector_file : str):
    """
        Function that uses gdal to vectorize a raster file.
    """

    # Open the raster dataset
    src_ds = gdal.Open(raster_file)
    band = src_ds.GetRasterBand(1)

    # Create the output shapefile
    driver = ogr.GetDriverByName("GeoJSON")
    out_ds = driver.CreateDataSource(vector_file)
    out_layer = out_ds.CreateLayer("layerName", geom_type=ogr.wkbPolygon)

    # Add a new field to store raster values
    field = ogr.FieldDefn("value", ogr.OFTInteger)
    out_layer.CreateField(field)

    # Perform the raster-to-vector conversion
    gdal.Polygonize(band, None, out_layer, 0, [], callback=None)

    # Cleanup
    src_ds = None
    out_ds = None

def extract_ee_values(gdf, img_col, dates, band_name = None, agg_fun='mean', column_name='extracted_value', scale = 10, 
                      slope = False, aspect= False, cloud_free = False, s1_corr = False, max_size = 4700*4700):
    """
    Extracts aggregated Earth Engine image values at points in a GeoDataFrame.

    Parameters:
    gdf (GeoDataFrame): A GeoDataFrame containing point geometries.
    img_col (str): The name of the Earth Engine image collection.
    dates (list): A list of two strings representing the start and end dates (e.g., ['2020-01-01', '2020-12-31']).
    agg_fun (str): Aggregation function to apply (e.g., 'mean', 'median', etc.).
    column_name (str): Name of the new column to store extracted values.
    scale (int): scale for data extraction.
    max_size (float): Maximum area of extent of GEDI shots over which values will be extracted.

    Returns:
    GeoDataFrame: The input GeoDataFrame with an additional column containing the extracted values.
    """

    minx, miny, maxx, maxy = gdf.to_crs(25830).total_bounds

    size = (maxx - minx) * (maxy - miny)

    def _ee_extraction(gdf, img_col, band_name, agg_fun, scale, column_name, slope, aspect, cloud_free, s1_corr):

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

            # warnings.warn("Using the entire time series for the image collection.", UserWarning)
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
        gdf['ee_id'] = gdf.index.astype(str)
        points = geemap.gdf_to_ee(gdf[['ee_id', 'geometry']])
        values = img.sampleRegions(collection=points, scale=scale).getInfo()

        # Check if the number of points in the GeoDataFrame matches the number of points in the EE FeatureCollection
        if points.size().getInfo() != len(gdf):
            raise ValueError("Mismatch between the number of points in the GeoDataFrame and the EE FeatureCollection.")

        # Map results back using the 'ee_id'
        if slope:
            results = {f['properties']['ee_id']: f['properties'].get('slope', None) for f in values['features']}
            gdf[column_name] = gdf['ee_id'].map(results)
        elif aspect:
            results = {f['properties']['ee_id']: f['properties'].get('aspect', None) for f in values['features']}
            gdf[column_name] = gdf['ee_id'].map(results)
        else:
            if band_name != None:
                for band in values['properties']['band_order']:
                    gdf[band] = gdf['ee_id'].map({f['properties']['ee_id']: f['properties'][band] for f in values['features']})
            else:
                results = {f['properties']['ee_id']: f['properties'] for f in values['features']}

        return gdf.drop(columns=['ee_id'])
    
    def _spatial_split(gdf, nx=2, ny=2):
        xmin, ymin, xmax, ymax = gdf.total_bounds

        x_bins = np.linspace(xmin, xmax, nx + 1)
        y_bins = np.linspace(ymin, ymax, ny + 1)

        gdf = gdf.copy()
        gdf["x_bin"] = np.clip(
            np.digitize(gdf.geometry.x, x_bins) - 1, 0, nx - 1
        )
        gdf["y_bin"] = np.clip(
            np.digitize(gdf.geometry.y, y_bins) - 1, 0, ny - 1
        )


        return {
            (i, j): gdf[(gdf.x_bin == i) & (gdf.y_bin == j)]
            for i in range(nx)
            for j in range(ny)
        }

    result_gdf = []

    for subset_gdf in _spatial_split(gdf, np.int16(np.ceil(np.sqrt(size/max_size))), np.int16(np.ceil(np.sqrt(size/max_size)))).values():
        result_gdf.append(_ee_extraction(subset_gdf.drop(columns = ['x_bin', 'y_bin']), img_col, band_name, agg_fun, scale, column_name, slope, aspect, cloud_free, s1_corr))

    result_gdf = pd.concat(result_gdf)

    return result_gdf
    


################ SPECIFIC FOR TOPOGRAPHIC VARIABLES ####################


def get_cop_dem_aoi(aoi_fn: str):
    """
        Function that gets the copernicus global DEM from GEE and clips it based on an area of interest.

        Input:
            - aoi_fn: Filename of the area of interest with which the image will be clipped.
    """

    aoi = geemap.geojson_to_ee(aoi_fn)

    dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').median().clip(aoi)

    return dem


def extract_topo_vars(dem : ee.Image, data_points : ee.FeatureCollection):
    """
        Function that extracts topographic variables (elevation and slope) from a DEM image.
        Input:
            - dem: DEM image from which to extract topographic variables.
            - data_points: FeatureCollection containing points for which to extract topographic variables.
        Output:
            - slope: FeatureCollection with extracted elevation and slope values.
    """

    elevation = dem.sampleRegions(collection = data_points,
                                 scale = 30,
                                 properties = ['id'],
                                 geometries = True)
    
    sl = ee.Terrain.slope(dem.reproject(crs='EPSG:4326', scale=30)).select('slope')

    slope = sl.sampleRegions(collection = elevation,
                             scale = 30,
                             properties = ['id', 'DEM'],
                             geometries = True
                             )

    return slope

def get_country_geojson(country_na : str):
    """
        Function to get
    """
    # Load the Large Scale International Boundary dataset
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")

    # Filter for Spain
    spain = countries.filter(ee.Filter.eq('country_na', country_na))

    # Get only the geometry
    spain_geometry = spain.first().geometry()

    # Convert to GeoJSON
    geojson_data = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": spain_geometry.getInfo(),
            "properties": {}
        }]
    }

    with open(country_na+'.geojson', 'w') as f:
        json.dump(geojson_data, f, indent=2)


def divide_in_fishnet(gdf: gpd.GeoDataFrame(), size: float = 2.5):
    # Get bounding box coordinates
    minx, miny, maxx, maxy = gdf.total_bounds

    # Define grid size (adjust depending on CRS: degrees for EPSG:4326, meters for projected)
    cell_width = size
    cell_height = size

    # Generate rows and columns
    cols = np.arange(minx, maxx, cell_width)
    rows = np.arange(miny, maxy, cell_height)

    # Create polygons for each grid cell
    grid_cells = []
    for x in cols:
        for y in rows:
            cell = box(x, y, x + cell_width, y + cell_height)
            grid_cells.append(cell)

    # Convert to GeoDataFrame
    fishnet = gpd.GeoDataFrame({'geometry': grid_cells}, crs=gdf.crs)

    fishnet = gpd.sjoin(fishnet, gdf, how='inner')

    return fishnet
####################################################




























#####################################################


# Inspired from https://code.earthengine.google.com/44a1d10e3dbe1731f30d4e2ab70e8d5d

# Implementation by Andreas Vollrath (ESA), inspired by Johannes Reiche (Wageningen)
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

def refined_lee(image):
    """
    Refined Lee speckle filter
    
    This is a sophisticated edge-preserving speckle filter that:
    - Identifies edge directions in 8 orientations
    - Applies directional filtering along edges
    - Preserves sharp boundaries while reducing speckle
    
    Best for: Forest boundaries, urban areas, any features with linear edges
    """
    
    band_names = image.bandNames()
    image = db_to_power(image)
    
    def process_band(b):
        img = image.select([b])
        
        # img must be in natural units, i.e. not in dB!
        # Set up 3x3 kernels
        weights3 = ee.List.repeat(ee.List.repeat(1, 3), 3)
        kernel3 = ee.Kernel.fixed(3, 3, weights3, 1, 1, False)
        
        mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3)
        variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3)
        
        # Use a sample of the 3x3 windows inside a 7x7 windows to determine gradients and directions
        sample_weights = ee.List([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])
        
        sample_kernel = ee.Kernel.fixed(7, 7, sample_weights, 3, 3, False)
        
        # Calculate mean and variance for the sampled windows and store as 9 bands
        sample_mean = mean3.neighborhoodToBands(sample_kernel)
        sample_var = variance3.neighborhoodToBands(sample_kernel)
        
        # Determine the 4 gradients for the sampled windows
        gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs()
        gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs())
        gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs())
        gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs())
        
        # And find the maximum gradient amongst gradient bands
        max_gradient = gradients.reduce(ee.Reducer.max())
        
        # Create a mask for band pixels that are the maximum gradient
        gradmask = gradients.eq(max_gradient)
        
        # Duplicate gradmask bands: each gradient represents 2 directions
        gradmask = gradmask.addBands(gradmask)
        
        # Determine the 8 directions
        directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(
            sample_mean.select(4).subtract(sample_mean.select(7))
        ).multiply(1)
        directions = directions.addBands(
            sample_mean.select(6).subtract(sample_mean.select(4)).gt(
                sample_mean.select(4).subtract(sample_mean.select(2))
            ).multiply(2)
        )
        directions = directions.addBands(
            sample_mean.select(3).subtract(sample_mean.select(4)).gt(
                sample_mean.select(4).subtract(sample_mean.select(5))
            ).multiply(3)
        )
        directions = directions.addBands(
            sample_mean.select(0).subtract(sample_mean.select(4)).gt(
                sample_mean.select(4).subtract(sample_mean.select(8))
            ).multiply(4)
        )
        # The next 4 are the not() of the previous 4
        directions = directions.addBands(directions.select(0).Not().multiply(5))
        directions = directions.addBands(directions.select(1).Not().multiply(6))
        directions = directions.addBands(directions.select(2).Not().multiply(7))
        directions = directions.addBands(directions.select(3).Not().multiply(8))
        
        # Mask all values that are not 1-8
        directions = directions.updateMask(gradmask)
        
        # "Collapse" the stack into a single band image
        directions = directions.reduce(ee.Reducer.sum())
        
        sample_stats = sample_var.divide(sample_mean.multiply(sample_mean))
        
        # Calculate localNoiseVariance
        sigma_v = sample_stats.toArray().arraySort().arraySlice(0, 0, 5).arrayReduce(
            ee.Reducer.mean(), [0]
        )
        
        # Set up the 7x7 kernels for directional statistics
        rect_weights = ee.List.repeat(ee.List.repeat(0, 7), 3).cat(
            ee.List.repeat(ee.List.repeat(1, 7), 4)
        )
        
        diag_weights = ee.List([
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1]
        ])
        
        rect_kernel = ee.Kernel.fixed(7, 7, rect_weights, 3, 3, False)
        diag_kernel = ee.Kernel.fixed(7, 7, diag_weights, 3, 3, False)
        
        # Create stacks for mean and variance using the original kernels
        dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(
            directions.eq(1)
        )
        dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(
            directions.eq(1)
        )
        
        dir_mean = dir_mean.addBands(
            img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2))
        )
        dir_var = dir_var.addBands(
            img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2))
        )
        
        # Add the bands for rotated kernels
        for i in range(1, 4):
            dir_mean = dir_mean.addBands(
                img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(
                    directions.eq(2 * i + 1)
                )
            )
            dir_var = dir_var.addBands(
                img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(
                    directions.eq(2 * i + 1)
                )
            )
            dir_mean = dir_mean.addBands(
                img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(
                    directions.eq(2 * i + 2)
                )
            )
            dir_var = dir_var.addBands(
                img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(
                    directions.eq(2 * i + 2)
                )
            )
        
        # "Collapse" the stack into a single band image
        dir_mean = dir_mean.reduce(ee.Reducer.sum())
        dir_var = dir_var.reduce(ee.Reducer.sum())
        
        # Finally generate the filtered value
        var_x = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigma_v)).divide(
            sigma_v.add(1.0)
        )
        
        b = var_x.divide(dir_var)
        
        return dir_mean.add(b.multiply(img.subtract(dir_mean))) \
            .arrayProject([0]) \
            .arrayFlatten([['sum']]) \
            .float()
    
    result = ee.ImageCollection(band_names.map(process_band)).toBands().rename(band_names)
    
    return power_to_db(ee.Image(result))


def scale_features(df, norm_df = pd.read_csv('data/standard_scaler.csv')):
    """

        Receives a pandas DataFrame with the values of features and scales them based on 
        the mean and standard deviation of the stratified sampling in Spanish forests.

    """
    
    df_ = df.copy()

    for col in df.columns:
        df_[col] = (df_[col] - norm_df[col].loc[0]) / (norm_df[col].loc[1])

    return df_





    # # Check if we need to divide the GeoDataFrame
    # if len(gdf) > max_points:
    #     # Get the bounds of the entire GeoDataFrame
    #     minx, miny, maxx, maxy = gdf.total_bounds
        
    #     # Calculate number of divisions needed in each dimension
    #     divisions_per_side = math.ceil(math.sqrt(total_divisions))
        
    #     # Calculate step sizes
    #     x_step = (maxx - minx) / divisions_per_side
    #     y_step = (maxy - miny) / divisions_per_side
        
    #     # Divide into grid cells
    #     results = []
    #     for i in range(divisions_per_side):
    #         for j in range(divisions_per_side):
    #             x_min = minx + i * x_step
    #             x_max = minx + (i + 1) * x_step
    #             y_min = miny + j * y_step
    #             y_max = miny + (j + 1) * y_step
                
    #             # Get points in this cell
    #             cell = gdf[(gdf.geometry.x >= x_min) & (gdf.geometry.x < x_max) & 
    #                       (gdf.geometry.y >= y_min) & (gdf.geometry.y < y_max)]
                
    #             # Handle edge case for the last column/row
    #             if i == divisions_per_side - 1:
    #                 cell = gdf[(gdf.geometry.x >= x_min) & (gdf.geometry.x <= x_max) & 
    #                           (gdf.geometry.y >= y_min) & (gdf.geometry.y < y_max)]
    #             if j == divisions_per_side - 1:
    #                 cell = gdf[(gdf.geometry.x >= x_min) & (gdf.geometry.x < x_max) & 
    #                           (gdf.geometry.y >= y_min) & (gdf.geometry.y <= y_max)]
    #             if i == divisions_per_side - 1 and j == divisions_per_side - 1:
    #                 cell = gdf[(gdf.geometry.x >= x_min) & (gdf.geometry.x <= x_max) & 
    #                           (gdf.geometry.y >= y_min) & (gdf.geometry.y <= y_max)]
                
    #             if len(cell) > 0:
    #                 result = extract_ee_values(cell, img_col, dates, band_name, agg_fun, column_name, 
    #                                          scale, slope, aspect, cloud_free, s1_corr, max_points)
    #                 results.append(result)
        
    #     # Concatenate results and restore original order
    #     gdf_result = pd.concat(results)
    #     return gdf_result.loc[gdf.index]
