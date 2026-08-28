import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import Point
from typing import Tuple, Optional
import warnings
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import math
import matplotlib.pyplot as plt
from rasterio.io import MemoryFile
from owslib.wcs import WebCoverageService
from scipy.ndimage import uniform_filter
from pyproj import Transformer

def plot_local_dem(local_dem, local_tf):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot DEM using imshow with transform (correct georeferencing)
    img = ax.imshow(
        local_dem,
        cmap='seismic',
        extent=(local_tf.c,                           # min x
                local_tf.c + local_tf.a * local_dem.shape[1],  # max x
                local_tf.f + local_tf.e * local_dem.shape[0],  # min y
                local_tf.f),                           # max y
        origin='upper',
        vmin = -10,
        vmax = 10
    )

    plt.colorbar(img, ax=ax, label="Elevation (m)")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_aspect('equal', 'box')
    ax.set_title("Local Difference with high-res DEM")
    plt.show()

def focal_mean_array(arr, pixel_size, window_size_m=25, sample_step_m=5):
    """
    Compute a focal mean with a square window on a NumPy array and sample every X meters.

    Parameters
    ----------
    arr : np.ndarray
        2D input array (e.g., elevation).
    pixel_size : float
        Pixel size in meters (assumed square).
    window_size_m : float, default 25
        Window size of focal mean in meters.
    sample_step_m : float, default 5
        Sampling resolution in meters.
        
    Returns
    -------
    np.ndarray
        2D NumPy array with focal means at sampled positions,
        np.nan elsewhere.
    """

    # Convert window size meters → pixels
    win_px = int(round(window_size_m / pixel_size))
    if win_px % 2 == 0:
        win_px += 1  # ensure odd window size for centered filter

    # Compute focal mean (uniform moving window)
    focal = uniform_filter(arr.astype(float), size=win_px, mode='nearest')

    # Prepare empty array with NaNs
    out = np.full_like(focal, np.nan, dtype=float)

    # Sampling step in pixels
    step_px = int(round(sample_step_m / pixel_size))

    # Fill sampled points only
    out[::step_px, ::step_px] = focal[::step_px, ::step_px]

    return out

def geogedi_reposition(
    gedi_gdf: gpd.GeoDataFrame,
    dem_path: str = "data/dem_.tif",
    on_the_fly_dem_spain: bool = True,
    elev_column: str = "elev_lowestmode",
    search_radius: float = 25.0,
    grid_spacing: float = 5.0,
    group_column: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Reposition GEDI shot locations by searching for the horizontal offset
    that best matches GEDI elevations with a DEM.

    The method performs a deterministic grid search around each GEDI shot
    (or group of shots) and finds the (dx, dy) offset that minimizes the
    mean absolute error (MAE) between GEDI elevations and DEM elevations.

    Two modes are supported:
    1. On-the-fly DEM download for Spain (EPSG:25830) using IDEE WCS.
    2. Local DEM file provided via `dem_path`.

    Parameters
    ----------
    gedi_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing GEDI shots as point geometries.
    dem_path : str, optional
        Path to a local DEM raster (used if on_the_fly_dem_spain=False).
    on_the_fly_dem_spain : bool, optional
        If True, download DEM tiles on-the-fly from IDEE (Spain only).
    elev_column : str, optional
        Column in `gedi_gdf` containing GEDI ground elevation values.
    search_radius : float, optional
        Maximum horizontal offset (meters) tested in x and y directions.
    grid_spacing : float, optional
        Step size (meters) for the grid search.
    group_column : str, optional
        Column used to group GEDI shots that should be shifted together.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of the input GeoDataFrame with updated geometries and
        additional columns:
        - offset_x, offset_y : applied horizontal offsets (meters)
        - mae : minimum mean absolute error
        - dem_elevation : DEM elevation at the final position
    """

    # ------------------------------------------------------------------
    # Input validation and output initialization
    # ------------------------------------------------------------------
    if elev_column not in gedi_gdf.columns:
        raise ValueError(f"Column '{elev_column}' not found in GeoDataFrame")

    result_gdf = gedi_gdf.copy()
    result_gdf["offset_x"] = 0.0
    result_gdf["offset_y"] = 0.0
    result_gdf["mae"] = np.nan
    result_gdf["dem_elevation"] = np.nan

    # ------------------------------------------------------------------
    # CASE 1: On-the-fly DEM download for Spain (EPSG:25830)
    # ------------------------------------------------------------------
    if on_the_fly_dem_spain:

        # Ensure GEDI data are in EPSG:25830
        if gedi_gdf.crs != 25830:
            warnings.warn(
                f"Reprojecting GEDI data from {gedi_gdf.crs} to EPSG:25830"
            )
            gedi_gdf_proj = gedi_gdf.to_crs(25830)
        else:
            gedi_gdf_proj = gedi_gdf.copy()

        def _download_dem_tile(x_coord, y_coord, buffer=100):
            """
            Download a DEM tile around a given coordinate using IDEE WCS.

            Returns a small DEM subset centered on (x, y).
            """
            # url = (
            #     "https://servicios.idee.es/wms-inspire/mdt"
            #     "?REQUEST=GetCapabilities&SERVICE=WMS&VERSION=1.3.0"
            # )
            # coverage_id = "EL.ElevationGridCoverage"
            url = "https://servicios.idee.es/wcs-inspire/mdt"
            coverage_id = "Elevacion25830_5"

            minx, maxx = x_coord - buffer, x_coord + buffer
            miny, maxy = y_coord - buffer, y_coord + buffer

            wcs = WebCoverageService(url, version="2.0.1")

            response = wcs.getCoverage(
                identifier=coverage_id,
                format="image/tiff",
                subsets=[("x", minx, maxx), ("y", miny, maxy)],
            )

            memfile = MemoryFile(response.read())
            dem = memfile.open()

            return dem.read(1), dem.transform, dem

        # ------------------------------------------------------------------
        # Build grid of candidate offsets
        # ------------------------------------------------------------------
        n_steps = int(2 * search_radius / grid_spacing) + 1
        offsets = np.linspace(-search_radius, search_radius, n_steps)
        offset_grid = np.array([(dx, dy) for dx in offsets for dy in offsets])

        # Group GEDI shots if requested
        if group_column and group_column in gedi_gdf.columns:
            groups = gedi_gdf.groupby(group_column)
        else:
            gedi_gdf["_temp_group"] = range(len(gedi_gdf))
            groups = gedi_gdf.groupby("_temp_group")

        # ------------------------------------------------------------------
        # Process each group independently
        # ------------------------------------------------------------------
        for group_id, group_data in groups:
            idxs = group_data.index
            coords = np.array(
                [[p.x, p.y] for p in gedi_gdf_proj.loc[idxs, "geometry"]]
            )
            gedi_elevs = group_data[elev_column].values

            if np.any(np.isnan(gedi_elevs)):
                continue

            best_mae = np.inf
            best_offset = np.array([0.0, 0.0])
            best_dem_elevs = None

            try:
                # Download and smooth local DEM
                local_dem, local_tf, dem = _download_dem_tile(
                    coords[0, 0], coords[0, 1], buffer=50
                )
                local_dem = focal_mean_array(local_dem, 5)

                # ----------------------------------------------------------
                # Grid search over offsets
                # ----------------------------------------------------------
                for offset in offset_grid:
                    test_coords = coords + offset
                    dem_vals = []
                    valid = True

                    for x, y in test_coords:
                        col, row = ~local_tf * (x, y)
                        col, row = int(col), int(row)

                        if (
                            0 <= row < local_dem.shape[0]
                            and 0 <= col < local_dem.shape[1]
                        ):
                            dem_vals.append(local_dem[row, col])
                        else:
                            valid = False
                            break

                    if not valid:
                        continue

                    dem_vals = np.array(dem_vals)
                    mae = np.mean(np.abs(gedi_elevs - dem_vals))

                    if mae < best_mae:
                        best_mae = mae
                        best_offset = offset
                        best_dem_elevs = dem_vals

            except Exception as e:
                print(f"Error processing group {group_id}. Error: {e}")
                continue

            # --------------------------------------------------------------
            # Apply best offset to geometries
            # --------------------------------------------------------------
            if best_dem_elevs is not None:
                for i, idx in enumerate(idxs):
                    orig = gedi_gdf_proj.loc[idx, "geometry"]
                    new_x = orig.x + best_offset[0]
                    new_y = orig.y + best_offset[1]

                    if gedi_gdf.crs != 25830:
                        new_geom = (
                            gpd.GeoDataFrame(
                                geometry=[Point(new_x, new_y)], crs=25830
                            )
                            .to_crs(gedi_gdf.crs)
                            .geometry.iloc[0]
                        )
                    else:
                        new_geom = Point(new_x, new_y)

                    result_gdf.loc[idx, "geometry"] = new_geom
                    result_gdf.loc[idx, "offset_x"] = best_offset[0]
                    result_gdf.loc[idx, "offset_y"] = best_offset[1]
                    result_gdf.loc[idx, "mae"] = best_mae
                    result_gdf.loc[idx, "dem_elevation"] = best_dem_elevs[i]

        if "_temp_group" in result_gdf.columns:
            result_gdf.drop(columns="_temp_group", inplace=True)

    # ------------------------------------------------------------------
    # CASE 2: Use local DEM file
    # ------------------------------------------------------------------
    else:
        with rasterio.open(dem_path) as dem:
            dem_crs = dem.crs

            if gedi_gdf.crs != dem_crs:
                warnings.warn(
                    f"Reprojecting GEDI data from {gedi_gdf.crs} to {dem_crs}"
                )
                gedi_gdf_proj = gedi_gdf.to_crs(dem_crs)
            else:
                gedi_gdf_proj = gedi_gdf.copy()

            def _extract_local_dem(x_coords, y_coords):
                """
                Extract and upsample a DEM window around a group of points.
                """
                min_x, max_x = float(x_coords.min()), float(x_coords.max())
                min_y, max_y = float(y_coords.min()), float(y_coords.max())

                extra = search_radius + 30.0
                win = from_bounds(
                    min_x - extra,
                    min_y - extra,
                    max_x + extra,
                    max_y + extra,
                    transform=dem.transform,
                    width=dem.width,
                    height=dem.height,
                )

                subset = dem.read(1, window=win)
                subset_transform = dem.window_transform(win)

                dst_transform = from_origin(
                    min_x - extra, max_y + extra, grid_spacing, grid_spacing
                )

                dst_width = int(
                    math.ceil((max_x - min_x + 2 * extra) / grid_spacing)
                )
                dst_height = int(
                    math.ceil((max_y - min_y + 2 * extra) / grid_spacing)
                )

                dst = np.full(
                    (dst_height, dst_width),
                    dem.nodata if dem.nodata is not None else np.nan,
                    dtype=np.float32,
                )

                reproject(
                    source=subset,
                    destination=dst,
                    src_transform=subset_transform,
                    src_crs=dem.crs,
                    dst_transform=dst_transform,
                    dst_crs=dem.crs,
                    resampling=Resampling.bilinear,
                    src_nodata=dem.nodata,
                    dst_nodata=dem.nodata,
                )

                return dst, dst_transform

            # Grid of offsets
            n_steps = int(2 * search_radius / grid_spacing) + 1
            offsets = np.linspace(-search_radius, search_radius, n_steps)
            offset_grid = np.array([(dx, dy) for dx in offsets for dy in offsets])

            if group_column and group_column in gedi_gdf.columns:
                groups = gedi_gdf.groupby(group_column)
            else:
                gedi_gdf["_temp_group"] = range(len(gedi_gdf))
                groups = gedi_gdf.groupby("_temp_group")

            for group_id, group_data in groups:
                idxs = group_data.index
                coords = np.array(
                    [[p.x, p.y] for p in gedi_gdf_proj.loc[idxs, "geometry"]]
                )
                gedi_elevs = group_data[elev_column].values

                if np.any(np.isnan(gedi_elevs)):
                    continue

                local_dem, local_tf = _extract_local_dem(
                    coords[:, 0], coords[:, 1]
                )

                best_mae = np.inf
                best_offset = np.array([0.0, 0.0])
                best_dem_elevs = None

                for offset in offset_grid:
                    test_coords = coords + offset
                    dem_vals = []
                    valid = True

                    for x, y in test_coords:
                        col, row = ~local_tf * (x, y)
                        col, row = int(col), int(row)

                        if (
                            0 <= row < local_dem.shape[0]
                            and 0 <= col < local_dem.shape[1]
                        ):
                            dem_vals.append(local_dem[row, col])
                        else:
                            valid = False
                            break

                    if not valid:
                        continue

                    dem_vals = np.array(dem_vals)
                    mae = np.mean(np.abs(gedi_elevs - dem_vals))

                    if mae < best_mae:
                        best_mae = mae
                        best_offset = offset
                        best_dem_elevs = dem_vals

                if best_dem_elevs is not None:
                    for i, idx in enumerate(idxs):
                        orig = gedi_gdf_proj.loc[idx, "geometry"]
                        new_x = orig.x + best_offset[0]
                        new_y = orig.y + best_offset[1]

                        if gedi_gdf.crs != dem_crs:
                            new_geom = (
                                gpd.GeoDataFrame(
                                    geometry=[Point(new_x, new_y)], crs=dem_crs
                                )
                                .to_crs(gedi_gdf.crs)
                                .geometry.iloc[0]
                            )
                        else:
                            new_geom = Point(new_x, new_y)

                        result_gdf.loc[idx, "geometry"] = new_geom
                        result_gdf.loc[idx, "offset_x"] = best_offset[0]
                        result_gdf.loc[idx, "offset_y"] = best_offset[1]
                        result_gdf.loc[idx, "mae"] = best_mae
                        result_gdf.loc[idx, "dem_elevation"] = best_dem_elevs[i]

            if "_temp_group" in result_gdf.columns:
                result_gdf.drop(columns="_temp_group", inplace=True)

    return result_gdf

def get_geoid_height(latitude, longitude, geoid_model='egm2008'):
    """
    Get geoid height (undulation) at a specific location.
    
    Parameters:
    -----------
    latitude : float
        Latitude in decimal degrees
    longitude : float
        Longitude in decimal degrees
    geoid_model : str
        Geoid model to use ('egm96' or 'egm2008')
        
    Returns:
    --------
    geoid_height : float
        Geoid height/undulation in meters
    """
    # Map geoid model to EPSG code
    geoid_epsg = {
        'egm96': '5773',     # EGM96 geoid
        'egm2008': '3855'    # EGM2008 geoid
    }
    
    if geoid_model.lower() not in geoid_epsg:
        raise ValueError(f"Unknown geoid model. Use 'egm96' or 'egm2008'")
    
    # Create transformer
    # From WGS84 3D to WGS84 + geoid model
    transformer = Transformer.from_crs(
        "EPSG:4979",  # WGS84 (lat, lon, ellipsoidal height)
        f"EPSG:4326+{geoid_epsg[geoid_model.lower()]}",  # WGS84 + geoid
        always_xy=False
    )
    
    # Use ellipsoidal height = 0 to get pure geoid height
    h_ellipsoidal = 0.0
    lat_out, lon_out, H_orthometric = transformer.transform(
        latitude, longitude, h_ellipsoidal
    )
    
    # Geoid height = ellipsoidal height - orthometric height
    geoid_height = h_ellipsoidal - H_orthometric
    
    return geoid_height

def resample_dem_simple(input_path: str, output_path: str, scale_factor: float = 6.0) -> None:
    """
    Simplified version using scale factor (30m to 5m = scale factor of 6).
    
    Parameters:
    -----------
    input_path : str
        Path to input DEM
    output_path : str
        Path to save resampled DEM
    scale_factor : float, default=6.0
        Upsampling factor (6 = 30m to 5m)
    """
    
    with rasterio.open(input_path) as src:
        # Read metadata
        data = src.read(1)
        
        # Calculate new dimensions
        new_height = int(src.height * scale_factor)
        new_width = int(src.width * scale_factor)
        
        # Update transform
        new_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )
        
        # Resample data
        from rasterio.warp import reproject, Resampling
        
        resampled_data = np.empty((new_height, new_width), dtype=data.dtype)
        
        reproject(
            source=data,
            destination=resampled_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=new_transform,
            dst_crs=src.crs,
            resampling=Resampling.bilinear
        )
        
        # Write output
        out_meta = src.meta.copy()
        out_meta.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform
        })
        
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(resampled_data, 1)
        
        print(f"Resampled from {src.width}x{src.height} to {new_width}x{new_height}")