from pyproj import Transformer
from scipy.spatial import cKDTree
import pandas as pd
import numpy as np
import dask
import geopandas as gpd
from tqdm import tqdm
from chap1_modules.geedal import geedal_utils, geedal_utils_on_steroids
from sklearn.preprocessing import StandardScaler
import threading

_ee_lock = threading.Lock()

def get_close_pairs(ds, max_distance=40, crs = 3857, idx = 'shot_number', enforce_exclusivity=True):

    print('Creating coords array for cKDTree...')

    trans = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    x, y = trans.transform(ds.longitude.values, ds.latitude.values)

    coords = np.array([(x[i],y[i]) for i in range(len(x))])

    tree = cKDTree(coords)

    print('Querying cKDTree for close pairs...')

    dist_matrix = tree.sparse_distance_matrix(tree, max_distance=max_distance, output_type='coo_matrix')

    i_indices = dist_matrix.row
    j_indices = dist_matrix.col
    distances = dist_matrix.data

    shot_num_1 = ds.isel({idx:i_indices})[idx].values
    shot_num_2 = ds.isel({idx:j_indices})[idx].values

    close_pairs = [(i, j, d) for i, j, d in zip(shot_num_1, shot_num_2, distances) if i < j]

    print(f'Found {len(close_pairs)} close pairs within {max_distance} meters.')
    
    df = pd.DataFrame(close_pairs, columns=['shot_num_1', 'shot_num_2', 'distance']).sort_values('distance')

    # print(len(df))

    # Get time between shots
    time_between = ds.sel({idx:df.shot_num_2.values}).time.values - ds.sel({idx:df.shot_num_1.values}).time.values 

    # print(len(time_between))

    # Get the rows with ind_1 shot after ind_2 shot and switch the order to include them on large DF
    df_switch = df[(time_between < 0)][[id1 not in df['shot_num_2'] for id1 in list(df[(time_between < 0)]['shot_num_1'])]]

    if len(df_switch) > 0:
        df_switch.shot_num_2, df_switch.shot_num_1 = df_switch.shot_num_1.copy(), df_switch.shot_num_2.copy()

        # Only get the shots with time(ind_1) < time(ind_2)
        df = df[(time_between > 0)]
        df = pd.concat([df, df_switch])
    else:
        df = df[(time_between > 0)]

    if enforce_exclusivity:
        df = df.drop_duplicates('shot_num_2', keep = 'first')

    return df

from joblib import Parallel, delayed

# def extract_ee_info(ds, shots_2, dates=['2018-01-01', '2019-01-01'],
#                        crs='EPSG:25830', max_distance=250, weight_geo=0.3,
#                        use_embeddings=False, use_baseline=True, use_slope=False,
#                        sel_feats=None, opt = True):
    
#     ## FEATURE NORMALIZATION SHOULD BE EQUAL ON EVERY PAIR OBTENTION
#     trans = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
#     x, y = trans.transform(ds.longitude.values, ds.latitude.values)
#     coords = np.array([(x[i], y[i]) for i in range(len(x))])
#     tree_coords = cKDTree(coords)

#     shots_2_crs = shots_2.to_crs(crs)
    
#     pair = [(shots_2_crs.iloc[0].geometry.x,
#                  shots_2_crs.iloc[0].geometry.y)]

#     close_pairs = tree_coords.query_ball_point(pair, max_distance)[0]

#     if len(close_pairs) == 0:
#         return None, [], None, None

#     x, y = trans.transform(
#         ds.isel(shot_number=close_pairs).longitude.values,
#         ds.isel(shot_number=close_pairs).latitude.values
#     )

#     close_pairs = list(ds.isel(shot_number=close_pairs).shot_number.values)

#     possible_fs_pairs = ds.sel(shot_number=close_pairs).to_dataframe()
#     possible_gdf = gpd.GeoDataFrame(
#         possible_fs_pairs,
#         geometry=gpd.points_from_xy(
#             possible_fs_pairs['longitude'],
#             possible_fs_pairs['latitude']
#         ),
#         crs=4326
#     )

#     features = []

#     if opt:
#         possible_gdf = geedal_utils_on_steroids.extract_ee_values(
#                         possible_gdf, 'COPERNICUS/S1_GRD',
#                         dates, agg_fun='median', scale=25,
#                         band_name=['VV', 'VH'], s1_corr=True
#                     )
#         features.extend(['VV', 'VH'])
#         possible_gdf = geedal_utils_on_steroids.extract_ee_values(
#             possible_gdf, 'COPERNICUS/S2_SR_HARMONIZED',
#             dates, agg_fun='median', scale=25,
#             band_name=['B' + str(i+1) for i in range(9)] + ['B' + str(i+11) for i in range(2)], cloud_free=True
#         )
#         features.extend(['B' + str(i+1) for i in range(9)] + ['B' + str(i+11) for i in range(2)])
#     else:
#         possible_gdf = geedal_utils.extract_ee_values(
#             possible_gdf, 'COPERNICUS/S1_GRD',
#             dates, agg_fun='median', scale=25,
#             band_name=['VV', 'VH'], s1_corr=True
#         )
#         features.extend(['VV', 'VH'])
#         possible_gdf = geedal_utils.extract_ee_values(
#             possible_gdf, 'COPERNICUS/S2_SR_HARMONIZED',
#             dates, agg_fun='median', scale=25,
#             band_name=['B' + str(i+1) for i in range(9)] + ['B' + str(i+11) for i in range(2)], cloud_free=True
#         )
#         features.extend(['B' + str(i+1) for i in range(9)] + ['B' + str(i+11) for i in range(2)])
    
#     return possible_gdf, features

def get_close_fs_pairs(ds, shots_2, dates=['2018-01-01', '2019-01-01'],
                       crs='EPSG:25830', max_distance=250, weight_geo=0.75,
                       use_embeddings=False, use_baseline=True, use_slope=False,
                       sel_feats=None, n_jobs=1, idx='shot_number', 
                       include_beam_info = False, 
                       all_feats = False, disturbed = True):
    """
    For each shot in shots_2, find the most similar historical GEDI shot from ds
    based on a weighted combination of remote sensing features and geographic proximity.

    Parameters:
        ds           : xarray Dataset of candidate GEDI shots (the "forest structure" reference pool)
        shots_2      : GeoDataFrame of query shots to find matches for
        dates        : Date range for extracting satellite imagery
        crs          : Projected CRS for distance calculations
        max_distance : Spatial search radius in CRS units (meters) for candidate filtering
        weight_geo   : Weight given to geographic coordinates vs. spectral features (0–1)
        use_embeddings: Whether to include Google satellite embedding features
        use_baseline : Whether to include Sentinel-1 (SAR) and Sentinel-2 (optical) features
        use_slope    : Whether to include terrain slope and aspect from Copernicus DEM
        sel_feats    : Optional explicit list of features to use; overrides auto-built list
        n_jobs       : Number of parallel jobs
        idx          : Name of the shot index dimension in ds
    """

    # --- Build a spatial index over all candidate shots in the projected CRS ---
    trans = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    x, y = trans.transform(ds.longitude.values, ds.latitude.values)
    coords = np.array([(x[i], y[i]) for i in range(len(x))])
    tree_coords = cKDTree(coords)  # KD-tree for fast spatial lookup

    shots_2_crs = shots_2.to_crs(crs)  # Reproject query shots to the same CRS

    def process_one(ind):
        """Find the best feature-space match for the ind-th query shot."""

        # Initialise Earth Engine (thread-safe via lock)
        with _ee_lock:
            geedal_utils.earthengine_init()

        pair = [(shots_2_crs.iloc[ind].geometry.x,
                 shots_2_crs.iloc[ind].geometry.y)]

        # --- Spatial pre-filter: only consider candidates within max_distance ---
        idxs = tree_coords.query_ball_point(pair, max_distance)[0]
        close_pairs = idxs  # indices into ds

        dists, _ = tree_coords.query(pair, k=len(idxs))  # distances, unused downstream

        if len(close_pairs) == 0:
            return None, [], None, None, None, None  # no spatial neighbours found

        # Retrieve projected coordinates of the candidate shots
        x, y = trans.transform(
            ds.isel({idx: close_pairs}).longitude.values,
            ds.isel({idx: close_pairs}).latitude.values
        )
        coords_close = np.array([(x[i], y[i]) for i in range(len(x))])

        # Map back to shot-number identifiers
        close_pairs = list(ds.isel({idx: close_pairs})[idx].values)

        # Locate the query shot itself within the candidate list
        s2_idx = close_pairs.index(np.uint64(shots_2.iloc[ind].shot_num_2))

        # Build a GeoDataFrame of candidate shots for Earth Engine extraction
        possible_fs_pairs = ds.sel({idx: close_pairs}).to_dataframe()
        possible_gdf = gpd.GeoDataFrame(
            possible_fs_pairs,
            geometry=gpd.points_from_xy(
                possible_fs_pairs['longitude'],
                possible_fs_pairs['latitude']
            ),
            crs=4326
        )

        possible_gdf['_row_id_'] = np.asarray(close_pairs)     # right after building possible_gdf

        if include_beam_info:
            query_beam_type = shots_2.iloc[ind]['beam_type']
            beam_mask = possible_gdf['beam_type'] == query_beam_type
            if not beam_mask.any():
                return None, [], None, None, None, None  # no candidates share the same beam_type

            # Keep only matching-beam candidates; update all dependent structures
            possible_gdf = possible_gdf[beam_mask]
            coords_close = coords_close[beam_mask.values]
            close_pairs = list(np.array(close_pairs)[beam_mask.values])

            # Recompute s2_idx after filtering (query shot may have shifted position)
            try:
                s2_idx = close_pairs.index(np.uint64(shots_2.iloc[ind].shot_num_2))
            except ValueError:
                # Query shot itself was filtered out (shouldn't happen, but guard anyway)
                return None, [], None, None, None, None

        features = []  # will accumulate column names as features are added

        # --- Extract Sentinel-1 (SAR) and Sentinel-2 (optical) baseline features ---
        if use_baseline:
            try:
                possible_gdf = geedal_utils_on_steroids.extract_ee_values(
                    possible_gdf, 'COPERNICUS/S1_GRD',
                    dates, agg_fun='median', scale=25,
                    band_name=['VV', 'VH'], s1_corr=True
                )
                
                assert len(possible_gdf) == len(close_pairs) and  np.array_equal(np.asarray(possible_gdf['_row_id_']), np.asarray(close_pairs)), 'extract_ee_values dropped/reordered rows'
                
                features.extend(['VV', 'VH'])
                possible_gdf = geedal_utils_on_steroids.extract_ee_values(
                    possible_gdf, 'COPERNICUS/S2_SR_HARMONIZED',
                    dates, agg_fun='median', scale=25,
                    band_name=['B' + str(i+1) for i in range(9)] + ['B' + str(i+11) for i in range(2)],
                    cloud_free=True
                )
                assert len(possible_gdf) == len(close_pairs) and  np.array_equal(np.asarray(possible_gdf['_row_id_']), np.asarray(close_pairs)), 'extract_ee_values dropped/reordered rows'
                features.extend(['B' + str(i+1) for i in range(9)] + ['B' + str(i+11) for i in range(2)])
            except Exception as e:
                # Retry with a smaller tile size if the EE request was too large
                print('Error: ' + str(e) + '. Trying again with smaller max points.')
                possible_gdf = geedal_utils_on_steroids.extract_ee_values(
                    possible_gdf, 'COPERNICUS/S1_GRD',
                    dates, agg_fun='median', scale=25,
                    band_name=['VV', 'VH'], s1_corr=True, max_size=3000*3000
                )
                features.extend(['VV', 'VH'])
                
                possible_gdf = geedal_utils_on_steroids.extract_ee_values(
                    possible_gdf, 'COPERNICUS/S2_SR_HARMONIZED',
                    dates, agg_fun='median', scale=25,
                    band_name=['B' + str(i+1) for i in range(9)] + ['B' + str(i+11) for i in range(2)],
                    cloud_free=True, max_size=4000*4000
                )
                features.extend(['B' + str(i+1) for i in range(9)] + ['B' + str(i+11) for i in range(2)])
                

        # --- Optionally add terrain features (slope, aspect) ---
        if use_slope:
            try:
                possible_gdf = geedal_utils_on_steroids.extract_ee_values(
                    possible_gdf, 'COPERNICUS/DEM/GLO30',
                    dates=None, agg_fun='mean', scale=25,
                    band_name='DEM', slope=True, column_name='slope'
                )
                assert len(possible_gdf) == len(close_pairs) and  np.array_equal(np.asarray(possible_gdf['_row_id_']), np.asarray(close_pairs)), 'extract_ee_values dropped/reordered rows'
                possible_gdf = geedal_utils_on_steroids.extract_ee_values(
                    possible_gdf, 'COPERNICUS/DEM/GLO30',
                    dates=None, agg_fun='mean', scale=25,
                    band_name='DEM', aspect=True, column_name='aspect'
                )
                assert len(possible_gdf) == len(close_pairs) and  np.array_equal(np.asarray(possible_gdf['_row_id_']), np.asarray(close_pairs)), 'extract_ee_values dropped/reordered rows'
                features.extend(['slope', 'aspect'])
            except Exception as e:
                print('Error: ' + str(e) + '. Trying again with smaller max points.')
                possible_gdf = geedal_utils_on_steroids.extract_ee_values(
                    possible_gdf, 'COPERNICUS/DEM/GLO30',
                    dates=None, agg_fun='mean', scale=25,
                    band_name='DEM', slope=True, column_name='slope', max_size=4000*4000
                )
                possible_gdf = geedal_utils_on_steroids.extract_ee_values(
                    possible_gdf, 'COPERNICUS/DEM/GLO30',
                    dates=None, agg_fun='mean', scale=25,
                    band_name='DEM', aspect=True, column_name='aspect', max_size=4000*4000
                )
                features.extend(['slope', 'aspect'])

        # --- Optionally add Google satellite embedding features (64-dim) ---
        if use_embeddings:
            try:
                possible_gdf = geedal_utils_on_steroids.extract_ee_values(
                    possible_gdf, 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL',
                    dates, scale=25,
                    band_name=['A' + "{:02d}".format(i) for i in range(64)]
                )
                features.extend(['A' + "{:02d}".format(i) for i in range(64)])
            except Exception as e:
                possible_gdf = geedal_utils_on_steroids.extract_ee_values(
                    possible_gdf, 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL',
                    dates, scale=25,
                    band_name=['A' + "{:02d}".format(i) for i in range(64)],
                    max_size=4000*4000
                )
                features.extend(['A' + "{:02d}".format(i) for i in range(64)])

        if sel_feats is not None:
            features = sel_feats  # Override with an explicit feature list if provided

        # --- Assemble the query shot's feature vector ---
        s2 = possible_gdf.iloc[[s2_idx]][features].copy()

        # Encode acquisition seasonality as a cosine (peak in June, trough in December)
        if sel_feats is None or ('seasonality' in sel_feats):
            s2['seasonality'] = np.cos(
                ((2*np.pi)/12) * (possible_gdf.time.dt.month.iloc[s2_idx] - 6)
            )

        if use_baseline or use_slope:
            s2 = geedal_utils.scale_features(s2)

        # --- Restrict candidates to shots acquired BEFORE the query shot (temporal filter) ---
        if disturbed:
            # Disturbed pairs: partner must be a PRE-disturbance observation,
            # so the match cannot land on a post-event shot (which would be
            # dropped downstream and cause a radius-dependent loss of pairs).
            fire_date = shots_2.iloc[ind]['dist']
            mask = possible_gdf.time < fire_date
        else:
            # Undisturbed pairs: partner must simply predate the query shot.
            mask = possible_gdf.time.iloc[s2_idx] > possible_gdf.time
        if not mask.any():
            return None, [], None, None, None, None  # no eligible earlier partner

        close_pairs = np.array(close_pairs)[mask]
        coords_close = coords_close[mask.values]
        possible_gdf = possible_gdf[mask]

        # --- Assemble candidate feature matrix ---
        possible = possible_gdf[features].copy()

        if sel_feats is None or ('seasonality' in sel_feats):
            possible['seasonality'] = np.cos(
                ((2*np.pi)/12) * (possible_gdf.time.dt.month.values - 6)
            )

        if use_baseline or use_slope:
            possible = geedal_utils.scale_features(possible)

        # Standardise spatial coordinates and append them as weighted features
        scaler = StandardScaler()
        possible_coords = scaler.fit_transform(coords_close)
        possible['X'] = possible_coords[:, 0]
        possible['Y'] = possible_coords[:, 1]

        s2_coords = scaler.transform(np.array(pair))[0]
        s2['X'] = s2_coords[0]
        s2['Y'] = s2_coords[1]

        # --- Apply feature/geo weights and find nearest neighbour in feature space ---
        # Spectral/terrain features are downweighted by (1 - weight_geo),
        # geographic coordinates are upweighted by weight_geo
        X_weighted = np.hstack([
            possible.values[:, :-2] * (1 - weight_geo),
            possible.values[:, -2:] * weight_geo
        ])

        s2_weighted = np.hstack([
            s2.values[:, :-2] * (1 - weight_geo),
            s2.values[:, -2:] * weight_geo
        ])

        fs_tree = cKDTree(X_weighted)
        fs_dist, ind = fs_tree.query(s2_weighted.squeeze()) # 1-NN in the weighted feature space

        if all_feats == True:
            all_features = possible.reset_index().drop(idx, axis=1).to_dict()
        else:
            all_features = None

        return (
            str(close_pairs[ind]),   # shot_number of the best match
            close_pairs,                # all spatially eligible candidates
            s2.reset_index().drop(idx, axis=1).to_dict(),           # query features
            possible.reset_index().drop(idx, axis=1).iloc[ind].to_dict(),  # matched features
            all_features,  # matched features
            float(fs_dist)
            # fs_dist
        )

    # --- Run process_one in parallel across all query shots ---
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(process_one)(i) for i in tqdm(range(len(shots_2)))
    )

    new_inds, poss_inds, s2_features, s1_features, all_pos_feats, match_dists = zip(*results)

    # Attach results back to the query GeoDataFrame
    shots_2['new_shot_num_1'] = list(new_inds)       # best-match shot ID
    shots_2['possible_pairs'] = list(poss_inds)      # all candidate shot IDs
    shots_2['s2_feats'] = list(s2_features)          # query feature vector
    shots_2['s1_feats'] = list(s1_features)          # matched feature vector
    shots_2['all_pos_feats'] = list(all_pos_feats)
    shots_2['match_dist']      = list(match_dists)

    return shots_2

