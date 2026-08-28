"""
    Module with all functions needed to compare two GEDI measurements (On-orbit or Simulated)
"""
import gedidb as gdb
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline
from shapely.geometry import Point
import contextily as ctx

def get_rh_data(shot_num, geo):

    rh_df = pd.read_csv('gedidb/relative_height_data.csv')

    if np.uint64(shot_num) in np.uint64(rh_df['shot_number'].values):
        rh_data = rh_df[np.uint64(rh_df['shot_number'].values) == np.uint64(shot_num)]
    else:
        # Instantiate the GEDIProvider
        provider = gdb.GEDIProvider(
            storage_type='s3',
            s3_bucket="dog.gedidb.gedi-l2-l4-v002",
            url="https://s3.gfz-potsdam.de"
        )

                    
        rh_data = provider.get_data(
                    variables=['rh'],
                    query_type="nearest",
                    num_shots = 1,
                    point=(geo.x, geo.y),
                    radius = 0.00001,
                    start_time="2018-01-01",
                    end_time="2024-12-31",
                    return_type='dataframe',
                    **{'shot_number': '== '+ shot_num})
        
        rh_data['shot_number'] = str(np.uint64(rh_data['shot_number'].values[0]))
        rh_df = pd.concat([rh_df, rh_data], ignore_index=True)
        rh_df.to_csv('gedidb/relative_height_data.csv', index = False)
    
    return rh_data

def get_wf_from_rh(rh, step = 2, include_spline = True, lam = 0.05, norm_1_elsum = True):
    """
        Function to get normalized pseudo waveform from relative height metrics.
    """
    frac = (np.linspace(0, 100, 101) / 100)[::step]
    
    w = np.gradient(frac, rh[::step])

    w = w - min(w)

    w = np.array([0] + list(w))
    rh = np.array([rh[0] + (rh[1] - rh[2])] + list(rh[::step]))

    if include_spline:
        spl = make_smoothing_spline(rh, w, lam = lam)
        w = spl(rh)

    if norm_1_elsum:
        w = w/np.max(w)
    else:
        w = w/np.sum(w)

    return w, rh

def interpolate_pair(int_1, int_2, elev_1, elev_2):
    """
        Function to interpolate and normalize two waveforms to make them comparable
    """
    
    bounds = [min(elev_1.min(), elev_2.min()),
              max(elev_1.max(), elev_2.max())]

    elevation = np.linspace(bounds[0], bounds[1], 200)

    int_1 = np.interp(elevation, elev_1, int_1)
    int_2 = np.interp(elevation, elev_2, int_2)

    # Normalize interpolated values
    int_1 = int_1/np.max(int_1)
    int_2 = int_2/np.max(int_2)

    return elevation, int_1, int_2

def dtw_distance(waveform1, waveform2):
    """
    Calculate Dynamic Time Warping distance between two waveforms.
    
    Parameters:
    -----------
    waveform1, waveform2 : array-like
        Input waveforms (can be different lengths)
    
    Returns:
    --------
    float : DTW distance (lower = more similar)
    """
    x = np.array(waveform1)
    y = np.array(waveform2)
    
    n, m = len(x), len(y)
    
    # Initialize cost matrix with infinity
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    # Fill the cost matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i-1] - y[j-1])
            dtw[i, j] = cost + min(
                dtw[i-1, j],      # insertion
                dtw[i, j-1],      # deletion
                dtw[i-1, j-1]     # match
            )
    
    return dtw[n, m]

def wasserstein_1d(z_bins, w1, w2):
    """
    z_bins: 1D array of bin centers (meters)
    w1, w2: waveform intensities on the same bins
    returns: W1 distance in same units as z_bins (meters)
    """
    dz = np.diff(z_bins)
    if not np.allclose(dz, dz[0]):
        # convert to bin widths per bin
        widths = np.concatenate([dz, [dz[-1]]])
    else:
        widths = np.full_like(z_bins, dz[0])

    a1 = w1 * widths
    a2 = w2 * widths
    
    c1 = np.cumsum(a1)
    c1 = c1/np.max(c1)
    c2 = np.cumsum(a2)
    c2 = c2/np.max(c2)

    W1 = np.sum(np.abs(c1 - c2) * widths)

    return W1

def calculate_wf_change_metrics(int_1, int_2, elev_1, elev_2):
    """
        Function to calculate change metrics: 
            - Integral of change
            - Integral of the absolute change
            - Cosine similarity
            - Euclidean distance
            - DTW
            - RMSD
            - r2
            - nRMSD: expresses uncertainty relative to the mean of int_1
            - wass: Wasserstein distance
            - Absolute Error profile
    """
        
    elevation, int_1, int_2 = interpolate_pair(int_1, int_2, elev_1, elev_2)

    intensity_change = np.asarray(int_2 - int_1)

    change_integral = np.trapezoid(intensity_change, elevation)
    abs_change_integral = np.trapezoid(np.abs(intensity_change), elevation)
    cos_sim = np.dot(int_1, int_2) / (np.linalg.norm(int_1) * np.linalg.norm(int_2))
    euc_dist = np.linalg.norm(int_1 - int_2)
    dtw = dtw_distance(int_1, int_2)
    rmsd = np.sqrt(np.sum((int_1 - int_2)**2) / len(int_1))
    r2 = np.corrcoef(int_1, int_2)[1,0]
    nrmsd = np.sqrt((np.sum((int_1 - int_2)**2) / len(int_1))) / np.mean(int_1)
    wass = wasserstein_1d(elevation, int_1, int_2)
    ae_profile = np.abs(int_1 - int_2)

    return change_integral, abs_change_integral, cos_sim, euc_dist, dtw, rmsd, r2, nrmsd, wass, ae_profile


def plot_pair_selection(pair, ds):

    pair['possible_pairs'] = np.uint64(pair.possible_pairs[1:-1].split(' '))

    poss = ds.sel(shot_number = list(pair.possible_pairs.iloc[0])).to_dataframe()
    selected = ds.sel(shot_number = np.uint64(pair.new_shot_num_1.iloc[0]))
    selected = gpd.GeoDataFrame(
        geometry=[Point(selected.longitude.values, selected.latitude.values)],
        crs="EPSG:4326"
    )
    previous = ds.sel(shot_number = np.uint64(pair.shot_num_1.iloc[0]))
    previous = gpd.GeoDataFrame(
        geometry=[Point(previous.longitude.values, previous.latitude.values)],
        crs="EPSG:4326"
    )

    fig, ax = plt.subplots(1,1, figsize = (7,7))

    # Your Blue 12.5m buffer
    h_small = pair.to_crs(25830).buffer(12.5).to_crs(4326).plot(
        color='b', alpha=0.5, edgecolor='red', ax=ax
    )

    # Your Orange 250m buffer
    h_large = pair.to_crs(25830).buffer(250).to_crs(4326).plot(
        color='orange', alpha=0.1, edgecolor='red', ax=ax
    )

    # Possible S1 (scatter) → this one works normally
    h_poss = ax.scatter(
        poss.longitude, poss.latitude, marker='x'
    )

    # New + Previous selections
    h_new = selected.to_crs(25830).buffer(12.5).to_crs(4326).plot(
        ax=ax, alpha=0.5, color='c'
    )
    h_prev = previous.to_crs(25830).buffer(12.5).to_crs(4326).plot(
        ax=ax, alpha=0.5, color='g'
    )

    # Build legend manually
    ax.legend(
        handles=[h_small, h_large, h_poss, h_new, h_prev],
        labels=[
            '12.5m buffer (S1 opt.)',
            '250m buffer (S2)',
            'Possible S1',
            'New selection S1',
            'Previous selection S1'
        ]
    )

    ctx.add_basemap(ax = ax, crs = 4326, source = ctx.providers.Esri.WorldImagery)

    return fig, ax

def individual_wf_change_plot(int_1, int_2, elev_1, elev_2):
    """
        Plot changes in normalized gedi waveform
    """

    elevation, int_1, int_2 = interpolate_pair(int_1, int_2, elev_1, elev_2)

    intensity_change = np.asarray(int_2 - int_1)

    plt.plot(intensity_change, elevation, c = 'k')
    
    plt.fill_betweenx(elevation, 0, intensity_change, where=(intensity_change >= 0),
                      facecolor='green', alpha=0.5, interpolate=True)
    plt.fill_betweenx(elevation, 0, intensity_change, where=(intensity_change < 0),
                      facecolor='red', alpha=0.5, interpolate=True)
    
    plt.axvline(0, color='k', linestyle='--')
    plt.axhline(0, ls = '--', c = 'gray')

    plt.xlim(-1.2*np.max(np.abs(intensity_change)), 1.2*np.max(np.abs(intensity_change)))
    plt.xlabel('Difference in GEDI waveform intensity\n(WF$_2$ - WF$_1$)')
    plt.ylabel('Height relative to ground [m]')

    return plt.gcf(), plt.gca()

def get_quick_gedi_wf(shotnum, geo):
    rh_data = get_rh_data(shotnum, geo)

    rh = rh_data[[col for col in rh_data.columns if 'rh' in col]].values[0]

    i_o, e_o = get_wf_from_rh(rh, step = 2)

    return i_o, e_o