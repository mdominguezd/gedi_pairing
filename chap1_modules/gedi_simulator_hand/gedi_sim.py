"""
Module with functions to work with Steven Hancock GEDI simulator using Laser Scanning data.
"""

import subprocess
import os

import numpy as np
import laspy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

from . import gedio


def gedi_metrics(wf_file = 'wf.txt', out = 'metric', ground = True):
    if ground:
        cmd = f"gediMetric -input {wf_file} -outRoot {out} -ground"
    else:
        cmd = f"gediMetric -input {wf_file} -outRoot {out}"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    
    return out+'.metric.txt'

def clip_gedi_footprint(file, x, y, shotnum, dir = 'als-data-sim/Descarga_PNOAD/las_files/', all_again = False):
    """
        Function to clip GEDI footprint on 
    """
    clipped = str(shotnum) + '_clipped.las'

    if (all_again) & (clipped in os.listdir(dir)):
        cmd = f"rm -rf {dir + clipped}"
        result = subprocess.run(cmd, shell = True, capture_output=True)
    
    if clipped not in os.listdir(dir):
        cmd = f"las2las64 -i {file} -o {dir + clipped} -keep_circle {x} {y} 12.5"
        result = subprocess.run(cmd, shell = True, capture_output=True)

        # Check for errors
        if result.returncode != 0:
            print(f"Error: {result.stderr}")

        las = unclassified_removal_las(dir + clipped)
        las.write(dir + clipped)

    return dir + clipped

def unclassified_removal_las(file):
    las = laspy.read(file)

    # Remove points outside 2 standard deviations from the mean height
    # Create mask to keep only points with classification 2-6
    mask = (las.classification != 0) & (las.classification != 1) & (las.classification < 7)

    # Apply mask to filter points
    las.points = las.points[mask]
    
    return las

def visua_laz(las_file):

    las = laspy.read(las_file)

    # Get coordinates
    x = las.x
    y = las.y
    z = las.z

    # Create 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points (subsample if file is large)
    step = max(1, len(x) // 100000)  # Limit to ~100k points for performance
    class_colors = {
    0: 'white',     # Never classified
    1: 'white',     # Unclassified
    2: 'brown',     # Ground
    3: 'lightgreen',# Low vegetation
    4: 'green',     # Medium vegetation
    5: 'darkgreen', # High vegetation
    6: 'red',       # Building
    7: 'purple',    # Low noise
    9: 'blue',      # Water
    }

    colors = [class_colors.get(c, 'black') for c in las.classification[::step]]

    ax.scatter(x[::step], y[::step], z[::step], c=colors)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('LAS Point Cloud')
    # plt.colorbar(ax.scatter(x[::step], y[::step], z[::step], c=z[::step], cmap='terrain', s=0.5))

    plt.tight_layout()

    plt.show()
    
    return ax

def correct_ground_als(file):

    # Read LAS file
    las = laspy.read(file)
    
    las_out = las

    # Extract ground points
    ground_mask = las.classification == 2
    ground_xyz = np.vstack((las.x[ground_mask], 
                            las.y[ground_mask], 
                            las.z[ground_mask])).T

    # Create grid for interpolation
    x_min, x_max = las.x.min(), las.x.max()
    y_min, y_max = las.y.min(), las.y.max()

    # 1m resolution grid
    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max, 1.0),
        np.arange(y_min, y_max, 1.0)
    )

    # Interpolate ground elevation for each point
    all_xy = np.vstack((las.x, las.y)).T
    ground_at_points = griddata(
        ground_xyz[:, :2],
        ground_xyz[:, 2],
        all_xy,
        method='nearest',
        fill_value=np.nan
    )

    las_out.header = las.header
    las_out.vlrs = las.vlrs
    
    las_out.header.offsets = [las.header.offsets[0], las.header.offsets[1], np.min(las.z[:]-ground_at_points[:])]
    las_out.header.scales = las.header.scales

    # Normalize
    las_out.z = las.z - ground_at_points

    o_file = file[:-4] + '_normalized.las'

    las_out.write(o_file)

    return o_file


def get_gedisim_wf(file, x, y, shotnum, dir = 'als-data-sim/Descarga_PNOAD/processed_wf/',
                   relative = True, algorithm = 'max', slope = False, normalize_als = False, 
                   all_again = False, norm_1_wf_elsum = True, ground = True, geogedi_cor = False):
    """
        Function to calculate the gedi simulated waveform from LS data.
    """

    if normalize_als:
        out = str(shotnum) + '_norm_wf.txt'
    elif geogedi_cor:
        out = str(shotnum)  + '_geogedi_corrected.txt'
    else:
        out = str(shotnum) + '_wf.txt'
        
    out_file = dir + out
    
    if (all_again) & (out in os.listdir(dir)):
            cmd = f"rm -rf {out_file}"
            result = subprocess.run(cmd, shell = True, capture_output=True)

    if out not in os.listdir(dir):

        las = clip_gedi_footprint(file, x, y, shotnum=shotnum, all_again= all_again)

        if normalize_als:
            las = correct_ground_als(las)

        if ground:
            cmd = f"gediRat -input {las} -coord {x} {y} -ground -output {out_file}"
        else:
            cmd = f"gediRat -input {las} -coord {x} {y} -output {out_file}"
        result = subprocess.run(cmd, shell = True, capture_output=True)

        # Check for errors
        if result.returncode != 0:
            print(f"Error: {result.stderr}")

    wf_df = gedio.read_waveform(out_file)

    intensity = wf_df[wf_df['discrete_intensity'] > 0]['discrete_intensity']
    elevation = wf_df[wf_df['discrete_intensity'] > 0]['elevation']

    if relative:
        o_met = dir + str(shotnum) +  '_metric'
        if o_met + '.metric.txt' not in os.listdir(dir):
            met = gedi_metrics(out_file, out = o_met)
        else:
            met = o_met + '.metric.txt'
        df_met = gedio.read_metrics(met)

    if norm_1_wf_elsum:
        intensity = intensity/np.max(intensity)
    else:
        intensity = intensity/np.sum(intensity)

    if algorithm == 'max':
        elevation = elevation - np.float16(df_met['7 maxGround'])
    elif algorithm == 'gauss':
        elevation = elevation - np.float16(df_met['6 gHeight'])
    elif algorithm == 'infl':
        elevation = elevation - np.float16(df_met['8 inflGround'])

    if slope:
        return intensity.values[::-1], elevation.values[::-1], np.float16(df_met['148 gSlope'])
    
    return intensity.values[::-1], elevation.values[::-1]

def get_rh_metrics_from_sim(intensity, elevation, proportion = 0.5):
    """
    Calculate RH from GEDI waveform data.
    
    Parameters:
    -----------
    intensity : array-like
        Waveform intensity values
    elevation : array-like
        Corresponding elevation values (in meters)
    proportion : float
        Proportion of cumulative energy
    
    Returns:
    --------
    rh : float
        Relative height at proportion% cumulative energy
    """

    dx = (elevation[1]-elevation[0])

    # Calculate cumulative energy
    cumulative_energy = np.cumsum(intensity*dx)
    total_energy = cumulative_energy[-1]
    
    # Normalize to get cumulative proportion
    cumulative_proportion = cumulative_energy / total_energy
    
    # Find indices for RH50 and RH98
    idx = np.argmin(np.abs(cumulative_proportion - proportion))
    
    # Calculate relative heights
    rh = elevation[idx] 
    
    return rh

def get_re_metrics_from_sim(intensity, elevation, height = 10):
    """
    Calculate RE from GEDI waveform data.
    
    Parameters:
    -----------
    intensity : array-like
        Waveform intensity values
    elevation : array-like
        Corresponding elevation values (in meters)
    height: float
        Corresponding to height below wich relative energy is calcualted
    
    Returns:
    --------
    re : float
        Relative energy below height
    """
    re = np.trapezoid(intensity[elevation < height], 
                      elevation[elevation < height])\
                        /np.trapezoid(intensity, elevation)
    
    return re