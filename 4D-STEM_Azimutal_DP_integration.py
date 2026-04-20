# ===============================================================
# Olivier Donzel-G. https://orcid.org/0000-0002-2101-3746
# Uppsala University, Sweden
#
# The following script comes with no warranty. 
# for Panta Rhei 0.25
# to perform azimutal integration on 4DSTEM data to quickly inspect the dataset. 
# 
# Steps:
# Assuming the datacube DP are correctly centered, for each DP:
#   1) calculate the polar transform (r, θ)
#   2) extract the first and most intense peak position
#      -for the most intense peak: calculate the FWHM and intensity
#   3) display the recalibrated rotationally integrated DP profiles with the peak main characteristics
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from panta_rhei.scripting import PRScriptingInterface, PRScriptingTypes
from panta_rhei.main.gui.utils import sliced_ndim
from scipy.signal import find_peaks#, peak_widths

api = PRScriptingInterface()
model = api.get_active_model()
data = api.get_active_data()

ndim = sliced_ndim(data.meta_data, data.ndim)
if ndim != 4:
    raise TypeError("This script applies only to 4DSTEM data (scan_y, scan_x, q_y, q_x).")

# -----------------------------
# Parameters
# -----------------------------
NUM_RADIAL  = 128 # radial resolution of the radial profile (e.g., 128 → averages over rings of 1 px)
NUM_ANGULAR = 90 # angular resolution of the radial profile (e.g., 90 → averages over sectors of 4°)
MIN_RADIUS  = 10

# ---------------------------------------------
# 1) Polar transformation
# ---------------------------------------------
def polar_transform(dp, num_radial=NUM_RADIAL, num_angular=NUM_ANGULAR):
    H, W = dp.shape
    cx, cy = W/2, H/2
    r_max = min(cx, cy)

    r = np.linspace(0, r_max, num_radial)
    theta = np.linspace(0, 2*np.pi, num_angular)

    R, Theta = np.meshgrid(r, theta, indexing='ij')
    X = R * np.cos(Theta) + cx
    Y = R * np.sin(Theta) + cy

    coords = np.array([Y.ravel(), X.ravel()])
    polar = map_coordinates(dp, coords, order=1, mode='nearest')
    return polar.reshape((num_radial, num_angular))

# ---------------------------------------------
# 2) FWHM
# ---------------------------------------------
def compute_fwhm(profile, peak_index):
    if peak_index <= 0 or peak_index >= len(profile)-1:
        return np.nan

    peak_val = profile[peak_index]
    if peak_val <= 0:
        return np.nan

    half = peak_val / 2
    y = profile

    # gauche
    k = peak_index
    while k > 0 and y[k] > half:
        k -= 1
    if k == peak_index:
        return np.nan
    try:
        x_left = k + (half - y[k]) / (y[k+1] - y[k])
    except:
        return np.nan

    # droite
    m = peak_index
    while m < len(y)-1 and y[m] > half:
        m += 1
    if m == peak_index:
        return np.nan
    try:
        x_right = (m-1) + (half - y[m-1]) / (y[m] - y[m-1])
    except:
        return np.nan

    return x_right - x_left if x_right > x_left else np.nan


# ============================================================
# Main
# ============================================================
Ny, Nx, Qy, Qx = data.shape

maxpeak_radius_map    = np.zeros((Ny, Nx))
maxpeak_intensity_map = np.zeros((Ny, Nx))
maxpeak_FWHM_map           = np.zeros((Ny, Nx))
first_peak_radius_map= np.zeros((Ny, Nx))

radial_profiles = np.zeros((NUM_RADIAL, Ny, Nx))

total = Ny * Nx
counter = 0
api.add_progress_bar(0, 100, "Progress")


for iy in range(Ny):
    for ix in range(Nx):

        dp = data[iy, ix].astype(float)

        polar = polar_transform(dp, NUM_RADIAL, NUM_ANGULAR)
        radial_profile = polar.mean(axis=1)
        radial_profiles[:, iy, ix] = radial_profile

        # peak detection
        prof = radial_profile.copy()
        prof[:MIN_RADIUS] = -np.inf     # direct beam masking
        
        #detection of the most intense peak
        maxpeak_index = np.argmax(prof) 
        maxpeak_value = radial_profile[maxpeak_index]
        maxfwhm_val = compute_fwhm(radial_profile, maxpeak_index)
        
        #detection of the first peak
        peaks, props = find_peaks(radial_profile, prominence=2) 
        # filtrage MIN_RADIUS
        valid = peaks[peaks >= MIN_RADIUS]
        if len(valid) > 0:
            first_peak_index = valid[0]             
        else:
            first_peak_index = None

        maxpeak_radius_map[iy, ix]    = maxpeak_index
        maxpeak_intensity_map[iy, ix] = maxpeak_value
        maxpeak_FWHM_map[iy, ix]      = maxfwhm_val
        first_peak_radius_map[iy, ix]   = first_peak_index
        
        counter += 1
        if counter % max(1, total//200) == 0:
            api.set_progress(int(100 * counter / total))


api.remove_progress_bar()

# ============================================================
# Display
# ============================================================
base = model.get_output_name()

api.data_to_repo(f"{base}_maxpeak_radius_px_map", maxpeak_radius_map, meta_data={"type":"image2D"})
api.data_to_repo(f"{base}_maxpeak_intensity_px_map", maxpeak_intensity_map, meta_data={"type":"image2D"})
api.data_to_repo(f"{base}_maxpeak_FWHM_px_map", maxpeak_FWHM_map, meta_data={"type":"image2D"})
api.data_to_repo(f"{base}_first_peak_radius_px_map", first_peak_radius_map, meta_data={"type":"image2D"})
api.data_to_repo(f"{base}_radial_profiles", radial_profiles,{"type":"image3D"})

def display_and_set_cmap(dataset_name, cmap):
    viewer = api.display_image(dataset_name, auto_size=True)
    dc = viewer.get_display_control()
    dc.set_parameters({'color_map': cmap})
    return viewer

viewer1 = display_and_set_cmap(f"{base}_maxpeak_radius_px_map", "flame")
viewer2 = display_and_set_cmap(f"{base}_maxpeak_intensity_px_map", "turbo")
viewer3 = display_and_set_cmap(f"{base}_maxpeak_FWHM_px_map", "bipolar")
viewer4 = display_and_set_cmap(f"{base}_first_peak_radius_px_map", "turbo")
viewer5 = display_and_set_cmap(f"{base}_radial_profiles", "grey")

#recalibration of the radial profiles
scaling = model.get_scaling()
main_calibration = scaling.get_parameters()
radial_profiles_3dcube = api.get_data_models_by_name(f"{base}_radial_profiles")[0]

radial_profiles_3dcube.set_user_calibrations(axes=0,
                                             values=main_calibration['calib'][2],
                                             units=main_calibration['unit'][2],
                                             origins=0.0,
                                             use_prefixes=main_calibration['use_prefix'][2],
                                             fixed_prefixes=main_calibration['fixed_prefix'][2],
                                             block=False
                                             )
radial_profiles_3dcube.set_user_calibrations(axes=1,
                                             values=main_calibration['calib'][3],
                                             units=main_calibration['unit'][3],
                                             origins=0.0,
                                             use_prefixes=main_calibration['use_prefix'][3],
                                             fixed_prefixes=main_calibration['fixed_prefix'][3],
                                             block=False
                                             )
radial_profiles_3dcube.set_user_calibrations(axes=2,
                                             values=main_calibration['calib'][0]*data.shape[2]/radial_profiles.shape[0], #works with different sampling for the azimutal integration
                                             units=main_calibration['unit'][0],
                                             origins=0.0,
                                             use_prefixes=main_calibration['use_prefix'][0],
                                             fixed_prefixes=main_calibration['fixed_prefix'][0],
                                             block=False
                                             )

# addition of depthscans
depthA = model.insert(
    PRScriptingTypes.ImageCubeDepthScan,
    parameters={"name": "ImageCubeDepthScan_A"}
)
dummy1 = viewer1.insert(PRScriptingTypes.DummyDepthScan,
    parameters={"name": "DummySource1"}
)
dummy2 = viewer2.insert(PRScriptingTypes.DummyDepthScan,
    parameters={"name": "DummySource2"}
)
dummy3 = viewer3.insert(PRScriptingTypes.DummyDepthScan,
    parameters={"name": "DummySource3"}
)
dummy4 = viewer4.insert(PRScriptingTypes.DummyDepthScan,
    parameters={"name": "DummySource4"}
)
depthB = viewer5.insert(
    PRScriptingTypes.DepthScan,
    parameters={"name": "DepthScan_B"}
)

depthscanA = api.display_image("ImageCubeDepthScan_A")
depthscanB = api.display_plot("DepthScan_B")

api.create_data_tool_link([dummy1, depthA])
api.create_data_tool_link([dummy2, depthA])
api.create_data_tool_link([dummy3, depthA])
api.create_data_tool_link([dummy4, depthA])
api.create_data_tool_link([depthA, depthB])