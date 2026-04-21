# ===============================================================
# Olivier Donzel-G.
# Uppsala University, Sweden
#
# for Panta Rhei 0.25
# to perform a subpixel centering of the direct beam in 4DSTEM nano-probe diffraction without beam stopper. 
# 
# Steps:
#   1) Sum of the first 20 DPs (or the first 10% if total < 20)
#   2) Gaussian blur + global max → initial center (cx_dyn, cy_dyn)
#   3) Serpentine scan; for each DP:
#        - disk mask centered at (cx_dyn, cy_dyn) -> outside disk = 0
#        - COM within the disk
#        - estimation of the Δx_abs = x_COM - cx_nom & Δy_abs = y_COM - cy_nom
#        - step limitation (10× median of the last 10) (dx_step, dy_step)
#        - cumulative update of the dynamic center (mask) to follow the direct beam
#   4) Recenter the DATACUBE using Δ_abs
# ===============================================================
# MIT License

# Copyright (c) 2026 Olivier Donzel-G.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===============================================================

import numpy as np
from collections import deque
from scipy.ndimage import shift, gaussian_filter
from panta_rhei.scripting import PRScriptingInterface
from panta_rhei.main.gui.utils import sliced_ndim

# ------------------------------ Parameters --------------------------------
R_PIXELS = 12              # mask radius (pixels) for the COM mask
GAUSS_SIGMA_INIT = 1.5     # sigma of the Gaussian blur to stabilize the initial peak detection
# --------------------------------------------------------------------------------------

api = PRScriptingInterface()
model = api.get_active_model()

data = api.get_active_data()
cube = data.copy()

ndim = sliced_ndim(cube.meta_data, cube.ndim)
if ndim != 4:
    raise TypeError("This script applies only to 4DSTEM data (scan_y, scan_x, q_y, q_x).")

Ny, Nx, Qy, Qx = cube.shape

# detector center
cx_nom = Qx // 2
cy_nom = Qy // 2

# Mask radius upper limit
R = min(R_PIXELS, min(Qx, Qy) // 2)

# COM and mask coordinates
Y, X = np.ogrid[:Qy, :Qx]
Xf = X.astype(np.float64)
Yf = Y.astype(np.float64)

base_name = model.get_output_name()

api.add_progress_bar(
    0, 100,
    additional_text=f"Initialization 10% -> Offset measurement 40% -> Centering 100%"
)

try:
    # ======================================================================
    # Initialization (20 first DPs, or 10% if total < 20)
    # ======================================================================
    total_positions = Ny * Nx
    if total_positions >= 20:
        n_init = 20
    else:
        n_init = max(1, int(np.ceil(0.10 * total_positions)))

    sum_img = np.zeros((Qy, Qx), dtype=np.float64)
    for i in range(n_init):
        iy = i // Nx
        ix = i % Nx
        sum_img += cube[iy, ix].astype(np.float64)

        api.set_progress(int(10 * (i + 1) / n_init))

    # Detection of the global max
    sum_blur = gaussian_filter(sum_img, sigma=GAUSS_SIGMA_INIT)
    y_peak, x_peak = np.unravel_index(np.argmax(sum_blur), sum_blur.shape)

    # initial dynamic center
    cx_dyn = float(x_peak)
    cy_dyn = float(y_peak)

    api.data_to_repo(f"{base_name}_sum_init", sum_img, meta_data={"type": "image2D"})
    api.display_image(f"{base_name}_sum_init")

    api.set_progress(10)

    # ======================================================================
    # Offset measurement with dynamic masked COM
    # ======================================================================
    # Offsets
    dx_abs = np.zeros((Ny, Nx), dtype=np.float64)
    dy_abs = np.zeros((Ny, Nx), dtype=np.float64)

    abs_dx_hist = deque(maxlen=10)
    abs_dy_hist = deque(maxlen=10)

    processed = 0
    total_to_process = Ny * Nx

    for iy in range(Ny):
        # Serpentine sequence: even rows L→R, odd rows R→L

        x_iter = range(0, Nx, 1) if (iy % 2 == 0) else range(Nx - 1, -1, -1)

        for ix in x_iter:
            dp = cube[iy, ix]

            mask_here = (((Xf - cx_dyn) ** 2 + (Yf - cy_dyn) ** 2) <= (R ** 2))
            # masking : outside disk = 0
            dp_masked = np.where(mask_here, dp, 0)

            # ----- COM -----
            S = dp_masked.sum()
            if S == 0.0:
                x_com = cx_dyn
                y_com = cy_dyn
                dx_step = 0.0
                dy_step = 0.0
            else:
                x_com = float((dp_masked * Xf).sum() / S)
                y_com = float((dp_masked * Yf).sum() / S)
                dx_step = x_com - cx_dyn
                dy_step = y_com - cy_dyn

            # ----- Step limitation : (10× median of the last 10) -----
            if len(abs_dx_hist) >= 10:
                med_abs_dx = float(np.median(abs_dx_hist))
                if med_abs_dx > 0.0 and abs(dx_step) > 10.0 * med_abs_dx:
                    dx_step = np.copysign(med_abs_dx, dx_step)

            if len(abs_dy_hist) >= 10:
                med_abs_dy = float(np.median(abs_dy_hist))
                if med_abs_dy > 0.0 and abs(dy_step) > 10.0 * med_abs_dy:
                    dy_step = np.copysign(med_abs_dy, dy_step)

            # ----- Offsets-----
            dx_abs[iy, ix] = x_com - cx_nom
            dy_abs[iy, ix] = y_com - cy_nom
            
            # ----- Update dynamic center (mask) -----
            cx_dyn += dx_step
            cy_dyn += dy_step

            abs_dx_hist.append(abs(dx_step))
            abs_dy_hist.append(abs(dy_step))

            processed += 1
            # Progression 10→40 %
            if processed % max(1, total_to_process // 200) == 0:
                api.set_progress(10 + int(30 * processed / total_to_process))

    api.data_to_repo(f"{base_name}_dx_COM_abs", dx_abs, meta_data={"type": "image2D"})
    api.data_to_repo(f"{base_name}_dy_COM_abs", dy_abs, meta_data={"type": "image2D"})
    api.display_image(f"{base_name}_dx_COM_abs")
    api.display_image(f"{base_name}_dy_COM_abs")

    # ======================================================================
    # 4) Recenter the DATACUBE using Δ_abs
    # ======================================================================
    cube_corr = np.empty_like(cube)

    for iy in range(Ny):
        p = 40 + int(60 * (iy + 1) / Ny)
        api.set_progress(p)
        for ix in range(Nx):
            cube_corr[iy, ix] = shift(
                cube[iy, ix],
                shift=(-dy_abs[iy, ix], -dx_abs[iy, ix]),
                order=1,
                mode="nearest"
            )

    api.data_to_repo(f"{base_name}_recentered", cube_corr, meta_data=cube.meta_data)
    api.display_image(f"{base_name}_recentered")
    api.set_progress(100)

finally:
    api.remove_progress_bar()