"""
================================
Workflow dialog plugin for 4D-STEM azimuthal integration in Panta Rhei 0.25.

Three sections:
  1. Background quality test — single DP, adjustable percentile, overlay plot.
  2. Full dataset run — uses percentile from section 1.
  3. Extra outputs — separate the amorphous rings from the background, compute the crystalline amorphous ratio
  
Install: copy to the user plugin folder
  Linux  : ~/.local/share/panta_rhei/plugins/
  Windows: C:\\Users\\<Username>\\AppData\\Local\\CEOS\\panta_rhei\\plugins\\
Then: Scripts > Load plugins
"""
# ===============================================================
# Copyright (C) <2026>  <Olivier Donzel-Gargand>
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ===============================================================
    
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from PyQt5 import QtWidgets

from panta_rhei.main.gui.base_workflow_dialog import BaseWorkflowDialog
from panta_rhei.scripting import PRScriptingInterface, PRScriptingTypes
from panta_rhei.main.gui.panta_rhei_interface import PantaRheiInterface
from panta_rhei.main.gui.utils import sliced_ndim
import copy
# ---------------------------------------------------------------------------
# Processing constants
# ---------------------------------------------------------------------------
NUM_RADIAL    = 128
NUM_ANGULAR   = 90
MIN_RADIUS    = 10
NUM_ANG_FREQS = NUM_ANGULAR // 2 + 1

# Repo key for the test overlay — fixed name, overwritten on each test run
_TEST_OVERLAY_KEY   = "azimuthal_test_overlay"
_BGFIT_SPEC_KEY     = "azimuthal_bgfit_spectrum"


# ---------------------------------------------------------------------------
# Processing functions (inlined — no external dependency)
# ---------------------------------------------------------------------------

def polar_transform(dp, num_radial=NUM_RADIAL, num_angular=NUM_ANGULAR):
    H, W  = dp.shape
    cx, cy = W / 2.0, H / 2.0
    r_max  = min(cx, cy)
    r      = np.linspace(0, r_max, num_radial)
    theta  = np.linspace(0, 2 * np.pi, num_angular, endpoint=False)
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    X = R * np.cos(Theta) + cx
    Y = R * np.sin(Theta) + cy
    coords = np.array([Y.ravel(), X.ravel()])
    polar  = map_coordinates(dp, coords, order=1, mode='nearest')
    return polar.reshape((num_radial, num_angular))


def compute_fwhm(profile, peak_index):
    if peak_index <= 0 or peak_index >= len(profile) - 1:
        return np.nan
    peak_val = profile[peak_index]
    if peak_val <= 0:
        return np.nan
    half = peak_val / 2.0
    y = profile
    k = peak_index
    while k > 0 and y[k] > half:
        k -= 1
    if k == peak_index:
        return np.nan
    try:
        x_left = k + (half - y[k]) / (y[k + 1] - y[k])
    except Exception:
        return np.nan
    m = peak_index
    while m < len(y) - 1 and y[m] > half:
        m += 1
    if m == peak_index:
        return np.nan
    try:
        x_right = (m - 1) + (half - y[m - 1]) / (y[m] - y[m - 1])
    except Exception:
        return np.nan
    return x_right - x_left if x_right > x_left else np.nan

def compute_angular_fingerprint(polar, profile_bgsubtracted):
    """
    Weighted azimuthal FFT fingerprint for one diffraction pixel.

    Each ring row is mean-subtracted to remove its azimuthally uniform
    (diffuse/amorphous) component, then weighted by the background-subtracted
    radial peak intensity so that rings carrying crystalline signal dominate.
    The result is L2-normalised and returned as its one-sided rfft.

    Note on background subtraction
    -------------------------------
    The per-ring mean (polar.mean(axis=1)) is the correct quantity to subtract
    here — not the percentile background used for radial profiles.  The mean
    is exactly the azimuthally isotropic component; removing it leaves only
    the angular modulation (Bragg spot positions).  The percentile background
    is the correct weight source (profile_bgsubtracted) but would leave a
    residual DC offset if used as the modulation baseline.

    Cross-correlation between two pixels at display time:
        C = np.fft.irfft(fp_a * fp_b.conj())   # (Ntheta,) real
    Peak position of C = projected misorientation angle between the two grains.

    Parameters
    ----------
    polar : (Nr, Ntheta) float
        Polar transform of one diffraction pattern.
    profile_bgsubtracted : (Nr,) float
        Background-subtracted radial profile (profile - percentile background).
        Used as per-ring weights; negative values are clipped to zero.

    Returns
    -------
    fingerprint : (Ntheta // 2 + 1,) complex
        rfft of the normalised weighted azimuthal signal.
        Returns a zero array for amorphous / vacuum pixels.
    """
    nr, n_theta = polar.shape

    weights    = np.maximum(profile_bgsubtracted[:nr], 0.0)
    modulation = polar - polar.mean(axis=1, keepdims=True)           # (Nr, Ntheta)
    weighted   = (weights[:, np.newaxis] * modulation).sum(axis=0)  # (Ntheta,)

    norm = np.linalg.norm(weighted)
    if norm < 1e-12:
        return np.zeros(n_theta // 2 + 1, dtype=complex)

    return np.fft.rfft(weighted / norm)

def compute_orientation_angle(polar, profile_bgsubtracted,
                               min_symmetry=2, max_symmetry=8):
    """
    Estimate the absolute orientation angle of a crystalline diffraction
    pattern relative to θ=0, modulo the inferred crystal symmetry order.

    The symmetry order is determined automatically from the autocorrelation
    of the weighted azimuthal signal — no prior knowledge of the crystal
    structure is required.

    Algorithm
    ---------
    1. Build the weighted background-subtracted azimuthal signal (same
       preprocessing as compute_angular_fingerprint).
    2. Compute its circular autocorrelation via rfft.  The first significant
       peak beyond Δθ=0 reveals the rotational period of the pattern.
    3. Infer the symmetry order n = round(Ntheta / first_lag), clamped to
       [min_symmetry, max_symmetry].
    4. Fold the signal into n equal sectors and average them — equivalent
       to phase-coherent stacking, which suppresses noise while preserving
       the orientation peak.
    5. Return the centre-of-mass angle of the folded signal in degrees,
       which gives the orientation modulo 360/n degrees.

    Parameters
    ----------
    polar : (Nr, Ntheta) float
    profile_bgsubtracted : (Nr,) float
        Same inputs as compute_angular_fingerprint.
    min_symmetry, max_symmetry : int
        Clamp range for the inferred symmetry order (default 2–8).

    Returns
    -------
    orientation_deg : float
        Orientation angle in degrees in [0, 360/n).
        Returns np.nan for amorphous / vacuum pixels or when no clear
        rotational symmetry is detected.
    """
    nr, n_theta = polar.shape

    # --- Weighted mean-subtracted azimuthal signal -----------------------
    # Per-ring mean subtraction removes the isotropic component; the
    # percentile-based profile_bgsubtracted weights the rings by crystalline
    # peak intensity (same rationale as compute_angular_fingerprint).
    weights    = np.maximum(profile_bgsubtracted[:nr], 0.0)
    modulation = polar - polar.mean(axis=1, keepdims=True)
    weighted   = (weights[:, np.newaxis] * modulation).sum(axis=0)

    norm = np.linalg.norm(weighted)
    if norm < 1e-12:
        return np.nan   # amorphous / vacuum
    weighted = weighted / norm

    # --- Infer symmetry order from circular autocorrelation --------------
    # AC[0] is the self-dot-product (always the global maximum).
    # The first significant peak at lag > 0 reveals the rotational period.
    ac = np.fft.irfft(np.abs(np.fft.rfft(weighted)) ** 2, n=n_theta)
    ac_half = ac[1 : n_theta // 2 + 1]
    peaks, _ = find_peaks(ac_half, height=0.1 * ac[0])

    if len(peaks) == 0:
        return np.nan   # no detectable periodicity

    first_lag     = int(peaks[0]) + 1   # +1: ac_half starts at lag=1
    sym_raw       = round(n_theta / first_lag)
    symmetry_order = int(np.clip(sym_raw, min_symmetry, max_symmetry))

    # --- Fold signal into symmetry sectors and average -------------------
    # Phase-coherent stacking: all n sectors are summed, boosting the
    # orientation signal while noise averages toward zero.
    period_bins = int(round(n_theta / symmetry_order))
    n_complete  = symmetry_order * period_bins
    if period_bins < 2 or n_complete > n_theta:
        return np.nan

    folded = weighted[:n_complete].reshape(symmetry_order, period_bins).mean(axis=0)

    # --- Orientation = centre-of-mass of positive part of folded signal --
    pos = np.maximum(folded, 0.0)
    total = pos.sum()
    if total < 1e-12:
        return np.nan

    bins            = np.arange(period_bins, dtype=float)
    orientation_bin = (bins * pos).sum() / total
    orientation_deg = orientation_bin * 360.0 / n_theta   # in [0, 360/n)

    return orientation_deg

_BG_MODELS = [
    "Power law",
    "Exponential",
    "Power law + constant",
    "Asymmetric polynomial",
    "Window minima",
]

# Hyper-parameters for the two new models (not exposed to UI — sensible defaults)
_ASYM_POLY_DEGREE = 4     # polynomial degree for the IRLS model
_ASYM_POLY_P      = 0.05  # weight for above-fit residuals (pushes curve downward)
_ASYM_POLY_NITER  = 20    # IRLS iterations
_WIN_MINIMA_WIDTH = 20    # window width (channels) for local-minima sampling


def fit_background_radial(profile, fit_start, fit_end, model):
    """
    Fit a smooth structural background to a radial profile.

    profile   : 1D array (NUM_RADIAL,)  — percentile background at one pixel
    fit_start : int channel index — lower bound of fit window (avoid direct beam)
    fit_end   : int channel index — upper bound of fit window
    model     : str — one of _BG_MODELS

    Returns
    -------
    fitted : (NUM_RADIAL,) smooth background estimate over the full range
    info   : str  brief parameter summary for the status label

    Scientific notes
    ----------------
    Power law (log-log linear):
        I(q) = A * q^r   fitted via np.polyfit in log-log space.
        Appropriate when the background decays monotonically as a power of q.
        Standard choice for electron diffraction diffuse background.

    Exponential (log-linear):
        I(q) = A * exp(-b*q)   fitted via np.polyfit(x, log(y), 1).
        Better for tails of the direct beam (Gaussian-like decay).

    Power law + constant:
        I(q) = A * q^r + C   fitted via scipy curve_fit.
        Handles detector dark current, multiple-scattering floor, or
        convergence-beam halos that leave a residual constant offset.
        Falls back to Power law if curve_fit fails.

    Asymmetric polynomial (IRLS):
        Fits a degree-4 polynomial by iterative reweighted least squares.
        Residuals above the current estimate are down-weighted by p=0.05 so
        the curve is pushed toward the lower envelope of the signal.
        Bragg peaks, which always sit above the background, are effectively
        ignored after the first few iterations.

    Window minima:
        Slides a window of _WIN_MINIMA_WIDTH channels across the fit range,
        retaining the minimum value in each window as a background anchor.
        A power law is then fitted through those anchor points in log-log
        space. Because the anchors sit below any Bragg peak that falls in
        their window, the resulting curve follows the diffuse background
        floor without clipping it.
    """
    nE      = len(profile)
    r_axis  = np.arange(nE, dtype=float)
    idx     = np.arange(int(fit_start), min(int(fit_end) + 1, nE))
    if len(idx) < 2:
        return np.zeros(nE, dtype=float), "fit range too narrow"
    x       = r_axis[idx]
    y       = profile[idx]
    y_safe  = np.maximum(y, 1e-10)

    fitted = np.zeros(nE, dtype=float)

    if model == "Power law":
        log_x  = np.log(np.maximum(x, 1.0))
        log_y  = np.log(y_safe)
        coeffs = np.polyfit(log_x, log_y, 1)
        r_exp, log_A = coeffs
        A = np.exp(log_A)
        fitted = A * np.power(np.maximum(r_axis, 1.0), r_exp)
        info   = f"A={A:.2g}  r={r_exp:.3f}"

    elif model == "Exponential":
        log_y  = np.log(y_safe)
        coeffs = np.polyfit(x, log_y, 1)
        b      = -coeffs[0]
        A      = np.exp(coeffs[1])
        fitted = A * np.exp(-b * r_axis)
        info   = f"A={A:.2g}  b={b:.4f}"

    elif model == "Power law + constant":
        def _model(q, A, r, C):
            return A * np.power(np.maximum(q, 1.0), r) + C
        try:
            p0   = [y_safe.max(), -2.0, max(y_safe.min(), 0.0)]
            popt, _ = curve_fit(_model, x, y, p0=p0, maxfev=2000,
                                bounds=([0, -10, 0], [np.inf, 0, np.inf]))
            fitted = _model(r_axis, *popt)
            info   = f"A={popt[0]:.2g}  r={popt[1]:.3f}  C={popt[2]:.2g}"
        except Exception:
            # Fallback to simple power law
            log_x  = np.log(np.maximum(x, 1.0))
            log_y  = np.log(y_safe)
            coeffs = np.polyfit(log_x, log_y, 1)
            r_exp, log_A = coeffs
            A = np.exp(log_A)
            fitted = A * np.power(np.maximum(r_axis, 1.0), r_exp)
            info   = f"fallback power law  A={A:.2g}  r={r_exp:.3f}"

    elif model == "Asymmetric polynomial":
        # Iterative Reweighted Least Squares with asymmetric loss.
        # Points sitting above the current estimate contribute only p=0.05 to
        # the next polynomial fit, so the curve is iteratively pushed down
        # toward the lower envelope.  Bragg peaks (always above background)
        # are down-weighted to near-irrelevance within a few iterations.
        w      = np.ones(len(x))
        coeffs = np.polyfit(x, y, _ASYM_POLY_DEGREE, w=w)
        for _ in range(_ASYM_POLY_NITER):
            y_hat  = np.polyval(coeffs, x)
            w      = np.where(y > y_hat, _ASYM_POLY_P, 1.0)
            coeffs = np.polyfit(x, y, _ASYM_POLY_DEGREE, w=w)
        fitted = np.polyval(coeffs, r_axis)
        info   = (f"deg={_ASYM_POLY_DEGREE}  "
                  f"p={_ASYM_POLY_P}  iter={_ASYM_POLY_NITER}")

    elif model == "Window minima":
        # Slide a window of _WIN_MINIMA_WIDTH channels across the fit range.
        # Keep the minimum of each window as a background anchor: these points
        # sit below any Bragg peak that lands in their window.
        # A power law is then fitted through the anchors in log-log space —
        # the physically motivated form for electron diffraction backgrounds.
        anchors_x, anchors_y = [], []
        i = int(fit_start)
        while i <= int(fit_end):
            j      = min(i + _WIN_MINIMA_WIDTH, int(fit_end) + 1)
            seg    = profile[i:j]
            k_min  = int(np.argmin(seg))
            anchors_x.append(r_axis[i + k_min])
            anchors_y.append(float(seg[k_min]))
            i = j

        ax      = np.array(anchors_x)
        ay_safe = np.maximum(anchors_y, 1e-10)

        if len(ax) >= 2:
            log_ax = np.log(np.maximum(ax, 1.0))
            log_ay = np.log(ay_safe)
            coeffs = np.polyfit(log_ax, log_ay, 1)
            r_exp, log_A = coeffs
            A = np.exp(log_A)
            fitted = A * np.power(np.maximum(r_axis, 1.0), r_exp)
            info   = f"n_anchors={len(ax)}  A={A:.2g}  r={r_exp:.3f}"
        else:
            # Fit range narrower than one window — fall back to power law
            log_x  = np.log(np.maximum(x, 1.0))
            log_y  = np.log(y_safe)
            coeffs = np.polyfit(log_x, log_y, 1)
            r_exp, log_A = coeffs
            A = np.exp(log_A)
            fitted = A * np.power(np.maximum(r_axis, 1.0), r_exp)
            info   = f"fallback power law  A={A:.2g}  r={r_exp:.3f}"

    # Ensure non-negative fitted background
    fitted = np.maximum(fitted, 0.0)
    return fitted, info

def compute_angular_sector_maps(polar, step=3):
    """
    Integrate one polar diffraction pattern over non-overlapping angular windows.

    Called once per pixel inside the main loop where polar is already computed.
    Windows are centred every `step` bins across the unique 180° half.
    Each window spans 2*(step//2)+1 bins. The Friedel pair at N+NUM_ANGULAR//2
    is summed into the same window before averaging.
    Radial integration excludes the central beam via MIN_RADIUS.

    Parameters
    ----------
    polar : (NUM_RADIAL, NUM_ANGULAR) float — polar transform of one DP
    step  : int — must be one of [1, 3, 5, 9, 15] for NUM_ANGULAR=90

    Returns
    -------
    sector_values : (n_maps,) float
        Mean intensity per angular sector for this pixel.
    """
    # VALID_STEPS = [1, 3, 5, 9, 15]
    # if step not in VALID_STEPS:
    #     raise ValueError(
    #         f"step={step} is invalid for NUM_ANGULAR={NUM_ANGULAR}. "
    #         f"Valid values are {VALID_STEPS}.")

    friedel_offset = NUM_ANGULAR // 2
    half_width     = step // 2
    n_bins_per_win = 2 * half_width + 1
    n_maps         = friedel_offset // step
    polar_crop     = polar[MIN_RADIUS:, :]

    sector_values = np.zeros(n_maps, dtype=float)
    for k in range(n_maps):
        N       = k * step
        primary = [(N - half_width + j) % NUM_ANGULAR
                   for j in range(n_bins_per_win)]
        friedel = [(N + friedel_offset - half_width + j) % NUM_ANGULAR
                   for j in range(n_bins_per_win)]
        sector_values[k] = polar_crop[:, primary + friedel].mean()

    return sector_values
    
def unpack_sector_maps(sector_maps, base):
    """
    Split (Ny, Nx, n_maps) into a list of n individual (Ny, Nx) arrays.

    Parameters
    ----------
    sector_maps : (Ny, Nx, n_maps) float

    Returns
    -------
    tuple of n (name,(Ny, Nx) arrays)
    """
    return [(f"{base}_sector_{k:02d}", sector_maps[:, :, k])
            for k in range(sector_maps.shape[2])]

# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class AzimuthalIntegrationDialog(BaseWorkflowDialog):
    """
    Two-section workflow dialog for 4D-STEM azimuthal integration.

    Section 1 — Background quality test:
      Select a single DP by scan coordinates, set the background percentile,
      and preview the radial profile / background / BG-subtracted overlay.

    Section 2 — Full dataset run:
      Runs the full azimuthal integration loop using the percentile from
      section 1 and pushes all output maps to the repository.
    """

    def __init__(self, parent=None):
        self._model        = None
        self._data         = None
        self._base         = ""
        self._interface    = None
        self._depth_scan   = None   # ImageCubeDepthScan on 4D model (section 1)
        self._scan_viewer  = None   # mean scan image viewer
        self._test_viewer  = None   # overlay plot viewer (section 1)
        self._full_viewers = {}     # name → viewer (section 2)
        # Section 4 — background ring isolation
        self._bg_model       = None   # 3D background cube model
        self._bg_data        = None   # 3D background cube data proxy
        self._bg_base        = ''     # output name prefix
        self._bg_depth_scan  = None   # DepthScan on background cube model
        self._fit_region     = None   # RegionROI on the 1D spectrum plot
        self._bg_spec_viewer = None   # single plot: spectrum + RegionROI + fit overlay
        super().__init__(parent=parent)

    # ------------------------------------------------------------------
    @classmethod
    def display_name(cls):
        return "4D-STEM Azimuthal Integration"

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QtWidgets.QVBoxLayout()
        self.setLayout(root)
        self.setWindowTitle("4D-STEM Azimuthal Integration")
        self.setMinimumWidth(360)

        # ---- Section 1 — Dataset -------------------------------------
        grp1  = QtWidgets.QGroupBox("1 — Dataset")
        form1 = QtWidgets.QFormLayout()
        grp1.setLayout(form1)

        self._btn_load = QtWidgets.QPushButton("Get the active model")
        self._btn_load.clicked.connect(self._on_load)
        form1.addRow(self._btn_load)

        self._lbl_shape = QtWidgets.QLabel("—")
        form1.addRow("Data Shape:", self._lbl_shape)

        self._lbl_load_status = QtWidgets.QLabel("—")
        self._lbl_load_status.setWordWrap(True)
        form1.addRow("Status:", self._lbl_load_status)

        root.addWidget(grp1)

        # ---- Section 1 -----------------------------------------------
        grp2 = QtWidgets.QGroupBox("2 — Background quality test (single DP)")
        form2 = QtWidgets.QFormLayout()
        grp2.setLayout(form2)

        # self._spin_ix = QtWidgets.QSpinBox()
        # self._spin_ix.setRange(0, 9999)
        # self._spin_ix.setValue(0)
        # form2.addRow("Scan col  ix:", self._spin_ix)

        self._lbl_dp_pos = QtWidgets.QLabel("Navigate the DepthScan ROI to select a DP")
        self._lbl_dp_pos.setWordWrap(True)
        form2.addRow("DepthScan pos:", self._lbl_dp_pos)


        pct_row = QtWidgets.QHBoxLayout()
        self._slider_pct = QtWidgets.QSlider(1)          # Qt.Horizontal = 1
        self._slider_pct.setRange(0, 100)
        self._slider_pct.setValue(55)
        self._slider_pct.setTickInterval(10)
        self._slider_pct.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self._spin_pct = QtWidgets.QSpinBox()
        self._spin_pct.setRange(0, 100)
        self._spin_pct.setValue(55)
        self._spin_pct.setSuffix(" %")
        self._spin_pct.setFixedWidth(72)
        # keep slider and spinbox in sync
        self._slider_pct.valueChanged.connect(self._spin_pct.setValue)
        self._spin_pct.valueChanged.connect(self._slider_pct.setValue)
        pct_row.addWidget(self._slider_pct)
        pct_row.addWidget(self._spin_pct)
        form2.addRow("Percentile:", pct_row)

        self._btn_test = QtWidgets.QPushButton("Test on selected DP")
        self._btn_test.clicked.connect(self._on_test)
        form2.addRow(self._btn_test)

        self._lbl_test_status = QtWidgets.QLabel("—")
        self._lbl_test_status.setWordWrap(True)
        form2.addRow("Status:", self._lbl_test_status)

        root.addWidget(grp2)

        # ---- Section 3 -----------------------------------------------
        grp3 = QtWidgets.QGroupBox("3 — Full dataset run")
        form3 = QtWidgets.QFormLayout()
        grp3.setLayout(form3)

        self._lbl_pct_used = QtWidgets.QLabel("Percentile: 55 %")
        # update label whenever spinbox changes
        self._spin_pct.valueChanged.connect(
            lambda v: self._lbl_pct_used.setText(f"Percentile: {v} %"))
        form3.addRow("Will use:", self._lbl_pct_used)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        form3.addRow(self._progress)

        self._btn_run = QtWidgets.QPushButton("Run full dataset")
        self._btn_run.clicked.connect(self._on_run_full)
        form3.addRow(self._btn_run)

        self._lbl_run_status = QtWidgets.QLabel("—")
        self._lbl_run_status.setWordWrap(True)
        form3.addRow("Status:", self._lbl_run_status)

        root.addWidget(grp3)

        # ---- Section 4 — Background ring isolation ----------------
        grp4 = QtWidgets.QGroupBox("4 — Background ring isolation")
        form4 = QtWidgets.QFormLayout()
        grp4.setLayout(form4)

        self._btn_load_bg = QtWidgets.QPushButton("Load background dataset (active model)")
        self._btn_load_bg.clicked.connect(self._on_load_bg)
        form4.addRow(self._btn_load_bg)

        self._lbl_bg_shape = QtWidgets.QLabel("—")
        form4.addRow("Shape (Ny, Nx, r):", self._lbl_bg_shape)

        self._lbl_bg_dp_pos = QtWidgets.QLabel("—")
        form4.addRow("DepthScan pos:", self._lbl_bg_dp_pos)

        # Fit model selector
        self._combo_bg_model = QtWidgets.QComboBox()
        for m in _BG_MODELS:
            self._combo_bg_model.addItem(m)
        form4.addRow("Fit model:", self._combo_bg_model)

        # Show spectrum + insert RegionROI for interactive fit-range selection
        _lbl_roi_hint = QtWidgets.QLabel(
            "Drag the RegionROI on the 1D plot to set the fit window.")
        _lbl_roi_hint.setWordWrap(True)
        form4.addRow(_lbl_roi_hint)

        self._btn_test_fit = QtWidgets.QPushButton("Test fit on current DP position")
        self._btn_test_fit.setEnabled(False)
        self._btn_test_fit.clicked.connect(self._on_test_bg_fit)
        form4.addRow(self._btn_test_fit)

        self._lbl_bg_test_status = QtWidgets.QLabel("—")
        self._lbl_bg_test_status.setWordWrap(True)
        form4.addRow("Fit status:", self._lbl_bg_test_status)

        self._lbl_bg_fit_info = QtWidgets.QLabel("Fit region and model from above.")
        self._lbl_bg_fit_info.setWordWrap(True)
        form4.addRow(self._lbl_bg_fit_info)

        self._progress_bg = QtWidgets.QProgressBar()
        self._progress_bg.setRange(0, 100)
        self._progress_bg.setValue(0)
        self._progress_bg.setVisible(False)
        form4.addRow(self._progress_bg)

        self._btn_fit_full = QtWidgets.QPushButton("Fit full background dataset")
        self._btn_fit_full.setEnabled(False)
        self._btn_fit_full.clicked.connect(self._on_fit_bg_full)
        form4.addRow(self._btn_fit_full)

        self._lbl_bg_run_status = QtWidgets.QLabel("—")
        self._lbl_bg_run_status.setWordWrap(True)
        form4.addRow("Run status:", self._lbl_bg_run_status)

        root.addWidget(grp4)

        # ---- Close button --------------------------------------------
        self._btn_close = QtWidgets.QPushButton("Close")
        self._btn_close.clicked.connect(self.close)
        root.addWidget(self._btn_close)

    # ------------------------------------------------------------------
    # Workflow setup — called by BaseWorkflowDialog after _setup_ui
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Section 1 — load model
    # ------------------------------------------------------------------

    def _on_load(self):
        try:
            self._interface = PantaRheiInterface.instance(is_intern=True)
            self._model     = self._interface.get_active_model()
            api             = PRScriptingInterface()
            self._data      = api.get_active_data()
            ndim            = sliced_ndim(self._data.meta_data, self._data.ndim)

            if ndim != 4:
                self._lbl_load_status.setText("ERROR: active dataset is not 4D.")
                self._btn_test.setEnabled(False)
                self._btn_run.setEnabled(False)
                return
            self._lbl_test_status.setText(
                    "4D-STEM selected")
            self._btn_test.setEnabled(True)
            self._btn_run.setEnabled(True)
            Ny, Nx, Qy, Qx = self._data.shape
            self._base = self._model.get_output_name()
            self._lbl_shape.setText(f"({Ny}, {Nx}, {Qy}, {Qx})")

            # Remove previous DepthScan if any
            if self._depth_scan is not None:
                try:
                    self._interface.remove(self._model, self._depth_scan)
                except Exception:
                    pass

            # Insert ImageCubeDepthScan on the 4D model — user navigates to pick DP
            self._depth_scan = self._interface.insert(self._model, PRScriptingTypes.ImageCubeDepthScan,
                parameters={"name": "AzimuthalDepthScan"})
            api.display_image("AzimuthalDepthScan", auto_size=True)

        except Exception as exc:
            self._lbl_load_status.setText(f"Load error: {exc}")

    def _setup_workflow(self):
        try:
            self._interface = PantaRheiInterface.instance(is_intern=True)
            self._model     = self._interface.get_active_model()
            api             = PRScriptingInterface()
            self._data      = api.get_active_data()
            ndim            = sliced_ndim(self._data.meta_data, self._data.ndim)
            if ndim != 4:
                self._lbl_test_status.setText(
                    "ERROR: active dataset is not 4D. Open a 4D-STEM cube first.")
                self._btn_test.setEnabled(False)
                self._btn_run.setEnabled(False)
                return

            Ny, Nx, Qy, Qx = self._data.shape
            self._base = self._model.get_output_name()
            self._lbl_shape.setText(f"({Ny}, {Nx}, {Qy}, {Qx})")

            # Insert ImageCubeDepthScan on the 4D model — user navigates to pick DP
            self._depth_scan = self._interface.insert(self._model, PRScriptingTypes.ImageCubeDepthScan,
                parameters={"name": "AzimuthalDepthScan"})
            api.display_image("AzimuthalDepthScan", auto_size=True)

            # Live connection: rerun test whenever depth scan position changes.
            # parameters_changed is the standard DataObject Qt signal in PR.
            # Ambiguity: signal name may vary — spinboxes + Test button are the fallback.
            try:
                self._depth_scan.parameters_changed.connect(self._on_depth_scan_moved)
                self._lbl_test_status.setText(
                    f"{Ny}\u00d7{Nx}  |  live update active — navigate the DepthScan")
            except AttributeError:
                self._lbl_test_status.setText(
                    f"{Ny}\u00d7{Nx}  |  live update unavailable — use Test button")

        except Exception as exc:
            self._lbl_test_status.setText(f"Setup error: {exc}")
            self._btn_test.setEnabled(False)
            self._btn_run.setEnabled(False)

    # ------------------------------------------------------------------
    # Live depth scan callback
    # ------------------------------------------------------------------

    def _read_dp_position(self):
        """
        Read DepthScan ROI centre in pixel coordinates.
        pos = top-left corner, size = (width_x, height_y).
        DP is taken at the centre of the ROI.
        Returns (iy, ix).
        """
        Ny, Nx = self._data.shape[:2]
        p    = self._interface.get_parameters(
                   self._depth_scan, scale_mode="pixel")
        pos  = p["pos"]
        size = p["size"]
        ix   = int(round(pos[0] + size[0] / 2.0))
        iy   = int(round(pos[1] + size[1] / 2.0))
        ix   = max(0, min(Nx - 1, ix))
        iy   = max(0, min(Ny - 1, iy))
        return iy, ix

    def _on_depth_scan_moved(self):
        """Called when the user moves the ImageCubeDepthScan cursor."""
        if self._depth_scan is None or self._interface is None:
            return
        try:
            iy, ix = self._read_dp_position()
            self._lbl_dp_pos.setText(f"iy={iy}  ix={ix}")
        except Exception:
            pass   # silent — live update is best-effort
        self._on_test()

    def _on_test(self):
        if self._data is None:
            self._lbl_test_status.setText("No data loaded.")
            return
        try:
            if self._depth_scan is not None and self._interface is not None:
                iy, ix = self._read_dp_position()
            else:
                iy, ix = 0, 0
            pct = self._spin_pct.value()

            dp    = np.array(self._data[iy, ix]).astype(float)
            polar = polar_transform(dp, NUM_RADIAL, NUM_ANGULAR)

            profile    = polar.mean(axis=1)
            background = np.percentile(polar, pct, axis=1)
            subtracted = profile - background

            # (3, NUM_RADIAL): row 0=profile, 1=background, 2=BG-subtracted
            overlay = np.stack([profile, background, subtracted], axis=0)

            api = PRScriptingInterface()
            api.data_to_repo(_TEST_OVERLAY_KEY, overlay, meta_data={"type": "1D"})
            # display_plot on an existing key updates the plot in-place if already open
            if self._test_viewer is None:
                self._test_viewer = api.display_plot(_TEST_OVERLAY_KEY)
            self._lbl_test_status.setText(
                f"DP [{ix},{iy}]  pct={pct}%  "
                f"profile_max={profile.max():.0f}  bg_max={background.max():.0f}"
            )
        except Exception as exc:
            self._lbl_test_status.setText(f"Test error: {exc}")

    def _on_run_full(self):
        if self._data is None or self._model is None:
            self._lbl_run_status.setText("No data loaded.")
            return

        pct = self._spin_pct.value()
        self._btn_run.setEnabled(False)
        self._btn_test.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        QtWidgets.QApplication.processEvents()

        try:
            data  = self._data
            model = self._model
            base  = self._base
            Ny, Nx, Qy, Qx = data.shape

            step = 1 # temporary hard coded 1,3,5,9,15
            maxpeak_radius_map    = np.zeros((Ny, Nx))
            maxpeak_intensity_map = np.zeros((Ny, Nx))
            maxpeak_FWHM_map      = np.zeros((Ny, Nx))
            first_peak_radius_map = np.zeros((Ny, Nx))
            orientation_map       = np.full((Ny, Nx), np.nan)
            #ang_fingerprint_map  = np.zeros((Ny, Nx, NUM_ANGULAR // 2 + 1), dtype=complex)
            radial_profiles       = np.zeros((Ny, Nx, NUM_RADIAL))
            radial_backgrounds    = np.zeros((Ny, Nx, NUM_RADIAL))
            #ang_profiles         = np.zeros((Ny, Nx, NUM_ANGULAR // 2 + 1))
            ang_power_spectra     = np.zeros((Ny, Nx, NUM_ANG_FREQS))
            radial_profiles_bgsubtracted = np.zeros((Ny, Nx, NUM_RADIAL))
            n_sector_maps         = (NUM_ANGULAR // 2) // step
            sector_maps           = np.zeros((Ny, Nx, n_sector_maps))
            name_list_2D_sector   = np.zeros((n_sector_maps))
            total   = Ny * Nx
            counter = 0

            for iy in range(Ny):
                for ix in range(Nx):
                    dp    = np.array(data[iy, ix]).astype(float)
                    polar = polar_transform(dp, NUM_RADIAL, NUM_ANGULAR)

                    profile    = polar.mean(axis=1)
                    background = np.percentile(polar, pct, axis=1)
                    profile_bgsubtracted = profile - background
                    #ang_profile    = np.mean(polar[MIN_RADIUS:,:],axis=0) # to exclude the central beam from the mean calculation
                    ang_fingerprint = compute_angular_fingerprint(
                        polar, profile_bgsubtracted)
                    orientation_map[iy, ix] = compute_orientation_angle(
                        polar, profile_bgsubtracted)
                    sector_maps[iy, ix, :] = compute_angular_sector_maps(polar, step=step)
                    radial_profiles[iy, ix, :]    = profile
                    radial_backgrounds[iy, ix, :] = background
                    radial_profiles_bgsubtracted [iy, ix, :] = profile_bgsubtracted
                    ang_power_spectra[iy, ix, :]      = np.abs(ang_fingerprint) ** 2
                    #ang_fingerprint_map[iy, ix, :]    = ang_fingerprint
                    #ang_profiles[iy, ix, :] = ang_profile

                    prof = profile_bgsubtracted.copy()
                    prof[:MIN_RADIUS] = 0 #-np.inf
                    maxpeak_index = np.argmax(prof)
                    maxpeak_value = profile_bgsubtracted[maxpeak_index]
                    maxfwhm_val   = compute_fwhm(profile_bgsubtracted, maxpeak_index)

                    peaks, _ = find_peaks(profile_bgsubtracted, prominence=1) #2
                    valid    = peaks[peaks >= MIN_RADIUS]
                    first_peak_index = valid[0] if len(valid) > 0 else 0

                    maxpeak_radius_map[iy, ix]    = maxpeak_index
                    maxpeak_intensity_map[iy, ix] = maxpeak_value
                    maxpeak_FWHM_map[iy, ix]      = maxfwhm_val
                    first_peak_radius_map[iy, ix] = first_peak_index

                    counter += 1
                    if counter % max(1, total // 100) == 0:
                        pct_done = int(100 * counter / total)
                        self._progress.setValue(pct_done)
                        QtWidgets.QApplication.processEvents()
            #Color coded maps for the sectors: first attempt
            threshold = 0.2 # threshold for contrast to determine if a sector is dominant
            max_vals  = sector_maps.max(axis=2)
            min_vals  = sector_maps.min(axis=2)
            contrast  = (max_vals - min_vals) / (max_vals + 1e-10)
            colour_map = np.where(contrast > threshold, np.argmax(sector_maps, axis=2), np.nan)
            # colour_map = np.where(contrast > threshold, np.argmax(sector_maps, axis=2)+1, 0)
            # ---- Push to repo ----------------------------------------
            api = PRScriptingInterface()
            
            #transfering the metadata:
            
            
            try:
                out_meta_data_2D = copy.deepcopy(self._data.meta_data)
                out_meta_data_3D = copy.deepcopy(self._data.meta_data)
                ref_size_list = list(self._data.meta_data['ref_size'])
                
                cut_factors_list = [data.shape[2] / NUM_RADIAL, 1.0, 1.0]
                pixels_factor_list = [1.0, 1.0, 1.0]
                try:
                    if self._data.meta_data['transform.cut_factors'] is not None:
                        cut_factors_list[1] = self._data.meta_data['transform.cut_factors'][2]
                        cut_factors_list[2] = self._data.meta_data['transform.cut_factors'][3]
                except Exception as metadata_exc:
                    self._lbl_run_status.setText(
                        f"Done (cut_factors failed: {metadata_exc})")
                try:
                    if self._data.meta_data['transform.pixel_factors'] is not None:
                        pixels_factor_list[:] = self._data.meta_data['transform.pixel_factors'][:]
                except Exception as metadata_exc:
                    self._lbl_run_status.setText(
                        f"Done (pixel_factors failed: {metadata_exc})")
                
                out_meta_data_2D["type"] = "2D"
                out_meta_data_2D['content.types'] = ['ScanX', 'ScanY']
                out_meta_data_3D["type"] = "2D"
                out_meta_data_3D['content.types'] = ['CameraX','ScanX', 'ScanY']
                out_meta_data_2D['ref_size'] = tuple(ref_size_list[2:])
                out_meta_data_3D['ref_size'] = tuple(ref_size_list[1:])
                out_meta_data_3D['transform.pixel_factors'] = tuple(pixels_factor_list)
                out_meta_data_3D['transform.cut_factors'] = tuple(cut_factors_list)
                
            except Exception as metadata_exc:
                self._lbl_run_status.setText(
                    f"Done (metadata failed: {metadata_exc})")
            #         # One thing that is intentionally:
            #         # The intensity must be set separately with `set_user_intensity()`.

            name_list_2D = [
                f"{base}_maxpeak_radius_px_map",
                f"{base}_maxpeak_intensity_px_map",
                f"{base}_maxpeak_FWHM_px_map",
                f"{base}_first_peak_radius_px_map",
                f"{base}_orientation_angle_map",
                f"{base}_colour_map",
            ]
            name_list_3D = [
                f"{base}_radial_profiles",
                f"{base}_ang_power_spectra",
                f"{base}_radial_backgrounds",
                f"{base}_radial_profiles_bgsubtracted",
                #f"{base}_ang_profiles",
            ]
            api.data_to_repo(f"{base}_maxpeak_radius_px_map",    maxpeak_radius_map,    meta_data=out_meta_data_2D)
            api.data_to_repo(f"{base}_maxpeak_intensity_px_map", maxpeak_intensity_map, meta_data=out_meta_data_2D)
            api.data_to_repo(f"{base}_maxpeak_FWHM_px_map",      maxpeak_FWHM_map,      meta_data=out_meta_data_2D)
            api.data_to_repo(f"{base}_first_peak_radius_px_map", first_peak_radius_map, meta_data=out_meta_data_2D)
            api.data_to_repo(f"{base}_orientation_angle_map",    orientation_map,       meta_data=out_meta_data_2D)
            #api.data_to_repo(f"{base}_ang_fingerprint",          ang_fingerprint_map,       meta_data=out_meta_data_3D)
            api.data_to_repo(f"{base}_radial_profiles",          radial_profiles,       meta_data=out_meta_data_3D)
            api.data_to_repo(f"{base}_radial_profiles_bgsubtracted", radial_profiles_bgsubtracted, meta_data=out_meta_data_3D)
            api.data_to_repo(f"{base}_ang_power_spectra",        ang_power_spectra,     meta_data=out_meta_data_3D)
            api.data_to_repo(f"{base}_radial_backgrounds",       radial_backgrounds,    meta_data=out_meta_data_3D)
            #api.data_to_repo(f"{base}_ang_profiles",             ang_profiles,          meta_data=out_meta_data_3D)
            api.data_to_repo(f"{base}_sector_maps",              sector_maps,           meta_data=out_meta_data_3D)
            api.data_to_repo(f"{base}_colour_map",               colour_map,            meta_data=out_meta_data_2D)
            
            #make individual maps for the sector map stack:
            # name_list_2D_sector = []
            # for name, map in unpack_sector_maps(sector_maps, base):
            #     name_list_2D_sector.append(name)
            #     api.data_to_repo(name, map, meta_data=out_meta_data_2D)
            # api.message(
            #     f"Dataset shape : {name_list_2D_sector}\n"
            # )
            # ---- Display ---------------------------------------------
            config_maps_2D = []
            config_maps_3D = []
            # config_sector_maps = []
            config_depth_scans = []
            
            for k, lbl in enumerate(name_list_2D):
                config_maps_2D.append((lbl,
                    False, False,
                    None, None))
            for k, lbl in enumerate(name_list_3D):
                config_maps_3D.append((lbl,
                    False, False,
                    None, None))
            # for k, lbl in enumerate(name_list_2D_sector):
            #     config_sector_maps.append((lbl,
            #         False, False,
            #         None, None))

            #api.display_image(f"{base}_ang_fingerprint_map", auto_size = True)
            #api.display_image(f"{base}_sector_maps", auto_size = True)
            # api.display_image(f"{base}_colour_map", auto_size = True)
            
            data_models_2D, _ = api.open_multi_view(
                config_maps_2D, title="Azimutal integration results:")
                
            data_models_3D, _ = api.open_multi_view(
                config_maps_3D, title="Azimutal integration datasets:")

            # data_models_sector, _ = api.open_multi_view(
            #     config_sector_maps, title="Orientation integration results:")

            for data_model in data_models_2D:
                dc = data_model.get_display_control()
                dc.set_parameters(color_map="flame")
            
            # ---- Calibration -----------------------------------------
            scaling          = self._interface.get_scaling(self._model)
            main_calibration = scaling.get_parameters()
            
            for data_model in data_models_3D:
                # data_model.set_user_calibrations(axes=0, values=main_calibration['calib'][0] * data.shape[2] / NUM_RADIAL, units=main_calibration['unit'][0], origins=0.0, use_prefixes=main_calibration['use_prefix'][0], fixed_prefixes=main_calibration['fixed_prefix'][0], block=False)
                data_model.set_user_calibrations(axes=0, values=main_calibration['calib'][0], units=main_calibration['unit'][0], origins=0.0, use_prefixes=main_calibration['use_prefix'][0], fixed_prefixes=main_calibration['fixed_prefix'][0], block=False)
                # data_model.set_user_calibrations(axes=1, values=main_calibration['calib'][2] * main_calibration['pixel_factor'][2],  units=main_calibration['unit'][2],  origins=0.0, use_prefixes=main_calibration['use_prefix'][2],  fixed_prefixes=main_calibration['fixed_prefix'][2],  block=False)
                # data_model.set_user_calibrations(axes=2, values=main_calibration['calib'][3] * main_calibration['pixel_factor'][3],  units=main_calibration['unit'][3],  origins=0.0, use_prefixes=main_calibration['use_prefix'][3],  fixed_prefixes=main_calibration['fixed_prefix'][3],  block=False)
            ang_power_spectra = api.get_data_models_by_name(f"{base}_ang_power_spectra")
            ang_power_spectra[0].set_user_calibrations(axes=0, values=1.0,  units='Symetry',  origins=0.0, use_prefixes=main_calibration['use_prefix'][0],  fixed_prefixes=main_calibration['fixed_prefix'][0],  block=False)


            # ---- Data tool links -------------------------------------
            # depthA = self._interface.insert(self._model, PRScriptingTypes.ImageCubeDepthScan, parameters={"name": "ImageCubeDepthScan_A"})
            name_list_3d_depth_scans = []
            for data_model in data_models_3D:
                name = data_model.get_output_name()
                depth = data_model.insert(PRScriptingTypes.DepthScan, parameters={"name": f"Depthscan_{name}"})
                api.create_data_tool_link([self._depth_scan, depth])
                #api.data_to_repo(f"Depthscan_{name}", depth, meta_data={"type": "1D"})
                name_list_3d_depth_scans.append(f"Depthscan_{name}")
                #api.display_plot(f"Depthscan_{name}")
            
            #name_list_3d_depth_scans = [f"Depthscan_{lbl}" for lbl in name_list_3D]
            for k, lbl in enumerate(name_list_3d_depth_scans):
                config_depth_scans.append((lbl,
                    True, False,
                    None, None))
            api.open_multi_view(
                config_depth_scans, title="3D maps Depth scans:")
            
            self._progress.setValue(100)
            self._lbl_run_status.setText(
                f"Done — {Ny}x{Nx} DPs  |  percentile={pct}%")

        except Exception as exc:
            self._lbl_run_status.setText(f"Run error: {exc}")

        finally:
            self._progress.setVisible(False)
            self._btn_run.setEnabled(True)
            self._btn_test.setEnabled(True)

    # ------------------------------------------------------------------
    # Section 4 — Background ring isolation
    # ------------------------------------------------------------------

    def _on_load_bg(self):
            """Load the active model as the 3D background dataset."""
            try:
                #reset the data
                self._bg_model       = None
                self._bg_data        = None
                self._bg_depth_scan  = None
                self._fit_region     = None
                self._bg_spec_viewer = None
                
                interface = PantaRheiInterface.instance(is_intern=True)
                api       = PRScriptingInterface()
                model     = interface.get_active_model()
                data      = api.get_active_data()
    
                if data.ndim != 3:
                    self._lbl_bg_shape.setText("ERROR: expected 3D dataset (Ny, Nx, r).")
                    return
    
                ny, nx, nr = data.shape
                self._bg_model = model
                self._bg_data  = data
                self._bg_base  = model.get_output_name()
    
                self._lbl_bg_shape.setText(f"({ny}, {nx}, {nr})")
    
                # Remove previous BG DepthScan if any
                if self._bg_depth_scan is not None:
                    try:
                        interface.remove(self._bg_model, self._bg_depth_scan)
                    except Exception:
                        pass
    
                self._bg_depth_scan = interface.insert(
                    self._bg_model,
                    PRScriptingTypes.DepthScan,
                    parameters={"name": _BGFIT_SPEC_KEY})
                # api.display_plot(_BGFIT_SPEC_KEY)
    
                # Live update of position label when DepthScan moves
                try:
                    self._bg_depth_scan.parameters_changed.connect(
                        self._on_bg_depth_scan_moved)
                except AttributeError:
                    pass
    
                # Open spectrum plot + RegionROI immediately at load time
                self._on_show_bg_spectrum()
                self._lbl_bg_test_status.setText(
                    f"Loaded  |  {self._bg_base}  |  drag RegionROI, then Test fit")
    
            except Exception as exc:
                self._lbl_bg_test_status.setText(f"Load error: {exc}")

    def _read_bg_pixel_pos(self):
        """
        Read spatial pixel (iy, ix) from the BG DepthScan ROI centre.
        pos = top-left corner, size = (width_x, height_y).
        """
        ny, nx, _ = self._bg_data.shape
        interface = PantaRheiInterface.instance(is_intern=True)
        p    = interface.get_parameters(self._bg_depth_scan, scale_mode="pixel")
        pos  = p["pos"]
        size = p["size"]
        ix   = max(0, min(nx - 1, int(round(pos[0] + size[0] / 2.0))))
        iy   = max(0, min(ny - 1, int(round(pos[1] + size[1] / 2.0))))
        return iy, ix

    def _on_bg_depth_scan_moved(self):
        """Update position label when the BG DepthScan cursor moves."""
        if self._bg_depth_scan is None:
            return
        try:
            iy, ix = self._read_bg_pixel_pos()
            self._lbl_bg_dp_pos.setText(f"iy={iy}  ix={ix}")
        except Exception:
            pass

    def _on_show_bg_spectrum(self):
        """
        Display the radial profile at the current BG DepthScan position and
        insert a RegionROI on the 1D plot for interactive fit-range selection.

        RegionROI is the confirmed PR pattern for plot models.
        scale_mode='calib' → region values are in channel indices (no energy calib here).
        """
        if self._bg_data is None:
            return
        try:
            iy, ix  = self._read_bg_pixel_pos()
            nr      = self._bg_data.shape[2]
            profile = np.array(self._bg_data[iy, ix, :]).astype(float)

            api = PRScriptingInterface()
            api.data_to_repo(_BGFIT_SPEC_KEY, profile, meta_data={"type": "1D"})

            if self._bg_spec_viewer is None:
                self._bg_spec_viewer = api.display_plot(_BGFIT_SPEC_KEY)

            # Recreate RegionROI so the range resets cleanly on each call
            if self._fit_region is not None:
                try:
                    self._bg_spec_viewer.remove(self._fit_region)
                except Exception:
                    pass

            self._fit_region = self._bg_spec_viewer.insert(
                PRScriptingTypes.RegionROI,
                parameters={"region": (float(MIN_RADIUS), float(nr - 1))},
                scale_mode="calib")

            self._lbl_bg_dp_pos.setText(f"iy={iy}  ix={ix}")
            self._btn_test_fit.setEnabled(True)
            self._btn_fit_full.setEnabled(True)
            self._lbl_bg_test_status.setText(
                f"Spectrum shown for [{iy},{ix}]  |  drag RegionROI, then Test fit")

        except Exception as exc:
            self._lbl_bg_test_status.setText(f"Show spectrum error: {exc}")

    def _read_fit_range(self, nr):
        """
        Read fit range channel indices from the RegionROI.
        region values come back in calibrated units (channel indices here).
        Returns (fit_start, fit_end) clamped to [0, nr-1].
        """
        params    = self._fit_region.get_parameters(scale_mode="pixel")
        lo, hi    = params["region"][0], params["region"][1]
        fit_start = max(0, min(nr - 1, int(round(lo))))
        fit_end   = max(0, min(nr - 1, int(round(hi))))
        if fit_start >= fit_end:
            fit_end = min(nr - 1, fit_start + 1)
        return fit_start, fit_end

    def _on_test_bg_fit(self):
        """Test fit on the pixel selected by the BG DepthScan.
        Updates _BGFIT_SPEC_KEY in place so the existing plot shows
        profile / fitted background / rings alongside the RegionROI.
        """
        if self._bg_data is None:
            self._lbl_bg_test_status.setText("No background dataset loaded.")
            return
        if self._fit_region is None:
            self._lbl_bg_test_status.setText(
                "Show spectrum first to create the fit region.")
            return
        try:
            iy, ix     = self._read_bg_pixel_pos()
            nr         = self._bg_data.shape[2]
            model_name = self._combo_bg_model.currentText()
            fit_start, fit_end = self._read_fit_range(nr)

            profile = np.array(self._bg_data[iy, ix, :]).astype(float)
            fitted, info = fit_background_radial(profile, fit_start, fit_end, model_name)
            rings   = np.maximum(profile - fitted, 0.0)

            # 3 rows pushed to the same key: the existing plot updates in place,
            # showing original profile / smooth fit / isolated rings + RegionROI.
            overlay = np.stack([profile, fitted, rings], axis=0)
            api = PRScriptingInterface()
            api.data_to_repo(_BGFIT_SPEC_KEY, overlay, meta_data={"type": "1D"})

            self._lbl_bg_test_status.setText(
                f"DP [{iy},{ix}]  model={model_name}  ch[{fit_start}:{fit_end}]  {info}  "
                f"rings_max={rings.max():.2g}")
            self._lbl_bg_fit_info.setText(
                f"Model: {model_name}  |  fit ch[{fit_start}:{fit_end}]")

        except Exception as exc:
            self._lbl_bg_test_status.setText(f"Test fit error: {exc}")

    def _on_fit_bg_full(self):
        """Fit all pixels in the background cube and push rings + fitted cubes."""
        if self._bg_data is None or self._bg_model is None:
            self._lbl_bg_run_status.setText("No background dataset loaded.")
            return
        if self._fit_region is None:
            self._lbl_bg_run_status.setText(
                "Use the Test fit step to define the fit region first.")
            return

        nr         = self._bg_data.shape[2]
        model_name = self._combo_bg_model.currentText()

        try:
            fit_start, fit_end = self._read_fit_range(nr)
        except Exception as exc:
            self._lbl_bg_run_status.setText(f"Cannot read fit region: {exc}")
            return

        self._btn_fit_full.setEnabled(False)
        self._btn_test_fit.setEnabled(False)
        self._progress_bg.setVisible(True)
        self._progress_bg.setValue(0)
        QtWidgets.QApplication.processEvents()

        try:
            # bg   = self._bg_data
            ny, nx, nr = self._bg_data.shape
            base = self._base

            fitted_cube = np.zeros((ny, nx, nr), dtype=float)
            amorphous_rings_cube  = np.zeros((ny, nx, nr), dtype=float)
            mean_amorphous_rings_cube = np.zeros((ny, nx), dtype=float)
            mean_crystalline_peaks = np.zeros((ny, nx), dtype=float)
            ratio_cube  = np.zeros((ny, nx), dtype=float)

            total   = ny * nx
            counter = 0
            for iy in range(ny):
                for ix in range(nx):
                    profile = np.array(self._bg_data[iy, ix, :]).astype(float)
                    fitted, _ = fit_background_radial(
                        profile, fit_start, fit_end, model_name)
                    fitted_cube[iy, ix, :] = fitted
                    amorphous_rings_cube[iy, ix, :]  = np.maximum(profile - fitted, 0.0)

                    counter += 1
                    if counter % max(1, total // 100) == 0:
                        self._progress_bg.setValue(int(100 * counter / total))
                        QtWidgets.QApplication.processEvents()
            api = PRScriptingInterface()
            radial_profiles_bgsubtracted = api.get_data_models_by_name(f"{base}_radial_profiles_bgsubtracted")[0]
            # # Temporary addition to catch the error
            # matches = api.get_data_models_by_name(f"{base}_radial_profiles_bgsubtracted")
            # if not matches:
            #     self._lbl_bg_run_status.setText(
            #         f"Cannot find dataset '{base}_radial_profiles_bgsubtracted'. ")
            #     return
            # radial_profiles_bgsubtracted = matches[0]
            
            crystalline_peaks = radial_profiles_bgsubtracted.get_topmost_data()

            mean_amorphous_rings_cube = amorphous_rings_cube[:, :, MIN_RADIUS:].mean(axis=2)
            mean_crystalline_peaks = crystalline_peaks[:, :, MIN_RADIUS:].mean(axis=2)
            denom = mean_amorphous_rings_cube+mean_crystalline_peaks
            ratio_cube  = np.where(denom > 0, mean_amorphous_rings_cube / denom, np.nan)
            
            
            #transfering the metadata:
            try:
                out_meta_data_3D = copy.deepcopy(self._bg_data.meta_data)
                out_meta_data_2D = copy.deepcopy(self._bg_data.meta_data)
                ref_size_list = list(self._bg_data.meta_data['ref_size'])

                out_meta_data_2D["type"] = "2D"
                out_meta_data_2D['content.types'] = ['ScanX', 'ScanY']
                out_meta_data_2D['ref_size'] = tuple(ref_size_list[1:])
                
            except Exception as exc:
                        self._lbl_bg_run_status.setText(f"Metadata transfer error: {exc}")

            key_fitted = f"{base}_fitted"
            key_rings  = f"{base}_rings"
            key_ratio = f"{base}_Amorphous/Crystalline_ratio"

            api.data_to_repo(key_fitted, fitted_cube, meta_data=out_meta_data_3D)
            api.data_to_repo(key_rings,  amorphous_rings_cube,  meta_data=out_meta_data_3D)
            api.data_to_repo(key_ratio,  ratio_cube,  meta_data=out_meta_data_2D)
            # Display both in a multi-view
            config = [
                (key_fitted, False, False, None, None),
                (key_rings,  False, False, None, None),
                (key_ratio,  False, False, None, None),
            ]
            data_models, _ = api.open_multi_view(config, title="Background fit results")

            self._progress_bg.setValue(100)
            self._lbl_bg_run_status.setText(
                f"Done  |  model={model_name}  |  {key_rings}  ({nr},{ny},{nx})")

        except Exception as exc:
            self._lbl_bg_run_status.setText(f"Fit error: {exc}")

        finally:
            self._progress_bg.setVisible(False)
            self._btn_fit_full.setEnabled(True)
            self._btn_test_fit.setEnabled(True)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _on_clear(self):
        try:
            if self._depth_scan is not None and self._model is not None and self._interface is not None:
                self._interface.remove(self._model, self._depth_scan)
        except Exception:
            pass
        self._depth_scan    = None
        self._test_viewer   = None
        self._model         = None
        self._data          = None
        self._interface     = None
        self._full_viewers.clear()
        self._bg_base       = None
        self._bg_model       = None
        self._bg_data        = None
        self._bg_depth_scan  = None
        self._fit_region     = None
        self._bg_spec_viewer = None

    def closeEvent(self, ev):
        self._on_clear()
        super().closeEvent(ev)