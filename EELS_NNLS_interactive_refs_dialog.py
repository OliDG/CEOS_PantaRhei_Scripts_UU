"""
for Panta Rhei 0.25

Interactive NNLS fitting for 3-D EELS spectral cubes, as a workflow dialog.

Workflow
--------
1. Load 3-D dataset. A DepthScan is inserted on the active model
2. Reference regions (default 6, adjustable 1-10). For each one, the user:
     - moves the DepthScan's ROI over the spatial area of interest,
     - positions the shared pre-edge and edge LinearRegions on the
       DepthScan's spectrum panel (created once, on first use),
     - clicks "Set ref_i" to store that component (label, ROI bounds,
       windows, background-subtracted & normalised reference vector).
   The same two LinearRegions are reused/moved for every reference
3. Fit window. Auto-proposed as the envelope of all stored reference
   windows
4. Run. NNLS runs pixel-by-pixel with a mixed reference matrix (clean
   references + E^-2..E^-5 background basis). Every output (reference
   spectra used, component maps, background maps, residual map) is pushed
   to the repo with data_to_repo() and shown afterwards, grouped into three multi-views.

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
from scipy.optimize import nnls

from PyQt5 import QtWidgets, QtCore

from panta_rhei.main.gui.base_workflow_dialog import BaseWorkflowDialog
from panta_rhei.scripting import PRScriptingInterface, PRScriptingTypes
from panta_rhei.main.gui.panta_rhei_interface import PantaRheiInterface
from panta_rhei.main.gui.utils import sliced_ndim

# ---------------------------------------------------------------------------
# Processing constants
# ---------------------------------------------------------------------------
MIN_REFS         = 1
MAX_REFS         = 10
DEFAULT_N_REFS   = 3
DEFAULT_PRE_POS  = [0.05, 0.20]   # normalised LinearRegion positions, first use only
DEFAULT_EDGE_POS = [0.30, 0.60]
BG_POWERS        = [2.0, 3.0, 4.0, 5.0]
BG_LABELS        = ["bg_E-2", "bg_E-3", "bg_E-4", "bg_E-5"]

# ---------------------------------------------------------------------------
# Core processing functions (same numerics as NNLS_interactive_refs.py)
# ---------------------------------------------------------------------------

def effective_ndim(data):
    return sum(1 for s in data.shape if s > 1)

def select_energy_calib(meta_data):
    """
    Pick the eV calibration dict to use, preferring a corrected calibration
    in meta_data['inherited.calib'] over the raw meta_data['device.calib'],
    which is used only as a fallback when no inherited calibration exists.
    Returns (calib_dict_or_None, source_key_or_None).
    """
    for key in ("inherited.calib", "device.calib"):
        calib_list = meta_data.get(key)
        if not calib_list:
            continue
        energy_calib = next(
            (e for e in calib_list if e is not None and e.get("unit") == "eV"),
            None,
        )
        if energy_calib is not None:
            return energy_calib, key
    return None, None


def energy_axis_from_meta(meta_data, n_channels=None):
    """
    Build energy axis: (np.arange(n_channels) + offset) * value * pixel_factor.
    Uses meta_data['inherited.calib'] in priority over meta_data['device.calib'].
    """
    energy_calib, _source = select_energy_calib(meta_data)
    if energy_calib is None or n_channels is None:
        return None
    offset       = float(energy_calib["offset"])
    value        = float(energy_calib["value"])
    pixel_factor = float(energy_calib.get("pixel_factor") or 1.0)
    return (np.arange(int(n_channels)) + offset) * value * pixel_factor
    

def norm_region_to_ev(lo_n, hi_n, energy_axis):
    """Convert normalised LinearRegion [0,1] positions to eV via channel lookup."""
    nE   = len(energy_axis)
    i_lo = max(0, min(nE - 1, int(round(lo_n * (nE - 1)))))
    i_hi = max(0, min(nE - 1, int(round(hi_n * (nE - 1)))))
    return float(energy_axis[i_lo]), float(energy_axis[i_hi])


def window_to_mask(energy_axis, window):
    """Boolean mask for channels within (e_lo, e_hi)."""
    e_lo, e_hi = min(window), max(window)
    mask = (energy_axis >= e_lo) & (energy_axis <= e_hi)
    if not mask.any():
        raise ValueError(
            f"Window [{e_lo:.1f}, {e_hi:.1f}] eV is outside "
            f"[{energy_axis[0]:.1f}, {energy_axis[-1]:.1f}] eV."
        )
    return mask


def fit_powerlaw_background(spectrum, energy_axis, pre_edge_window):
    """Fit A*E^-r to spectrum in pre_edge_window via log-log linear regression."""
    mask = window_to_mask(energy_axis, pre_edge_window)
    E    = energy_axis[mask]
    S    = np.maximum(spectrum[mask], 1e-10)
    c    = np.polyfit(np.log(E), np.log(S), 1)
    r    = -c[0]
    A    = np.exp(c[1])
    return A * np.power(np.maximum(energy_axis, 1e-10), -r)


def build_clean_reference(local_spectrum, energy_axis, pre_edge_window, edge_window):
    """Background-subtract local_spectrum in edge_window, clip negatives to 0."""
    bg    = fit_powerlaw_background(local_spectrum, energy_axis, pre_edge_window)
    nE    = len(energy_axis)
    ref   = np.zeros(nE, dtype=float)
    emask = window_to_mask(energy_axis, edge_window)
    ref[emask] = np.maximum(local_spectrum[emask] - bg[emask], 0.0)
    return ref, bg


def build_powerlaw_background_basis(energy_axis, fit_window):
    """4 power-law background basis columns (E^-2..E^-5), zero outside fit_window."""
    nE       = len(energy_axis)
    fit_mask = window_to_mask(energy_axis, fit_window)
    E_fit    = np.maximum(energy_axis[fit_mask], 1e-10)
    cols = []
    for r in BG_POWERS:
        col           = np.zeros(nE, dtype=float)
        col[fit_mask] = E_fit ** (-r)
        cols.append(col)
    return cols


def build_reference_matrix(clean_refs, energy_axis, fit_window):
    """Assemble (nE, N_refs+4): clean chemical refs + power-law background basis."""
    bg_cols  = build_powerlaw_background_basis(energy_axis, fit_window)
    all_cols = clean_refs + bg_cols
    return np.column_stack(all_cols)


def fit_cube(data, R, energy_axis, fit_window, progress_callback=None):
    """NNLS pixel-by-pixel fit. Returns dict with coefficients/residuals/reconstructed."""
    ny, nx, nE = data.shape
    N_total    = R.shape[1]

    fit_mask = window_to_mask(energy_axis, fit_window)
    n_fit    = fit_mask.sum()
    if n_fit <= N_total:
        raise ValueError(
            f"Fit window has {n_fit} channels for {N_total} reference columns "
            "- system is underdetermined. Widen the fit window."
        )

    R_fit    = R[fit_mask, :].copy()
    data_fit = data[:, :, fit_mask]

    col_norms = np.linalg.norm(R_fit, axis=0)
    zero_cols = np.where(col_norms == 0)[0]
    if zero_cols.size > 0:
        raise ValueError(
            f"R columns {zero_cols.tolist()} have zero norm inside the fit window. "
            "Check that each reference window overlaps the fit window."
        )
    R_fit_norm = R_fit / col_norms[np.newaxis, :]

    coeff_maps    = np.zeros((ny, nx, N_total), dtype=float)
    residual_map  = np.zeros((ny, nx),          dtype=float)
    reconstructed = np.zeros_like(data,         dtype=float)

    n_pixels = ny * nx
    for idx, (iy, ix) in enumerate(np.ndindex(ny, nx)):
        spectrum          = data_fit[iy, ix, :].astype(float)
        coeffs_norm, res  = nnls(R_fit_norm, spectrum)
        coeffs            = coeffs_norm / col_norms
        coeff_maps[iy, ix, :]           = coeffs
        residual_map[iy, ix]            = res
        reconstructed[iy, ix, fit_mask] = R_fit_norm @ coeffs_norm

        if progress_callback and idx % max(1, n_pixels // 100) == 0:
            progress_callback(int(100 * idx / n_pixels))

    if progress_callback:
        progress_callback(100)

    return {
        "coefficients":  coeff_maps,
        "residuals":     residual_map,
        "reconstructed": reconstructed,
    }


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class NNLSReferenceDialog(BaseWorkflowDialog):
    """
    Four-section workflow dialog for interactive NNLS fitting of 3-D EELS cubes.

    Section 1 — Load 3-D dataset (DepthScan only - no separate ROI)
    Section 2 — Reference regions (shared pre-edge/edge LinearRegions on the
                DepthScan's own spectrum panel; one Set/Clear pair per slot)
    Section 3 — Fit window (auto-proposed, user-adjustable, confirmed)
    Section 4 — Run NNLS fit (all previews/results grouped, shown only
                after this step)
    """

    def __init__(self, parent=None):
        # Section 1 — dataset
        self._interface   = None
        self._model       = None
        self._data        = None      # raw cube, numpy (ny, nx, nE)
        self._base        = ''
        self._depth_scan  = None      # also acts as the spatial ROI
        self._energy_axis = None
        self._spec_meta   = None

        # Section 2 — references
        self._n_refs        = DEFAULT_N_REFS
        self._ref_rows        = []    # UI widgets, one dict per slot
        self._refs             = {}   # idx -> stored reference dict
        self._spec_view        = None # DepthScan's own spectrum panel
        self._pre_region       = None
        self._edge_region      = None
        self._windows_ready    = False

        # Section 3 — fit window
        self._fit_region  = None
        self._fit_window  = None

        super().__init__(parent=parent)

    # ------------------------------------------------------------------
    @classmethod
    def display_name(cls):
        return "EELS - NNLS Interactive Reference Fitting"

    def _setup_workflow(self):
        # Try to auto-load the active model when the dialog opens; non-fatal.
        self._on_load()
        self._on_show_windows()


    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QtWidgets.QVBoxLayout()
        self.setLayout(root)
        self.setWindowTitle("EELS - NNLS Interactive Reference Fitting")
        self.setMinimumWidth(450)

        # ---- Section 1 — dataset --------------------------------------
        grp1  = QtWidgets.QGroupBox("1 — 3D EELS dataset")
        form1 = QtWidgets.QFormLayout()
        grp1.setLayout(form1)

        self._btn_load = QtWidgets.QPushButton("Load active 3D dataset")
        self._btn_load.clicked.connect(self._on_load)
        form1.addRow(self._btn_load)

        self._lbl_shape = QtWidgets.QLabel("—")
        form1.addRow("Shape (ny, nx, nE):", self._lbl_shape)

        self._lbl_erange = QtWidgets.QLabel("—")
        form1.addRow("Energy range:", self._lbl_erange)

        self._lbl_load_status = QtWidgets.QLabel("—")
        self._lbl_load_status.setWordWrap(True)
        form1.addRow("Status:", self._lbl_load_status)

        root.addWidget(grp1)

        # ---- Section 2 — references ------------------------------------
        grp2  = QtWidgets.QGroupBox("2 — Reference regions")
        v2    = QtWidgets.QVBoxLayout()
        grp2.setLayout(v2)

        self._btn_show_windows = QtWidgets.QPushButton(
            "Show background/edge windows")
        self._btn_show_windows.setEnabled(False)
        self._btn_show_windows.clicked.connect(self._on_show_windows)
        v2.addWidget(self._btn_show_windows)
        
        hint = QtWidgets.QLabel(
            "For each reference: move the DepthScan over the area of interest, "
            "position the pre-edge and edge regions then click 'Set'.")
        hint.setWordWrap(True)
        v2.addWidget(hint)

        count_row = QtWidgets.QHBoxLayout()
        count_row.addWidget(QtWidgets.QLabel("Number of references:"))
        self._spin_n_refs = QtWidgets.QSpinBox()
        self._spin_n_refs.setRange(MIN_REFS, MAX_REFS)
        self._spin_n_refs.setValue(DEFAULT_N_REFS)
        count_row.addWidget(self._spin_n_refs)
        self._btn_apply_n = QtWidgets.QPushButton("Apply")
        self._btn_apply_n.clicked.connect(self._on_apply_n_refs)
        count_row.addWidget(self._btn_apply_n)
        count_row.addStretch(1)
        v2.addLayout(count_row)

        self._lbl_window_status = QtWidgets.QLabel("—")
        self._lbl_window_status.setWordWrap(True)
        v2.addWidget(self._lbl_window_status)

        self._ref_rows_container = QtWidgets.QVBoxLayout()
        v2.addLayout(self._ref_rows_container)

        self._btn_clear_all_refs = QtWidgets.QPushButton("Clear all references")
        self._btn_clear_all_refs.clicked.connect(self._on_clear_all_refs)
        v2.addWidget(self._btn_clear_all_refs)

        root.addWidget(grp2)
        self._build_ref_rows(DEFAULT_N_REFS)

        # ---- Section 3 — fit window -------------------------------------
        grp3  = QtWidgets.QGroupBox("3 — Fit window")
        form3 = QtWidgets.QFormLayout()
        grp3.setLayout(form3)
        
        fit_row = QtWidgets.QHBoxLayout()
        self._btn_propose_fit_window = QtWidgets.QPushButton(
            "Propose fit window from stored references")
        self._btn_propose_fit_window.setEnabled(False)
        self._btn_propose_fit_window.clicked.connect(self._on_propose_fit_window)
        fit_row.addWidget(self._btn_propose_fit_window)

        self._btn_confirm_fit_window = QtWidgets.QPushButton(
            "Refresh")
        self._btn_confirm_fit_window.setEnabled(False)
        self._btn_confirm_fit_window.clicked.connect(self._on_confirm_fit_window)
        fit_row.addWidget(self._btn_confirm_fit_window)
        form3.addRow(fit_row)
        self._lbl_fit_window = QtWidgets.QLabel("_")
        self._lbl_fit_window.setWordWrap(True)
        form3.addRow("Status:", self._lbl_fit_window)

        root.addWidget(grp3)

        # ---- Section 4 — run ---------------------------------------------
        grp4  = QtWidgets.QGroupBox("4 — Run NNLS fit")
        form4 = QtWidgets.QFormLayout()
        grp4.setLayout(form4)

        self._btn_run = QtWidgets.QPushButton("Run NNLS fit")
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._on_run)
        form4.addRow(self._btn_run)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        form4.addRow(self._progress)

        self._lbl_run_status = QtWidgets.QLabel("—")
        self._lbl_run_status.setWordWrap(True)
        form4.addRow("Status:", self._lbl_run_status)

        root.addWidget(grp4)

        # ---- Clear -----------------------------------------------------
        btn_clear = QtWidgets.QPushButton("Clear all")
        btn_clear.clicked.connect(self._on_clear)
        root.addWidget(btn_clear)

    # ------------------------------------------------------------------
    # Section 1 — Load dataset
    # ------------------------------------------------------------------

    def _on_load(self):
        try:
            self._interface = PantaRheiInterface.instance(is_intern=True)
            self._model     = self._interface.get_active_model()
            api              = PRScriptingInterface()
            data_obj         = api.get_active_data()
            ndim             = sliced_ndim(data_obj.meta_data, data_obj.ndim)

            if ndim != 3:
                self._lbl_load_status.setText(
                    "ERROR: expected a 3D EELS cube (ny, nx, nE).")
                return

            raw_data  = np.array(data_obj).astype(float)
            meta_data = data_obj.meta_data
            energy_axis = energy_axis_from_meta(meta_data, n_channels=raw_data.shape[-1])
            if energy_axis is None:
                self._lbl_load_status.setText(
                    "ERROR: could not build an energy axis from "
                    "meta_data['device.calib'] (need an entry with unit == 'eV').")
                return
            if not np.all(np.diff(energy_axis) > 0):
                self._lbl_load_status.setText(
                    "ERROR: energy axis is not monotonically increasing "
                    "- check calibration.")
                return

            self._data        = raw_data
            self._energy_axis = energy_axis
            self._base        = self._model.get_output_name()
            ny, nx, nE         = raw_data.shape
            self._lbl_shape.setText(f"({ny}, {nx}, {nE})")
            self._lbl_erange.setText(
                f"{energy_axis[0]:.2f} - {energy_axis[-1]:.2f} eV "
                f"({energy_axis[1] - energy_axis[0]:.4f} eV/ch)")

            _energy_calib, _calib_source = select_energy_calib(meta_data)
            self._spec_meta = {
                "type": "1D",
                _calib_source: [_energy_calib],
            }

            # Reset per-dataset state (DepthScan, regions, references, fit window).
            self._teardown_dataset_widgets()
            self._refs.clear()
            self._fit_window = None
            self._windows_ready = False
            self._refresh_ref_row_labels(reset=True)
            self._lbl_fit_window.setText("_")
            self._btn_propose_fit_window.setEnabled(False)
            self._btn_confirm_fit_window.setEnabled(False)
            self._btn_run.setEnabled(False)

            # DepthScan: the single tool used to browse the cube, place the
            # spatial ROI and view/place windows on the resulting spectrum.
            self._depth_scan = self._interface.insert(
                self._model, PRScriptingTypes.DepthScan,
                parameters={"name": "NNLS_DepthScan"})
            self._spec_view = api.display_plot("NNLS_DepthScan")

            self._btn_show_windows.setEnabled(True)
            self._lbl_load_status.setText(
                f"Loaded  |  {self._base}  |  ({ny}, {nx}, {nE}) ")

        except Exception as exc:
            self._lbl_load_status.setText(f"Load error: {exc}")

    def _teardown_dataset_widgets(self):
        """Remove regions and the DepthScan belonging to the previous dataset."""
        for attr in ("_pre_region", "_edge_region", "_fit_region"):
            obj = getattr(self, attr)
            if obj is not None and self._spec_view is not None:
                try:
                    self._spec_view.remove(obj)
                except Exception:
                    pass
            setattr(self, attr, None)
        self._spec_view = None

        if self._depth_scan is not None and self._model is not None \
                and self._interface is not None:
            try:
                self._interface.remove(self._model, self._depth_scan)
            except Exception:
                pass
        self._depth_scan = None

    # ------------------------------------------------------------------
    # Section 2 — Reference regions
    # ------------------------------------------------------------------

    def _confirm(self, text):
        box = QtWidgets.QMessageBox.question(
            self, "Confirm", text,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        return box == QtWidgets.QMessageBox.Yes

    def _build_ref_rows(self, n):
        """(Re)create the N reference rows in the UI."""
        while self._ref_rows_container.count():
            item = self._ref_rows_container.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._ref_rows = []

        for i in range(n):
            row = QtWidgets.QGroupBox()
            row.setStyleSheet(f"QGroupBox {{ border: 1px solid {'#CACACA'}; border-radius: 8px; margin-top: 1px; }}")
            # row.setStyleSheet(f"QGroupBox {{ border: 1px solid {REF_COLORS[i % len(REF_COLORS)]}; border-radius: 4px; margin-top: 2px; }}")
            h = QtWidgets.QVBoxLayout()
            row.setLayout(h)

            top = QtWidgets.QHBoxLayout()
            label_edit = QtWidgets.QLineEdit(f"Component_{i}")
            top.addWidget(QtWidgets.QLabel(f"ref_{i+1}:"))
            top.addWidget(label_edit)
            set_btn = QtWidgets.QPushButton(f"Set ref_{i+1}")
            set_btn.clicked.connect(lambda _, idx=i: self._on_set_ref(idx))
            top.addWidget(set_btn)
            h.addLayout(top)

            status_lbl = QtWidgets.QLabel("_")
            status_lbl.setWordWrap(True)
            h.addWidget(status_lbl)

            self._ref_rows_container.addWidget(row)
            self._ref_rows.append({
                "label_edit": label_edit,
                "set_btn": set_btn,
                "status_lbl": status_lbl,
            })
        self._n_refs = n

    def _on_apply_n_refs(self):
        new_n = self._spin_n_refs.value()
        dropped = [i for i in self._refs if i >= new_n]
        if dropped:
            if not self._confirm(
                    f"Reducing to {new_n} references will discard "
                    f"{len(dropped)} already-stored reference(s). Continue?"):
                self._spin_n_refs.setValue(self._n_refs)
                return
            for i in dropped:
                del self._refs[i]
        self._build_ref_rows(new_n)
        self._refresh_ref_row_labels()
        self._btn_propose_fit_window.setEnabled(len(self._refs) > 0)

    def _on_show_windows(self):
        """Create (once) the shared pre-edge/edge LinearRegions on the
        DepthScan's own spectrum panel. Safe to click again - a no-op if
        the windows already exist, so it also doubles as a quick check
        that the panel is available."""
        if self._depth_scan is None:
            self._lbl_window_status.setText("Load a dataset first.")
            return
        if self._windows_ready:
            self._lbl_window_status.setText(
                "Windows already shown - move them on the DepthScan panel.")
            return
        try:
            api = PRScriptingInterface()
            # The DepthScan panel displays the live spectrum for its own ROI;
            # grab that plot view so the shared windows live on it directly.
            self._spec_view = api.display_plot("NNLS_DepthScan")
            self._pre_region = self._spec_view.insert(
                PRScriptingTypes.LinearRegion,
                parameters={"name": "NNLS_pre_edge",
                            "region": DEFAULT_PRE_POS,
                            "region_color": "#e49ecf"})
            self._edge_region = self._spec_view.insert(
                PRScriptingTypes.LinearRegion,
                parameters={"name": "NNLS_edge",
                            "region": DEFAULT_EDGE_POS,
                            "region_color": "#f1f094"})
            self._windows_ready = True
            self._lbl_window_status.setText(
                "Pre-edge = pink | Edge = yellow")
        except Exception as exc:
            self._lbl_window_status.setText(f"Window setup error: {exc}")

    def _current_roi_bounds(self):
        """
        Read the DepthScan's own spatial ROI and return pixel bounds
        x0, x1, y0, y1 (clamped), plus a warning string if the parameter
        layout wasn't as expected (falls back to the full frame average).
        """
        ny, nx, _ = self._data.shape
        try:
            roi_params = self._depth_scan.get_parameters(scale_mode="pixel")
        except Exception:
            roi_params = {}

        if "pos" in roi_params and "size" in roi_params:
            cx, cy = roi_params["pos"]
            w,  h  = roi_params["size"]
            x0 = max(0, int(round(cx - w / 2)))
            x1 = min(nx, int(round(cx + w / 2)))
            y0 = max(0, int(round(cy - h / 2)))
            y1 = min(ny, int(round(cy + h / 2)))
            warning = None
        else:
            x0, x1, y0, y1 = 0, nx, 0, ny
            warning = (f"unexpected DepthScan parameter keys "
                       f"{list(roi_params.keys())} - used full-frame average.")

        if x0 >= x1 or y0 >= y1:
            x0, x1, y0, y1 = 0, nx, 0, ny
            warning = "ROI collapsed - used full-frame average."

        return x0, x1, y0, y1, warning

    def _on_set_ref(self, idx):
        if self._data is None:
            self._ref_rows[idx]["status_lbl"].setText("Load a dataset first.")
            return

        just_created = not self._windows_ready
        if just_created:
            self._on_show_windows()
            self._ref_rows[idx]["status_lbl"].setText(
                "Background/edge windows created with default positions - "
                "position them over this reference area, then click "
                "'Set ref_%d' again." % (idx + 1))
            return

        try:
            label = self._ref_rows[idx]["label_edit"].text().strip() or f"Component_{idx}"
            x0, x1, y0, y1, warning = self._current_roi_bounds()
            local_spec = self._data[y0:y1, x0:x1, :].mean(axis=(0, 1))

            pre_params  = self._pre_region.get_parameters()
            pre_lo, pre_hi = norm_region_to_ev(
                pre_params["region"][0], pre_params["region"][1], self._energy_axis)
            edge_params = self._edge_region.get_parameters()
            edge_lo, edge_hi = norm_region_to_ev(
                edge_params["region"][0], edge_params["region"][1], self._energy_axis)

            clean_ref, bg_vec = build_clean_reference(
                local_spec, self._energy_axis,
                pre_edge_window=(pre_lo, pre_hi),
                edge_window=(edge_lo, edge_hi))
            ref_max = clean_ref.max()
            if ref_max <= 0:
                self._ref_rows[idx]["status_lbl"].setText(
                    "No positive signal after background subtraction here - "
                    "reposition the ROI or the pre-edge/edge windows.")
                return

            overwritten = idx in self._refs
            self._refs[idx] = {
                "label": label,
                "roi_bounds": (x0, x1, y0, y1),
                "pre_window": (pre_lo, pre_hi),
                "edge_window": (edge_lo, edge_hi),
                "clean_ref": clean_ref / ref_max,
                "bg_vec": bg_vec,
                "ref_max": ref_max,
            }

            status = (
                f"{'Updated |' if overwritten else ''} ROI x=[{x0},{x1}] y=[{y0},{y1}] | "
                f"Pre-edge {pre_lo:.1f}-{pre_hi:.1f} eV | Edge {edge_lo:.1f}-{edge_hi:.1f} eV")
            if warning:
                status += f"\nWARNING: {warning}"
            self._ref_rows[idx]["status_lbl"].setText(status)

            self._btn_propose_fit_window.setEnabled(True)

        except Exception as exc:
            self._ref_rows[idx]["status_lbl"].setText(f"Set error: {exc}")

    def _on_clear_ref(self, idx):
        if idx in self._refs:
            del self._refs[idx]
        self._ref_rows[idx]["status_lbl"].setText("_")
        self._btn_propose_fit_window.setEnabled(len(self._refs) > 0)
        self._btn_run.setEnabled(False)

    def _on_clear_all_refs(self):
        if not self._refs:
            return
        if not self._confirm(f"Clear all {len(self._refs)} stored reference(s)?"):
            return
        self._refs.clear()
        self._refresh_ref_row_labels()
        self._btn_propose_fit_window.setEnabled(False)
        self._btn_confirm_fit_window.setEnabled(False)
        self._btn_run.setEnabled(False)
        self._lbl_fit_window.setText("_")

    def _refresh_ref_row_labels(self, reset=False):
        for i, row in enumerate(self._ref_rows):
            if reset or i not in self._refs:
                row["status_lbl"].setText("_")

    # ------------------------------------------------------------------
    # Section 3 — Fit window
    # ------------------------------------------------------------------

    def _on_propose_fit_window(self):
        if not self._refs:
            self._lbl_fit_window.setText("Set at least one reference first.")
            return
        try:
            lo = min(min(r["pre_window"][0], r["edge_window"][0]) for r in self._refs.values())
            hi = max(max(r["pre_window"][1], r["edge_window"][1]) for r in self._refs.values())
            nE = len(self._energy_axis)
            i_lo = int(np.argmin(np.abs(self._energy_axis - lo)))
            i_hi = int(np.argmin(np.abs(self._energy_axis - hi)))
            region_pos = [i_lo / (nE - 1), i_hi / (nE - 1)]

            if self._fit_region is None:
                self._fit_region = self._spec_view.insert(
                    PRScriptingTypes.LinearRegion,
                    parameters={"name": "NNLS_fit_window", "region": region_pos,
                                "region_color": "#9ee4ff"})
            else:
                self._fit_region.set_parameters({"region": region_pos})

            self._btn_confirm_fit_window.setEnabled(True)
            self._lbl_fit_window.setText(
                f"Proposed {lo:.1f}-{hi:.1f} eV (envelope of stored references). "
                "Widen if needed, then click 'Confirm fit window'.")
        except Exception as exc:
            self._lbl_fit_window.setText(f"Propose error: {exc}")

    def _on_confirm_fit_window(self):
        if self._fit_region is None:
            self._lbl_fit_window.setText("Propose a fit window first.")
            return
        try:
            fw_params = self._fit_region.get_parameters()
            fit_win = norm_region_to_ev(
                fw_params["region"][0], fw_params["region"][1], self._energy_axis)

            problems = []
            for i, r in self._refs.items():
                lo, hi = r["edge_window"]
                if hi < fit_win[0] or lo > fit_win[1]:
                    problems.append(r["label"])
            if problems:
                self._lbl_fit_window.setText(
                    "ERROR: edge window(s) outside the fit window for: "
                    + ", ".join(problems) + ". Widen the fit window or "
                    "reposition those references.")
                self._btn_run.setEnabled(False)
                return

            self._fit_window = fit_win
            self._lbl_fit_window.setText(
                f"Confirmed: {fit_win[0]:.1f}-{fit_win[1]:.1f} eV "
                f"({len(self._refs)} reference(s) ready).")
            self._btn_run.setEnabled(len(self._refs) > 0)
        except Exception as exc:
            self._lbl_fit_window.setText(f"Confirm error: {exc}")

    # ------------------------------------------------------------------
    # Section 4 — Run
    # ------------------------------------------------------------------

    def _on_run(self):
        if self._data is None:
            self._lbl_run_status.setText("No dataset loaded.")
            return
        if not self._refs:
            self._lbl_run_status.setText("No references set.")
            return
        if self._fit_window is None:
            self._lbl_run_status.setText("Confirm the fit window first.")
            return
        labels = [self._refs[i]["label"] for i in sorted(self._refs)]
        if len(set(labels)) != len(labels):
            self._lbl_run_status.setText(
                "ERROR: duplicate reference labels - rename before running.")
            return
        if len(self._refs) < 2:
            if not self._confirm(
                    "Only one reference component is set. NNLS will still run "
                    "(reference + power-law background), but a single-component "
                    "fit rarely separates chemistry from background well. "
                    "Continue anyway?"):
                return

        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        QtWidgets.QApplication.processEvents()

        try:
            order       = sorted(self._refs)
            labels      = [self._refs[i]["label"] for i in order]
            clean_refs  = [self._refs[i]["clean_ref"] for i in order]

            R = build_reference_matrix(clean_refs, self._energy_axis, self._fit_window)
            N_REFS = len(labels)

            def show_progress(pct):
                self._progress.setValue(pct)
                QtWidgets.QApplication.processEvents()

            results = fit_cube(self._data, R, self._energy_axis, self._fit_window,
                                progress_callback=show_progress)

            coeff_maps   = results["coefficients"]
            residual_map = results["residuals"]

            api = PRScriptingInterface()
            base = self._base
            map_meta = {"type": "2D"}   # 2D maps; self._spec_meta already says "1D" for plots
            # self._spec_meta = {"type": "1D"}
            # ---- Reference spectra used, grouped (pushed to repo only -
            # ---- nothing was shown while they were being placed) ----------
            config_refs = []
            for i in order:
                r = self._refs[i]
                name = f"{base}_NNLS_ref_{r['label']}"
                api.data_to_repo(name, r["clean_ref"], meta_data=self._spec_meta)
                config_refs.append((name, True, False, None, None))
            api.open_multi_view(config_refs, title=f"{base} — Reference spectra used")

            # ---- Chemical maps, grouped ----------------------------------
            config_components = []
            for k, lbl in enumerate(labels):
                name = f"{base}_NNLS_coeff_{lbl}"
                api.data_to_repo(name, coeff_maps[:, :, k],
                                  meta_data={**map_meta, "NNLS_component": lbl})
                config_components.append((name, False, False, None, None))
            component_models, _ = api.open_multi_view(
                config_components, title=f"{base} — Component models")
            for data_model in component_models:
                dc = data_model.get_display_control()
                dc.set_parameters(color_map="flame")

            # ---- Background + residual maps, grouped -----------------------
            config_bg = []
            for j, bg_lbl in enumerate(BG_LABELS):
                name = f"{base}_NNLS_{bg_lbl}"
                api.data_to_repo(name, coeff_maps[:, :, N_REFS + j],
                                  meta_data={**map_meta, "NNLS_component": bg_lbl})
                config_bg.append((name, False, False, None, None))
            res_name = f"{base}_NNLS_residual"
            api.data_to_repo(res_name, residual_map, meta_data=map_meta)
            config_bg.append((res_name, False, False, None, None))
            bg_models, _ = api.open_multi_view(
                config_bg, title=f"{base} — Background and residual")
            for data_model in bg_models:
                dc = data_model.get_display_control()
                dc.set_parameters(color_map="flame")

            self._progress.setValue(100)
            self._lbl_run_status.setText(
                "Done.\n"
                f"Components: {', '.join(labels)}\n"
                f"Fit window: {self._fit_window[0]:.1f}-{self._fit_window[1]:.1f} eV\n"
                f"Mean residual: {residual_map.mean():.4g}"
            )

        except Exception as exc:
            self._lbl_run_status.setText(f"Run error: {exc}")

        finally:
            self._progress.setVisible(False)
            self._btn_run.setEnabled(True)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _on_clear(self):
        self._teardown_dataset_widgets()
        self._interface   = None
        self._model       = None
        self._data        = None
        self._energy_axis = None
        self._spec_meta   = None
        self._refs.clear()
        self._fit_window  = None
        self._windows_ready = False
        self._build_ref_rows(DEFAULT_N_REFS)
        self._spin_n_refs.setValue(DEFAULT_N_REFS)
        self._lbl_shape.setText("—")
        self._lbl_erange.setText("—")
        self._lbl_load_status.setText("—")
        self._lbl_window_status.setText("—")
        self._lbl_fit_window.setText("_")
        self._lbl_run_status.setText("—")
        self._btn_show_windows.setEnabled(False)
        self._btn_propose_fit_window.setEnabled(False)
        self._btn_confirm_fit_window.setEnabled(False)
        self._btn_run.setEnabled(False)

    def closeEvent(self, ev):
        self._on_clear()
        super().closeEvent(ev)