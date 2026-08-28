"""
==================================
for Panta Rhei 0.25

Computes local strain (εxx, εyy, εxy) and rigid-body rotation (ω) from
4D-STEM data using two diffraction spot tracking with masked COM,
following the deformation gradient approach.

Theory
------
For each scan pixel, the COM positions of two g-vectors (g1, g2) are
measured within disk masks that track the spots across the serpentine scan.
These positions form columns of a 2x2 matrix:

    A = [[g1x_ref, g2x_ref],    (reference, measured once from ref pattern)
         [g1y_ref, g2y_ref]]

    B = [[g1x_loc, g2x_loc],    (local, measured per pixel)
         [g1y_loc, g2y_loc]]

    F = B · A⁻¹                 (deformation gradient)
    ε = (F + Fᵀ)/2 - I         (symmetric strain tensor)
    ω = (F - Fᵀ)/2              (antisymmetric rotation tensor, radians)

Output maps (Ny, Nx): εxx, εyy, εxy, ω
NaN is returned for pixels where either spot COM collapses (weak/absent signal).

Sections
--------
1. Load 4D dataset (active model, must be 4D)
2. Load 2D reference pattern (active model, must be 2D)
3. Select 2 diffraction spots via PointROI on the reference pattern
4. Run strain calculation

Spot tracking
-------------
Static masked COM: each pixel uses a fixed disk mask centred on the
COM-refined reference position (grabbed in section 3).
The local COM within the mask gives the instantaneous g-vector position for that pixel.

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

from PyQt5 import QtWidgets

from panta_rhei.main.gui.base_workflow_dialog import BaseWorkflowDialog
from panta_rhei.scripting import PRScriptingInterface, PRScriptingTypes
from panta_rhei.main.gui.panta_rhei_interface import PantaRheiInterface
from panta_rhei.main.gui.utils import sliced_ndim

# ---------------------------------------------------------------------------
# Processing constants
# ---------------------------------------------------------------------------
R_SPOT          = 10   # default COM mask radius in pixels
N_SPOTS         = 2    # fixed: exactly 2 g-vectors

# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------

def measure_spot_com(dp, cx, cy, r, Xf, Yf):
    """
    Measure the COM of one diffraction spot within a disk mask.

    Parameters
    ----------
    dp       : (Qy, Qx) float — diffraction pattern
    cx, cy   : float — mask centre (pixels)
    r        : float — mask radius (pixels)
    Xf, Yf  : (Qy, Qx) float — coordinate grids (pre-computed, shared)

    Returns
    -------
    x_com, y_com : float  COM position; equals (cx, cy) if signal is zero
    dx, dy       : float  shift from mask centre
    valid        : bool   False when integrated signal is zero (→ NaN pixel)
    """
    mask    = ((Xf - cx) ** 2 + (Yf - cy) ** 2) <= r ** 2
    dp_mask = np.where(mask, dp, 0.0)
    S       = dp_mask.sum()

    if S < 1e-12:
        return cx, cy, 0.0, 0.0, False

    x_com = float((dp_mask * Xf).sum() / S)
    y_com = float((dp_mask * Yf).sum() / S)
    return x_com, y_com, x_com - cx, y_com - cy, True


def compute_strain_rotation(g_ref, g_loc):
    """
    Compute the 2D strain tensor and rotation from two g-vector pairs.

    Parameters
    ----------
    g_ref : (2, 2) float — reference g-vectors as columns
                           [[g1x, g2x], [g1y, g2y]]
    g_loc : (2, 2) float — local g-vectors as columns (same layout)

    Returns
    -------
    exx, eyy, exy : float — strain components (dimensionless)
    omega         : float — rigid-body rotation (radians, + = anticlockwise)

    All values are NaN when A is singular (collinear g-vectors).
    """
    try:
        A_inv = np.linalg.inv(g_ref)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan

    cond = np.linalg.cond(g_ref)
    if not np.isfinite(cond) or cond > 1e10:
        return np.nan, np.nan, np.nan, np.nan

    F   = g_loc @ A_inv
    eps = (F + F.T) / 2.0 - np.eye(2)   # symmetric part
    omg = (F - F.T) / 2.0               # antisymmetric part

    return float(eps[0, 0]), float(eps[1, 1]), float(eps[0, 1]), float(omg[1, 0])


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class StrainMappingDialog(BaseWorkflowDialog):
    """
    Four-section workflow dialog for 4D-STEM strain mapping.

    Section 1 — Load 4D dataset
    Section 2 — Load 2D reference pattern and display it
    Section 3 — Place PointROIs on the reference pattern to select 2 spots
    Section 4 — Run the strain calculation
    """

    def __init__(self, parent=None):
        # Section 1 — 4D dataset
        self._interface  = None
        self._model      = None
        self._data       = None
        self._base       = ''
        self._depth_scan = None   # ImageCubeDepthScan on 4D model
        # Section 2 — reference pattern
        self._ref_model  = None
        self._ref_data   = None
        self._ref_viewer = None   # viewer where PointROIs are inserted
        # Section 3 — spot ROIs
        self._spot_rois       = []   # list of up to N_SPOTS CircleAnnotation objects
        self._g_ref_positions = []   # list of (x_com, y_com, r) after COM grab
        super().__init__(parent=parent)

    # ------------------------------------------------------------------
    @classmethod
    def display_name(cls):
        return "4D-STEM Strain Mapping"

    def _setup_workflow(self):
        try:
            self._interface = PantaRheiInterface.instance(is_intern=True)
            self._model     = self._interface.get_active_model()
            api             = PRScriptingInterface()
            self._data      = api.get_active_data()
            ndim            = sliced_ndim(self._data.meta_data, self._data.ndim)

            if ndim != 4:
                self._lbl_load_status.setText(
                    "ERROR: expected 4D dataset (Ny, Nx, Qy, Qx).")
                return

            Ny, Nx, Qy, Qx = self._data.shape
            self._base = self._model.get_output_name()
            self._lbl_shape.setText(f"({Ny}, {Nx}, {Qy}, {Qx})")

            # Remove previous DepthScan if any
            if self._depth_scan is not None:
                try:
                    self._interface.remove(self._model, self._depth_scan)
                except Exception:
                    pass

            # Insert ImageCubeDepthScan via model.insert (correct PR pattern for 4D)
            self._depth_scan = self._interface.insert(self._model,
                PRScriptingTypes.ImageCubeDepthScan,
                parameters={"name": "StrainDepthScan"})
            api.display_image("StrainDepthScan")

            self._lbl_load_status.setText(
                f"Loaded  |  {self._base}  |  ({Ny}, {Nx}, {Qy}, {Qx})")

        except Exception as exc:
            self._lbl_load_status.setText(f"Load error: {exc}")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QtWidgets.QVBoxLayout()
        self.setLayout(root)
        self.setWindowTitle("4D-STEM Strain Mapping")
        self.setMinimumWidth(380)

        # ---- Section 1 — 4D dataset ----------------------------------
        grp1  = QtWidgets.QGroupBox("1 — 4D dataset")
        form1 = QtWidgets.QFormLayout()
        grp1.setLayout(form1)

        self._btn_load = QtWidgets.QPushButton("Load active 4D model")
        self._btn_load.clicked.connect(self._on_load)
        form1.addRow(self._btn_load)

        self._lbl_shape = QtWidgets.QLabel("—")
        form1.addRow("Shape (Ny, Nx, Qy, Qx):", self._lbl_shape)

        self._lbl_load_status = QtWidgets.QLabel("—")
        self._lbl_load_status.setWordWrap(True)
        form1.addRow("Status:", self._lbl_load_status)

        root.addWidget(grp1)

        # ---- Section 2 — Reference pattern ---------------------------
        grp2  = QtWidgets.QGroupBox("2 — Reference pattern")
        form2 = QtWidgets.QFormLayout()
        grp2.setLayout(form2)

        self._btn_load_ref = QtWidgets.QPushButton("Load active 2D reference pattern")
        self._btn_load_ref.clicked.connect(self._on_load_ref)
        form2.addRow(self._btn_load_ref)

        self._lbl_ref_shape = QtWidgets.QLabel("—")
        form2.addRow("Shape (Qy, Qx):", self._lbl_ref_shape)

        self._lbl_ref_status = QtWidgets.QLabel("—")
        self._lbl_ref_status.setWordWrap(True)
        form2.addRow("Status:", self._lbl_ref_status)

        root.addWidget(grp2)

        # ---- Section 3 — Spot selection ------------------------------
        grp3  = QtWidgets.QGroupBox("3 — Spot selection")
        form3 = QtWidgets.QFormLayout()
        grp3.setLayout(form3)

        _lbl_hint = QtWidgets.QLabel(
            "Add one CircleROI per diffraction spot on the reference pattern.\n"
            "Drag each ROI onto the spot and resize to cover it, then\n"
            "click Grab to refine positions by COM before running.")
        _lbl_hint.setWordWrap(True)
        form3.addRow(_lbl_hint)

        self._btn_add_spot = QtWidgets.QPushButton("Add spot ROI")
        self._btn_add_spot.setEnabled(False)
        self._btn_add_spot.clicked.connect(self._on_add_spot)
        form3.addRow(self._btn_add_spot)

        self._btn_grab_spots = QtWidgets.QPushButton(
            "Grab spot positions (COM refinement)")
        self._btn_grab_spots.setEnabled(False)
        self._btn_grab_spots.clicked.connect(self._on_grab_spots)
        form3.addRow(self._btn_grab_spots)

        self._lbl_spots = QtWidgets.QLabel("No spots selected.")
        self._lbl_spots.setWordWrap(True)
        form3.addRow("Spots:", self._lbl_spots)

        self._btn_clear_spots = QtWidgets.QPushButton("Clear all spots")
        self._btn_clear_spots.clicked.connect(self._on_clear_spots)
        form3.addRow(self._btn_clear_spots)

        root.addWidget(grp3)

        # ---- Section 4 — Run -----------------------------------------
        grp4  = QtWidgets.QGroupBox("4 — Strain calculation")
        form4 = QtWidgets.QFormLayout()
        grp4.setLayout(form4)

        self._btn_run = QtWidgets.QPushButton("Run strain calculation")
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

        # ---- Clear ---------------------------------------------------
        btn_clear = QtWidgets.QPushButton("Clear all")
        btn_clear.clicked.connect(self._on_clear)
        root.addWidget(btn_clear)

    # ------------------------------------------------------------------
    # Section 1 — Load 4D dataset
    # ------------------------------------------------------------------

    def _on_load(self):
        try:
            self._interface = PantaRheiInterface.instance(is_intern=True)
            self._model     = self._interface.get_active_model()
            api             = PRScriptingInterface()
            self._data      = api.get_active_data()
            ndim            = sliced_ndim(self._data.meta_data, self._data.ndim)

            if ndim != 4:
                self._lbl_load_status.setText(
                    "ERROR: expected 4D dataset (Ny, Nx, Qy, Qx).")
                return

            Ny, Nx, Qy, Qx = self._data.shape
            self._base = self._model.get_output_name()
            self._lbl_shape.setText(f"({Ny}, {Nx}, {Qy}, {Qx})")

            # Remove previous DepthScan if any
            if self._depth_scan is not None:
                try:
                    self._interface.remove(self._model, self._depth_scan)
                except Exception:
                    pass

            # Insert ImageCubeDepthScan via model.insert (correct PR pattern for 4D)
            self._depth_scan = self._interface.insert(self._model,
                PRScriptingTypes.ImageCubeDepthScan,
                parameters={"name": "StrainDepthScan"})
            api.display_image("StrainDepthScan", auto_size=True)

            self._lbl_load_status.setText(
                f"Loaded  |  {self._base}  |  ({Ny}, {Nx}, {Qy}, {Qx})")

        except Exception as exc:
            self._lbl_load_status.setText(f"Load error: {exc}")

    # ------------------------------------------------------------------
    # Section 2 — Load reference pattern
    # ------------------------------------------------------------------

    def _on_load_ref(self):
        try:
            interface       = PantaRheiInterface.instance(is_intern=True)
            self._ref_model = interface.get_active_model()
            api             = PRScriptingInterface()
            self._ref_data  = api.get_active_data()
            ndim            = sliced_ndim(self._ref_data.meta_data, self._ref_data.ndim)

            if ndim != 2:
                self._lbl_ref_status.setText(
                    "ERROR: expected 2D reference pattern (Qy, Qx).")
                return

            ref_name = self._ref_model.get_output_name()
            Qy, Qx   = self._ref_data.shape
            self._lbl_ref_shape.setText(f"({Qy}, {Qx})")

            # Display and store viewer for PointROI insertion
            self._ref_viewer = api.display_image(ref_name)

            self._btn_add_spot.setEnabled(True)
            self._lbl_ref_status.setText(
                f"Loaded  |  {ref_name}  |  add PointROIs on the pattern")

        except Exception as exc:
            self._lbl_ref_status.setText(f"Load error: {exc}")

    # ------------------------------------------------------------------
    # Section 3 — Spot selection
    # ------------------------------------------------------------------
    # to be modified: must grab the COM coordinates from the ROI
    def _on_add_spot(self):
        if self._ref_viewer is None or self._ref_data is None:
            return
        if len(self._spot_rois) >= N_SPOTS:
            self._lbl_spots.setText(
                f"Already {N_SPOTS} spots selected. "
                f"Clear first to reselect.")
            return
        try:
            Qy, Qx = self._ref_data.shape
            # Default position at pattern centre — user drags it to the spot 
                
            roi1 = self._ref_viewer.insert(
                PRScriptingTypes.CircleAnnotation,
                parameters={"center": (float(Qx // 1.33), float(Qy // 2)), "color": "#52f552","line width": 2, "radius": 10},
                scale_mode="pixel")
            roi2 = self._ref_viewer.insert(
                PRScriptingTypes.CircleAnnotation,
                parameters={"center": (float(Qx // 2), float(Qy // 4)), "color": "#52eaf5","line width": 2, "radius": 10},
                scale_mode="pixel")            

            self._spot_rois.append(roi1)
            self._spot_rois.append(roi2)
            self._update_spot_label()

            if len(self._spot_rois) == N_SPOTS:
                self._btn_grab_spots.setEnabled(True)
                self._lbl_spots.setText(
                    self._lbl_spots.text() +
                    "\nBoth spots placed — click Grab to refine by COM.")

        except Exception as exc:
            self._lbl_spots.setText(f"Add spot error: {exc}")

    def _on_clear_spots(self):
        if self._ref_viewer is not None:
            for roi in self._spot_rois:
                try:
                    self._ref_viewer.remove(roi)
                except Exception:
                    pass
        self._spot_rois.clear()
        self._g_ref_positions.clear()
        self._btn_grab_spots.setEnabled(False)
        self._btn_run.setEnabled(False)
        self._lbl_spots.setText("No spots selected.")

    def _update_spot_label(self):
        if not self._spot_rois:
            self._lbl_spots.setText("No spots selected.")
            return
        lines = []
        for k, roi in enumerate(self._spot_rois):
            try:
                p   = roi.get_parameters(scale_mode="pixel")
                pos = p["center"]
                r   = p.get("radius")
                lines.append(
                    f"Spot {k + 1}: x={pos[0]:.1f}  y={pos[1]:.1f}  r={r}")
            except Exception:
                lines.append(f"Spot {k + 1}: (unreadable)")
        self._lbl_spots.setText("\n".join(lines))

    def _on_grab_spots(self):
        """
        Read each CircleROI centre and radius, measure COM on the reference
        pattern within that radius, and store the refined (x_com, y_com, r)
        in _g_ref_positions.  These are the positions used as initial tracker
        centres at run time — making the result robust to imprecise ROI
        placement as long as the correct spot is enclosed.
        """
        if self._ref_data is None or len(self._spot_rois) < N_SPOTS:
            return
        try:
            for i in range(3): # COM repeated three times to ensure convergence for weak spots
                ref_dp = np.array(self._ref_data).astype(np.float64)
                Qy, Qx = ref_dp.shape
                Y, X   = np.ogrid[:Qy, :Qx]
                Xf     = X.astype(np.float64)
                Yf     = Y.astype(np.float64)

                self._g_ref_positions.clear()
                lines = []

                for k, roi in enumerate(self._spot_rois):
                    p   = roi.get_parameters(scale_mode="pixel")
                    pos = p["center"]
                    r   = float(p["radius"])
                    sx  = float(pos[0])
                    sy  = float(pos[1])

                    xc, yc, _, _, valid = measure_spot_com(
                        ref_dp, sx, sy, r, Xf, Yf)

                    if not valid:
                        self._lbl_spots.setText(
                            f"Grab failed: spot {k + 1} has no signal within "
                            f"the ROI. Reposition the CircleROI over the spot.")
                        self._g_ref_positions.clear()
                        self._btn_run.setEnabled(False)
                        return

                    self._g_ref_positions.append((xc, yc, r))

                    # Recenter the CircleROI on the COM-refined position so the
                    # user can visually confirm the spot was correctly identified
                    try:
                        roi.set_parameters({"center": (xc, yc)}, scale_mode="pixel")
                    except Exception:
                        pass   # non-fatal — position stored regardless

                    lines.append(
                        f"Spot {k + 1}: COM x={xc:.1f}  y={yc:.1f}  r={r:.1f} px" #f"  (ROI was x={sx:.1f} y={sy:.1f})"
                        )

            self._btn_run.setEnabled(True)
            self._lbl_spots.setText(
                "\n".join(lines) +
                "\nPositions grabbed — ready to run.")

        except Exception as exc:
            self._lbl_spots.setText(f"Grab error: {exc}")

    def _read_spot_positions(self):
        """Return list of (x, y) pixel positions from the PointROIs."""
        positions = []
        for roi in self._spot_rois:
            p   = roi.get_parameters(scale_mode="pixel")
            pos = p["center"]
            positions.append((float(pos[0]), float(pos[1])))
        return positions

    # ------------------------------------------------------------------
    # Section 4 — Run
    # ------------------------------------------------------------------

    def _on_run(self):
        if self._data is None:
            self._lbl_run_status.setText("No 4D dataset loaded.")
            return
        if self._ref_data is None:
            self._lbl_run_status.setText("No reference pattern loaded.")
            return
        if len(self._spot_rois) < N_SPOTS:
            self._lbl_run_status.setText(
                f"Need {N_SPOTS} spots — {len(self._spot_rois)} selected.")
            return

        if len(self._g_ref_positions) < N_SPOTS:
            self._lbl_run_status.setText(
                "Grab spot positions first (COM refinement button).")
            return

        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        QtWidgets.QApplication.processEvents()

        try:
            data = self._data
            Ny, Nx, Qy, Qx = data.shape
            base = self._base

            # Coordinate grids — float64, shared across all DPs
            Y, X = np.ogrid[:Qy, :Qx]
            Xf   = X.astype(np.float64)
            Yf   = Y.astype(np.float64)

            # ----------------------------------------------------------
            # Reference g-vectors — already COM-refined at grab time
            # ----------------------------------------------------------
            g_ref_cols = []
            spot_radii = []
            for xc, yc, r in self._g_ref_positions:
                g_ref_cols.append([xc, yc])
                spot_radii.append(r)

            # g_ref columns = g-vectors: [[g1x, g2x], [g1y, g2y]]
            g_ref = np.array(g_ref_cols).T

            # Sanity check: g-vectors must not be collinear
            if abs(np.linalg.det(g_ref)) < 1e-6:
                self._lbl_run_status.setText(
                    "ERROR: the two spots appear collinear — "
                    "choose non-collinear g-vectors.")
                return

            self._lbl_run_status.setText(
                f"Reference  g1=({g_ref[0,0]:.2f}, {g_ref[1,0]:.2f})  "
                f"g2=({g_ref[0,1]:.2f}, {g_ref[1,1]:.2f})  "
                f"— running…")
            QtWidgets.QApplication.processEvents()

            # ----------------------------------------------------------
            # Output maps — initialised to NaN
            # ----------------------------------------------------------
            exx_map   = np.full((Ny, Nx), np.nan)
            eyy_map   = np.full((Ny, Nx), np.nan)
            exy_map   = np.full((Ny, Nx), np.nan)
            omega_map = np.full((Ny, Nx), np.nan)

            # ----------------------------------------------------------
            # Main loop — static mask, no tracking between pixels
            # ----------------------------------------------------------
            total     = Ny * Nx
            processed = 0

            for iy in range(Ny):
                for ix in range(Nx):
                    dp = np.array(data[iy, ix]).astype(np.float64)

                    g_loc_cols  = []
                    any_invalid = False

                    for s in range(N_SPOTS):
                        cx_s, cy_s, r_s = self._g_ref_positions[s]
                        xc, yc, _, _, valid = measure_spot_com(
                            dp, cx_s, cy_s, r_s, Xf, Yf)

                        if not valid:
                            any_invalid = True
                            g_loc_cols.append([cx_s, cy_s])
                        else:
                            g_loc_cols.append([xc, yc])

                    if not any_invalid:
                        g_loc = np.array(g_loc_cols).T
                        exx, eyy, exy, omega = compute_strain_rotation(
                            g_ref, g_loc)
                        exx_map[iy, ix]   = exx
                        eyy_map[iy, ix]   = eyy
                        exy_map[iy, ix]   = exy
                        omega_map[iy, ix] = omega

                    processed += 1
                    if processed % max(1, total // 200) == 0:
                        self._progress.setValue(
                            int(95 * processed / total))
                        QtWidgets.QApplication.processEvents()

            # ----------------------------------------------------------
            # Push outputs and display
            # ----------------------------------------------------------
            api      = PRScriptingInterface()
            out_meta = {"type": "image2D"}

            key_exx   = f"{base}_strain_exx"
            key_eyy   = f"{base}_strain_eyy"
            key_exy   = f"{base}_strain_exy"
            key_omega = f"{base}_rotation_omega"

            api.data_to_repo(key_exx,   exx_map,   meta_data=out_meta)
            api.data_to_repo(key_eyy,   eyy_map,   meta_data=out_meta)
            api.data_to_repo(key_exy,   exy_map,   meta_data=out_meta)
            api.data_to_repo(key_omega, omega_map, meta_data=out_meta)

            config = [
                (key_exx,   False, False, None, None),
                (key_eyy,   False, False, None, None),
                (key_exy,   False, False, None, None),
                (key_omega, False, False, None, None),
            ]
            strain_maps,_ = api.open_multi_view(config, title=f"{base} — Strain maps")
            
            for data_model in strain_maps:
                dc = data_model.get_display_control()
                dc.set_parameters(color_map="bipolar")
                
            self._progress.setValue(100)
            n_valid = int(np.isfinite(exx_map).sum())
            self._lbl_run_status.setText(
                f"Done  |  {n_valid}/{total} valid pixels  |  "
                f"εxx  εyy  εxy  ω  →  {base}_strain_*")

        except Exception as exc:
            self._lbl_run_status.setText(f"Run error: {exc}")

        finally:
            self._progress.setVisible(False)
            self._btn_run.setEnabled(True)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _on_clear(self):
        self._on_clear_spots()
        if self._depth_scan is not None and self._model is not None \
                and self._interface is not None:
            try:
                self._interface.remove(self._model, self._depth_scan)
            except Exception:
                pass
        self._interface       = None
        self._model           = None
        self._data            = None
        self._depth_scan      = None
        self._ref_model       = None
        self._ref_data        = None
        self._ref_viewer      = None
        self._g_ref_positions = []
        self._lbl_shape.setText("—")
        self._lbl_load_status.setText("—")
        self._lbl_ref_shape.setText("—")
        self._lbl_ref_status.setText("—")
        self._lbl_run_status.setText("—")
        self._btn_add_spot.setEnabled(False)
        self._btn_grab_spots.setEnabled(False)
        self._btn_run.setEnabled(False)

    def closeEvent(self, ev):
        self._on_clear()
        super().closeEvent(ev)