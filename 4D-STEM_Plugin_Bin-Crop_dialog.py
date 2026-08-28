"""
Workflow dialog plugin for 4D-STEM cropping and binning in Panta Rhei 0.25.
Crop and Bin both operate on the originally loaded dataset independently.

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
import copy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_bin_factors(n, max_factor=9):
    """Return all divisors of n that are <= max_factor, as a sorted list."""
    return sorted(f for f in range(1, max_factor + 1) if n % f == 0)


def _bin_axis(arr, factor, axis):
    """
    Block-mean bin along one axis.
    Trims to the nearest lower multiple of factor before reshaping.
    """
    n     = (arr.shape[axis] // factor) * factor
    arr   = np.take(arr, np.arange(n), axis=axis)
    shape = list(arr.shape)
    shape[axis:axis + 1] = [n // factor, factor]
    return arr.reshape(shape).mean(axis=axis + 1)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class FourDimBinCropDialog(BaseWorkflowDialog):
    """
    Three-section workflow dialog for 4D-STEM crop and binning.

    See module docstring for usage.
    """

    def __init__(self, parent=None):
        self._model      = None
        self._data       = None
        self._base       = ""
        self._interface  = None
        self._depth_scan = None
        super().__init__(parent=parent)

    @classmethod
    def display_name(cls):
        return "4D-STEM Bin / Crop"

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QtWidgets.QVBoxLayout()
        self.setLayout(root)
        self.setWindowTitle("4D-STEM Bin / Crop")
        self.setMinimumWidth(400)

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

        # ---- Section 2 — Crop ----------------------------------------
        grp2  = QtWidgets.QGroupBox("2 — Crop (real space only)")
        form2 = QtWidgets.QFormLayout()
        grp2.setLayout(form2)

        info2 = QtWidgets.QLabel(
            "Move and resize the DepthScan ROI on the 4D-STEM image\n"
            )
        info2.setWordWrap(True)
        form2.addRow(info2)

        self._lbl_roi_info = QtWidgets.QLabel("ROI: —")
        self._lbl_roi_info.setWordWrap(True)
        form2.addRow("Current ROI:", self._lbl_roi_info)

        self._btn_read_roi = QtWidgets.QPushButton("Update ROI info")
        self._btn_read_roi.setEnabled(False)
        self._btn_read_roi.clicked.connect(self._on_read_roi)
        form2.addRow(self._btn_read_roi)

        self._btn_crop = QtWidgets.QPushButton("Crop")
        self._btn_crop.setEnabled(False)
        self._btn_crop.clicked.connect(self._on_crop)
        form2.addRow(self._btn_crop)

        self._lbl_crop_status = QtWidgets.QLabel("—")
        self._lbl_crop_status.setWordWrap(True)
        form2.addRow("Status:", self._lbl_crop_status)

        root.addWidget(grp2)

        # ---- Section 3 — Bin -----------------------------------------
        grp3  = QtWidgets.QGroupBox("3 — Bin (real + reciprocal space)")
        vbox3 = QtWidgets.QVBoxLayout()
        grp3.setLayout(vbox3)

        # Real space spinboxes
        real_grp  = QtWidgets.QGroupBox("Real space:")
        real_grid = QtWidgets.QGridLayout()
        real_grp.setLayout(real_grid)
        
        self._lbl_factors_rx = QtWidgets.QLabel("")
        real_grid.addWidget(self._lbl_factors_rx, 0, 0)
        self._spin_brx = self._make_bin_spin()
        real_grid.addWidget(self._spin_brx, 0, 1)
        
        self._lbl_factors_ry = QtWidgets.QLabel("")
        real_grid.addWidget(self._lbl_factors_ry, 2, 0)
        self._spin_bry = self._make_bin_spin()
        real_grid.addWidget(self._spin_bry, 2, 1)

        vbox3.addWidget(real_grp)

        # Reciprocal space spinboxes
        recip_grp  = QtWidgets.QGroupBox("Reciprocal space")
        recip_grid = QtWidgets.QGridLayout()
        recip_grp.setLayout(recip_grid)

        self._lbl_factors_qx = QtWidgets.QLabel("")
        recip_grid.addWidget(self._lbl_factors_qx, 0, 0)
        self._spin_bqx = self._make_bin_spin()
        recip_grid.addWidget(self._spin_bqx, 0, 1)

        self._lbl_factors_qy = QtWidgets.QLabel("")
        recip_grid.addWidget(self._lbl_factors_qy, 2, 0)
        self._spin_bqy = self._make_bin_spin()
        recip_grid.addWidget(self._spin_bqy, 2, 1)

        vbox3.addWidget(recip_grp)

        self._lbl_bin_preview = QtWidgets.QLabel("Output shape: —")
        for sp in (self._spin_bry, self._spin_brx, self._spin_bqy, self._spin_bqx):
            sp.valueChanged.connect(self._update_bin_preview)
        vbox3.addWidget(self._lbl_bin_preview)

        self._btn_bin = QtWidgets.QPushButton("Bin")
        self._btn_bin.setEnabled(False)
        self._btn_bin.clicked.connect(self._on_bin)
        vbox3.addWidget(self._btn_bin)

        self._lbl_bin_status = QtWidgets.QLabel("—")
        self._lbl_bin_status.setWordWrap(True)
        vbox3.addWidget(self._lbl_bin_status)

        root.addWidget(grp3)

        # ---- Progress bar (shared) -----------------------------------
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        # ---- Close ---------------------------------------------------
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        root.addWidget(btn_close)

    def _make_bin_spin(self):
        sp = QtWidgets.QSpinBox()
        sp.setRange(1, 64)
        sp.setValue(1)
        return sp

    # ------------------------------------------------------------------
    # Workflow setup
    # ------------------------------------------------------------------

    def _setup_workflow(self):
        self._on_load()
        self._on_read_roi()
        return   # user triggers section 1 manually

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

            self._depth_scan = self._interface.insert(
                self._model,
                PRScriptingTypes.ImageCubeDepthScan,
                parameters={"name": "BinCropDepthScan"})
            api.display_image("BinCropDepthScan", auto_size=True)

            # Populate valid bin factor hints
            self._lbl_factors_rx.setText(f"X:  {_valid_bin_factors(Nx)}")
            self._lbl_factors_ry.setText(f"Y:  {_valid_bin_factors(Ny)}")
            self._lbl_factors_qx.setText(f"Qx:  {_valid_bin_factors(Qx)}")
            self._lbl_factors_qy.setText(f"Qy:  {_valid_bin_factors(Qy)}")
            self._update_bin_preview()

            self._btn_read_roi.setEnabled(True)
            self._btn_crop.setEnabled(True)
            self._btn_bin.setEnabled(True)
            self._lbl_load_status.setText(
                f"Loaded  |  {self._base}  |  resize DepthScan ROI to define crop")

        except Exception as exc:
            self._lbl_load_status.setText(f"Load error: {exc}")

    # ------------------------------------------------------------------
    # Section 2 — crop
    # ------------------------------------------------------------------

    def _read_roi_pixels(self):
        """
        Read DepthScan position and size in pixel coordinates.
        pos  = top-left corner
        size = (width_x, height_y)   ← axis order to verify visually on first use
        Returns (ry0, rx0, ry1, rx1) pixel indices.
        """
        Ny, Nx = self._data.shape[:2]
        size = np.zeros(2)
        try:
            p    = self._interface.get_parameters(
                       self._depth_scan, scale_mode="pixel")
            pos  = p["pos"]
            size[0] = round(p["size"][0])
            size[1] = round(p["size"][1])
            #correct to even numbers to help further binning by 2
            for i,value in enumerate(size):
                if value % 2 != 0:
                    size[i] = value+1

            rx0  = int(round(pos[0]))
            ry0  = int(round(pos[1]))
            rx1  = int(round(pos[0]) + size[0])
            ry1  = int(round(pos[1]) + size[1])
        except Exception as exc:
            self._lbl_roi_info.setText(f"Read ROI error: {exc}")

        # Clamp to valid range
        ry0 = max(0, min(Ny - 1, ry0))
        ry1 = max(ry0 + 1, min(Ny, ry1))
        rx0 = max(0, min(Nx - 1, rx0))
        rx1 = max(rx0 + 1, min(Nx, rx1))
        return ry0, rx0, ry1, rx1

    def _on_read_roi(self):
        """Read ROI and display pixel coordinates without processing."""
        if self._data is None or self._depth_scan is None:
            return
        try:
            ry0, rx0, ry1, rx1 = self._read_roi_pixels()
            Ny, Nx, Qy, Qx = self._data.shape
            out_nx = (rx1 - rx0)
            out_ny = (ry1 - ry0)
            out_qx = Qx
            out_qy = Qy
            size_mb = out_ny * out_nx * out_qy * out_qx * 4 / 1024**2   # float32

            self._lbl_roi_info.setText(
                f"Output shape  : {out_ny} x {out_nx} scan | {out_qy} x {out_qx} diffraction\n"
                f"Est. size     : {size_mb:.1f} MB (float32)"
                )
        except Exception as exc:
            self._lbl_roi_info.setText(f"Read error: {exc}")

    def _on_crop(self):
        if self._data is None or self._model is None:
            self._lbl_crop_status.setText("No data loaded.")
            return
        try:
            ry0, rx0, ry1, rx1 = self._read_roi_pixels()
            Ny, Nx, Qy, Qx = self._data.shape
            out_nx = (rx1 - rx0)
            out_ny = (ry1 - ry0)
            out_qx = Qx
            out_qy = Qy
            size_mb = out_ny * out_nx * out_qy * out_qx * 4 / 1024**2   # float32

            self._lbl_roi_info.setText(
                f"Output shape  : {out_ny} x {out_nx} scan | {out_qy} x {out_qx} diffraction\n"
                f"Est. size     : {size_mb:.1f} MB (float32)"
                )

            self._btn_crop.setEnabled(False)
            self._progress.setVisible(True)
            self._progress.setValue(10)
            QtWidgets.QApplication.processEvents()

            cropped  = np.array(self._data[ry0:ry1, rx0:rx1, :, :]).astype(float)
            out_key  = f"{self._base}_crop"

            self._progress.setValue(80)
            QtWidgets.QApplication.processEvents()

            try:
                out_meta_data = copy.deepcopy(self._data.meta_data)
                #correcting the scale bars after cropping
                ref_size_list = list(out_meta_data['ref_size'])
                ref_size_list[2] = out_nx * self._data.meta_data['scan_generator.pixel_factors'][0]
                ref_size_list[3] = out_ny * self._data.meta_data['scan_generator.pixel_factors'][1]
                out_meta_data['ref_size'] = tuple(ref_size_list)
                out_meta_data['transform.cut_factors'] = (1.0, 1.0, out_nx/Nx, out_ny/Ny)

            except Exception as metadata_exc:
                self._lbl_crop_status.setText(
                    f"Done (metadata failed: {metadata_exc})")
            
            api = PRScriptingInterface()
            api.data_to_repo(out_key, cropped, meta_data=out_meta_data)
            api.display_image(out_key, auto_size=True)

            self._progress.setValue(100)
            ny = ry1 - ry0
            nx = rx1 - rx0
            _, _, Qy, Qx = self._data.shape
            self._lbl_crop_status.setText(
                f"Done  →  {out_key}  ({ny}, {nx}, {Qy}, {Qx})")

        except Exception as exc:
            self._lbl_crop_status.setText(f"Crop error: {exc}")

        finally:
            self._progress.setVisible(False)
            self._btn_crop.setEnabled(True)

    # ------------------------------------------------------------------
    # Section 3 — bin
    # ------------------------------------------------------------------

    def _update_bin_preview(self):
        if self._data is None:
            return
        Ny, Nx, Qy, Qx = self._data.shape
        bry = self._spin_bry.value()
        brx = self._spin_brx.value()
        bqy = self._spin_bqy.value()
        bqx = self._spin_bqx.value()
        self._lbl_bin_preview.setText(
            f"Output shape: ({Ny//bry}, {Nx//brx}, {Qy//bqy}, {Qx//bqx})  ")

    def _on_bin(self):
        if self._data is None or self._model is None:
            self._lbl_bin_status.setText("No data loaded.")
            return

        bry = self._spin_bry.value()
        brx = self._spin_brx.value()
        bqy = self._spin_bqy.value()
        bqx = self._spin_bqx.value()

        self._btn_bin.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(5)
        QtWidgets.QApplication.processEvents()

        try:
            raw = np.array(self._data).astype(float)

            self._progress.setValue(20)
            QtWidgets.QApplication.processEvents()

            binned = _bin_axis(raw,    bry, 0)
            self._progress.setValue(40)
            QtWidgets.QApplication.processEvents()
            binned = _bin_axis(binned, brx, 1)
            self._progress.setValue(60)
            QtWidgets.QApplication.processEvents()
            binned = _bin_axis(binned, bqy, 2)
            self._progress.setValue(75)
            QtWidgets.QApplication.processEvents()
            binned = _bin_axis(binned, bqx, 3)

            self._progress.setValue(85)
            QtWidgets.QApplication.processEvents()

            out_key = f"{self._base}_bin{bry}x{brx}_q{bqy}x{bqx}"
            
            try:
                out_meta_data = copy.deepcopy(self._data.meta_data)
                out_meta_data['transform.pixel_factors'] = (bqx, bqy, brx, bry)
                
            except Exception as metadata_exc:
                self._lbl_crop_status.setText(
                    f"Done (metadata failed: {metadata_exc})")
            
            api = PRScriptingInterface()
            api.data_to_repo(out_key, binned, meta_data=out_meta_data)
            api.display_image(out_key, auto_size=True)

            self._progress.setValue(100)
            self._lbl_bin_status.setText(
                f"Done  →  {out_key}  {binned.shape}")

        except Exception as exc:
            self._lbl_bin_status.setText(f"Bin error: {exc}")

        finally:
            self._progress.setVisible(False)
            self._btn_bin.setEnabled(True)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _on_clear(self):
        try:
            if (self._depth_scan is not None
                    and self._model     is not None
                    and self._interface is not None):
                self._interface.remove(self._model, self._depth_scan)
        except Exception:
            pass
        self._depth_scan = None
        self._model      = None
        self._data       = None
        self._interface  = None

    def closeEvent(self, ev):
        self._on_clear()
        super().closeEvent(ev)