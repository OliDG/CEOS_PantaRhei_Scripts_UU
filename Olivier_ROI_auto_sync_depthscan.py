"Multi synchronized depth scans at once"
# ===============================================================
# for Panta Rhei 0.25
# Set and display multiple synchronized depth scans at once
#
# Steps:
# 	Use shift + clic to select the models (2D-3D-4D)
#   The script will: 
#   - Set-up the correct depth scan for the model dimension
#   - Display the plot for 3D data model, images for 4D data model.
#   - Synchronize them
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

from panta_rhei.main.gui.utils import sliced_ndim
from panta_rhei.scripting import PRScriptingInterface, PRScriptingTypes
api = PRScriptingInterface()

# Select all models 
models = api.get_selected_models()
name_list = []
if len(models) <= 1:
    raise TypeError("Select at least 2 models (shift+clic)")

def add_depth_scan_correct_dimension(model):
#get the correct type of depth scan and display it
    base = model.get_output_name()
    depth_scan_image_name = f"{base}_"+str(i)
    data = model.get_topmost_data()
    ndim = sliced_ndim(data.meta_data, data.ndim)
    if ndim not in (2, 3, 4):
        raise TypeError("Only working for 2D,3D and 4D datasets")

    elif ndim == 2:
        depth_scan = model.insert(
                PRScriptingTypes.DummyDepthScan,
                parameters={"name": depth_scan_image_name}
        )

    elif ndim == 3:
        depth_scan = model.insert(
                PRScriptingTypes.DepthScan,
                parameters={"name": depth_scan_image_name}
        )
        api.display_plot(depth_scan_image_name)

    elif ndim == 4:
        depth_scan = model.insert(
                PRScriptingTypes.ImageCubeDepthScan,
                parameters={"name": depth_scan_image_name}
        )
        api.display_image(depth_scan_image_name)

    return depth_scan

# add the right depth scan to each model selected
for i, model in enumerate(models):
    depth_scan = add_depth_scan_correct_dimension(model)
    name_list.append(depth_scan)

# link the depth scans created
for i in range(len(name_list)-1):
    api.create_data_tool_link([name_list[0], name_list[i+1]])