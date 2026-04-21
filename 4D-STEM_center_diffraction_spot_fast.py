# ===============================================================
# Olivier Donzel-G.
# Uppsala University, Sweden
#
# for Panta Rhei 0.25
# to quickly center the direct beam. Originally developped for 4D-STEM nano-beam diffraction without beam stopper.
#
# Steps:
# 	for each pixel, detect the brightest spot in the DP and center it.
#	display the x and y shifts used for the centered dataset. 
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
from panta_rhei.scripting import PRScriptingInterface
from panta_rhei.main.gui.utils import sliced_ndim
# Add progress bar
api.add_progress_bar(0, 100, additional_text="Progress")
api = PRScriptingInterface()
model = api.get_active_model()
data = api.get_active_data().copy() # copy the data to enable writing
ndim = sliced_ndim(data.meta_data, data.ndim)
shift_x_map = np.zeros(data.shape[:2])
shift_y_map = np.zeros(data.shape[:2]) 

#Check the data is 4D-STEM
if ndim == 4:
	# Add progress bar
	#api.add_progress_bar(0, 100, additional_text="Progress")
	# Assume data is a 4D array: (scan_y, scan_x, diff_y, diff_x)
	for i in range(data.shape[0]):
		api.set_progress(int(i/data.shape[0]*100))
		for j in range(data.shape[1]):
			pattern = data[i, j]
			cy, cx = np.array(np.unravel_index(np.argmax(pattern), pattern.shape))
			shift_y = pattern.shape[0] // 2 - cy
			shift_x = pattern.shape[1] // 2 - cx
			data[i, j] = np.roll(np.roll(pattern, shift_y, axis=0), shift_x, axis=1)
			shift_x_map[i, j] = shift_x
			shift_y_map[i, j] = shift_y
	   
	newname = model.get_output_name() #+ '_' + data.meta_data['syncscan.data']['data_name']
	api.data_to_repo(newname +'_centered', data) #export to repo 
	api.data_to_repo('shift_x_map', shift_x_map) #export shift maps to repo
	api.data_to_repo('shift_y_map', shift_y_map)
	viewer_4D = api.display_image(newname +'_centered')	#display the result
	viewer_x =	api.display_image('shift_x_map')	#display shift maps
	viewer_y = api.display_image('shift_y_map')	#display shift maps
	api.remove_progress_bar()

	# addition of depthscans
	depthA = model.insert(
		PRScriptingTypes.ImageCubeDepthScan,
		parameters={"name": "DepthScan_Original"}
	)
	depth_centered = viewer_4D.insert(
		PRScriptingTypes.ImageCubeDepthScan,
		parameters={"name": "DepthScan_Centered"}
	)
	dummy1 = viewer_x.insert(PRScriptingTypes.DummyDepthScan,
		parameters={"name": "DummySource1"}
	)
	dummy2 = viewer_y.insert(PRScriptingTypes.DummyDepthScan,
		parameters={"name": "DummySource2"}
	)

	api.display_image("DepthScan_Original")
	api.display_image("DepthScan_Centered")

	api.create_data_tool_link([depth_centered, depthA])
	api.create_data_tool_link([dummy1, depthA])
	api.create_data_tool_link([dummy2, depthA])

else:
	raise TypeError('Diffraction centering is only applicable to 4D-data')
