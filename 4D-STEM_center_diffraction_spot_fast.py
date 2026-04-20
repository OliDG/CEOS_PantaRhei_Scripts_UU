# ===============================================================
# Olivier Donzel-G. https://orcid.org/0000-0002-2101-3746
# Uppsala University, Sweden
#
# The following script comes with no warranty. 
# for Panta Rhei 0.25
# to quickly center the direct beam. Originally developped for 4D-STEM nano-beam diffraction without beam stopper.
# Steps:
# 	for each pixel, detect the brightest spot in the DP and center it.
# ===============================================================

import numpy as np
import time
from panta_rhei.scripting import PRScriptingInterface, PRScriptingTypes
from panta_rhei.gui.utils import sliced_ndim

api = PRScriptingInterface()
model = api.get_active_model()
data = api.get_active_data().copy() # copy the data to enable writing
ndim = sliced_ndim(data.meta_data, data.ndim)

#Check the data is 4D-STEM
if ndim == 4:
	# Add progress bar
	api.add_progress_bar(0, 100, additional_text="Progress")
	# Assume data is a 4D array: (scan_y, scan_x, diff_y, diff_x)
	for i in range(data.shape[0]):
		api.set_progress(int(i/data.shape[0]*100))
		for j in range(data.shape[1]):
			pattern = data[i, j]
			cy, cx = np.array(np.unravel_index(np.argmax(pattern), pattern.shape)) # find the coordinates of the brightest spot
			shift_y = pattern.shape[0] // 2 - cy
			shift_x = pattern.shape[1] // 2 - cx
			data[i, j] = np.roll(np.roll(pattern, shift_y, axis=0), shift_x, axis=1) # center the pattern by rolling the array
	   
	newname = model.get_output_name() #+ '_' + data.meta_data['syncscan.data']['data_name']
	api.data_to_repo(newname +'_centered', data) #export to repo 
	api.display_image(newname +'_centered')	#display the result
	api.remove_progress_bar()


else:
    raise TypeError('Diffraction centering is only applicable to 4D-data (scan_y, scan_x, diff_y, diff_x)')
