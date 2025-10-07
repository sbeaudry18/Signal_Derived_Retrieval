#### boundary_layer_index.py ####

# Author: Sam Beaudry
# Last changed: 2025-03-11
# Location: Signal_Derived_Retrieval/TEMPO/main/functions
# Contact: samuel_beaudry@berkeley.edu

##################################

import numpy as np
import xarray as xr

def boundary_layer_index(scan_ds: xr.Dataset):
    '''
    Determines the boundary layer index for TEMPO pixels

    Parameters
    ----------
    scan_ds : xr.Dataset
        Dataset with variables 'boundary_layer_height' and 'interface_heights'

    Returns
    -------
    scan_ds : xr.Dataset
        scan_ds with the added variable 'boundary_layer_index'
    '''
    boundary_level_layer = np.full(scan_ds['amf_troposphere'].shape, -9999, dtype=int)

    layer_indices = np.arange(0, 72)

    loop_size = scan_ds.mirror_step.size * scan_ds.xtrack.size
    nan_size = 0

    for ms in range(scan_ds.mirror_step.size):
        for xt in range(scan_ds.xtrack.size):

            if scan_ds['main_data_quality_flag'].data[ms, xt] > 0:
                continue

            blh = scan_ds['boundary_layer_height'].data[ms, xt] # m

            if (blh == -99.) | (blh == -123.):
                continue

            height_above_bl = scan_ds['interface_heights'].data[ms, xt] - blh

            # The first positive height is the top interface of the layer containing the boundary layer pause
            filtered_layers = layer_indices[height_above_bl >= 0]

            #if len(filtered_layers) == len(layer_indices):
            try:
                if filtered_layers[0] == 0:
                    # Boundary layer is contained in the lowest model level
                    # Set as -1 to indicate that no model levels are fully contained in the boundary layer
                    bll = -1
                    boundary_level_layer[ms, xt] = bll

                else:
                    # Set as the highest layer completely contained in the boundary layer
                    bll = filtered_layers[0]
                    assert bll > 0
                    boundary_level_layer[ms, xt] = bll - 1

            except IndexError:
                nan_size += 1
                
                continue # leave as nan

            #pixels_completed += 1


    scan_ds['boundary_layer_index'] = (
                                            ['mirror_step', 'xtrack'],
                                            boundary_level_layer.astype(int),
                                            {
                                                'units': '1',
                                                'description': "Index of the highest layer in GEOS-CF which is completely inside the convective/planetary boundary layer",
                                                'ancillary_vars': ['boundary_layer_height', 'interface_heights']
                                            }
    )

    #print(nan_size/loop_size, ' were nan')
    
    return scan_ds