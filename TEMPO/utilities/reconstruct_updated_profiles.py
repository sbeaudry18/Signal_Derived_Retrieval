#### reconstruct_updated_profiles.py ####

# Author: Sam Beaudry
# Last changed: 2025-10-07
# Location: Signal_Derived_Retrieval/TEMPO/utlities
# Contact: samuel_beaudry@berkeley.edu

##########################################

import numpy as np
import xarray as xr

def reconstruct_updated_profiles(scan_ds: xr.Dataset, update_modes: list=['Standard']):
    '''
    scan_ds: Dataset with standard vertical profiles but not updated profiles
    update_modes: Optional
    '''
    shape_2d = scan_ds.vertical_column_troposphere.shape
    size_2d = shape_2d[0] * shape_2d[1]
    itr_size = scan_ds.iteration.size
    swt_size = scan_ds.swt_level.size

    updated_profiles = {}
    for mode in update_modes:
        updated_profiles[mode] = np.full((shape_2d[0], shape_2d[1], itr_size, swt_size), np.nan)

    for j in range(size_2d):
        ms, xt = np.unravel_index(j, shape_2d)

        try:
            gas_profile_standard = scan_ds.gas_profile.data[ms, xt, :] # ['swt_level']
            bl_idx = scan_ds.boundary_layer_index.data[ms, xt]
            bl_vcd_standard = scan_ds.model_no2_boundary_layer_vcd.data[ms, xt]

            bl_vert_shape = gas_profile_standard[:(bl_idx+1)] / bl_vcd_standard 

            # model_no2_boundary_layer_vcd_updated_Standard
            for mode in update_modes:
                bl_vcd_updated = scan_ds['model_no2_boundary_layer_vcd_updated_{}'.format(mode)].data[ms, xt, :] # ['iteration']
                
                gas_profile_bl_updated = np.expand_dims(bl_vert_shape, axis=0) * np.expand_dims(bl_vcd_updated, axis=1)
                
                gas_profile_total_updated = gas_profile_standard.copy()
                gas_profile_total_updated = np.tile(np.expand_dims(gas_profile_total_updated, axis=0), (itr_size, 1))
                gas_profile_total_updated[:, :(bl_idx+1)] = gas_profile_bl_updated

                updated_profiles[mode][ms, xt, :, :] = gas_profile_total_updated

        except Exception as e:
            raise e

    for mode in update_modes:
        scan_ds['gas_profile_updated_{}'.format(mode)] = (['mirror_step', 'xtrack', 'iteration', 'swt_level'], updated_profiles[mode])

    return scan_ds