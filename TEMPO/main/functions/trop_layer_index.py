#### trop_layer_index.py ####

# Author: Sam Beaudry
# Last changed: 2025-03-26
# Location: Signal_Derived_Retrieval/TEMPO/main/functions
# Contact: samuel_beaudry@berkeley.edu

#############################

# SB 2025-03-26
# After some trial and error, the way that I can replicate the provided TEMPO tropospheric AMF is by:
#  1) Locating the vertical layer which contains the provided tropopause pressure
#  2) Finding the difference between the layer interface pressures and the tropopause pressure
#  3) If the tropopause is in the middle or closer to the lower interface, exclude the matching layer.
#     If the tropopause is closer to the upper interface, include the matching layer.
# This function adds the variable "geoscf_tropopause_layer_index". This can be thought of as the "inclusion"
# layer: it is the highest layer to include in calculations for the troposphere.

import numpy as np
import xarray as xr

def trop_layer_index(scan_ds: xr.Dataset):
    '''
    Determines tropopause layer index for pixels in TEMPO Dataset

    Parameters
    ----------
    scan_ds : xr.Dataset
        Dataset with variables 'interface_pressures' and 'tropopause_pressure'

    Returns
    -------
    scan_ds : xr.Dataset
        scan_ds with the added variable 'geoscf_tropopause_layer_index'
    '''

    # Since we are working with integers, use a fill value rather than nan
    trop_layer_index = np.full((scan_ds['mirror_step'].size, scan_ds['xtrack'].size), -9999, dtype=int)

    layer_indices = np.arange(72, dtype=int)

    for ms in range(scan_ds.mirror_step.size):
        for xt in range(scan_ds.xtrack.size):
            trop_pres = scan_ds.tropopause_pressure.data[ms, xt] # hPa
            interface_pres = scan_ds.interface_pressures.data[ms, xt] # hPa
            # Reconstruct the full interface pressure array by inserting the surface pressure at position 0
            surf_pres = scan_ds.surface_pressure.data[ms, xt] # hPa
            interface_pres = np.insert(interface_pres, 0, surf_pres)

            if (np.isnan(trop_pres) or np.any(np.isnan(interface_pres))):
                continue

            # Filter layers to those with lower interfaces of greater pressure than the tropopause pressure

            # Exclude the "top" layer if the lower interface equals the tropopause. As we will see, this layer would 
            # be excluded from calculations even if the tropopause pressure was slightly lower, so it makes sense to 
            # define the inequality as a simple greater than
            layers_trop = layer_indices[interface_pres[:-1] > trop_pres]

            layer_with_tp = layers_trop[-1]

            # Unless layer_with_tp is at the same pressure as the upper interface, only some of this layer is within the 
            # troposphere. To determine if this layer should be used as the tropopause layer index (i.e., whether or not
            # to include it in troposphere calculations), determine how close the tropopause pressure is to the interface pressures
            pres_diff_lower = np.abs(interface_pres[layer_with_tp] - trop_pres)
            pres_diff_upper = np.abs(interface_pres[layer_with_tp+1] - trop_pres)

            # If tropopause is in middle of layer or lower, do not include layer_with_tp in troposphere calculations and use the layer below it
            if pres_diff_lower <= pres_diff_upper:
                trop_layer_index[ms, xt] = layer_with_tp - 1

            # If tropopause is in upper half of layer, do include layer_with_tp in troposphere calculations
            else:
                trop_layer_index[ms, xt] = layer_with_tp

    scan_ds['geoscf_tropopause_layer_index'] = (['mirror_step', 'xtrack'], trop_layer_index, {'description': 'index in swt_level corresponding to the highest layer that should be included in troposphere calculations', 'ancillary variable': "['tropopause_pressure', 'interface_pressures', 'surface_pressure']"})
    return scan_ds