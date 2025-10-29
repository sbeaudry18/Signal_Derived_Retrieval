#### prune_dataset.py ####

# Author: Sam Beaudry
# Last changed: 2025-10-15
# Location: Signal_Derived_Retrieval/TEMPO/main/functions
# Contact: samuel_beaudry@berkeley.edu

##########################

import xarray as xr

def prune_dataset(scan_ds: xr.Dataset, update_modes: tuple, remove_originals=True):
    '''
    Drops vertically resolved variables from update dataset to reduce file size

    Parameters
    ----------
    scan_ds : xr.Dataset
        Finalized TEMPO dataset from SDR workflow
    update_modes : tuple
        List of update modes to identify as profile variable names 
    remove_originals : bool (Optional)
        Default is True. Controls whether the original TEMPO variables 'gas_profile', 'temperature_profile', and 'scattering_weights' are removed.

    Returns
    -------
    xr.Dataset
        scan_ds without fewer vertically-resolved variables
    '''
    vars_to_remove = [
                        'interface_pressures', 
                        'midpoint_pressures', 
                        'interface_heights', 
                        'vertical_layer_thickness', 
                        'TemperatureCorrection', 
                        ]
    
    for mode in update_modes:
        vars_to_remove.append('gas_profile_updated_{}'.format(mode))
    
    if remove_originals:
        vars_to_remove.append('gas_profile')
        vars_to_remove.append('temperature_profile')
        vars_to_remove.append('scattering_weights')

    scan_ds = scan_ds.drop_vars(vars_to_remove)

    return scan_ds