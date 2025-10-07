#### a_priori_division.py ####

# Author: Sam Beaudry
# Last changed: 2025-03-11
# Location: Signal_Derived_Retrieval/TEMPO/main/functions
# Contact: samuel_beaudry@berkeley.edu

################################
import numpy as np
import xarray as xr
                                                                                    
def a_priori_division(scan_ds: xr.Dataset):
    '''
    Calculates a priori vertical column densities in the boundary layer and troposphere

    Parameters
    ----------
    scan_ds : xr.Dataset
        Dataset with variables 'boundary_layer_index' and 'geoscf_tropopause_layer_index'

    Returns
    -------
    scan_ds : xr.Dataset
        scan_ds with the added variables 'model_no2_tropospheric_vcd', 'model_no2_boundary_layer_vcd', 'model_no2_total_bcd'
    '''
    # Start by defining the columns as zero
    model_tropospheric_vcd = np.full(scan_ds['amf_troposphere'].shape, np.nan, dtype=float) # molecules / cm^2
    model_bl_vcd = np.full(scan_ds['amf_troposphere'].shape, np.nan, dtype=float) # molecules / cm^2


    for ms in range(scan_ds.mirror_step.size):
        for xt in range(scan_ds.xtrack.size):

            if scan_ds['main_data_quality_flag'].data[ms, xt] > 0:
                continue

            #### Loop content ####
            # Find the tropospheric column density
            trop_layer_index = scan_ds['geoscf_tropopause_layer_index'].data[ms, xt]

            if not np.isnan(trop_layer_index):
                trop_layer_index = int(trop_layer_index)
                model_tropospheric_vcd[ms, xt] = np.sum(scan_ds['gas_profile'].data[ms, xt, :(trop_layer_index+1)])

            # Find the boundary layer column density
            bl_index = scan_ds['boundary_layer_index'].data[ms, xt]

            # Can only find this if the boundary layer index was identified from HRRR
            if np.isnan(bl_index):
                continue

            try:
                if bl_index < 0:
                    model_bl_vcd[ms, xt] = scan_ds['gas_profile'].data[ms, xt, 0]

                else:
                    model_bl_vcd[ms, xt] = np.sum(scan_ds['gas_profile'].data[ms, xt, :(bl_index+1)])

                    #####################

            except TypeError as e:
                model_bl_vcd[ms, xt] = np.nan

            #pixels_completed += 1

    scan_ds['model_no2_tropospheric_vcd'] = (
                                                ['mirror_step', 'xtrack'],
                                                model_tropospheric_vcd,
                                                {
                                                    'units': 'molecules/cm^2',
                                                    'description': 'tropospheric vertical column density of the a priori profile used in the original retrieval',
                                                    'ancillary_vars': ['geoscf_tropopause_layer_index', 'gas_profile']
                                                }
        )                

    scan_ds['model_no2_boundary_layer_vcd'] = (
                                            ['mirror_step', 'xtrack'],
                                            model_bl_vcd,
                                            {
                                                'units': 'molecules/cm^2',
                                                'description': 'boundary layer vertical column density of the a priori profile used in the original retrieval',
                                                'ancillary_vars': ['boundary_layer_index', 'boundary_layer_position', 'vertical_layer_thickness', 'gas_profile']
                                            }
    )       

    scan_ds['model_no2_total_vcd'] = (
                                            ['mirror_step', 'xtrack'],
                                            np.sum(scan_ds['gas_profile'].data, axis=2),
                                            {
                                                'units': 'molecules/cm^2',
                                                'description': 'total vertical column density of the a priori profile used in the original retrieval',
                                                'ancillary_vars': ['gas_profile']
                                            }
    )    

    return scan_ds