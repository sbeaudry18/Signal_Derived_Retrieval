#### prepare_for_update.py ####

# Author: Sam Beaudry
# Last changed: 2025-10-15
# Location: Signal_Derived_Retrieval/TEMPO/main/functions
# Contact: samuel_beaudry@berkeley.edu

###############################

import xarray as xr
import numpy as np
import pandas as pd

def prepare_for_update(scan_ds: xr.Dataset, prior_match_condition : np.ndarray, ms_match : np.ndarray, xt_match : np.ndarray, n_swt_levels: int, n_updates: int=50, sw_var: str='scattering_weights', uqf_var: str='update_quality_flags_Standard'):
    '''
    Collects information from pixels with shared prior for use in the recursive update

    The AMF recursive update is unaware of, for example, how the pixels are grouped or how the
    boundary layer is characterized. This function takes the relevant information from scan_ds
    and turns it into simple arrays that the recursive update can work with.

    Parameters
    ----------
    scan_ds : xr.Dataset
        TEMPO Dataset with the variables sufficient for performing the update
    prior_match_condition : np.ndarray
        Array with dimensions [mirror_step, xtrack] indicating if each pixel is matched with the selected GEOS-CF cell
    ms_match : np.ndarray
        Mirrorstep indices corresponding to prior_match_condition
    xt_match : np.ndarray
        XTrack indices corresponding to prior_match_condition
    n_swt_levels : int
        The number of GEOS-CF scattering weight levels for the vertical dimension
    n_updates : int (Optional)
        Number of times to update the a priori profiles based on the retrieved values
    sw_var : str (Optional)
        Variable to use for scattering weights
    uqf_var : str (Optional)
        Variable to use for update quality filtering
        
    Returns
    -------
    dict
        Arguments to amf_recursive_update
    '''      

    # Arrays needed to determine pixel quality and perform update
    update_quality_flags = scan_ds[uqf_var].data[prior_match_condition]

    model_partial_columns = scan_ds['gas_profile'].data[prior_match_condition]
    model_partial_columns /= 6.022e19 # mol m^-2
    original_amf = scan_ds['amf_troposphere'].data[prior_match_condition]
    original_retrieved_vcd = scan_ds['vertical_column_troposphere'].data[prior_match_condition]
    original_retrieved_vcd /= 6.022e19 # mol m^-2
    trop_index = scan_ds['geoscf_tropopause_layer_index'].data[prior_match_condition]
    boundary_layer_index = scan_ds['boundary_layer_index'].data[prior_match_condition]
    model_boundary_layer_vcd = scan_ds['model_no2_boundary_layer_vcd'].data[prior_match_condition]
    model_boundary_layer_vcd /= 6.022e19 # mol m^-2
    model_tropospheric_vcd = scan_ds['model_no2_tropospheric_vcd'].data[prior_match_condition]
    model_tropospheric_vcd /= 6.022e19 # mol m^-2
    scattering_weights = scan_ds[sw_var].data[prior_match_condition]
    pixel_area = scan_ds['area'].data[prior_match_condition]

    if sw_var == 'scattering_weights':
        # SB 2025-03-25: My read of the TEMPO PUM is that the provided scattering weights do not
        # include the temperature correction factors. As of this date, the calculation of custom
        # scattering weights already incorporates the temperature factors. Add them in for the 
        # standard mode here
        temperature_corrections = scan_ds['TemperatureCorrection'].data[prior_match_condition]
        scattering_weights *= temperature_corrections

    # Remove pixels from the arrays that do not meet "calculation quality"
    # In other words, remove pixels which will cause issues if we try to recalculate
    # their AMF

    # The first 5 conditions in update_quality_flags must be satisfied to meet calculation quality
    # This means if any of the first 5 bits are raised (1), the pixel does not meet calculation quality
    # Calculation quality pixels therefore have update_quality_flags evenly divisible by 32, since the 
    # first 5 bits define the integer up to a value of 31

    calculation_quality = ((update_quality_flags % 32) == 0)
    # Define new ms and xt matching values for use in the main script
    ms_match_new = ms_match[calculation_quality]
    xt_match_new = xt_match[calculation_quality]

    # Redefine arrays without the problematic pixels
    model_partial_columns = model_partial_columns[calculation_quality]
    scattering_weights = scattering_weights[calculation_quality]
    original_amf = original_amf[calculation_quality]
    original_retrieved_vcd = original_retrieved_vcd[calculation_quality]
    trop_index = trop_index[calculation_quality]
    boundary_layer_index = boundary_layer_index[calculation_quality]
    model_boundary_layer_vcd = model_boundary_layer_vcd[calculation_quality]
    model_tropospheric_vcd = model_tropospheric_vcd[calculation_quality]
    pixel_area = pixel_area[calculation_quality]

    # Now, as a subset of calculation quality pixels, define "good quality" pixels
    # which can be used to redistribute the prior
    update_quality_flags_narrower = update_quality_flags[calculation_quality]

    # A good quality pixel meets all of the conditions, so none of the bits should be raised. This corresponds to a 0 integer
    good_quality = (update_quality_flags_narrower == 0)

    # Store the indices of the pixels meeting this qa condition
    pixel_indices = np.arange(ms_match_new.size)
    good_pixels = pixel_indices[good_quality]

    # We will leave the other arrays at calculation_quality filter for now. When these 
    # arrays are passed to amf_recursive_update, the good_pixels array will be used to identify
    # which pixels should actually be used to determine the prior distribution

    args_for_amf_update = {
        "model_partial_columns": model_partial_columns,
        "box_amfs": scattering_weights,
        "original_amf": original_amf,
        "original_retrieved_vcd": original_retrieved_vcd,
        "trop_index": trop_index,
        "boundary_layer_index": boundary_layer_index,
        "model_boundary_layer_vcd": model_boundary_layer_vcd,
        "model_tropospheric_vcd": model_tropospheric_vcd,
        "pixel_area": pixel_area,
        "good_pixels": good_pixels,
        "Ni": n_updates,
        "initial_final_only": True
    }

    return args_for_amf_update, ms_match_new, xt_match_new