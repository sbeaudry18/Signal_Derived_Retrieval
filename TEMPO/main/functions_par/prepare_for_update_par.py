#### prepare_for_update_par.py ####

# Author: Sam Beaudry
# Last changed: 2025-08-11
# Location: Signal_Derived_Retrieval/TEMPO/main/functions_par
# Contact: samuel_beaudry@berkeley.edu

######################################

# Parallel version of prepare_for_update

def prepare_for_update_par(subset_ds: dict, ms_match, xt_match, n_swt_levels: int, n_updates: int=50, sw_var: str='scattering_weights'):
    '''
    Collects information from pixels with shared prior for use in the recursive update

    The AMF recursive update is unaware of, for example, how the pixels are grouped or how the
    boundary layer is characterized. This function takes the relevant information from subset_ds
    and turns it into simple areas that the recursive update can work with.

    Parameters
    ----------
    subset_ds : dict
        Dict of TEMPO variables necessary for update
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
        
    Returns
    -------
    dict
        Arguments to amf_recursive_update
    '''      
    import numpy as np
    import pandas as pd

    # Arrays needed to determine pixel quality and perform update
    model_partial_columns = subset_ds['gas_profile']
    model_partial_columns /= 6.022e19 # mol m^-2
    original_amf = subset_ds['amf_troposphere']
    original_retrieved_vcd = subset_ds['vertical_column_troposphere']
    original_retrieved_vcd /= 6.022e19 # mol m^-2
    trop_index = subset_ds['geoscf_tropopause_layer_index']
    boundary_layer_index = subset_ds['boundary_layer_index']
    model_boundary_layer_vcd = subset_ds['model_no2_boundary_layer_vcd']
    model_boundary_layer_vcd /= 6.022e19 # mol m^-2
    model_tropospheric_vcd = subset_ds['model_no2_tropospheric_vcd']
    model_tropospheric_vcd /= 6.022e19 # mol m^-2
    scattering_weights = subset_ds[sw_var]
    data_quality_flag = subset_ds['main_data_quality_flag']
    eff_cloud_fraction = subset_ds['eff_cloud_fraction']
    sza = subset_ds['solar_zenith_angle']
    pixel_area = subset_ds['area']
    snow_ice_fraction = subset_ds['snow_ice_fraction']

    if sw_var == 'scattering_weights':
        # SB 2025-03-25: My read of the TEMPO PUM is that the provided scattering weights do not
        # include the temperature correction factors. As of this date, the calculation of custom
        # scattering weights already incorporates the temperature factors. Add them in for the 
        # standard mode here
        temperature_corrections = subset_ds['TemperatureCorrection']
        scattering_weights *= temperature_corrections

    # SB 2025-03-24: We are going to implement a series of filters, building up, first, basic fundamentals for
    # calculating AMF from the provided quantities, then additional requirements for "good" pixels which can
    # inform the redistribution scheme.

    # Filter 0: Pixels with existing positive AMFs 
    original_amf_exists = ~np.isnan(original_amf) & ~(original_amf <= 0) 

    # Filter 1: Pixels with existing VCD values (i.e. successful DOAS fits)
    vcd_trop_exists = ~np.isnan(original_retrieved_vcd)

    # Filter 2: Tropopause layer index is known
    # Missing tropopause layers are marked by the fill value -9999
    trop_index_known = trop_index > 0

    # Filter 3: Scattering weight profile is complete
    scattering_weights_good = ~np.any(np.isnan(scattering_weights), axis=1)

    # Filter 4: AMF calculation will produce non-zero values
    # Since this does not exactly follow the NASA method and may use custom scattering weights,
    # there is the risk that the calculated AMF will be zero. This is most likely in scenarios where the
    # cloud radiance fraction is 1 and the cloud layer is at or above the tropopause layer
    nonzero_amf_calc = np.array([], dtype=bool)
    for pi in range(trop_index.size):
        if trop_index_known[pi] & scattering_weights_good[pi]:
            trop = trop_index[pi]
            m = scattering_weights[pi, :trop+1]
            v = model_partial_columns[pi, :trop+1]

            numerator = np.sum(m * v)
            denominator = np.sum(v)
            calculated_amf = numerator / denominator
            if calculated_amf > 0:
                nonzero_amf_calc = np.append(nonzero_amf_calc, True)

            else:
                nonzero_amf_calc = np.append(nonzero_amf_calc, False)
        else:
            nonzero_amf_calc = np.append(nonzero_amf_calc, False)

    # Filters 0-4 remove "critically bad" pixels; pixels that cannot be allowed to enter the AMF update process
    # or they will cause runtime warnings and or Exceptions

    # The remaining filters indicate "good pixels". These remove pixels that are okay to enter the algorithm,
    # but should not be used to redistribute the prior (either because they are poor quality or missing necessary 
    # information such as the boundary layer height).

    # Filter 5: main_data_quality_flag indicates good quality data
    main_data_quality_0 = data_quality_flag < 1

    # Filter 6: strict requriement for low effective cloud fraction
    low_eff_cloud_fraction = eff_cloud_fraction <= 0.1

    # Filter 7: solar zenith angle (SZA) is less than 70 degrees (following TEMPO PUM)
    low_sza = sza < 70

    # Filter 8: model boundary layer VCD was found successfully
    model_bl_vcd_exists = ~np.isnan(model_boundary_layer_vcd)

    # Filter 9: boundary layer index was found successfully. Note that the pixels with missing boundary layer
    # indices feature the fill value of -9999
    bl_index_exists = boundary_layer_index > 0

    # Filter 10: boundary layer index is below the tropopause layer index (i.e. free tropospheric component exists)
    bl_index_below_trop = boundary_layer_index < trop_index

    # Filter 11: valid value for pixel area. Invalid values include unrealistic areas or negative areas
    # pixel areas are in m^2
    valid_pixel_area = (pixel_area >= 5e6) & (pixel_area <= 4e7)

    # Filter 12: snow ice fraction is zero
    zero_snow_ice = snow_ice_fraction == 0

    # Collect quality filters into a single array
    quality_filter = np.stack([
        original_amf_exists, #0
        vcd_trop_exists, #1
        trop_index_known, #2
        scattering_weights_good, #3
        nonzero_amf_calc, #4 
        main_data_quality_0, #5 
        low_eff_cloud_fraction, #6
        low_sza, #7
        model_bl_vcd_exists, #8
        bl_index_exists, #9
        bl_index_below_trop, # 10
        valid_pixel_area, # 11
        zero_snow_ice # 12
    ], axis=1)

    # Remove pixels from the arrays that do not meet "calculation quality"
    # In other words, remove pixels which will cause issues if we try to recalculate
    # their AMF
    calculation_quality = np.all(quality_filter[:, :5], axis=1)
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
    quality_filter_narrower = quality_filter[calculation_quality]
    good_quality = np.all(quality_filter_narrower, axis=1)
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

    #qf_columns = ['original_amf_exists', 'vcd_trop_exists', 'trop_index_known', 'scattering_weights_good', 'nonzero_amf_calc',
    #              'main_data_quality_0', 'low_eff_cloud_fraction', 'low_sza', 'model_bl_vcd_exists', 'bl_index_exists']

    # It is easier to interpret the resulting bit array flags if 1 corresponds to bad quality and 0 to good (since all flags at 0 yield an int
    # of 0 regardless of the length of the bit array). 
    quality_filter_inverted = ~quality_filter # Now True corresponds to bad quality

    qf_columns = ['original_trop_amf_invalid', 'original_trop_vcd_invalid', 'trop_index_unknown', 'scattering_weights_bad', 'calculated_trop_amf_invalid',
                  'main_data_quality_above_0', 'high_eff_cloud_fraction', 'high_sza', 'model_bl_vcd_unknown', 'bl_index_unknown', 'bl_index_above_tp', 
                  'invalid_pixel_area', 'nonzero_snow_ice']
    
    quality_df = pd.DataFrame(quality_filter_inverted.astype(int), columns=qf_columns, dtype=str)

    return args_for_amf_update, ms_match_new, xt_match_new, quality_df