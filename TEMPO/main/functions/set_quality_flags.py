#### set_quality_flags.py ####

# Author: Sam Beaudry
# Last changed: 2025-10-15
# Location: Signal_Derived_Retrieval/TEMPO/main/functions
# Contact: samuel_beaudry@berkeley.edu

##############################

import numpy as np
import pandas as pd
import xarray as xr

def set_quality_flags(scan_ds, update_mode="Standard", nonzero_amf_calc=None):
    '''
    Creates bit array flag for troubleshooting and quality filtering

    Parameters
    ----------
    scan_ds : xr.Dataset
        TEMPO dataset with all variables needed to perform signal-derived update
    update_mode : str (Optional)
        Scattering weight mode to determine flags for. Default is Standard
    nonzero_amf_calc : array (Optional)
        Predetermined nonzero_amf_calc to use instead of entering loop in this function
    '''

    # Set the scattering_weight variable
    if update_mode == "Standard":
        sw_var = 'scattering_weights'

        # SB 2025-03-25: My read of the TEMPO PUM is that the provided scattering weights do not
        # include the temperature correction factors. As of this date, the calculation of custom
        # scattering weights already incorporates the temperature factors. Add them in for the 
        # standard mode here
        temperature_corrections = scan_ds['TemperatureCorrection'].data
        scattering_weights *= temperature_corrections

    else:
        sw_var = 'ScatteringWeightsIPA_{}'.format(update_mode)


    # Arrays needed to determine pixel quality for update
    model_partial_columns = scan_ds['gas_profile'].data
    original_amf = scan_ds['amf_troposphere'].data
    original_retrieved_vcd = scan_ds['vertical_column_troposphere'].data
    trop_index = scan_ds['geoscf_tropopause_layer_index'].data
    boundary_layer_index = scan_ds['boundary_layer_index'].data
    model_boundary_layer_vcd = scan_ds['model_no2_boundary_layer_vcd'].data
    scattering_weights = scan_ds[sw_var].data
    data_quality_flag = scan_ds['main_data_quality_flag'].data
    eff_cloud_fraction = scan_ds['eff_cloud_fraction'].data
    sza = scan_ds['solar_zenith_angle'].data
    pixel_area = scan_ds['area'].data
    snow_ice_fraction = scan_ds['snow_ice_fraction'].data

    shape_2d = original_amf.shape

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

    # Check if we have a precalculated value for this provided 
    if not isinstance(nonzero_amf_calc, np.ndarray):

        # If not, loop through the pixels and determine this
        nonzero_amf_calc = np.array([], dtype=bool)
        for ms in range(shape_2d[0]):
            for xt in range(shape_2d[1]):
                if trop_index_known[ms, xt] & scattering_weights_good[ms, xt]:
                    trop = trop_index[ms, xt]
                    m = scattering_weights[ms, xt, :trop+1]
                    v = model_partial_columns[ms, xt, :trop+1]

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
        original_amf_exists.flatten(), #0
        vcd_trop_exists.flatten(), #1
        trop_index_known.flatten(), #2
        scattering_weights_good.flatten(), #3
        nonzero_amf_calc, #4 
        main_data_quality_0.flatten(), #5 
        low_eff_cloud_fraction.flatten(), #6
        low_sza.flatten(), #7
        model_bl_vcd_exists.flatten(), #8
        bl_index_exists.flatten(), #9
        bl_index_below_trop.flatten(), # 10
        valid_pixel_area.flatten(), # 11
        zero_snow_ice.flatten() # 12
    ], axis=1)

    # It is easier to interpret the resulting bit array flags if 1 corresponds to bad quality and 0 to good (since all flags at 0 yield an int
    # of 0 regardless of the length of the bit array). 
    quality_filter_inverted = ~quality_filter # Now True corresponds to bad quality

    qf_columns = ['original_trop_amf_invalid', 'original_trop_vcd_invalid', 'trop_index_unknown', 'scattering_weights_bad', 'calculated_trop_amf_invalid',
                  'main_data_quality_above_0', 'high_eff_cloud_fraction', 'high_sza', 'model_bl_vcd_unknown', 'bl_index_unknown', 'bl_index_above_tp', 
                  'invalid_pixel_area', 'nonzero_snow_ice']
    
    quality_df = pd.DataFrame(quality_filter_inverted.astype(int), columns=qf_columns, dtype=str)

    ## Quality flags (bit-array)
    # Flip so that the most severe issues are at the earlier bit positions 
    quality_df = quality_df[quality_df.columns[::-1]]

    bit_sign_series = pd.Series(np.tile('0b', len(quality_df)), dtype=str)
    quality_series = bit_sign_series.str.cat(quality_df)
    quality_series_int = quality_series.apply(lambda x: int(x, 2))

    # Reconstruct the Series of quality flags into the 2D dataset shape
    update_quality_flags = np.reshape(quality_series_int.to_numpy(), shape_2d)

    # Store as a variable in scan_ds with the bit positions and meanings as variable attributes
    scan_ds['update_quality_flags_{}'.format(update_mode)] = (
                                                        ['mirror_step', 'xtrack'],
                                                        update_quality_flags,
                                                        {
                                                            'description': 'bit flag indicating quality of pixel for update algorithm',
                                                            'bit_positions': np.flip(np.arange(quality_df.shape[1])),
                                                            'bit_meanings': list(quality_df.columns)
                                                        }
    )

    return scan_ds