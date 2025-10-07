#### synthesize_tempo_behr.py ####

# Author: Sam Beaudry
# Last changed: 2025-05-25
# Location: Signal_Derived_Retrieval/TEMPO/main/functions
# Contact: samuel_beaudry@berkeley.edu

###################################

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import xarray as xr
import pickle

def synthesize_tempo_behr(tempo_ds_path: str, tempo_pickle_path: str, vars_path: str, full_FOR: bool, ms_min, ms_max, xt_min, xt_max, use_behr_output=False, behr_output=None, fill_behr_vars=False, verbose=False, upper_bound_inclusive=True):
    '''
    Opens partially-completed MATLAB structure from the BEHR workflow, reformats as an xarray dataset, and adds additional information

    Parameters
    ----------
    tempo_ds_path: str
        Full file name of the original TEMPO dataset
    tempo_pickle_path: str
        Path to the standard TEMPO pickle produced by TEMPO_L2_NO2_on_date
    vars_path: str
        Full file name of the .csv file containing TEMPO variable names, locations, and corresponding OMI variables
    full_FOR: bool
        whether to process for the full field of regard
    ms_min: int
        Lower mirrorstep bound
    ms_max: int
        Upper mirrorstep bound
    xt_min: int
        Lower xtrack bound
    xt_max: int
        Upper xtrack bound
    use_behr_output: bool (Optional)
        Set to True to include data from a BEHR-prepared dataset. Default is False.
    behr_output : str or dict (Optional)
        If use_behr_output=True, the pickled or unpickled BEHR dictionary to use. 
    fill_behr_vars : bool (Optional)
        If use_behr_output=False, whether to include the BEHR variables and add fill values. Default is False.
    verbose: bool (default=False)
        Controls printing behavior of function for debugging
    upper_bound_inclusive: bool (default=True)
        Include ms_max and xt_max values

    Returns
    -------
    xr.Dataset
        Dataset with both S5P and BEHR variables
    '''

    if use_behr_output:
        # Open the BEHR output dictionary produced by MATLAB
        if isinstance(behr_output, str):
            with open(behr_output, 'rb') as handle:
                tempo_behr_dict = pickle.load(handle)
            handle.close()
    
        elif isinstance(behr_output, dict):
            tempo_behr_dict = behr_output.copy()
            del behr_output # Remove from memory
    
        else:
            raise Exception("behr_output is of unsupported type '{}'".format(type(behr_output)))

        # Pull out the keys of the BEHR output dictionary
        behr_output_keys = list(tempo_behr_dict.keys())

    # Open the TEMPO initialized dictionary
    if isinstance(tempo_pickle_path, str):
        with open(tempo_pickle_path, 'rb') as handle:
            tempo_init_dict = pickle.load(handle)
        handle.close()

    elif isinstance(tempo_pickle_path, dict):
        tempo_init_dict = tempo_pickle_path.copy()
        del tempo_pickle_path

    else:
        raise Exception("tempo_pickle_path is of unsupported type '{}'".format(type(tempo_pickle_path)))

    # Open the standard TEMPO product
    tempo_standard_ds = Dataset(tempo_ds_path, mode='r')

    try:
        if full_FOR:
            ms_min = 0
            ms_max = tempo_standard_ds['mirror_step'].size
            xt_min = 0
            xt_max = tempo_standard_ds['xtrack'].size

        else:
            if upper_bound_inclusive:
                ms_max += 1
                xt_max += 1

            # Adjust the mirrorstep bounds to account for the offset
            ms_granule_start = tempo_standard_ds['mirror_step'][:].min()
            ms_min = ms_min - ms_granule_start
            ms_max = ms_max - ms_granule_start

        # Open the table which corresponds BEHR and TEMPO variable names
        variable_table = pd.read_csv(vars_path)

        # It is recommended to read the from the TEMPO dataset when possible, in case error were introduced into TEMPO variables when performing the BEHR portion
        use_behr_for_standard_vars = False

        # Initiliaze dictionaries which will map dataset variable names to data
        tempo_data_main = {}
        tempo_data_coords = {}
        
        sp_variable_list = [
                        'longitude', 'latitude', 'time', 'viewing_zenith_angle', 'solar_zenith_angle', 'viewing_azimuth_angle', 'solar_azimuth_angle', 'relative_azimuth_angle', 'amf_stratosphere', 'amf_troposphere', 'amf_total', 'eff_cloud_fraction', 'amf_cloud_fraction', 'terrain_height', 'surface_pressure', 'albedo', 'amf_cloud_pressure', 'snow_ice_fraction', 'vertical_column_total', 'vertical_column_total_uncertainty', 'fitted_slant_column', 'fitted_slant_column_uncertainty', 'vertical_column_troposphere', 'vertical_column_troposphere_uncertainty', 'vertical_column_stratosphere', 'tropopause_pressure', 'main_data_quality_flag', 'ground_pixel_quality_flag', 'amf_diagnostic_flag', 'longitude_bounds', 'latitude_bounds', 'gas_profile', 'temperature_profile', 'scattering_weights', 'mirror_step', 'xtrack'
        ]

        behr_variable_list = [
            'MODISCloud', 'MODISAlbedo', 'MODISAlbedoQuality', 'MODISAlbedoFillFlag', 'GLOBETerrainHeight', 'AlbedoOceanFlag'
        ]

        misc_variable_list = [
            'BadGeoMask', 'i_mirror_step', 'i_xtrack'
        ]

        variable_attrs = {
            # Lowercase variable names come from the TEMPO standard product
            # Most of the following attributes are copied over from the TEMPO standard product attributes
            'longitude': {'units': 'degrees E', 'source': 'TEMPO'},
            'latitude': {'units': 'degrees N', 'source': 'TEMPO'},
            'time': {'source': 'TEMPO', 'description': 'seconds since 1980-01-06T00:00:00Z'},
            'viewing_zenith_angle': {'units': 'degree', 'source': 'TEMPO', 'description': 'Viewing zenith angle at pixel center'},
            'solar_zenith_angle': {'units': 'degree', 'source': 'TEMPO', 'description': 'Solar zenith angle at pixel center'},
            'viewing_azimuth_angle': {'units': 'degree', 'source': 'TEMPO', 'description': 'Viewing azimuth angle at pixel center'},
            'solar_azimuth_angle': {'units': 'degree', 'source': 'TEMPO', 'description': 'Solar azimuth angle at pixel center'},
            'relative_azimuth_angle': {'units': 'degrees', 'source': 'TEMPO', 'description': 'Relative azimuth angle at pixel center'},
            'amf_stratosphere': {'units': 1, 'source': 'TEMPO'},
            'amf_troposphere': {'units': 1}, 'source': 'TEMPO',
            'amf_total': {'units': 1, 'source': 'TEMPO'},
            'eff_cloud_fraction': {'units': 1, 'source': 'TEMPO', 'description': 'effective cloud fraction from cloud retrieval'},
            'amf_cloud_fraction': {'units': 1, 'source': 'TEMPO', 'description': 'cloud radiance fraction for AMF computation'},
            'snow_ice_fraction': {'units': 1, 'source': 'TEMPO', 'description': 'fraction of pixel area covered by snow and/or ice'},
            'terrain_height': {'units': 'm', 'source': 'TEMPO'},
            'surface_pressure': {'units': 'hPa', 'source': 'TEMPO'},
            'albedo': {'units': 1, 'source': 'TEMPO', 'description': 'surface albedo'},
            'amf_cloud_pressure': {'units': 'hPa', 'source': 'TEMPO', 'description': 'cloud pressure for AMF computation'},
            'vertical_column_total': {'units': 'molecules cm-2', 'source': 'TEMPO', 'description': 'nitrogen dioxide vertical column determined from fitted slant column and total AMF calculated from surface to top of atmosphere'},
            'vertical_column_total_uncertainty': {'units': 'molecules cm-2', 'source': 'TEMPO', 'description': 'nitrogen dioxide vertical column uncertainty'},
            'fitted_slant_column': {'units': 'molecules/cm^2', 'source': 'TEMPO'},
            'fitted_slant_column_uncertainty': {'units': 'molecules/cm^2', 'source': 'TEMPO'},
            'vertical_column_troposphere': {'units': 'molecules/cm^2', 'source': 'TEMPO'},
            'vertical_column_troposphere_uncertainty': {'units': 'molecules/cm^2', 'source': 'TEMPO'},
            'vertical_column_stratosphere': {'units': 'molecules/cm^2', 'source': 'TEMPO'},
            'tropopause_pressure': {'units': 'hPa', 'source': 'TEMPO'},
            'main_data_quality_flag': {'units': 1, 'source': 'TEMPO', 'flag_meanings': 'normal suspicious bad', 'flag_values': np.array([0, 1, 2], dtype=int)},
            'ground_pixel_quality_flag': {'source': 'TEMPO', 'flag_meanings': 'shallow_ocean land shallow_inland_water shoreline intermittent_water deep_inland_water continental_shelf_ocean deep_ocean land_water_error sun_glint_possibility solar_eclipse_possibility water evergreen_needleleaf_forest evergreen_broadleaf_forest deciduous_needleleaf_forest deciduous_broadleaf_forest mixed_forest closed_shrublands open_shrublands woody_savannas savannas grasslands permanent_wetlands croplands urban_and_built_up cropland_natural_vegetation_mosaic snow_and_ice barren_or_sparsely_vegetated unclassified fill_value', 'flag_values': np.array([0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 32, 0, 65536, 131072, 196608, 262144, 327680, 393216, 458752, 524288, 589824, 655360, 720896, 786432, 851968, 917504, 983040, 1048576, 16646144, 16711680], dtype=int)},
            'amf_diagnostic_flag': {'source': 'TEMPO', 'flag_meanings': 'good_AMF no_AMF glint climatological_cloud_pressure adjusted_surface_pressure adjusted_cloud_pressure no_albedo no_cloud_fraction no_gas_profile no_scattering_weights no_geolocation_information', 'flag_masks': np.array([1, 2, 4, 8, 16, 32, 1024, 2048, 4096, 8192, 16384], dtype=int)},
            'longitude_bounds': {'units': 'degrees E', 'source': 'TEMPO', 'description': 'longitude at pixel corners (SW,SE,NE,NW)'},
            'latitude_bounds': {'units': 'degrees N', 'source': 'TEMPO', 'description': 'latitude at pixel corners (SW,SE,NE,NW)'},
            'scattering_weights': {'units': 1, 'source': 'TEMPO', 'description': 'vertical profile of scattering weights'},
            'gas_profile': {'units': 'molecules/cm^2', 'source': 'TEMPO', 'description': 'vertical profile of nitrogen dioxide partial column (from GEOS-CF)'},
            'temperature_profile': {'units': 'K', 'source': 'TEMPO', 'description': 'air temperature'},
            'mirror_step': {'units': 1, 'source': 'TEMPO', 'description': 'scan mirror position index'},
            'xtrack': {'units': 1, 'source': 'TEMPO', 'description': 'pixel index along slit'},

            # Uppercase variable names are added by BEHR
            # Many of the following attributes are copied over from the BEHR NO2 product user guide
            'RelativeAzimuthAngle': {'units': 'degree', 'source': 'BEHR', 'description': 'Calculated as: 1) Take the absolute value of the difference of solar_azimuth_angle and viewing_azimuth_angle. 2) Where this angle is greater than 180 degrees, recalculate as 360 degrees minus this angle. 3) Take the absolute value of this angle minus 180 degrees.'},
            'MODISCloud': {'units': 1, 'source': 'BEHR', 'description': 'The cloud fraction of the pixel averaged from Aqua MODIS MYD06 cloud fraction data'},
            'MODISAlbedo': {'units': 1, 'source': 'BEHR', 'description': 'The average surface reflectance of the pixel calculated from Band 3 of the MCD43D combined MODIS BRDF product for the viewing geometry of this pixel.'},
            'MODISAlbedoQuality': {'source': 'BEHR', 'description': 'Indicates quality of albedo based on code in avg_modis_alb_to_pixels.m'},
            'MODISAlbedoFillFlag': {'units': 'bool', 'source': 'BEHR', 'description': 'Indicates if more than 50% of albedo quality values are fills for the pixel'},
            'GLOBETerrainHeight': {'units': 'hPa', 'source': 'BEHR', 'description': 'The average surface elevation of the pixel calculated from the GLOBE database'},
            'AlbedoOceanFlag': {'units': 'bool', 'source': 'BEHR'},
            'BadGeoMask': {'description': 'indicates positions where latitude/longitude values are invalid', 'source': 'BEHR'},
            'i_mirror_step': {'description': 'integer positions of mirror_step'},
            'i_xtrack': {'description': 'integer positions of xtrack'}
        }

        variable_dims = {
            'longitude': ['mirror_step', 'xtrack'],
            'latitude': ['mirror_step', 'xtrack'],
            'time': ['mirror_step'],
            'viewing_zenith_angle': ['mirror_step', 'xtrack'],
            'solar_zenith_angle': ['mirror_step', 'xtrack'],
            'viewing_azimuth_angle': ['mirror_step', 'xtrack'],
            'solar_azimuth_angle': ['mirror_step', 'xtrack'],
            'relative_azimuth_angle': ['mirror_step', 'xtrack'],
            'amf_stratosphere': ['mirror_step', 'xtrack'],
            'amf_troposphere': ['mirror_step', 'xtrack'],
            'amf_total': ['mirror_step', 'xtrack'],
            'eff_cloud_fraction': ['mirror_step', 'xtrack'],
            'amf_cloud_fraction': ['mirror_step', 'xtrack'],
            'terrain_height': ['mirror_step', 'xtrack'],
            'surface_pressure': ['mirror_step', 'xtrack'],
            'albedo': ['mirror_step', 'xtrack'],
            'amf_cloud_pressure': ['mirror_step', 'xtrack'],
            'snow_ice_fraction': ['mirror_step', 'xtrack'],
            'vertical_column_total': ['mirror_step', 'xtrack'],
            'vertical_column_total_uncertainty': ['mirror_step', 'xtrack'],
            'fitted_slant_column': ['mirror_step', 'xtrack'],
            'fitted_slant_column_uncertainty': ['mirror_step', 'xtrack'],
            'vertical_column_troposphere': ['mirror_step', 'xtrack'],
            'vertical_column_troposphere_uncertainty': ['mirror_step', 'xtrack'],
            'vertical_column_stratosphere': ['mirror_step', 'xtrack'],
            'tropopause_pressure': ['mirror_step', 'xtrack'],
            'main_data_quality_flag': ['mirror_step', 'xtrack'],
            'ground_pixel_quality_flag': ['mirror_step', 'xtrack'],
            'amf_diagnostic_flag': ['mirror_step', 'xtrack'],
            'longitude_bounds': ['mirror_step', 'xtrack', 'corner'],
            'latitude_bounds': ['mirror_step', 'xtrack', 'corner'],
            'scattering_weights': ['mirror_step', 'xtrack', 'swt_level'],
            'gas_profile': ['mirror_step', 'xtrack', 'swt_level'],
            'temperature_profile': ['mirror_step', 'xtrack', 'swt_level'],
            'mirror_step': ['mirror_step'],
            'xtrack': ['xtrack'],
            'RelativeAzimuthAngle': ['mirror_step', 'xtrack'],
            'MODISCloud': ['mirror_step', 'xtrack'],
            'MODISAlbedo': ['mirror_step', 'xtrack'],
            'MODISAlbedoQuality': ['mirror_step', 'xtrack'],
            'MODISAlbedoFillFlag': ['mirror_step', 'xtrack'],
            'GLOBETerrainHeight': ['mirror_step', 'xtrack'],
            'AlbedoOceanFlag': ['mirror_step', 'xtrack'],
            'BadGeoMask': ['mirror_step', 'xtrack'],
            'i_mirror_step': ['mirror_step'],
            'i_xtrack': ['xtrack']
        }



        # Create version of the variable table with the TEMPO variable name set as the index
        tempo_variable_table = variable_table.copy()
        tempo_variable_table = tempo_variable_table[tempo_variable_table['TEMPO Name'] != 'none']
        tempo_variable_table.set_index('TEMPO Name', inplace=True)

        # We need to iterate over the two variable lists defined above. At each step, we need to determine if the variable is in the BEHR dictionary or if we should read it from the dataset itself.
        try:
            for vr in (sp_variable_list + behr_variable_list + misc_variable_list):
                if verbose:
                    print(vr)
                # Start by assuming we will read the variable from the TEMPO dataset
                use_behr_var = False
                # The data does not exist until we pull it from one of the sources
                data_exists = False

                if vr in sp_variable_list:
                    if use_behr_for_standard_vars:
                        # Find the corresponding BEHR name
                        behr_vr_equivalent = variable_table[variable_table['TEMPO Name'] == vr]['OMI Variable']     

                        if len(behr_vr_equivalent) > 0:
                            assert len(behr_vr_equivalent) == 1, 'if there are any matches, there should only be one'
                            # Convert from series to string
                            behr_vr_equivalent = behr_vr_equivalent.item()

                            if behr_vr_equivalent in behr_output_keys:       
                                if verbose:        
                                    print('Using BEHR')
                                # Now we know that there is an equivalent BEHR variable and that it is present in the output file
                                use_behr_var = True

                                # Pull the data variable from the BEHR output
                                data = tempo_behr_dict[behr_vr_equivalent][ms_min:ms_max, xt_min:xt_max]
                                data_exists = True

                                # If we are reading one of the corner variables, the BEHR dimension order should be [corner, scanline, ground_pixel]
                                if (behr_vr_equivalent == 'TiledCornerLongitude') | (behr_vr_equivalent == 'TiledCornerLatitude'):
                                    assert data.shape[0] == 4, 'the first dimension of a BEHR corner variable should have size 4'
                                    # This means we have to change the axis order so the corner dimension is last: [scanline, ground_pixel, corner]
                                    data = np.moveaxis(data, [0, 1, 2], [2, 0, 1])

                    if not use_behr_var:
                        if verbose:
                            print("Using TEMPO")
                        # Pull the data from the TEMPO dataset

                        if (vr != 'mirror_step') & (vr != 'xtrack'):
                            tempo_var_location = tempo_variable_table.loc[vr, 'TEMPO Group']

                        if vr == 'mirror_step': 
                            data = tempo_standard_ds['mirror_step'][ms_min:ms_max]
                            data_exists = True

                        elif vr == 'xtrack': 
                            data = tempo_standard_ds['xtrack'][xt_min:xt_max]
                            data_exists = True

                        elif variable_dims[vr] == ['mirror_step']:
                            data = tempo_standard_ds[tempo_var_location][vr][ms_min:ms_max]
                            data_exists = True

                        elif variable_dims[vr] == ['mirror_step', 'xtrack']:
                            data = tempo_standard_ds[tempo_var_location][vr][ms_min:ms_max, xt_min:xt_max]
                            data_exists = True

                        elif variable_dims[vr] == ['mirror_step', 'xtrack', 'corner']:
                            data = tempo_standard_ds[tempo_var_location][vr][ms_min:ms_max, xt_min:xt_max, :]
                            data_exists = True

                        elif variable_dims[vr] == ['mirror_step', 'xtrack', 'swt_level']:
                            data = tempo_standard_ds[tempo_var_location][vr][ms_min:ms_max, xt_min:xt_max, :]
                            data_exists = True

                        else:
                            raise Exception('Unrecognized dimensions for variable {}:'.format(vr), variable_dims[vr])

                elif vr in behr_variable_list:
                    missing_behr_allowed = ['MODISCloud', 'GLOBETerrainHeight']
                    
                    if use_behr_output:
                        if verbose:
                            print('Using BEHR')
    
                        if vr in behr_output_keys: # Check that this variable is included in the BEHR output
                            try:
                                data = tempo_behr_dict[vr][ms_min:ms_max, xt_min:xt_max]
                                data_exists = True
    
                            except TypeError as e:
                                if vr in missing_behr_allowed:
                                    # Move on without loading in this variable
                                    data_exists = False
    
                                else:
                                    raise e
                    else:
                        if fill_behr_vars:
                            if vr in missing_behr_allowed:
                                data_exists = False

                            else:
                                # Add a dummy array with nan values
                                data = np.full((ms_max-ms_min, xt_max-xt_min), np.nan)
                                data_exists = True


                        else: # If we are not using the BEHR output nor filling values, we can move on
                            data_exists = False

                # Miscellaneous variables are generally available from the initialized dictionary
                elif vr in misc_variable_list:
                    if vr == 'BadGeoMask':
                        data_exists = True
                        if isinstance(tempo_init_dict[vr], float):
                            data = np.full((ms_max-ms_min, xt_max-xt_min), False, dtype=bool)

                        elif isinstance(tempo_init_dict[vr], np.ndarray):
                            data = tempo_init_dict[vr][ms_min:ms_max, xt_min:xt_max].astype(bool)

                    elif vr == 'RelativeAzimuthAngle':
                        data = tempo_init_dict[vr][ms_min:ms_max, xt_min:xt_max]
                        data_exists = True

                    elif vr == 'i_mirror_step':
                        data = np.arange(tempo_data_main['mirror_step'][1].size)
                        data_exists = True

                    elif vr == 'i_xtrack':
                        data = np.arange(tempo_data_main['xtrack'][1].size)
                        data_exists = True                    
                            
                    else:
                        raise Exception('Functionality does not exist for variable {}'.format(vr))
                        
                else:
                    raise Exception('Variable {} is not in sp_variable_list, behr_variable_list, or misc_variable_list'.format(vr))

                if data_exists:
                    if verbose:
                        print(data.shape)
                    if (vr == 'latitude') | (vr == 'longitude'):
                        # Add the dimensions, data, and attributes to the coordinate map
                        tempo_data_coords[vr] = (variable_dims[vr], data, variable_attrs[vr])

                    else:
                        # Add the dimensions, data, and attributes to the main data map
                        tempo_data_main[vr] = (variable_dims[vr], data, variable_attrs[vr])
                
                if verbose:
                    print('')

        except Exception as e:
            print(vr)
            print(data.shape)
            raise e

        #global_attributes = {
        #    'TEMPO_scan_num': tempo_standard_ds.scan_num,
        #    'TEMPO_granule_num': tempo_standard_ds.granule_num,
        #    'TEMPO_standard_id': tempo_standard_ds.local_granule_id,
        #    'TEMPO_version_id': tempo_standard_ds.version_id,
        #    'BEHR_Date': tempo_behr_dict['Date'],
        #    'BEHR_LatBdy': tempo_behr_dict['LatBdy'],
        #    'BEHR_LonBdy': tempo_behr_dict['LonBdy'],
        #    'BEHR_Region': tempo_behr_dict['BEHRRegion'],
        #    'BEHR_MODISCloudFiles': tempo_behr_dict['MODISCloudFiles'],
        #    'BEHR_MODISAlbedoFile': tempo_behr_dict['MODISAlbedoFile']
        #}

        granule = tempo_standard_ds.granule_num

        global_attributes = {
            'TEMPO_standard_id_G{:02d}'.format(granule): tempo_standard_ds.local_granule_id,
            'TEMPO_version_id_G{:02d}'.format(granule): tempo_standard_ds.version_id,
            'LatBdy_G{:02d}'.format(granule): tempo_init_dict['LatBdy'],
            'LonBdy_G{:02d}'.format(granule): tempo_init_dict['LonBdy'],
        }

        if use_behr_output:
            global_attributes['BEHR_Date_G{:02d}'.format(granule)] = tempo_behr_dict['Date']
            global_attributes['BEHR_Region_G{:02d}'.format(granule)] = tempo_behr_dict['BEHRRegion']
            global_attributes['BEHR_MODISCloudFiles_G{:02d}'.format(granule)] = tempo_behr_dict['MODISCloudFiles']
            global_attributes['BEHR_MODISAlbedoFile_G{:02d}'.format(granule)] = tempo_behr_dict['MODISAlbedoFile']

        tempo_combined_ds = xr.Dataset(tempo_data_main, coords=tempo_data_coords, attrs=global_attributes)

        tempo_standard_ds.close()

    except Exception as e:
        print('Preemptively closing active dataset')
        tempo_standard_ds.close()
        raise e

    return tempo_combined_ds