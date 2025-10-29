#### amf_update_one_scan.py ####

# Author: Sam Beaudry
# Last changed: 2025-10-05
# Location: Signal_Derived_Retrieval/TEMPO/main
# Contact: samuel_beaudry@berkeley.edu

################################

def amf_update_one_scan(PY_TO_MAT_SUITCASE, MAT_TO_PY_SUITCASE, scan_df, tempo_dir_head, vars_path, constants_path, save_path, minimize_output_size, full_FOR, N_updates: int=2, sdr_product: bool=True, pblh=750, hrrr_grib=None, save_path_partial="", git_commit="None", verbosity=5):
    '''
    Main function of the signal-derived retrieval. Takes TEMPO data, finds boundary layer/free troposphere division, and redistributes prior.

    Parameters
    ----------
    PY_TO_MAT_SUITCASE : str
        path to directory containing the pickle files from TEMPO_L2_NO2_on_date.py
    MAT_TO_PY_SUITCASE : str
        path to directory containing the pickle files from read_main_single.m (allowed to be empty)
    scan_df : pd.DataFrame
        DataFrame with column 'Name' and 'Granule' for each pickle file
    tempo_dir_head : str
        TEMPO directory head 
    vars_path : str
        path to the csv file containing TEMPO dataset variable names and locations
    constants_path : str
        path to constant values
        
    save_path : str
        location to save the completed dataset
    minimize_output_size : bool
        if True, will remove vertically-resolved variables when able to to reduce size of output dataset
    full_FOR : bool
        whether to process for the full field of regard
    N_updates : int (Optional)
        number of iterations to perform
    sdr_product : int (Optional)
        if True, will store the first iteration as the signal derived product
    pblh : float or str (Optional)
        in meters, height to use for planetary boundary layer or 'hrrr' to pull boundary layer height from reanalysis
    hrrr_grib : str (Optional) 
        path to HRRR grib files
    save_path_partial : str (Optional)
        path to save partially completed scan_ds when the function fails to finish
    git_commit : str (Optional)
        the commit of Signal_Derived_Retrieval repository used
    verbosity : int (Optional)
        controls print statements for debugging
    '''
    import os
    import re
    from datetime import datetime
    from datetime import timedelta
    import numpy as np
    import pandas as pd
    import xarray as xr
    import shapely
    import pickle
    import warnings

    from functions.synthesize_tempo_behr import synthesize_tempo_behr
    from functions.build_scan_ds import build_scan_ds
    from functions.add_pblh_hrrr import add_pblh_hrrr
    from functions.layer_heights import layer_heights
    from functions.trop_layer_index import trop_layer_index
    from functions.pixel_area import pixel_area
    from functions.temperature_correction import temperature_correction
    from functions.calc_scattering_weights_s5p import calc_scattering_weights_s5p
    from functions.cloud_lyr_amf_adjustment import cloud_lyr_amf_adjustment
    from functions.boundary_layer_index import boundary_layer_index
    from functions.a_priori_division import a_priori_division
    from functions.group_shared_priors import group_shared_priors
    from functions.great_circle_distance import great_circle_distance
    from functions.prepare_for_update import prepare_for_update
    from functions.amf_recursive_update_sf import amf_recursive_update
    from functions.amf_recursive_update_sf import amf_recursive_update_no_good_pixels
    from functions.build_geobounds_str import build_geobounds_str

    running_main_algorithm = False
    scan_ds_created = False


    # Function to save progress if an exception is encountered after preparing scan_ds
    def save_partial_ds():
        current_time = datetime.now()
        current_time_string = current_time.strftime('%Y%m%dT%H%M')

        try:
            partial_file_name = 'UNFINISHED-SDR-TEMPO_' + date_string + "_S{:03d}_".format(scan) + geobounds_str + '_n{:02d}_'.format(N_updates) + pblh_save_string + '_proc_' + current_time_string + '.nc'

        except Exception as e:
            print('Exception encountered while naming partial file:')
            print(e)
            print('')
            
            partial_file_name = 'UNFINISHED-SDR-TEMPO_proc_' + current_time_string + '.nc'
            print('Saving unfinished file as {}'.format(partial_file_name))

        if len(save_path_partial) > 0:
            if not os.path.exists(save_path_partial):
                os.makedirs(save_path_partial)
            
        scan_ds.to_netcdf(os.path.join(save_path_partial, partial_file_name), mode='w')
        print('Partially completed scan_ds stored at {}'.format(save_path_partial))

    try:
        # Set some options
        if pblh.lower() == 'hrrr':
            constant_boundary_layer_height = False

            if hrrr_grib is None:
                raise Exception("If running in 'hrrr' mode, path to grib data must be pass as 'hrrr_grib' argument")
            if not os.path.exists(hrrr_grib):
                raise Exception("Path for 'hrrr_grib' does not exist: {}".format(hrrr_grib))
            
            # Load HRRR coordinates from the constants folder
            hrrr_coords = "{}/hrrr_reanalysis_coordinates.nc".format(constants_path)
            
            if hrrr_coords is None:
                raise Exception("If running in 'hrrr' mode, path to coordinate data must be pass as 'hrrr_coords' argument")
            if not os.path.exists(hrrr_coords):
                raise Exception("Path for 'hrrr_coords' does not exist: {}".format(hrrr_coords))
            
            from herbie import Herbie

        else:
            constant_boundary_layer_height = True
            try:
                pblh_value = float(pblh)
                if pblh_value < 10:
                    raise ValueError("Choice of 'pblh' interpreted as {} m and is too low. Be sure to enter as meters.".format(pblh_value))

            except ValueError:
                raise ValueError("Argument for 'pblh', {}, could not be interpreted as 'hrrr' or a fixed height.".format(pblh))

        #############################################################
        #### Combine BEHR and Standard Datasets for each Granule ####
        #############################################################
        if verbosity > 2:
            print('Synthesizing BEHR and TEMPO data')

        grans_with_behr = len(scan_df['BEHR Name'].dropna())

        granule_dict = {}

        first_granule = True
        for granule in scan_df.index:
            # Open TEMPO pickle
            pickle_name = scan_df.loc[granule, 'TEMPO Name']
            tempo_pickle_path = '{}/{}'.format(PY_TO_MAT_SUITCASE, pickle_name)
            
            with open(tempo_pickle_path, 'rb') as handle:
                tempo_init_dict = pickle.load(handle)
            handle.close()        

            # Get dimension bounds
            ms_min = int(tempo_init_dict['MirrorStepBdy'][0])
            ms_max = int(tempo_init_dict['MirrorStepBdy'][1])
            xt_min = int(tempo_init_dict['XTrackBdy'][0])
            xt_max = int(tempo_init_dict['XTrackBdy'][1])
            
            original_file = tempo_init_dict['TEMPOProductID']
            scan = scan_df.loc[granule, 'Scan']
            # granule = granule
            
            date_string = re.search(r'^TEMPO_NO2_L2_V\d{2}_(\d{8})T\d{6}Z_S\d{3}G\d{2}\.nc$', original_file).group(1)
            year_str = date_string[:4]
            month_str = date_string[4:6]
            collection = re.search(r'^TEMPO_NO2_L2_(V\d{2})_\d{8}T\d{6}Z_S\d{3}G\d{2}\.nc$', original_file).group(1)

            date_of_interest = datetime.strptime(date_string, '%Y%m%d')
            
            original_file_path = '{}/NO2/L2/{}/{}/{}/{}'.format(tempo_dir_head, collection, year_str, month_str, original_file)
            
            if isinstance(scan_df.loc[granule, 'BEHR Name'], float):
                if np.isnan(scan_df.loc[granule, 'BEHR Name']):
                    if grans_with_behr == 0:
                        # None of the granules have BEHR data.
                        # No need to add BEHR fill values for the sake of combining the granule datasets
                        fill_behr_vars = False
                
                    else:
                        # Since other granules have BEHR data, we want to include those variables here
                        fill_behr_vars = True
                        
                    granule_ds = synthesize_tempo_behr(original_file_path, tempo_init_dict, vars_path, full_FOR, ms_min, ms_max, xt_min, xt_max, use_behr_output=False, fill_behr_vars=fill_behr_vars)
            
            
                else:
                    raise Exception('Unable to interpret value for BEHR Name')
            
            else:
                behr_pickle_path = '{}/{}'.format(MAT_TO_PY_SUITCASE, scan_df.loc[granule, 'BEHR Name'])
                granule_ds = synthesize_tempo_behr(original_file_path, tempo_init_dict, vars_path, full_FOR, ms_min, ms_max, xt_min, xt_max, use_behr_output=True, behr_output=behr_pickle_path)

            # Add a variable to associate these mirror_step values with the appropriate granule
            granule_ds['granule'] = (['mirror_step'], np.full(granule_ds['mirror_step'].shape, granule, dtype=int), {'description': 'granule number for TEMPO scan'})

            # Store dataset in the granule dataset dictionary
            granule_dict[granule] = granule_ds

            if first_granule:
                lat_domain = granule_ds.attrs['LatBdy_G{:02d}'.format(granule)]
                lon_domain = granule_ds.attrs['LonBdy_G{:02d}'.format(granule)]
                first_granule = False

        if full_FOR:
            geobounds_str = "full-FOR"

        else:
            geobounds_str = build_geobounds_str(lat_domain, lon_domain)

        #########################################################
        #### Concatenate Granules along Mirrorstep Dimension ####
        #########################################################
        if verbosity > 2:
            print('Building scan_ds')

        date_string = datetime.strftime(date_of_interest, '%Y%m%d')

        scan_ds = build_scan_ds(granule_dict, scan)
        scan_ds_created = True

        shape_2d = (scan_ds.mirror_step.size, scan_ds.xtrack.size)
        n_swt_levels = scan_ds.gas_profile.shape[2]

        #############################
        #### Set Operating Modes ####
        #############################

        # This section controls which sets of values are used to produce scattering weight 
        # profiles, and which scattering weight profiles to use in the update of 
        # air mass factors

        # Standard: use the scattering weights provided in the standard TEMPO L2 product
        # Validation: calculate scattering weights from the TROPOMI NO2 LUT using inputs of the standard TEMPO product
        # BEHR-A: same as Validation, except the surface albedo is replaced with MODISAlbedo added by BEHR

        # First case: some BEHR data available
        if grans_with_behr > 0:
            scattering_modes = {
                                "Validation": {"Albedo": "TEMPO", "Terrain": "TEMPO", "Cloud": "TEMPO"}, 
                                "BEHR_A": {"Albedo": "BEHR", "Terrain": "TEMPO", "Cloud": "TEMPO"}
            }

            update_modes = ["Standard", "Validation", "BEHR_A"]

            behr_mode = 'with_MODIS'

        # Second case: no BEHR data available
        else:
            scattering_modes = {
                        "Validation": {"Albedo": "TEMPO", "Terrain": "TEMPO", "Cloud": "TEMPO"}, 
            }

            update_modes = ["Standard", "Validation"]

            behr_mode = 'without_MODIS'

        # Options to avoid calculation of custom scattering weights
        custom_sw = False

        if not custom_sw:
            # If we don't compute our own scattering weights, we can only use the standard ones in the retrieval
            update_modes = ["Standard"]

        ##################
        #### Time UTC ####
        ##################
        if verbosity > 2:
            print('Converting to time UTC')

        def tempo_utc_conversion(seconds):
            tempo_time_ref = datetime(1980, 1, 6)
            
            if np.isnan(seconds):
                tempo_time = np.datetime64('NaT')

            else:
                tempo_time = tempo_time_ref + timedelta(seconds=seconds)
                
            return tempo_time

        tempo_utc_conversion_v = np.vectorize(tempo_utc_conversion)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # If we allow the following two lines to run with warnings, we will get a repeated warning
            # about converting non-nanosecond precision values to nanosecond precision. This is pretty much
            # irrelevant for the use of the time_utc variable, so we suppress the warning.

            time_utc = tempo_utc_conversion_v(scan_ds['time']).astype(dtype=np.datetime64)
            scan_ds['time_utc'] = (['mirror_step'], time_utc, {'description': 'UTC Time based on time delta given in standard time variable'})

        ####################################
        #### Add Boundary Layer Heights ####
        ####################################

        # ---------------------------------------------------------------------------------------------------------------------------------------

        if constant_boundary_layer_height:
            nearest_pblh = np.full(shape_2d, pblh_value, dtype='f8')
            pblh_var_description = 'constant boundary layer height imposed on retrieval of updated AMF'
            if verbosity > 2:
                print('Using constant boundary layer height of ' + str(pblh_value) + ' m')
            pblh_save_string = "fixed_bl_" + str(pblh_value)

        # ---------------------------------------------------------------------------------------------------------------------------------------

        else:
            if verbosity > 2:
                print('Determining boundary layer heights from colocated HRRR reanalysis')

            pblh_save_string = "variable_bl_HRRR"
            hrrr_coordinates = xr.open_dataset("{}/hrrr_reanalysis_coordinates.nc".format(constants_path))
            with open(os.path.join(constants_path, 'hrrr_domain.pickle'), 'rb') as handle:
                hrrr_polygon = pickle.load(handle)

            scan_ds = add_pblh_hrrr(scan_ds, hrrr_polygon, hrrr_coordinates, hrrr_grib, constant_pblh=False)
            
        ###############################################
        #### Prepare layer thicknesses and heights ####
        ###############################################

        if verbosity > 2:
            print('Calculating layer heights')

        with open('{}/tempo_eta_a.npy'.format(constants_path), 'rb') as handle:
            eta_a = np.load(handle)
        handle.close()

        with open('{}/tempo_eta_b.npy'.format(constants_path), 'rb') as handle:
            eta_b = np.load(handle)
        handle.close()

        scan_ds = layer_heights(scan_ds, eta_a, eta_b)

        ##########################################
        #### Determine Tropopause Layer Index ####
        ##########################################

        if verbosity > 2:
            print('Identifying tropopause layers')

        scan_ds = trop_layer_index(scan_ds)

        ##############################
        #### Calculate Pixel Area ####
        ##############################

        if verbosity > 2:
            print('Calculating pixel areas')
        # Determine the area covered by each pixel
        scan_ds = pixel_area(scan_ds)

        ###################################
        #### Custom Scattering Weights ####                                         
        ###################################

        # Store the temperature correction values (used to account for temperature dependence of absorption cross section)
        # Necessary for both standard and custom scattering weights
        temperature_corrections = temperature_correction(scan_ds['temperature_profile'].data, mode='s5p')
        scan_ds['TemperatureCorrection'] = (['mirror_step', 'xtrack', 'swt_level'], temperature_corrections, {'units': 1, 'description': 'The vectors of temperature correction values for use in the AMF calculation'})

        if custom_sw:
            if verbosity > 2:
                print('Calculating custom scattering weights')

            # Open the LUT
            bamf_lut_path = constants_path + "/S5P_OPER_LUT_NO2AMF_00000000T000000_99999999T999999_20160527T173500.nc"
            bamf_lut = xr.open_dataset(bamf_lut_path)

            # LUT takes 6 inputs parameters
            lut_inputs = {}

            # Set inputs that are invariant to calculation mode
            lut_inputs['solar_zenith_angle'] = scan_ds['solar_zenith_angle'].data
            lut_inputs['viewing_zenith_angle'] = scan_ds['viewing_zenith_angle'].data
            lut_inputs['RelativeAzimuthAngle'] = scan_ds['RelativeAzimuthAngle'].data
            lut_inputs['midpoint_pressures'] = scan_ds['midpoint_pressures'] # hPa

            # Set the cloud parameters
            lut_inputs['cloud_albedo'] = np.full(scan_ds['solar_zenith_angle'].shape, 0.8) # Consider changing to cloud_albedo_crb
            lut_inputs['cloud_pressure'] = scan_ds['amf_cloud_pressure'].data # hPa

            # Lookup scattering weights for each operating mode
            for mode in list(scattering_modes.keys()):
                print('----> {}'.format(mode))
                # Surface albedo
                if scattering_modes[mode]["Albedo"] == "TEMPO":
                    lut_inputs['surface_albedo'] = scan_ds['albedo'].data
                elif scattering_modes[mode]["Albedo"] == "BEHR":
                    lut_inputs['surface_albedo'] = scan_ds['MODISAlbedo'].data
                else:
                    raise ValueError("Invalid option for albedo: '{}'".format(scattering_modes[mode]["Albedo"]))
                
                # Surface pressure
                if scattering_modes[mode]["Terrain"] == "TEMPO":
                    lut_inputs['surface_pressure'] = scan_ds['surface_pressure'].data # hPa
                elif scattering_modes[mode]["Terrain"] == "GLOBE":
                    #terrain_height = scan_ds['GLOBETerrainHeight'].data
                    raise ValueError("Invalid option for terrain: '{}'".format(scattering_modes[mode]["Terrain"]))
                else:
                    raise ValueError("Invalid option for terrain: '{}'".format(scattering_modes[mode]["Terrain"]))
                
                # Cloud
                if scattering_modes[mode]["Cloud"] == "TEMPO":
                    lut_inputs['cloud_pressure'] = scan_ds['amf_cloud_pressure'].data # hPa
                    w = np.expand_dims(scan_ds['amf_cloud_fraction'].data, axis=2) # Cast to sw profile shape for IPA calculation
                else:
                    raise ValueError("Invalid option for terrain: '{}'".format(scattering_modes[mode]["Cloud"]))

                ###############
                #### CLEAR ####
                ###############
                scattering_weights_clear = calc_scattering_weights_s5p(
                                                                    bamf_lut,
                                                                    lut_inputs['midpoint_pressures'],
                                                                    lut_inputs['solar_zenith_angle'],
                                                                    lut_inputs['viewing_zenith_angle'],
                                                                    lut_inputs['RelativeAzimuthAngle'],
                                                                    lut_inputs['surface_albedo'],
                                                                    lut_inputs['surface_pressure'],
                                                                    linear_6d=True                                                            
                )

                ################
                #### CLOUDY ####
                ################
                if mode == 'Validation':
                    # For cloudy scattering weights, we need to adjust the cloud layer midpoint pressure
                    # since the cloud pressure is treated as the surface pressure in the LUT interpolation
                    lut_inputs['midpoint_pressures_cloudy'] = lut_inputs['midpoint_pressures'].copy()
                    p_cloud_greater_than_p_surf = np.full(lut_inputs['cloud_pressure'].shape, -9999, dtype=int)

                    lyrs = np.arange(72, dtype=int)

                    # For each pixel, determine which layer contains the cloud pressure
                    cloud_layer_index = np.full(lut_inputs['cloud_pressure'].shape, -9999, dtype=int)
                    cloud_layer_adjustment = np.full(lut_inputs['cloud_pressure'].shape, -9999, dtype=int)

                    for i in range(lut_inputs['cloud_pressure'].size):
                        ms, xt = np.unravel_index(i, lut_inputs['cloud_pressure'].shape)

                        cloud_pressure = lut_inputs['cloud_pressure'][ms, xt] # hPa
                        surface_pressure = lut_inputs['surface_pressure'][ms, xt] # hPa

                        if cloud_pressure < 0:
                            raise Exception('Cloud pressure is negative')
                        
                        elif np.isnan(cloud_pressure):
                            # Leave the index value as nan and don't change the cloud layer midpoint pressure
                            continue
                        
                        if cloud_pressure >= surface_pressure:
                            cloud_layer_index[ms, xt] = 0

                            # Flag this for later
                            p_cloud_greater_than_p_surf[ms, xt] = 1
                                    
                        else:
                            p_cloud_greater_than_p_surf[ms, xt] = 0
                            lower_interface_pressures = scan_ds['interface_pressures'].data[ms, xt, :-1] # hPa
                            lower_interface_pressures = np.insert(lower_interface_pressures, 0, surface_pressure)
                            upper_interface_pressures = scan_ds['interface_pressures'].data[ms, xt, :] # hPa

                            lyrs_filt = lyrs[ (lower_interface_pressures > cloud_pressure) & (upper_interface_pressures <= cloud_pressure) ]
                            # Since the cloud pressure is the cloud top pressure, I include it in the layer if it is the same as the upper interface pressure
                            if len(lyrs_filt) != 1:
                                raise Exception('Cloud not assigned to a single unique layer!')
                            
                            cld_lyr = lyrs_filt.item()
                            cloud_layer_index[ms, xt] = cld_lyr

                            # Now we have to define a new midpoint pressure for the cloud containing layer
                            cld_lyr_p_midpoint = (cloud_pressure + upper_interface_pressures[cloud_layer_index[ms, xt]]) / 2 # hPa
                            lut_inputs['midpoint_pressures_cloudy'][ms, xt, cloud_layer_index[ms, xt]] = cld_lyr_p_midpoint

                            # Find the cloud_layer_adjusment. This is a value on the domain [0, 1] which speicifies
                            # how much of the non-zero cloudy bAMF applies to the cloudy layer in the cloudy part of the pixel.

                            # Instead, if we have accurately computed clear and cloudy scattering weights, have the cloud radiance fraction,
                            # and know the true scattering weight profile, we can solve for the cloud_layer_adjustment of the provided retrieval.
                            cloud_layer_adjustment[ms, xt] = cloud_lyr_amf_adjustment(upper_interface_pressures[cld_lyr], lower_interface_pressures[cld_lyr], cloud_pressure)

                    scan_ds['cloud_layer_index'] = (['mirror_step', 'xtrack'], cloud_layer_index, {'units': 1, 'description': 'Vertical layer index corresponding to the layer containing cloud_pressure_crb.'})
                    scan_ds['cloud_p_greater_than_surf'] = (['mirror_step', 'xtrack'], p_cloud_greater_than_p_surf, {'description': '0: Cloud pressu$re is less than surface pressure. 1: Cloud pressure is greater than surface pressure.'})
                    scan_ds['cloud_layer_adjustment'] = (['mirror_step', 'xtrack'], cloud_layer_adjustment, {"units": 1, "description": "factor to apply to cloudy AMF fraction at cloud layer in order to account for portion of layer obscured by cloud. Valid from 0 to 1."})

                if mode == "BEHR-A":
                    # Altering the surface albedo doesn't change the cloudy scattering weights
                    scattering_weights_cloudy = scan_ds['ScatteringWeightsCloudy_Validation'].data

                else:
                    scattering_weights_cloudy = calc_scattering_weights_s5p(
                                                                        bamf_lut,
                                                                        lut_inputs['midpoint_pressures_cloudy'],
                                                                        lut_inputs['solar_zenith_angle'],
                                                                        lut_inputs['viewing_zenith_angle'],
                                                                        lut_inputs['RelativeAzimuthAngle'],
                                                                        lut_inputs['cloud_albedo'],
                                                                        lut_inputs['cloud_pressure'],
                                                                        linear_6d=True
                    )

                # Apply temperature corrections
                scattering_weights_clear = scattering_weights_clear * scan_ds['TemperatureCorrection'].data
                scattering_weights_cloudy = scattering_weights_cloudy * scan_ds['TemperatureCorrection'].data

                scan_ds['ScatteringWeightsClear_{}'.format(mode)] = (
                                                        ['mirror_step', 'xtrack', 'swt_level'], 
                                                        scattering_weights_clear, 
                                                        {
                                                            'units': 1, 
                                                            'description': 'The vectors of scattering weights for clear-sky (non-cloudy) conditions. Includes temperature correction.',
                                                            'albedo': scattering_modes[mode]["Albedo"],
                                                            'terrain': scattering_modes[mode]["Terrain"],
                                                            'cloud': scattering_modes[mode]["Cloud"]}
                                                        )
                scan_ds['ScatteringWeightsCloudy_{}'.format(mode)] = (
                                                        ['mirror_step', 'xtrack', 'swt_level'],
                                                        scattering_weights_cloudy, 
                                                        {
                                                            'units': 1, 
                                                            'description': 'The vectors of scattering weights for cloudy conditions. Includes temperature correction.',
                                                            'albedo': scattering_modes[mode]["Albedo"],
                                                            'terrain': scattering_modes[mode]["Terrain"],
                                                            'cloud': scattering_modes[mode]["Cloud"]}
                                                        )
                
                m_clr = scan_ds['ScatteringWeightsClear_{}'.format(mode)].data
                m_cld = scan_ds['ScatteringWeightsCloudy_{}'.format(mode)].data

                # Use the independent pixel approximation to get the combined profile
                scattering_weights_ipa = ((1-w) * m_clr) + (w * m_cld)

                # Use the previously determined cloud_layer_adjustment to find the final scattering weight profile
                for i in range(scan_ds['cloud_layer_adjustment'].size):
                    ms, xt = np.unravel_index(i, scan_ds['cloud_layer_adjustment'].shape)
                    cld_lyr = scan_ds['cloud_layer_index'].data[ms, xt]
                    
                    if np.isnan(cld_lyr):
                        # Don't adjust the IPA if we can't place the cloud layer
                        continue

                    elif w[ms, xt, 0] == 0:
                        # We don't have to do IPA if the cloud fraction is zero
                        continue

                    elif cld_lyr < 0:
                        # Issue with this being set as -9223372036854775808 instead of nan
                        continue

                    try:
                        scattering_weights_ipa[ms, xt, cld_lyr] = ((1-w[ms, xt, 0]) * m_clr[ms, xt, cld_lyr]) + (scan_ds['cloud_layer_adjustment'].data[ms, xt] * w[ms, xt, 0] * m_cld[ms, xt, cld_lyr])

                    except IndexError as e:
                        print(ms)
                        print(xt)
                        print('')
                        print(cld_lyr)
                        print(scan_ds['amf_cloud_pressure'].data[ms, xt])
                        scan_ds.to_netcdf('scan_ds.nc')
                        raise e

                scan_ds['ScatteringWeightsIPA_{}'.format(mode)] = (
                                                                    ['mirror_step', 'xtrack', 'swt_level'], 
                                                                    scattering_weights_ipa, 
                                                                    {
                                                                        "units": 1, 
                                                                        "description": "scattering weights determined from the S5P look-up table",
                                                                        'albedo': scattering_modes[mode]["Albedo"],
                                                                        'terrain': scattering_modes[mode]["Terrain"],
                                                                        'cloud': scattering_modes[mode]["Cloud"]
                                                                        }
                                                                        )   
            # End of scattering weight loop

        ###################################################
        #### Characterize Boundary Layer in TM5 Layers ####
        ###################################################
        if verbosity > 2:
            print('Characterizing boundary layer')

        # For conservation of mass in the update, we need to know the mass of NO2 available
        # for reallocation. To determine this, calculate the modeled boundary layer column

        # First, the index corresponding to the boundary layer height should be found
        scan_ds = boundary_layer_index(scan_ds)

        #################################################################
        #### Calculcate Boundary Layer and Tropospheric VCD of Prior ####
        #################################################################

        if verbosity > 2:
            print('Calculating model column values')

        scan_ds = a_priori_division(scan_ds)

        ######################################
        #### Group Pixels by Shared Prior ####
        ######################################

        if verbosity > 2:
            print('Finding shared prior groups')

        scan_ds = group_shared_priors(scan_ds)


        ############################################
        #### Run amf_recursive_update algorithm ####
        ############################################

        if verbosity > 2:
            print('Running recursive update')

        if full_FOR:
            # Restrict lat_domain and lon_domain to a rectangle containing the FOR
            lat_domain = np.array([15, 70])
            lon_domain = np.array([-180, -20])

        # Construct a DataFrame so that we can group pixels by nearest parallel and nearest meridian
        ms_mesh, xt_mesh = np.meshgrid(np.arange(scan_ds['mirror_step'].size), np.arange(scan_ds['xtrack'].size), indexing='ij')

        total_number_priors = np.arange(lat_domain[0], lat_domain[1]+0.25, 0.25).size * np.arange(lon_domain[0], lon_domain[1]+0.25, 0.25).size

        flat_shape_update_invariant = (scan_ds['amf_troposphere'].shape[0], scan_ds['amf_troposphere'].shape[1])
        flat_shape = (scan_ds['amf_troposphere'].shape[0], scan_ds['amf_troposphere'].shape[1], 3)
        vert_resolve_shape = (scan_ds['gas_profile'].shape[0], scan_ds['gas_profile'].shape[1], 3, scan_ds['gas_profile'].shape[2])

        for mode in update_modes:
            print('----> {}'.format(mode))
            # Set the scattering weight variable to use
            if mode == "Standard":
                sw_var = 'scattering_weights'

            else:
                sw_var = 'ScatteringWeightsIPA_{}'.format(mode)

            # Initialize final arrays
            no2_partial_columns_updated = np.full(vert_resolve_shape, np.nan, dtype=float)
            tropospheric_amf_updated = np.full(flat_shape, np.nan, dtype=float)
            no2_tropospheric_vcd_updated = np.full(flat_shape, np.nan, dtype=float)
            vcd_iteration_differences = np.full(flat_shape, np.nan, dtype=float)
            coverage_of_model_pixel = np.full(flat_shape_update_invariant, np.nan, dtype=float)
            proportion_free_troposphere = np.full(flat_shape_update_invariant, np.nan, dtype=float)
            removed_free_troposphere_in_practice = np.full(flat_shape, np.nan, dtype=float)
            update_quality_flags = np.full(flat_shape_update_invariant, 0, dtype=int)
            no2_boundary_layer_prior_updated = np.full(flat_shape, np.nan, dtype=float)
            retrieved_over_apriori_gridcell_arr = np.full(flat_shape_update_invariant, np.nan, dtype=float)
            retrieved_model_mismatch_flag_arr = np.full(flat_shape_update_invariant, -9999, dtype=int)

            iteration_record = np.array([0, 1, N_updates], dtype=int) # Added to avoid UnboundLocalError (presumably for when no good pixels are found for lat/lon pair

            # Each prior is distinguished by a lat/lon pair on the GEOS-CF grid

            # For full field of regard (FOR) runs, lat_domain and lon_domain can be set
            # arbitrarily large to ensure all pixels are selected for

            # We can quickly determine if a selected (lat, lon) coordinate is outside the FOR
            # using a rough outline of the FOR

            with open(os.path.join(constants_path, 'TEMPO_rough_FOR.pickle'), 'rb') as handle:
                field_of_regard = pickle.load(handle)

            for lat in np.arange(lat_domain[0], lat_domain[1]+0.25, 0.25):
                for lon in np.arange(lon_domain[0], lon_domain[1]+0.25, 0.25): 
                    running_main_algorithm = True

                    if shapely.geometry.Point(lon, lat).within(field_of_regard):
                        try:
                            # Round to 2 decimals. TEMPO data should already be rounded
                            lat = np.round(lat, decimals=2)
                            lon = np.round(lon, decimals=2)

                            ms_mesh, xt_mesh = np.meshgrid(np.arange(scan_ds.mirror_step.size), np.arange(scan_ds.xtrack.size), indexing='ij')

                            # Locate the matching indices for mirror_step and xtrack
                            prior_match_condition = (scan_ds['nearest_geoscf_latitude'].data == lat) & (scan_ds['nearest_geoscf_longitude'].data == lon)
                            prior_match_condition = prior_match_condition & ~scan_ds['BadGeoMask'].data
                            ms_match = ms_mesh[prior_match_condition]
                            xt_match = xt_mesh[prior_match_condition]

                            if len(xt_match) == 0:
                                continue

                            # Prepare information for recursive update
                            # The dimension match arrays are updated to remove any pixels with critical issues (i.e. missing AMF or VCD)
                            aru_args, ms_match_filt, xt_match_filt, quality_df = prepare_for_update(scan_ds, prior_match_condition, ms_match, xt_match, n_swt_levels, N_updates, sw_var)

                            ## Quality flags (bit-array)
                            # Flip so that the most severe issues are at the earlier bit positions 
                            quality_df = quality_df[quality_df.columns[::-1]]

                            bit_sign_series = pd.Series(np.tile('0b', len(quality_df)), dtype=str)
                            quality_series = bit_sign_series.str.cat(quality_df)
                            quality_series_int = quality_series.apply(lambda x: int(x, 2))

                            for p in range(len(ms_match)):
                                update_quality_flags[ms_match[p], xt_match[p]] = quality_series_int.loc[p]

                            ########################
                            # Perform the AMF update
                            # In prepare_for_update, we determined if there are any good pixels
                            if aru_args['good_pixels'].size == 0:
                                if len(ms_match_filt) == 0:
                                    # If we also have no pixels to work with at all, move on from this prior
                                    continue

                                else:
                                    # Otherwise, calculate values of AMF and VCD for the "initial" step,
                                    # even though we have no good pixels to update the prior
                                    apriori_partial_columns, trop_amfs, retrieved_trop_vcd, iteration_record, vcd_iter_diff, portion_free_trop, removed_ft_in_practice, retrieved_over_apriori_gridcell, retrieved_model_mismatch_flag = amf_recursive_update_no_good_pixels(**aru_args)

                            else:
                                # If we do have good pixels, run the full recursive update with changes to the 
                                # prior at each iteration
                                apriori_partial_columns, trop_amfs, retrieved_trop_vcd, iteration_record, vcd_iter_diff, portion_free_trop, removed_ft_in_practice, retrieved_over_apriori_gridcell, retrieved_model_mismatch_flag  = amf_recursive_update(**aru_args)
                            ########################

                            # Rearrange axes so that the iteration dimension is after the pixel dimension
                            apriori_partial_columns = np.moveaxis(apriori_partial_columns, [0, 1, 2], [1, 0, 2])
                            trop_amfs = trop_amfs.T
                            retrieved_trop_vcd = retrieved_trop_vcd.T
                            vcd_iter_diff = vcd_iter_diff.T

                            # To evaluate ability of observation to update the prior, we can consider
                            # what percentange of the model pixel was observed with good quality
                            # Compute model area as a trapezoid
                            top_length = great_circle_distance(lat + 0.125, lon - 0.125, lat + 0.125, lon + 0.125, units='m', float_mode=True)
                            bottom_length = great_circle_distance(lat - 0.125, lon - 0.125, lat - 0.125, lon + 0.125, units='m', float_mode=True)
                            height = great_circle_distance(lat - 0.125, lon, lat + 0.125, lon, units='m')
                            model_area = ((top_length + bottom_length) / 2) * height

                            coverage_of_model_pixel_value = np.sum(aru_args['pixel_area'][aru_args['good_pixels']]) / model_area

                            # Reconstruct data into the [mirror_step, xtrack] format
                            for p in range(len(ms_match_filt)):
                                ms = ms_match_filt[p]
                                xt = xt_match_filt[p]
                                
                                no2_partial_columns_updated[ms, xt, :, :] = apriori_partial_columns[p, :, :] # [pixel, iteration, lev]
                                tropospheric_amf_updated[ms, xt, :] = trop_amfs[p, :] # [pixel, iteration]
                                no2_tropospheric_vcd_updated[ms, xt, :] = retrieved_trop_vcd[p, :]
                                vcd_iteration_differences[ms, xt, :] = vcd_iter_diff[p, :] # [pixel, iteration]
                                coverage_of_model_pixel[ms, xt] = coverage_of_model_pixel_value
                                proportion_free_troposphere[ms, xt] = portion_free_trop
                                removed_free_troposphere_in_practice[ms, xt, :] = removed_ft_in_practice
                                retrieved_over_apriori_gridcell_arr[ms, xt] = retrieved_over_apriori_gridcell
                                retrieved_model_mismatch_flag_arr[ms, xt] = retrieved_model_mismatch_flag

                                # Change units back from mol/m^2 to molecules/cm^2
                                no2_partial_columns_updated[ms, xt, :, :] = no2_partial_columns_updated[ms, xt, :, :] * 6.022e+19
                                no2_tropospheric_vcd_updated[ms, xt, :] = no2_tropospheric_vcd_updated[ms, xt, :] * 6.022e+19
                                removed_free_troposphere_in_practice[ms, xt, :] = removed_free_troposphere_in_practice[ms, xt, :] * 6.022e+19

                                # 2025-04-24: Added boundary layer prior so that user can reconstruct the prior from the original gas profile and boundary_layer_idnex
                                # even if they are lacking the updated gas_profile
                                no2_boundary_layer_prior_updated[ms, xt, :] = np.sum(
                                                                                    no2_partial_columns_updated[ms, xt, :, :(scan_ds.boundary_layer_index.data[ms, xt]+1)], 
                                                                                    axis=1
                                                                                    ) # molecules/cm^2

                        except Exception as e:
                            debug_mode = True
                            if debug_mode:
                                raise e
                            
                            else:
                                print('Issue performing update on this prior')


            # Store the new values in the dataset

            scan_ds['gas_profile_updated_{}'.format(mode)] = (
                                                                ['mirror_step', 'xtrack', 'iteration', 'swt_level'],
                                                                no2_partial_columns_updated,
                                                                {
                                                                    'units': 'molecules/cm^2',
                                                                    'description': 'updated iterations of no2_partial_columns'
                                                                }
            )

            scan_ds['amf_troposphere_updated_{}'.format(mode)] = (
                                                                ['mirror_step', 'xtrack', 'iteration'],
                                                                tropospheric_amf_updated,
                                                                {
                                                                    'units': 1,
                                                                    'description': 'updated iterations of tropospheric_amf'
                                                                }
            )

            scan_ds['vertical_column_troposphere_updated_{}'.format(mode)] = (
                                                                ['mirror_step', 'xtrack', 'iteration'],
                                                                no2_tropospheric_vcd_updated,
                                                                {
                                                                    'units': 'molecules/cm^2',
                                                                    'description': 'updated iterations of no2_tropospheric_vcd'
                                                                }
            )

            scan_ds['vcd_iteration_differences_{}'.format(mode)] = (
                                                                ['mirror_step', 'xtrack', 'iteration'],
                                                                vcd_iteration_differences,
                                                                {
                                                                    'units': '%',
                                                                    'description': 'percent difference between vertical_column_troposphere_updated compared to previous iteration'
                                                                }
            )

            scan_ds['coverage_of_model_pixel_{}'.format(mode)] = (
                                                                ['mirror_step', 'xtrack'],
                                                                coverage_of_model_pixel, 
                                                                {
                                                                    'units': '1',
                                                                    'description': 'percent of model pixel area used in the update process'
                                                                }
            )

            scan_ds['proportion_free_troposphere_{}'.format(mode)] = (
                                                                ['mirror_step', 'xtrack'],
                                                                proportion_free_troposphere, 
                                                                {
                                                                    'units': '1',
                                                                    'description': 'proportion of tropospheric NO2 in the coarse model resolution (1 degree) allocated to the free troposphere'
                                                                }
            )

            scan_ds['removed_free_troposphere_in_practice_{}'.format(mode)] = (
                                                                ['mirror_step', 'xtrack', 'iteration'],
                                                                removed_free_troposphere_in_practice,
                                                                {
                                                                    'units': 'molecules/cm^2',
                                                                    'description': 'vertical column density removed as the free tropospheric portion during the spatial shape-factor calculation'
                                                                }
            )

            scan_ds['update_quality_flags_{}'.format(mode)] = (
                                                                ['mirror_step', 'xtrack'],
                                                                update_quality_flags,
                                                                {
                                                                    'description': 'bit flag indicating quality of pixel for update algorithm',
                                                                    'bit_positions': np.flip(np.arange(quality_df.shape[1])),
                                                                    'bit_meanings': list(quality_df.columns)
                                                                }
            )

            scan_ds['model_no2_boundary_layer_vcd_updated_{}'.format(mode)] = (
                                                                                ['mirror_step', 'xtrack', 'iteration'],
                                                                                no2_boundary_layer_prior_updated,
                                                                                {
                                                                                    'units': 'molecules/cm^2',
                                                                                    'description': 'Sum of updated gas profile over boundary layer levels. Use in combination with boundary_layer_index and gas_profile to reconstruct gas_profile_updated.',
                                                                                    'vertical_shape_mode': 'GEOS-CF Vertical Shape Factor'
                                                                                }
            )

            scan_ds['retrieved_over_apriori_gridcell_{}'.format(mode)] = (
                                                                            ['mirror_step', 'xtrack'],
                                                                            retrieved_over_apriori_gridcell_arr,
                                                                            {
                                                                                'units': 1,
                                                                                'description': 'Ratio of retrieved trop VCD to a priori trop VCD at model grid cell resolution'
                                                                            }
            )

            scan_ds['retrieved_model_mismatch_flag_{}'.format(mode)] = (
                                                                            ['mirror_step', 'xtrack'],
                                                                            retrieved_model_mismatch_flag_arr,
                                                                            {
                                                                                'description': 'flag indicating if severe disagreement exists between retrieved trop VCD and trop VCD modeled by GEOS-CF based on if retrieved_over_apriori_gridcell exceeds a factor of 3',
                                                                                'values': np.array([-1, 0, 1, -9999], dtype=int),
                                                                                'meanings': ['model underestimates', 'good', 'model overestimates', 'fill_value']
                                                                            }
            )

        scan_ds['iteration'] = (
                                        ['iteration'],
                                        iteration_record
        )

        #################################
        #### Updated Total AMFs/VCDs ####
        #################################

        for mode in update_modes:
            # Set the scattering weight variable to use
            if mode == "Standard":
                sw_var = 'scattering_weights'

            else:
                sw_var = 'ScatteringWeightsIPA_{}'.format(mode)

            v = scan_ds['gas_profile_updated_{}'.format(mode)].data # ['mirror_step', 'xtrack', 'iteration', 'swt_level']
            m = np.expand_dims(scan_ds[sw_var].data, axis=2) # ['mirror_step', 'xtrack', 'iteration', 'swt_level']
            c = np.expand_dims(scan_ds['TemperatureCorrection'].data, axis=2) # ['mirror_step', 'xtrack', 'iteration', 'swt_level']

            # Invalid values for the prior should be marked by NaNs
            bad_v = np.any(np.isnan(v), axis=3) # ['mirror_step', 'xtrack', 'iteration']

            amf_total_upd = np.where(bad_v, np.nan, np.sum(v * m * c, axis=3) / np.sum(v, axis=3)) # ['mirror_step', 'xtrack', 'iteration']
            vcd_total_upd = np.where(bad_v, np.nan, np.expand_dims(scan_ds.fitted_slant_column.data, axis=2) / amf_total_upd )

            scan_ds['amf_total_updated_{}'.format(mode)] = (
                                                            ['mirror_step', 'xtrack', 'iteration'],
                                                            amf_total_upd,
                                                            {
                                                                'units': 1,
                                                                'description': 'updated iterations of total AMF'
                                                            }
            )

            scan_ds['vertical_column_total_updated_{}'.format(mode)] = (
                                                                        ['mirror_step', 'xtrack', 'iteration'],
                                                                        vcd_total_upd,
                                                                        {
                                                                            'units': 'molecules/cm^2',
                                                                            'descritption': 'updated iterations of total VCD'
                                                                        }
            )

        ################
        #### Saving ####
        ################

        # Removal of vertically-resolved variables if indicated by user
        if minimize_output_size:
            from functions.prune_dataset import prune_dataset

            # With this setting, all vertically resolved variables (aside from custom scattering weights)
            # will be removed. This includes original variables which cannot otherwise be reconstructed, but
            # which are available in the original datasets
            scan_ds = prune_dataset(scan_ds, update_modes, remove_originals=True)

        # Naming of output file

        current_time = datetime.now()
        current_time_string = current_time.strftime('%Y%m%dT%H%M')

        new_file_name = 'SDR-TEMPO_' + date_string + "_S{:03d}_".format(scan) + geobounds_str + '_n{:02d}_'.format(N_updates) + pblh_save_string + '_proc_' + current_time_string + '.nc'

        global_attrs = {
            "version": "2stat",
            "version_notes": "Retrieval version 2stat is version 2b for TEMPO. 2b retains the vertical shape factors of no2 partial columns from GEOS-CF. Unlike version 2, the layer containing the boundary layer pause is uniformly treated as part of the free troposphere in this version. The removed free tropospheric VCD is allowed to be negative, if one of the pixels with main_data_quality flag == 0 has a negative retrieved VCD.",
            "Signal_Derived_Retrieval__commit": git_commit
        }

        scan_ds = scan_ds.assign_attrs(global_attrs)
        
        # behr_mode sub path is not included since it was set as part of save_path in the director script.
        final_save_path = os.path.join(save_path, geobounds_str, year_str, month_str)
        
        # Example: for a file generated on the square CONUS bounds with no BEHR-MODIS data for April 2024:
        # save_path/without_MODIS/lat_N25-N50_lon_W125-W065/2024/04/SDR-TEMPO...

        # Check if the sub-directories exist. If they do not, then make them.
        if not os.path.exists(final_save_path):
            os.makedirs(final_save_path)

        final_save_name = os.path.join(final_save_path, new_file_name)

        scan_ds.to_netcdf(final_save_name, mode='w')


    except AssertionError as ae:
        print('AssertionError encountered while processing scan. Will not finish process for this scan.')
        print('{}'.format(repr(ae))) 

        if running_main_algorithm:
            print('GEOS-CF grid cell at point of failure (lat, lon):')
            print('({}, {})'.format(lat, lon))

        if scan_ds_created:
            save_partial_ds()

    except Exception as e:
        print('General Exception encountered while processing scan. Will not finish process for this scan.')
        print('{}'.format(repr(e)))

        if running_main_algorithm:
            print('GEOS-CF grid cell at point of failure (lat, lon):')
            print('({}, {})'.format(lat, lon))

        if scan_ds_created:
            save_partial_ds()