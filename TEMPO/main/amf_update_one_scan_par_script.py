#### amf_update_one_scan_par_script.py ####

# Author: Sam Beaudry
# Last changed: 2025-10-05
# Location: Signal_Derived_Retrieval/TEMPO/main
# Contact: samuel_beaudry@berkeley.edu

###########################################

'''
Main function of the signal-derived retrieval, parallelized version. Takes TEMPO data, finds boundary layer/free troposphere division, and redistributes prior.

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
num_engines : int
    the number of engines to distribute work across for parallelized tasks
N_updates : int (Optional)
    number of iterations to perform
pblh : float or str (Optional)
    in meters, height to use for planetary boundary layer or 'hrrr' to pull boundary layer height from reanalysis
hrrr_grib : str (Optional) 
    path to HRRR grib files
save_path_partial : str (Optional)
    path to save partially completed scan_ds when the function fails to finish
git_commit : str (Optional)
    the commit of Singal_Derived_Retrieval repository used
verbosity : int (Optional)
    controls print statements for debugging
'''

import argparse

# Define arguments for the script
parser = argparse.ArgumentParser()
parser.add_argument('--scan_df_file', type=str)
parser.add_argument('--PY_TO_MAT_SUITCASE', type=str)
parser.add_argument('--MAT_TO_PY_SUITCASE', type=str)
parser.add_argument('--tempo_dir_head', type=str)
parser.add_argument('--vars_path', type=str)
parser.add_argument('--constants_path', type=str)
parser.add_argument('--save_path', type=str)
parser.add_argument('--minimize_output_size', type=int)
parser.add_argument('--full_FOR', type=int)
parser.add_argument('--num_engines', type=int)
parser.add_argument('--N_updates', type=int)
parser.add_argument('--pblh', type=str)
parser.add_argument('--hrrr_grib', type=str)
parser.add_argument('--save_path_partial', type=str)
parser.add_argument('--git_commit', type=str)
parser.add_argument('--verbosity', type=int)

# Read from command line
args = vars(parser.parse_args())

# Parse arguments 
scan_df_file = args['scan_df_file']
PY_TO_MAT_SUITCASE = args['PY_TO_MAT_SUITCASE']
MAT_TO_PY_SUITCASE = args['MAT_TO_PY_SUITCASE']
tempo_dir_head = args['tempo_dir_head']
vars_path = args['vars_path']
constants_path = args['constants_path']
save_path = args['save_path']
minimize_output_size = bool(int(args['minimize_output_size']))
full_FOR = bool(int(args['full_FOR']))
num_engines = int(args['num_engines'])
N_updates = int(args['N_updates'])
pblh = args['pblh']
hrrr_grib = args['hrrr_grib']
save_path_partial = args['save_path_partial']
git_commit = args['git_commit']
verbosity = int(args['verbosity'])

# Open DataFrame with the TEMPO files to be used
import pandas as pd
scan_df = pd.read_csv(scan_df_file)
scan_df.set_index('Granule', inplace=True)

# Import remaining modules
import os
import re
from datetime import datetime
from datetime import timedelta
import numpy as np
import xarray as xr
import shapely
import pickle
import warnings
import ipyparallel as ipp

from functions.great_circle_distance import great_circle_distance
from functions.build_geobounds_str import build_geobounds_str
from functions.synthesize_tempo_behr import synthesize_tempo_behr

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
        
    #############################
    #############################
    #### DATASET PREPARATION ####
    #############################
    #############################

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

    from functions.build_scan_ds import build_scan_ds
    scan_ds = build_scan_ds(granule_dict, scan)
    scan_ds_created = True

    shape_2d = (scan_ds.mirror_step.size, scan_ds.xtrack.size)
    n_swt_levels = scan_ds.gas_profile.shape[2]


    ############################
    ############################
    #### STANDARD EXECUTION ####
    ############################
    ############################

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


    ######################################
    #### Group Pixels by Shared Prior ####
    ######################################

    if verbosity > 2:
        print('Finding shared prior groups')

    from functions.group_shared_priors import group_shared_priors
    scan_ds = group_shared_priors(scan_ds)


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

    from functions.layer_heights import layer_heights
    scan_ds = layer_heights(scan_ds, eta_a, eta_b)


    ##############################
    #### Calculate Pixel Area ####
    ##############################

    if verbosity > 2:
        print('Calculating pixel areas')
    # Determine the area covered by each pixel
    from functions.pixel_area import pixel_area
    scan_ds = pixel_area(scan_ds)


    ###################################
    #### Custom Scattering Weights ####                                         
    ###################################

    # Store the temperature correction values (used to account for temperature dependence of absorption cross section)
    # Necessary for both standard and custom scattering weights

    from functions.temperature_correction import temperature_correction
    temperature_corrections = temperature_correction(scan_ds['temperature_profile'].data, mode='s5p')
    scan_ds['TemperatureCorrection'] = (['mirror_step', 'xtrack', 'swt_level'], temperature_corrections, {'units': 1, 'description': 'The vectors of temperature correction values for use in the AMF calculation'})

    if custom_sw:
        from functions.calc_scattering_weights_s5p import calc_scattering_weights_s5p
        from functions.cloud_lyr_amf_adjustment import cloud_lyr_amf_adjustment

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
        

    ############################
    ############################
    #### PARALLEL EXECUTION ####
    ############################
    ############################

    # Start the client and engines
    mycluster = ipp.Cluster(n = num_engines) # Cluster
    c = mycluster.start_and_connect_sync() # Client

    # Assign a DirectView of the engines to the name "dview"
    dview = c[:]
    # Block execution
    dview.block = True

    dview.execute('import numpy as np')
    dview.execute('import pandas as pd')

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
        pblh_var_description = 'boundary layer height from nearest HRRR analysis location'
        hrrr_coordinates = xr.open_dataset("{}/hrrr_reanalysis_coordinates.nc".format(constants_path))
        with open(os.path.join(constants_path, 'hrrr_domain.pickle'), 'rb') as handle:
            hrrr_polygon = pickle.load(handle)

                                                                    
        from scipy.interpolate import NearestNDInterpolator
        from shapely.geometry import Point

        hrrr_save_directory = hrrr_grib
        pblh_value = 750. # m
        
        # Load in the coordinates of HRRR cells and flatten to 1D arrays
        hrrr_lats = hrrr_coordinates.latitude.data.flatten()
        hrrr_lons = hrrr_coordinates.longitude.data.flatten()

        # Chunk the data based on the closest hour (since HRRR analysis is stepped at 1 hr)
        all_times = scan_ds.time_utc.data # [mirror_step]
        time_df = pd.DataFrame({'Time UTC': all_times}, index=scan_ds.mirror_step)
        time_df.index.rename('mirror_step', inplace=True)

        # dropna is included since not all granules may be included in every scan
        time_df = time_df.dropna()

        time_df['Bottom Hour'] = time_df['Time UTC'].dt.hour
        time_df['Nearest Hour'] = time_df['Time UTC'].apply(lambda t: (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)+timedelta(hours=t.minute//30)).hour)
        time_df['Day'] = time_df['Time UTC'].dt.day
        time_df['Month'] = time_df['Time UTC'].dt.month
        time_df['Year'] = time_df['Time UTC'].dt.year

        # If nearest hour is lower than the bottom hour, we know that we switched a day and have to increase by one
        time_df['Day'] = time_df['Day'].where(time_df['Bottom Hour'] <= time_df['Nearest Hour'], time_df['Day'] + 1)

        unique_nearest_hours = time_df['Nearest Hour'].unique()

        # Right now I'm using a loop to manually assign values to cells based on mirrorstep value instead of passing arrays to the interpolator
        # This is because:
        #     1) Not all scans have all granules, so there are nan placeholders that get removed and offset the indexing
        #     2) Some TEMPO pixels lack geolocation and also have to be skipped entirely
        #     3) Nearest hour may straddle the scan dimension, making it complicated to move between 1D and 2D spaces

        # Start by filling with nan
        nearest_pblh = np.full((scan_ds['mirror_step'].size, scan_ds['xtrack'].size), np.nan, dtype='f8')

        for h in unique_nearest_hours:
            ms_during_h = time_df[time_df['Nearest Hour'] == h]

            first_ms = ms_during_h.index[0]

            year_in = ms_during_h.loc[first_ms, 'Year']
            month_in = ms_during_h.loc[first_ms, 'Month']
            day_in = ms_during_h.loc[first_ms, 'Day']

            print("Opening UTC Hour ", h)
            hrrr_time = "{year}-{month}-{day} {hour}:00".format(year = '%04d' % year_in, month = '%02d' % month_in, day = '%02d' % day_in, hour = '%02d' % h)
            H = Herbie(
                hrrr_time, # model run date
                model="hrrr", # model name
                product="sfc", # model product name (model dependent)
                fxx=0, # forecast lead time
                save_dir=hrrr_save_directory
                )
            hrrr_pbl = H.xarray(":HPBL:surface:anl", remove_grib=False)

            # Flatten pbl to 1D array
            hrrr_pbl_1D = hrrr_pbl.blh.data.flatten()

            # Create interpolator
            nearest_hrrr_interpolator = NearestNDInterpolator(list(zip(hrrr_lons, hrrr_lats)), hrrr_pbl_1D)

            # Scatter data
            ms_values = ms_during_h.index
            if not np.all(np.diff(ms_values) == 1):
                raise Exception('mirror_step values should monotonically increase')

            num_values = len(ms_during_h)
            values_per_engine = int(np.floor(num_values / num_engines))
            remainder = num_values % num_engines

            start_ms = ms_values[0]
            for eng in range(num_engines):
                end_ms = start_ms + values_per_engine

                if remainder > 0:
                    end_ms += 1
                    remainder -= 1

                latitude_subset = scan_ds['latitude'].sel(mirror_step=range(start_ms, end_ms)).data
                longitude_subset = scan_ds['longitude'].sel(mirror_step=range(start_ms, end_ms)).data
                badgeomask_subset = scan_ds['BadGeoMask'].sel(mirror_step=range(start_ms, end_ms)).data

                dview.push(dict(latitude_subset=latitude_subset, longitude_subset=longitude_subset, badgeomask_subset=badgeomask_subset), targets=eng)

                start_ms = end_ms

            # Push variables shared by all workers
            dview.push(dict(
                Point=Point,
                hrrr_polygon=hrrr_polygon,
                nearest_hrrr_interpolator=nearest_hrrr_interpolator,
                pblh_value=pblh_value,
                xtrack_size=scan_ds['xtrack'].size
            ))

            # Call the function on the workers
            def pblh_hrrr_loop():
                import numpy as np
                
                nearest_pblh_subset = np.full(latitude_subset.shape, np.nan, dtype='f8')

                for i_ms in range(latitude_subset.shape[0]):
                    for i_xt in range(latitude_subset.shape[1]):
                        # First, check if the geolocation flag is false so we can proceed
                        if not badgeomask_subset[i_ms, i_xt]:
                            sat_lat = latitude_subset[i_ms, i_xt]
                            sat_lon = longitude_subset[i_ms, i_xt]

                            if Point(sat_lon, sat_lat).within(hrrr_polygon):
                                nearest_pblh_subset[i_ms, i_xt] = nearest_hrrr_interpolator(sat_lon, sat_lat)

                            else:
                                # Use the default value of 750 m
                                nearest_pblh_subset[i_ms, i_xt] = pblh_value


                return nearest_pblh_subset

            nearest_pblh_subset_list = dview.apply_sync(pblh_hrrr_loop)

            # Transform worker output to the PBLH array
            # Reset remainder and starting index
            remainder = num_values % num_engines
            start_ms = ms_values[0]
            start_i_ms = scan_ds.i_mirror_step.sel(mirror_step=start_ms).item()
            
            for eng in range(num_engines):
                end_i_ms = start_i_ms + values_per_engine

                if remainder > 0:
                    end_i_ms += 1
                    remainder -= 1

                nearest_pblh[start_i_ms:end_i_ms, :] = nearest_pblh_subset_list[eng]

                start_i_ms = end_i_ms 

            hrrr_pbl.close()

    # ---------------------------------------------------------------------------------------------------------------------------------------

    scan_ds['boundary_layer_height'] = (['mirror_step', 'xtrack'], nearest_pblh, {'units': 'm', 'description': pblh_var_description})


    ##########################################
    #### Determine Tropopause Layer Index ####
    ##########################################

    if verbosity > 2:
        print('Identifying tropopause layers')

    layer_indices = np.arange(72, dtype=int)

    dview.push(dict(layer_indices=layer_indices))
    dview.scatter('trop_pres_array', scan_ds.tropopause_pressure.data.flatten())
    dview.scatter('interface_pres_array', scan_ds.interface_pressures.data.reshape((scan_ds.tropopause_pressure.size, 72)))
    dview.scatter('surf_pres_array', scan_ds.surface_pressure.data.flatten())

    def tropopause_layer_index_loop():
        import numpy as np
        
        num_pixels = trop_pres_array.size

        # Since we are working with integers, use a fill value rather than nan
        trop_layer_index = np.full(trop_pres_array.shape, -9999, dtype=int)

        for i in range(num_pixels):
            trop_pres = trop_pres_array[i] # hPa
            interface_pres = interface_pres_array[i, :] # hPa
            # Reconstruct the full interface pressure array by inserting the surface pressure at position 0
            surf_pres = surf_pres_array[i] # hPa
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
                trop_layer_index[i] = layer_with_tp - 1

            # If tropopause is in upper half of layer, do include layer_with_tp in troposphere calculations
            else:
                trop_layer_index[i] = layer_with_tp

        return trop_layer_index

    trop_layer_index = dview.apply_sync(tropopause_layer_index_loop)
    trop_layer_index = np.concatenate(trop_layer_index).reshape(scan_ds.tropopause_pressure.shape)

    scan_ds['geoscf_tropopause_layer_index'] = (['mirror_step', 'xtrack'], trop_layer_index, {'description': 'index in swt_level corresponding to the highest layer that should be included in troposphere calculations', 'ancillary variable': "['tropopause_pressure', 'interface_pressures', 'surface_pressure']"})


    ###################################################
    #### Characterize Boundary Layer in TM5 Layers ####
    ###################################################
    if verbosity > 2:
        print('Characterizing boundary layer')

    # For conservation of mass in the update, we need to know the mass of NO2 available
    # for reallocation. To determine this, calculate the modeled boundary layer column

    # First, the index corresponding to the boundary layer height should be found
    
    dview.scatter('boundary_layer_height', scan_ds.boundary_layer_height.data.flatten())
    dview.scatter('main_data_quality_flag', scan_ds.main_data_quality_flag.data.flatten())
    dview.scatter('interface_heights', scan_ds.interface_heights.data.reshape((scan_ds.boundary_layer_height.size, 72)))

    def boundary_layer_index_loop():
        import numpy as np
        
        num_pixels = boundary_layer_height.size
        boundary_level_layer = np.full(num_pixels, -9999, dtype=int)

        for i in range(num_pixels):
            if main_data_quality_flag[i] > 0:
                continue

            blh = boundary_layer_height[i] # m

            if (blh == -99.) | (blh == -123.):
                continue                

            height_above_bl = interface_heights[i, :] - blh

            # The first positive height is the top interface of the layer containing the boundary layer pause
            filtered_layers = layer_indices[height_above_bl >= 0]

            #if len(filtered_layers) == len(layer_indices):
            try:
                if filtered_layers[0] == 0:
                    # Boundary layer is contained in the lowest model level
                    # Set as -1 to indicate that no model levels are fully contained in the boundary layer
                    bll = -1
                    boundary_level_layer[i] = bll

                else:
                    # Set as the highest layer completely contained in the boundary layer
                    bll = filtered_layers[0]
                    assert bll > 0
                    boundary_level_layer[i] = bll - 1

            except IndexError:
                continue # leave as nan

        return boundary_level_layer

    boundary_level_layer = dview.apply_sync(boundary_layer_index_loop)
    boundary_level_layer = np.concatenate(boundary_level_layer).reshape(scan_ds.boundary_layer_height.shape)

    scan_ds['boundary_layer_index'] = (
                                            ['mirror_step', 'xtrack'],
                                            boundary_level_layer.astype(int),
                                            {
                                                'units': '1',
                                                'description': "Index of the highest layer in GEOS-CF which is completely inside the convective/planetary boundary layer",
                                                'ancillary_vars': ['boundary_layer_height', 'interface_heights']
                                            }
    )

    #################################################################
    #### Calculcate Boundary Layer and Tropospheric VCD of Prior ####
    #################################################################

    if verbosity > 2:
        print('Calculating model column values')
    
    dview.scatter('geoscf_tropopause_layer_index', scan_ds.geoscf_tropopause_layer_index.data.flatten())
    dview.scatter('gas_profile', scan_ds.gas_profile.data.reshape((scan_ds.geoscf_tropopause_layer_index.size, 72)))
    dview.scatter('boundary_layer_index', scan_ds.boundary_layer_index.data.flatten())

    def a_priori_division_loop():
        import numpy as np
        
        num_pixels = main_data_quality_flag.size

        # Start by defining the columns as zero
        model_tropospheric_vcd = np.full(num_pixels, np.nan, dtype=float) # molecules / cm^2
        model_bl_vcd = np.full(num_pixels, np.nan, dtype=float) # molecules / cm^2

        for i in range(num_pixels):
            if main_data_quality_flag[i] > 0:
                continue

            #### Loop content ####
            # Find the tropospheric column density
            trop_layer_index = geoscf_tropopause_layer_index[i]

            if not np.isnan(trop_layer_index):
                trop_layer_index = int(trop_layer_index)
                model_tropospheric_vcd[i] = np.sum(gas_profile[i, :(trop_layer_index+1)])

            # Find the boundary layer column density
            bl_index = boundary_layer_index[i]

            # Can only find this if the boundary layer index was identified from HRRR
            if np.isnan(bl_index):
                continue

            try:
                if bl_index < 0:
                    model_bl_vcd[i] = gas_profile[i, 0]

                else:
                    model_bl_vcd[i] = np.sum(gas_profile[i, :(bl_index+1)])

                    #####################

            except TypeError as e:
                model_bl_vcd[i] = np.nan

            #pixels_completed += 1

        return model_tropospheric_vcd, model_bl_vcd
    
    a_priori_division_results = dview.apply_sync(a_priori_division_loop) # a list of tuples
    model_tropospheric_vcd = []
    model_bl_vcd = []

    for eng in range(len(a_priori_division_results)):
        model_tropospheric_vcd.append( a_priori_division_results[eng][0] )
        model_bl_vcd.append( a_priori_division_results[eng][1] )

    model_tropospheric_vcd = np.concatenate(model_tropospheric_vcd).reshape(scan_ds.boundary_layer_index.shape)
    model_bl_vcd = np.concatenate(model_bl_vcd).reshape(scan_ds.boundary_layer_index.shape)

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


    ##########################
    #### Clean up Engines ####
    ##########################

    # From matching with HRRR PBLH
    dview.execute('del latitude_subset')
    dview.execute('del longitude_subset')
    dview.execute('del badgeomask_subset')

    # From tropopuase layer selection
    dview.execute('del trop_pres_array')
    dview.execute('del interface_pres_array')
    dview.execute('del surf_pres_array')

    # From boundary layer selection
    dview.execute('del boundary_layer_height')
    dview.execute('del main_data_quality_flag')
    dview.execute('del interface_heights')

    # From summation over model columns
    dview.execute('del geoscf_tropopause_layer_index')
    dview.execute('del gas_profile')
    dview.execute('del boundary_layer_index')


    ############################################
    #### Run amf_recursive_update algorithm ####
    ############################################

    if verbosity > 2:
        print('Running recursive update')

    ms_mesh, xt_mesh = np.meshgrid(np.arange(scan_ds['mirror_step'].size), np.arange(scan_ds['xtrack'].size), indexing='ij')

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

        ms_mesh, xt_mesh = np.meshgrid(np.arange(scan_ds.mirror_step.size), np.arange(scan_ds.xtrack.size), indexing='ij')

        # Divide data among workers

        # Store the names of variables needed for the algorithm
        subset_vars = [
            'gas_profile',
            'amf_troposphere',
            'vertical_column_troposphere',
            'geoscf_tropopause_layer_index',
            'boundary_layer_index',
            'model_no2_boundary_layer_vcd',
            'model_no2_tropospheric_vcd',
            'main_data_quality_flag',
            sw_var,
            'eff_cloud_fraction',
            'solar_zenith_angle',
            'area',
            'snow_ice_fraction',
            'TemperatureCorrection',
        ]

        subset_ds_list = []
        ms_match_list = []
        xt_match_list = []
        lat_list = []
        lon_list = []

        # Compose a DataFrame to group the pixels by shared GEOS-CF priors
        geoscf_df = pd.DataFrame({
            'mirror_step': ms_mesh.flatten(),
            'xtrack': xt_mesh.flatten(),
            'BadGeoMask': scan_ds['BadGeoMask'].data.flatten(),
            'nearest_geoscf_latitude': scan_ds['nearest_geoscf_latitude'].data.flatten(),
            'nearest_geoscf_longitude': scan_ds['nearest_geoscf_longitude'].data.flatten()
        })

        # Drop pixels with no geolocation info
        geoscf_df = geoscf_df[~geoscf_df['BadGeoMask']]
        geoscf_df.drop(columns=['BadGeoMask'], inplace=True)

        # Group by prior
        geoscf_grouping = geoscf_df.groupby(['nearest_geoscf_latitude', 'nearest_geoscf_longitude'])
        # Return a DataFrame where the 'mirror_step' column gives a list of indices for the matching pixels
        ms_match_df = geoscf_grouping['mirror_step'].apply(list).reset_index()

        # Store as lists
        ms_match_list = ms_match_df['mirror_step'].to_list()
        xt_match_list = geoscf_grouping['xtrack'].apply(list).to_list()
        lat_list = ms_match_df['nearest_geoscf_latitude'].to_list()
        lon_list = ms_match_df['nearest_geoscf_longitude'].to_list()
        
        # Slice a subset of scan_ds for each prior
        for prior in range(len(ms_match_list)):
            subset_ds = {}
            for v in subset_vars:
                # Slice will reduce scalar variables to 1D [pixel,] and vertically-resolved variables to 2D [pixel, swt_level]
                subset_ds[v] = scan_ds[v].data[ms_match_list[prior], xt_match_list[prior]]

            # Append to the list which will be scattered
            subset_ds_list.append( subset_ds )

        # Scatter subset datasets across engines
        dview.scatter('subset_ds_list', subset_ds_list)
        dview.scatter('ms_match_list', ms_match_list)
        dview.scatter('xt_match_list', xt_match_list)
        dview.scatter('lat_list', lat_list)
        dview.scatter('lon_list', lon_list)

        # Push remaining necessary variables
        dview.push(dict(n_swt_levels=int(72)))
        dview.push(dict(N_updates=N_updates))
        dview.push(dict(sw_var=sw_var))

        # On each engine, import the functions which are going to be called in the main_algorithm_loop
        dview.execute("import numpy as np")
        dview.execute("import pandas as pd")
        dview.execute("from functions_par.prepare_for_update_par import prepare_for_update_par")
        dview.execute("from functions.amf_recursive_update_sf import amf_recursive_update, amf_recursive_update_no_good_pixels, amf_calculator, mismatch_check")
        dview.execute("from functions.great_circle_distance import great_circle_distance")

        # Run the algoirthm on each engine
        def main_algorithm_loop():
            import numpy as np
            import pandas as pd
            
            # Function will output a list of the dicts, with each dict containing the results of the GEOS-CF subset
            subset_results_list = []
            
            for j in range(len(subset_ds_list)):
                subset_ds = subset_ds_list[j] # dict
                ms_match = np.array(ms_match_list[j], dtype=int) # array
                xt_match = np.array(xt_match_list[j], dtype=int) # array
                lat = lat_list[j] # float (?)
                lon = lon_list[j] # float (?)     

                try:
                    # Prepare information for recursive update
                    # The dimension match arrays are updated to remove any pixels with critical issues (i.e. missing AMF or VCD)
                    aru_args, ms_match_filt, xt_match_filt, quality_df = prepare_for_update_par(subset_ds, ms_match, xt_match, n_swt_levels, N_updates, sw_var)

                    ## Quality flags (bit-array)
                    # Flip so that the most severe issues are at the earlier bit positions 
                    quality_df = quality_df[quality_df.columns[::-1]]

                    bit_sign_series = pd.Series(np.tile('0b', len(quality_df)), dtype=str)
                    quality_series = bit_sign_series.str.cat(quality_df)
                    quality_series_int = quality_series.apply(lambda x: int(x, 2))

                    update_quality_flags = quality_series_int.loc[:].to_numpy()

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

                    # Collect results for this subset
                    subset_results = {}
                    subset_results['ms_match_filt'] = ms_match_filt
                    subset_results['xt_match_filt'] = xt_match_filt
                    subset_results['update_quality_flags'] = update_quality_flags
                    subset_results['apriori_partial_columns'] = apriori_partial_columns
                    subset_results['trop_amfs'] = trop_amfs
                    subset_results['retrieved_trop_vcd'] = retrieved_trop_vcd
                    subset_results['vcd_iter_diff'] = vcd_iter_diff
                    subset_results['coverage_of_model_pixel_value'] = coverage_of_model_pixel_value
                    subset_results['portion_free_trop'] = portion_free_trop
                    subset_results['removed_ft_in_practice'] = removed_ft_in_practice
                    subset_results['retrieved_over_apriori_gridcell'] = retrieved_over_apriori_gridcell
                    subset_results['retrieved_model_mismatch_flag'] = retrieved_model_mismatch_flag

                    subset_results_list.append( subset_results )

                except Exception as e:
                    debug_mode = True
                    if debug_mode:
                        raise e
                    
                    else:
                        print('Issue performing update on this prior')

            return subset_results_list


        retrieval_results = dview.apply_sync(main_algorithm_loop)
        # "retrieval_results" is a list of lists of dicts. 

        # Each entry in the first list corresponds to an engine.
        for eng in range(len(retrieval_results)):
            subset_results_list = retrieval_results[eng]

            # Each entry in the second list corresponds to a GEOS-CF subset. The dict contains results.
            for i_subset in range(len(subset_results_list)):
                subset_results = subset_results_list[i_subset] # dict

                # Reconstruct data into the [mirror_step, xtrack] format
                for p in range(len(subset_results['ms_match_filt'])):
                    ms = subset_results['ms_match_filt'][p]
                    xt = subset_results['xt_match_filt'][p]
                    
                    # Assign values to final arrays we initialized earlier
                    update_quality_flags[ms, xt] = subset_results['update_quality_flags'][p]
                    no2_partial_columns_updated[ms, xt, :, :] = subset_results['apriori_partial_columns'][p, :, :] # [pixel, iteration, lev]
                    tropospheric_amf_updated[ms, xt, :] = subset_results['trop_amfs'][p, :] # [pixel, iteration]
                    no2_tropospheric_vcd_updated[ms, xt, :] = subset_results['retrieved_trop_vcd'][p, :]
                    vcd_iteration_differences[ms, xt, :] = subset_results['vcd_iter_diff'][p, :] # [pixel, iteration]
                    coverage_of_model_pixel[ms, xt] = subset_results['coverage_of_model_pixel_value']
                    proportion_free_troposphere[ms, xt] = subset_results['portion_free_trop']
                    removed_free_troposphere_in_practice[ms, xt, :] = subset_results['removed_ft_in_practice']
                    retrieved_over_apriori_gridcell_arr[ms, xt] = subset_results['retrieved_over_apriori_gridcell']
                    retrieved_model_mismatch_flag_arr[ms, xt] = subset_results['retrieved_model_mismatch_flag']

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
                                                                'bit_positions': np.array([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=int),
                                                                'bit_meanings': ['nonzero_snow_ice', 'invalid_pixel_area', 'bl_index_above_tp', 'bl_index_unknown', 'model_bl_vcd_unknown', 'high_sza', 'high_eff_cloud_fraction', 'main_data_quality_above_0', 'calculated_trop_amf_invalid', 'scattering_weights_bad', 'trop_index_unknown', 'original_trop_vcd_invalid', 'original_trop_amf_invalid']
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

    # Stop the cluster
    mycluster.stop_cluster_sync()

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

    if scan_ds_created:
        save_partial_ds()

except Exception as e:
    print('General Exception encountered while processing scan. Will not finish process for this scan.')
    print('{}'.format(repr(e)))

    if scan_ds_created:
        save_partial_ds()