#### finalize_product_file.py ####

# Author: Sam Beaudry
# Last changed: 2025-10-30
# Location: Signal_Derived_Retrieval/TEMPO/main
# Contact: samuel_beaudry@berkeley.edu

##################################

def finalize_product_file(processing_dataset, product_dir, sdr_version, product_itr: int=1, sw_mode: str='Standard', trace_gas: str='NO2'):
    '''
    Takes processing datasets produced by amf_update_one_scan and reformats to be published as a data product

    scan_ds : str or xr.Dataset
        Dataset produced by amf_update_one_scan (or path to this dataset)
    product_dir : str
        Location to save the product file at
    sdr_version : str
        Version of the signal-derived retrieval used to produce the processing_dataset
    product_itr : int  
        Iteration of the AMF recursive update to use as the signal-derived product
    sw_mode : str
        Scattering weight mode used for signal-derived retrieval
    trace_gas : str
        Default NO2. Trace gas of retrieval.
    '''

    import numpy as np
    import xarray as xr
    import pandas as pd
    import datetime
    import os
    import re

    if isinstance(processing_dataset, str):
        processing_file_path = processing_dataset
        scan_ds = xr.open_dataset(processing_file_path)

    elif isinstance(processing_dataset, xr.Dataset):
        # SB 2025-10-30: Could implement this later; problem now is that it's easy to rename
        # the file with the existing name, but this is not stored in datasets at the moment.
        # This should probably be added to the metadata in future versions anyway
        raise ValueError("'processing_dataset' must be a path leading to a netCDF file")

    else:
        raise ValueError("'processing_dataset' must be a path leading to a netCDF file")
    
    ###################
    #### VARIABLES ####
    ###################

    #### Renaming SDR Variables ####

    # Define a function which will replace variable names and collapse to a single iteration
    def redefine_sdr_vars(scan_ds, old_name, new_name):
        old_var_data = scan_ds[old_name].data
        var_attrs = scan_ds[old_name].attrs # dictionary
        var_attrs['product'] = "Signal-Derived Retrieval (SDR)"
        
        # Select update iteration if necessary
        if len(old_var_data.shape) == 3:
            var_data = scan_ds[old_name].data[:, :, product_itr]
            # Store the iteration we report as SDR product
            var_attrs['iteration_of_update'] = product_itr

        else:
            # Keep data as is
            var_data = old_var_data

        # Save as a new variable in scan_ds
        scan_ds[new_name] = (
                                ['mirror_step', 'xtrack'],
                                var_data,
                                var_attrs
        )

        # Drop the old variable
        scan_ds = scan_ds.drop_vars([old_name])

        return scan_ds

    # Define another function which will remove any other variables produced with another scattering weight mode
    # (e.g. using BEHR-MODIS albedo values)
    def remove_other_sw_modes(scan_ds, var_names_no_sw):
        # We will call this after the other function where the unwanted scattering weight modes remain
        # See if there are any
        scan_ds_var_list = list(scan_ds.variables.keys())
        vars_to_drop = []

        for var in var_names_no_sw:
            var_pat = re.compile(r'^{}_.*$'.format(var))
            matching_vars = [mv for mv in scan_ds_var_list if var_pat.match(mv)]

            if len(matching_vars) > 0:
                for mv in matching_vars:
                    vars_to_drop.append( mv )

        scan_ds = scan_ds.drop_vars(vars_to_drop)

        return scan_ds

    #### Updated Variables ####

    # These variables have standard analogues in the product
    # A list of the variable names to be changed...
    vars_to_change = [
        'amf_troposphere_updated', 
        'vertical_column_troposphere_updated', 
        'amf_total_updated', 
        'vertical_column_total_updated',
        'model_no2_boundary_layer_vcd_updated'
        ]

    # Define a pattern which extracts the part of the base name we want to keep (i.e. without 'updated')
    base_name_pat = re.compile(r'^(.*)_[^_]*$')
    var_base_names = [base_name_pat.match(var).group(1) for var in vars_to_change]

    # Now add 'sdr' at the front to indicate result from signal-derived retrieval
    var_names_sdr = ["sdr_{}".format(var) for var in var_base_names]

    # For the existing vars, need to add scattering weight mode to the updated name
    vars_to_change_w_sw = ["{}_{}".format(var, sw_mode) for var in vars_to_change]

    # Loop through product variables and change to new format
    for i in range(len(vars_to_change)):
                            # old name             # new name
        scan_ds = redefine_sdr_vars(scan_ds, vars_to_change_w_sw[i], var_names_sdr[i])

    # Drop for other sw_modes
    scan_ds = remove_other_sw_modes(scan_ds, vars_to_change)

    #### SDR Reprocessing Difference ####

    # Before removing all iterations of vcd_iteration_differences, we might want to know what happens
    # if we run the SDR again with the new VCDs
    # By keeping vertical_column_troposphere and sdr_vertical_column_troposphere, we have the i=0 to i=1 change

    # Store the i=1 to i=2 change
    one_vs_two_data = scan_ds["vcd_iteration_differences_{}".format(sw_mode)].data[:, :, 2]

    scan_ds['sdr_reprocessing_difference'] = (
                                                ['mirror_step', 'xtrack'],
                                                one_vs_two_data,
                                                {
                                                    'description': 'percent change from sdr_vertical_column_troposphere if put back into AMF redistribution method (i.e. another iteration of AMF recursive update)',
                                                    'units': '%'
                                                    }
    )

    # Drop for other sw_modes
    scan_ds = remove_other_sw_modes(scan_ds, ["vcd_iteration_differences"])

    #### Ancillary Variables ####

    # Unlike the previous variables, these don't have "updated" in the title
    # Besides the update_quality_flags, these are unnecessary for most users but may be helpful 
    # for interpreting the retrieval results

    # A new list of the variable names to be updated...
    vars_to_change = [
        'coverage_of_model_pixel', 
        'proportion_free_troposphere', 
        'removed_free_troposphere_in_practice', 
        'update_quality_flags',
        'retrieved_over_apriori_gridcell',
        'retrieved_model_mismatch_flag'
        ]

    # Add the scattering weight mode
    vars_to_change_w_sw = ["{}_{}".format(var, sw_mode) for var in vars_to_change]

    # Loop, rename, and select iteration
    for i in range(len(vars_to_change)):
                                                # old name             # new name
        scan_ds = redefine_sdr_vars(scan_ds, vars_to_change_w_sw[i], vars_to_change[i])

    # Drop for other sw_modes
    scan_ds = remove_other_sw_modes(scan_ds, vars_to_change)

    #### Drop iteration dimension ####
    scan_ds = scan_ds.drop_dims('iteration')

    ###################
    #### FILE NAME ####
    ###################

    processing_file_name = os.path.basename(processing_file_path)
    processing_pat = re.compile(r'^BEHR-RED-TEMPO_(\d{8})_S(\d{3})_([^_]+)_n\d+_variable_bl_HRRR_proc_(\d{8}T\d{4})\.nc$')

    # Capture groups:
    # 1: TEMPO measurement date (YYYYMMDD, UTC)
    # 2: TEMPO scan (e.g. 010)
    # 3: Geographic scope of product (e.g. full-FOR)
    # 4: Processing time (e.g. YYYYMMDDTHHMM)

    # Some notes:
    # - Processing time is for the processing dataset, not the finalization here
    # - I am not capturing the number of iterations or treatement of the boundary layer
    #   since this function requires that the SDR version is passed as an argument and
    #   these are taken as given for the version. They should also be capture in the 
    #   product metadata: boundary layers are labeled as coming from HRRR and the SDR
    #   product interation is included.

    regex_results = processing_pat.match(processing_file_name)
    meas_date = regex_results.group(1)
    meas_year = meas_date[:4]
    meas_month = meas_date[4:6]
    scan_num = regex_results.group(2)
    geo_scope = regex_results.group(3)
    processing_time = regex_results.group(4)

    # Store processing time as an attribute to clean up the file name
    scan_ds.attrs['sdr_processing_time'] = processing_time
    scan_ds.attrs['intermediate_dataset_id'] = processing_file_name

    # Add the start and end times of the scan (which is more descriptive than the measurement date)
    # Format: YYYYMMDDTHHMMSS (same as S5P products)
    start_time = pd.to_datetime(str(scan_ds.time_utc.data[0])).strftime('%Y%m%dT%H%M%S') # annoying solution but it does work
    end_time = pd.to_datetime(str(scan_ds.time_utc.data[-1])).strftime('%Y%m%dT%H%M%S')

    # Assemble new file name
    product_name = "SDR-TEMPO_NO2_L2_{}_{}_{}_S{}_{}.nc".format(
        sdr_version,
        start_time,
        end_time,
        scan_num,
        geo_scope
    )

    # Save to product directory
    # product_dir should be at level "SDR", below which we specify:
    # product_dir / trace_gas / level / sdr_version / geographic_scope / year / month
    save_dir = os.path.join(product_dir, trace_gas, 'L2', sdr_version, geo_scope, meas_year, meas_month)

    if not os.path.exists(product_dir):
        raise ValueError("{} does not exist".format(product_dir))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, product_name)
    scan_ds.to_netcdf(save_path, mode='w')