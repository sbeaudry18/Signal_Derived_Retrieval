#### finalize_product_file.py ####

# Author: Sam Beaudry
# Last changed: 2026-03-22
# Location: Signal_Derived_Retrieval/TEMPO/main
# Contact: samuel_beaudry@berkeley.edu

##################################

def finalize_product_file(processing_dataset, product_dir, sdr_version, scan_num, geo_scope, product_itr: int=1, sw_mode: str='Standard', trace_gas: str='NO2', convert_from_behr=False, name_w_commit=False, name_w_proctime=False):
    '''
    Takes processing datasets produced by amf_update_one_scan and reformats to be published as a data product

    scan_ds : str or xr.Dataset
        Dataset produced by amf_update_one_scan (or path to this dataset)
    product_dir : str
        Location to save the product file at
    sdr_version : str
        Version of the signal-derived retrieval used to produce the processing_dataset
    scan : int
        Scan number of dataset
    geo_scope : str
        Geographic scope of the scan dataset
    product_itr : int  
        Iteration of the AMF recursive update to use as the signal-derived product
    sw_mode : str
        Scattering weight mode used for signal-derived retrieval
    trace_gas : str
        Default NO2. Trace gas of retrieval.
    convert_from_behr : bool
        Default False. If True, pass processing_dataset as a string point to a BEHR-RED-TEMPO result
    name_w_commit : bool (Optional)
        if True, will add the Signal_Derived_Retrieval repository commit to the output filename
    name_w_proctime : bool (Optional)
        if True, will add the processing time to the output filename
    '''

    import numpy as np
    import xarray as xr
    import pandas as pd
    import datetime
    import os
    import re

    try:
        from name_product_file import name_product_file
    except ModuleNotFoundError:
        from functions.name_product_file import name_product_file
        
    if isinstance(processing_dataset, str):
        if convert_from_behr:
            processing_file_path = processing_dataset
            scan_ds = xr.open_dataset(processing_file_path)

        else:
            raise ValueError("'processing_dataset' is a string but convert_from_behr is set to False. Pass an xarray Dataset instead")

    elif isinstance(processing_dataset, xr.Dataset):
        scan_ds = processing_dataset

    else:
        raise ValueError("'processing_dataset' must be a path leading to a netCDF file or an xarray Dataset")
    
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
        if 'iteration' in scan_ds[old_name].dims:
            var_data = scan_ds[old_name].sel(iteration=product_itr).data # will remove the iteration dim
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

    if convert_from_behr:
        # Pull this information from the BEHR-RED-TEMPO file name
        processing_file_name = os.path.basename(processing_file_path)
        processing_pat = re.compile(r'^BEHR-RED-TEMPO_(\d{8})_(S\d{3})_([^_]+)_n\d+_variable_bl_HRRR_proc_(\d{8}T\d{4})\.nc$')

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
        scan_num = regex_results.group(2)
        geo_scope = regex_results.group(3)
        current_time_string = regex_results.group(4)

        scan_ds.attrs['intermediate_dataset_id'] = processing_file_name

    else:
        # Get the processing time (now with second precision)
        current_time = datetime.datetime.now(tz=datetime.timezone.utc)
        current_time_string = current_time.strftime('%Y%m%dT%H%M%SZ')

        scan_num = "S{:03d}".format(scan_num)


    # Store processing time as an attribute to clean up the file name
    scan_ds.attrs['sdr_processing_time'] = current_time_string
    
    # Add the start and end times of the scan (which is more descriptive than the measurement date)
    # Format: YYYYMMDDTHHMMSSZ (same as S5P products)
    # start_time = pd.to_datetime(str(scan_ds.time_utc.data[0])).strftime('%Y%m%dT%H%M%SZ') # annoying solution but it does work
    end_time = pd.to_datetime(str(scan_ds.time_utc.data[-1])).strftime('%Y%m%dT%H%M%SZ')

    # SB 2026-03-22: The start time of mirror_step == 0  is not the same as the start time of the first granule, which means this method
    # produces filenames that are not easy to match up with standard product files. I still like having the end time for usability, but
    # let's use the granule ID stored in attributes to get a start time that is the same as the standard product files.
    # Record the lowest granule
    lowest_gran = int(np.nanmin(scan_ds.granule.data))
    # Get the ID of this granule
    lowest_gran_id = scan_ds.attrs['TEMPO_standard_id_G{:02d}'.format(lowest_gran)]
    # Capture the start time
    gran_id_pat = re.compile(r'TEMPO_[^_]+_[^_]+_[^_]+_(\d{8}T\d{6}Z)_S\d{3}G\d{2}\.nc')
    start_time = gran_id_pat.match(lowest_gran_id).group(1)

    # Assemble new file name
    if name_w_commit:
        commit = scan_ds.git_commit
    else:
        commit = None

    if name_w_proctime:
        proc_time = current_time_string
    else:
        proc_time = False

    product_name = name_product_file(sdr_version, start_time, end_time, scan_num, geo_scope, commit=commit, proc_time=proc_time)

    meas_year = start_time[:4]
    meas_month = start_time[4:6]

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