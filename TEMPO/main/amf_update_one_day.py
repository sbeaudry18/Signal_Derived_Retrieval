#### amf_update_one_day.py ####

# Author: Sam Beaudry
# Last changed: 2025-10-05
# Location: Signal_Derived_Retrieval/TEMPO/main
# Contact: samuel_beaudry@berkeley.edu

################################

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import xarray as xr
import pickle
import os
import sys
import re
from datetime import datetime
from datetime import timedelta
import warnings
from scipy.interpolate import NearestNDInterpolator

from functions.build_geobounds_str import build_geobounds_str

def amf_update_one_day(date_string, py_to_mat_textfile, PY_TO_MAT_SUITCASE, MAT_TO_PY_SUITCASE, run_matlab, TEMPO, vars_path, constants_path, save_path, minimize_output_size, reprocess_if_exists, N_updates, pblh, full_FOR, N_workers=0, lon_domain=np.array([-180, 180]), lat_domain=np.array([-90, 90]), hrrr_grib=None, save_path_partial="", git_commit="None", prioritize_latest=True, remove_matlab=False, scanlist=None):
    '''
    Calls amf_update_one_scan across the provided day.

    Parameters
    ----------
    date_string : str
        YYYYMMDD to process
    py_to_mat_textfile : str 
        path to .txt file containing the names of the dictionaries produced by TEMPO_L2_NO2_on_date.py
    PY_TO_MAT_SUITCASE : str 
        path of the directory containing dictionaries produced by TEMPO_L2_NO2_on_date.py
    MAT_TO_PY_STUICASE : str
        path of the directory containing dictionaries produced by read_main_single.m
    run_matlab : bool
        whether MATLAB is run during the process to get MODIS albedo values
    TEMPO : str 
        path to unprocessed TEMPO data
    vars_path : str 
        path to the .csv file containing TEMPO variable names and groups
    constants_path : str
        path of the directory containing constant values
    save_path : str
        path to the directory containing processed data
    minimize_output_size : bool
        whether to remove vertical information from final dataset to reduce storage size
    reprocess_if_exists : bool
        whether to process scans even if an equivalent dataset is found in save_path
    N_updates : int
        number of update iterations to perform during the retrieval-redistribution method
    pblh : str or float
        hrrr or value to use for planetary boundary layer height (m)
    full_FOR : bool
        where processing is done for the full field of regard
    N_workers : int (Optional)
        the number of workers to distribute tasks across. If 1 (Default) algorithm is run in serial.
    lon_domain : np.ndarray (Optional)
        array of mimimum and maximum longitude values to process
    lat_domain : np.ndarray (Optional)
        array of mimimum and maximum latitude values to process
    hrrr_grib : str (Optional)
        if pblh == 'hrrr' then this specifies the path to the HRRR grib files
    save_path_partial : str (Optional)
        path to save partially completed scan_ds when the function fails to finish
    git_commit : str (Optional)
        the commit of Signal_Derived_Retrieval repository used
    prioritize_latest : bool (Optional)
        if True, will use last generated SDR outputs when there are multiple options
    remove_matlab : bool (Optional)
        if True, will remove dictionaries in MAT_TO_PY_SUITCASE after reading their data
    scanline : list (Optional)
        list of scans to restrict processing to
    '''

    N_workers = int(N_workers)
    if N_workers == 1:
        parallel_algorithm = False
        # Import main processing function
        from amf_update_one_scan import amf_update_one_scan
        print('Will process days using serial algorithm')

    elif N_workers > 1:
        parallel_algorithm = True

        print('Will process days using parallel algorithm')

        scan_df_list = []

    else:
        raise ValueError("Value for 'N_workers' of {} is not greater than or equal to zero.".format(N_workers))
    
    print('')

    if isinstance(scanlist, list):
        check_against_list = True

    elif isinstance(scanlist, np.ndarray):
        check_against_list = True
        scanlist = list(scanlist)

    else:
        check_against_list = False
    
    # Determine the list of granule datasets to work with for this day
    file_list_tempo = []
    with open(py_to_mat_textfile) as file_record:
        for line in file_record:
            file_list_tempo.append(line[:-1])

    if file_list_tempo[0][:4] == '9999':
        print('No TEMPO files in PY_TO_MAT_SUITCASE for date {}'.format(date_string))
    
    else:        
        if len(file_list_tempo) > 0:
            file_df = pd.DataFrame({'TEMPO Name': file_list_tempo})
            file_df['Scan'] = file_df['TEMPO Name'].str.extract(r'S(\d{3})G\d{2}').astype(int)
            file_df['Granule'] = file_df['TEMPO Name'].str.extract(r'S\d{3}G(\d{2})').astype(int)

            file_list_behr = os.listdir(MAT_TO_PY_SUITCASE)
        
            for scan in file_df['Scan'].unique():
                if check_against_list:
                    if scan not in scanlist:
                        # If not in the provided list, skip the processing for this scan
                        continue

                if not reprocess_if_exists:
                    # Setting to restrict matching outputs to those with the same number of update iterations
                    require_n_updates = True

                    # Determine if an equivalent dataset exists
                    # If so, do not process this scan

                    # Start by writing a regex pattern to match equivalent datasets
                    if full_FOR:
                        geobounds_str = "full-FOR"

                    else:
                        geobounds_str = build_geobounds_str(lat_domain, lon_domain)

                    if pblh == 'hrrr':
                        bl_setting = 'variable'
                        bl_value = 'HRRR'

                    else:
                        bl_setting = 'fixed'
                        bl_value = pblh

                    if require_n_updates:
                        dataset_name_pat = re.compile(r'^SDR-TEMPO_{date_string}_S{scan:03d}_{geo}_n{N:02d}_{bl_setting}_bl_{bl_value}_proc_\d{{8}}T\d{{4}}\.nc$'.format(date_string=date_string, scan=scan, geo=geobounds_str, N=N_updates, bl_setting=bl_setting, bl_value=bl_value))

                    else:
                        dataset_name_pat = re.compile(r'^SDR-TEMPO_{date_string}_S{scan:03d}_{geo}_n\d{{2}}_{bl_setting}_bl_{bl_value}_proc_\d{{8}}T\d{{4}}\.nc$'.format(date_string=date_string, scan=scan, geo=geobounds_str, bl_setting=bl_setting, bl_value=bl_value))

                    # Check against existing files to see if any equivalents exist

                    # The below lines were commented out since the results subdirectory setting occurs in the director script
                    #if run_matlab:
                    #    behr_mode = 'with_MODIS'
                    #else:
                    #    behr_mode = 'without_MODIS'

                    eqv_save_loc = os.path.join(save_path, geobounds_str, date_string[:4], date_string[4:6])

                    # First see if this director exists. If not, then there are no equivalent files
                    if os.path.exists(eqv_save_loc):
                        eqv_file_list = os.listdir(eqv_save_loc)
                        eqv_file_list.sort()

                        matching_file_list = [f for f in eqv_file_list if dataset_name_pat.match(f)]

                        # If any files match these conditions, then skip them
                        if len(matching_file_list) > 0:
                            print('Encountered existing dataset. Skipping this scan:')
                            print('Date: {}'.format(date_string))
                            print('Scan: {:03d}'.format(scan))
                            for f in matching_file_list:
                                print('Existing File: {}'.format(f))
                            print('')

                            continue

                scan_df = file_df[file_df['Scan'] == scan].copy()

                # Each row contains a granule we want to use
                scan_df.set_index('Granule', inplace=True)

                behr_dfs = {}

                for g in scan_df.index:
                    # SB 2025-03-31: BEHR file matching is now on granule level since the UTC date may vary from one granule to the next, even for the same scan
                    utc_date_string = scan_df.loc[g, 'TEMPO Name'][17:25]

                    # Get any matching BEHR datasets
                    pattern_granule_behr = re.compile(r'^TEMPO_SP_[A-Z]+_REDv\d-\d_{DATE}_S{SCAN:03d}G{GRAN:02d}_proc_\d{{8}}T\d{{6}}\.pickle$'.format(DATE=utc_date_string, SCAN=scan, GRAN=g))
                    behr_matching = [f for f in file_list_behr if pattern_granule_behr.match(f)]
                    behr_matching.sort()

                    if len(behr_matching) > 0:
                        behr_dfs[g] = pd.DataFrame({'BEHR Name': behr_matching})
                        behr_dfs[g]['Granule'] = behr_dfs[g]['BEHR Name'].str.extract(r'S\d{3}G(\d{2})_proc').astype(int)
                        behr_dfs[g]['Major Version'] = behr_dfs[g]['BEHR Name'].str.extract(r'REDv(\d)-\d').astype(int)
                        behr_dfs[g]['Minor Version'] = behr_dfs[g]['BEHR Name'].str.extract(r'REDv\d-(\d)').astype(int)
                        behr_dfs[g]['Version'] = behr_dfs[g].agg('{0[Major Version]}.{0[Minor Version]}'.format, axis=1).astype(float) # https://stackoverflow.com/questions/11858472/string-concatenation-of-two-pandas-columns
                        behr_dfs[g]['Processing Time'] = behr_dfs[g]['BEHR Name'].str.extract(r'proc_(\d{8}T\d{6})')
                        behr_dfs[g]['Processing Time'] = pd.to_datetime(behr_dfs[g]['Processing Time'], format='%Y%m%dT%H%M%S')

                        # Code block to deal with multiple versions of MATLAB outputs
                        if len(behr_matching) > 1:
                            warnings.warn('Multiple MATLAB outputs corresponding to TEMPO scan {} and granule {}.'.format(scan, g))

                            # Start a list of the files that we want to pass to amf_update_one_scan
                            behr_files_to_include = []
                            
                            if prioritize_latest:
                                print('Using most recently generated output')
                                behr_dfs[g] = behr_dfs[g].sort_values('Processing Time', ascending=False)
                                behr_files_to_include.append(behr_dfs[g]['BEHR Name'].to_list()[0])
    
                            else:
                                # Also consider version number
                                print('Using most recentely generated output of highest version')
                                version_df = behr_dfs[g][behr_dfs[g]['Version'] == behr_dfs[g]['Version'].max()]
                                version_df = version_df.sort_values('Processing Time', ascending=False)
                                behr_files_to_include.append(version_df['BEHR Name'].to_list()[0])

                            # Keep only the BEHR outputs we want to use
                            behr_dfs[g] = behr_dfs[g][behr_dfs[g]['BEHR Name'].isin(behr_files_to_include)]

                    else: # len(behr_matching) == 0
                        behr_dfs[g] = pd.DataFrame(np.array([[g, np.nan, np.nan, np.nan, np.nan, np.nan]]), columns=['Granule', 'BEHR Name', 'Major Version', 'Minor Version', 'Version', 'Processing Time'])

                # Concat granule DataFrames for one scan into a single DataFrame
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=FutureWarning)
                    behr_df = pd.concat(behr_dfs, axis=0, ignore_index=False)

                behr_df.set_index('Granule', inplace=True)

                # Join TEMPO and BEHR DataFrames
                scan_df = scan_df.join(behr_df, on='Granule', how='left')


                if parallel_algorithm:
                    scan_df_filename = 'scan_{:03d}_df.csv'.format(scan)
                    scan_df.to_csv(scan_df_filename)
                    scan_df_list.append(scan_df_filename)
                    #amf_update_one_scan_par(PY_TO_MAT_SUITCASE, MAT_TO_PY_SUITCASE, scan_df, TEMPO, vars_path, constants_path, save_path, minimize_output_size, full_FOR, N_workers, N_updates=N_updates, pblh=pblh, hrrr_grib=hrrr_grib, save_path_partial=save_path_partial, git_commit=git_commit, verbosity=5)
                else:
                    print('Starting process for scan {}'.format(scan))
                    print('------------------------------')                    
                    amf_update_one_scan(PY_TO_MAT_SUITCASE, MAT_TO_PY_SUITCASE, scan_df, TEMPO, vars_path, constants_path, save_path, minimize_output_size, full_FOR, N_updates=N_updates, pblh=pblh, hrrr_grib=hrrr_grib, save_path_partial=save_path_partial, git_commit=git_commit, verbosity=5)
                    print('------------------------------')
                    print('')

                if remove_matlab:
                    if len(behr_df) > 0:
                        for i in behr_df.index:
                            matlab_output_path = "{}/{}".format(MAT_TO_PY_SUITCASE, behr_df.loc[i, 'BEHR Name'])
                            os.remove(matlab_output_path)

            
            if parallel_algorithm:
                with open("scan_df_file_list_transient.txt", 'w') as file:
                    for i in range(len(scan_df_list)):
                        file.write("{}\n".format(scan_df_list[i]))
                file.close()

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--current_date', type=str)
    parser.add_argument('--py_to_mat_textfile', type=str)
    parser.add_argument('--PY_TO_MAT_SUITCASE', type=str)
    parser.add_argument('--MAT_TO_PY_SUITCASE', type=str)
    parser.add_argument('--run_matlab', type=int)
    parser.add_argument('--TEMPO', type=str)
    parser.add_argument('--vars_path', type=str)
    parser.add_argument('--constants_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--minimize_output_size', type=int)
    parser.add_argument('--reprocess_if_exists', type=int)
    parser.add_argument('--N_updates', type=int)
    parser.add_argument('--first_scan_num', type=int)
    parser.add_argument('--last_scan_num', type=int)
    parser.add_argument('--fullfor', type=int)
    parser.add_argument('--N_workers', type=int)
    parser.add_argument('--lonmin', type=float)
    parser.add_argument('--lonmax', type=float)
    parser.add_argument('--latmin', type=float)
    parser.add_argument('--latmax', type=float)
    parser.add_argument('--pblh', type=str)
    parser.add_argument('--hrrr_grib', type=str)
    parser.add_argument('--save_path_partial', type=str)
    parser.add_argument('--git_commit', type=str)

    args = vars(parser.parse_args())

    # Limit scans to be processed
    scanlist = list(np.arange(args['first_scan_num'], args['last_scan_num']+1))

    if args['pblh'].lower() != 'hrrr':
        args['pblh'] = float(args['pblh'])

    # Processing domain
    full_FOR = bool(args['fullfor'])

    if full_FOR:
        amf_update_one_day(args['current_date'], args['py_to_mat_textfile'], args['PY_TO_MAT_SUITCASE'], args['MAT_TO_PY_SUITCASE'], bool(args['run_matlab']), args['TEMPO'], args['vars_path'], args['constants_path'], args['save_path'], bool(args['minimize_output_size']), bool(args['reprocess_if_exists']), args['N_updates'], args['pblh'], full_FOR, args['N_workers'], hrrr_grib=args['hrrr_grib'], save_path_partial=args['save_path_partial'], git_commit=args['git_commit'], prioritize_latest=True, scanlist=scanlist)

    else:
        lon_domain = np.array([args['lonmin'], args['lonmax']], dtype=float)
        lat_domain = np.array([args['latmin'], args['latmax']], dtype=float)

        amf_update_one_day(args['current_date'], args['py_to_mat_textfile'], args['PY_TO_MAT_SUITCASE'], args['MAT_TO_PY_SUITCASE'], bool(args['run_matlab']), args['TEMPO'], args['vars_path'], args['constants_path'], args['save_path'], bool(args['minimize_output_size']), bool(args['reprocess_if_exists']), args['N_updates'], args['pblh'], full_FOR, args['N_workers'], lon_domain=lon_domain, lat_domain=lat_domain, hrrr_grib=args['hrrr_grib'], save_path_partial=args['save_path_partial'], git_commit=args['git_commit'], prioritize_latest=True, scanlist=scanlist)

if __name__ == "__main__":
    main()