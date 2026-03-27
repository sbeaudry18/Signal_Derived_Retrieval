#### amf_update_one_day.py ####

# Author: Sam Beaudry
# Last changed: 2026-03-27
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

from TEMPO_L2_NO2_on_date import TEMPO_L2_NO2_on_date

from functions.build_geobounds_str import build_geobounds_str
from functions.name_product_file import name_product_file

def amf_update_one_day(date_string, TEMPO, collection, sdr_letter, vars_path, constants_path, save_path, minimize_output_size, reprocess_if_exists, N_updates, pblh, full_FOR, N_workers=0, lon_domain=np.array([-180, 180]), lat_domain=np.array([-90, 90]), hrrr_grib=None, save_path_partial="", git_commit="None", prioritize_latest=True, remove_matlab=False, scanlist=None, py_to_mat_textfile=None, PY_TO_MAT_SUITCASE=None, MAT_TO_PY_SUITCASE=None, run_matlab=False, file_df=None):
    '''
    Calls amf_update_one_scan across the provided day.

    Parameters
    ----------
    date_string : str
        YYYYMMDD to process
    TEMPO : str 
        path to unprocessed TEMPO data
    collection : str
        TEMPO processor version 
    sdr_letter : str
        SDR processer version letter
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
    py_to_mat_textfile : str 
        path to .txt file containing the names of the dictionaries produced by TEMPO_L2_NO2_on_date.py
    PY_TO_MAT_SUITCASE : str 
        path of the directory containing dictionaries produced by TEMPO_L2_NO2_on_date.py
    MAT_TO_PY_STUICASE : str
        path of the directory containing dictionaries produced by read_main_single.m
    run_matlab : bool
        whether MATLAB is run during the process to get MODIS albedo values
    file_df : pd.DataFrame (Optional)
        DataFrame produced by TEMPO_L2_NO2_on_date
    '''

    N_workers = int(N_workers)
    if N_workers == 1:
        parallel_algorithm = False
        # Import main processing function
        from amf_update_one_scan import amf_update_one_scan
        print('Will process scans using serial algorithm')

    elif N_workers > 1:
        parallel_algorithm = True

        print('Will process scans using parallel algorithm')

        scan_df_list = []
        save_options_list = []

    else:
        raise ValueError("Value for 'N_workers' of {} is not greater than or equal to 1.".format(N_workers))
    
    print('')

    if isinstance(scanlist, list):
        check_against_list = True

    elif isinstance(scanlist, np.ndarray):
        check_against_list = True
        scanlist = list(scanlist)

    else:
        check_against_list = False
    
    # Check if we have already called TEMPO_L2_NO2_on_date and provided the DataFrame with matched files
    if file_df is None:
        # Call TEMPO_L2_NO2_on_date for the provided day
        file_df = TEMPO_L2_NO2_on_date(date_string, TEMPO, collection, vars_path, full_FOR)

    elif not isinstance(file_df, pd.DataFrame):
        raise ValueError("'file_df' must be None or pd.DataFrame, not {}".format(type(file_df)))

    if len(file_df) > 0:
        file_df['Scan'] = file_df['TEMPO Name'].str.extract(r'S(\d{3})G\d{2}').astype(int)
        file_df['Granule'] = file_df['TEMPO Name'].str.extract(r'S\d{3}G(\d{2})').astype(int)
    
        for scan in file_df['Scan'].unique():
            if check_against_list:
                if scan not in scanlist:
                    # If not in the provided list, skip the processing for this scan
                    continue

            scan_df = file_df[file_df['Scan'] == scan].copy()

            # Each row contains a granule we want to use
            scan_df.set_index('Granule', inplace=True)

            # Check if an equivalent file exists
            # "Equivalent": Same date and scan, same TEMPO processor version (e.g. V03), same SDR version (e.g. V03-A)
            # Start by assuming it doesn't
            name_w_commit = False
            name_w_proctime = False

            # Generate a file name using the earliest granule
            earliest_granule = scan_df.loc[scan_df.index[0], 'TEMPO Name']
            tempo_file_pat = re.compile(r'^TEMPO_NO2_L2_V\d{2}_(\d{8}T\d{6}Z)_(S\d{3})G\d{2}\.nc$')
            start_time = tempo_file_pat.match(earliest_granule).group(1)
            scan_num = tempo_file_pat.match(earliest_granule).group(2)
            
            if full_FOR:
                geo_scope = "full-FOR"

            else:
                geo_scope = build_geobounds_str(lat_domain, lon_domain)

            test_file_name = name_product_file("{}-{}".format(collection, sdr_letter), start_time, "19700101T000000Z", scan_num, geo_scope) # end time doesn't matter for this check
            sdr_file_pat = re.compile(r'SDR-TEMPO_NO2_L2_([^_]+)_(\d{4})(\d{2})(\d{2}T\d{6}Z)_[^_]+_([^_]+)_([^_\.]+)')
            name_groups = sdr_file_pat.match(test_file_name)
            sdr_file_eqv = re.compile(r'SDR-TEMPO_NO2_L2_{sdr_version}_{year}{mo}{day_time}_\d{{8}}T\d{{6}}Z_{scan}_{geo}'.format(sdr_version=name_groups.group(1), year=name_groups.group(2), mo=name_groups.group(3), day_time=name_groups.group(4), scan=name_groups.group(5), geo=name_groups.group(6)))

            sdr_path = os.path.join(save_path, 'NO2', 'L2', name_groups.group(1), name_groups.group(6), name_groups.group(2), name_groups.group(3))

            if os.path.exists(sdr_path):
                sdr_existing_files = os.listdir(sdr_path)

                if len(sdr_existing_files) > 0:
                    sdr_matching_files = [f for f in sdr_existing_files if sdr_file_eqv.match(f)]

                    n_matching_files = len(sdr_matching_files)
                    if n_matching_files > 0:
                        print('{} equivalent SDR file(s) for scan {} at save_path'.format(n_matching_files, scan))

                        if reprocess_if_exists:
                            print('Will reprocess')
                            name_w_commit = False
                            name_w_proctime = True # this one will help prevent overwriting files

                            if (not name_w_commit) & (not name_w_proctime):
                                warnings.warn('Reprocessed file will overwrite existing file')

                        else:
                            print('Will not reprocess')
                            continue # continues to next scan

            if 'BEHR Name' in list(scan_df.columns):
                file_list_behr = scan_df['BEHR Name'].to_list()
                MAT_TO_PY_SUITCASE = scan_df['BEHR Location'].to_list()[0]
                
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
                        behr_dfs[g]['BEHR Location'] = MAT_TO_PY_SUITCASE
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

            sdr_version = "{}-{}".format(collection, sdr_letter)

            if parallel_algorithm:
                scan_df_filename = 'scan_{:03d}_df.csv'.format(scan)
                scan_df.to_csv(scan_df_filename)
                scan_df_list.append(scan_df_filename)

                save_options_filename = 'scan_{:03d}_save_options.pickle'.format(scan)
                with open(save_options_filename, 'wb') as handle:
                    pickle.dump(dict(name_w_commit=name_w_commit, name_w_proctime=name_w_proctime), handle, protocol=pickle.HIGHEST_PROTOCOL)
                save_options_list.append(save_options_filename)

            else:
                print('Starting process for scan {}'.format(scan))
                print('------------------------------')                    
                amf_update_one_scan(scan_df, TEMPO, vars_path, constants_path, save_path, sdr_version, minimize_output_size, full_FOR, N_updates=N_updates, pblh=pblh, hrrr_grib=hrrr_grib, save_path_partial=save_path_partial, git_commit=git_commit, name_w_commit=name_w_commit, name_w_proctime=name_w_proctime, verbosity=5)
                print('------------------------------')
                print('')

            if remove_matlab:
                if len(behr_df) > 0:
                    for i in behr_df.index:
                        matlab_output_path = "{}/{}".format( behr_df.loc[i, 'BEHR Location'], behr_df.loc[i, 'BEHR Name'])
                        os.remove(matlab_output_path)

        
        if parallel_algorithm:
            with open("scan_file_list_transient.txt", 'w') as file:
                for i in range(len(scan_df_list)):
                    file.write("scan_df: {} save_options: {}\n".format(scan_df_list[i], save_options_list[i]))
            file.close()
    else:
        print('No files on date')

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--current_date', type=str)
    parser.add_argument('--TEMPO', type=str)
    parser.add_argument('--collection', type=str)
    parser.add_argument('--sdr_letter', type=str)
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
        amf_update_one_day(args['current_date'], args['TEMPO'], args['collection'], args['sdr_letter'], args['vars_path'], args['constants_path'], args['save_path'], bool(args['minimize_output_size']), bool(args['reprocess_if_exists']), args['N_updates'], args['pblh'], full_FOR, args['N_workers'], hrrr_grib=args['hrrr_grib'], save_path_partial=args['save_path_partial'], git_commit=args['git_commit'], prioritize_latest=True, scanlist=scanlist)

    else:
        lon_domain = np.array([args['lonmin'], args['lonmax']], dtype=float)
        lat_domain = np.array([args['latmin'], args['latmax']], dtype=float)

        amf_update_one_day(args['current_date'], args['py_to_mat_textfile'], args['PY_TO_MAT_SUITCASE'], args['MAT_TO_PY_SUITCASE'], bool(args['run_matlab']), args['TEMPO'], args['vars_path'], args['constants_path'], args['save_path'], bool(args['minimize_output_size']), bool(args['reprocess_if_exists']), args['N_updates'], args['pblh'], full_FOR, args['N_workers'], lon_domain=lon_domain, lat_domain=lat_domain, hrrr_grib=args['hrrr_grib'], save_path_partial=args['save_path_partial'], git_commit=args['git_commit'], prioritize_latest=True, scanlist=scanlist)

if __name__ == "__main__":
    main()