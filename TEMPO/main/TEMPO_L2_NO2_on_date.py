#### TEMPO_L2_NO2_on_date.py ####

# Author: Sam Beaudry
# Last changed: 2025-10-07
# Location: Signal_Derived_Retrieval/TEMPO/main
# Contact: samuel_beaudry@berkeley.edu

##################################

import numpy as np
import os
import re
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from datetime import datetime
from datetime import timedelta
import warnings

from functions.build_tempo_struct import build_tempo_struct
from functions.build_geobounds_str import build_geobounds_str

def TEMPO_L2_NO2_on_date(date_string, tempo_location, collection, vars_path, intermediate_save_location, full_FOR, lonmin=-180, lonmax=180, latmin=-90, latmax=90, replace_existing=False):
    '''
    Prepares pickled dictionaries to be read by read_main_single.m or amf_update_one_scan.py

    Parameters
    ----------
    date_string : str
        Date as YYYYMMDD
    tempo_location : str
        Path to directory with standard TEMPO datasets
    collection : str
        Version of TEMPO product to use as VXX
    vars_path : str
        Path to csv file containing group assignments for TEMPO variables
    intermediate_save_location : str
        Path of directory to save pickled dictionaries
    full_FOR : bool
        If True, prepares process for entire field of regard
    lonmin : float (Optional)
        Minimum longitude boundary if not full_FOR
    lonmax : float (Optional)
        Maximum longitude boundary if not full_FOR
    latmin : float (Optional)
        Minimum latitude boundary if not full_FOR
    latmax : float (Optional)
        Maximum latitude boundary if not full_FOR
    replace_existing : bool (Optional)
        If set to True, will overwrite existing pickle files even if they are equivalent

    Returns
    -------
    str
        Name of file listing the names of the pickled files
    
    '''
    script_dir = os.path.dirname(os.path.realpath(__file__))

    lat_domain = np.array([latmin, latmax])
    lon_domain = np.array([lonmin, lonmax])
    geobounds_str = build_geobounds_str(lat_domain, lon_domain)

    date_of_interest = datetime.strptime(date_string, "%Y%m%d")

    next_date_of_interest = date_of_interest + timedelta(days=1)
    next_date_string = next_date_of_interest.strftime("%Y%m%d")

    year_of_interest = date_of_interest.year
    month_of_interest = date_of_interest.month

    # We have to check both the selected day and the next because the daylight hours TEMPO measures over North America
    # straddle UTC midnight
    # First UTC Day
    tempo_product_location_month = "{TEMPO}/NO2/L2/{collection}/{yr:04d}/{mo:02d}".format(TEMPO=tempo_location, collection=collection, yr=year_of_interest, mo=month_of_interest)
    file_list = os.listdir(tempo_product_location_month)

    date_pattern = re.compile(r'^TEMPO_NO2_L2_{collection}_{date}T\d{{6}}Z_S\d{{3}}G\d{{2}}\.nc$'.format(collection=collection, date=date_string))
    files_on_date = [f for f in file_list if date_pattern.match(f)]
    files_on_date.sort()

    dir_list = [tempo_product_location_month for f in files_on_date]
    ###

    # Second UTC Day
    tempo_product_location_month = "{TEMPO}/NO2/L2/{collection}/{yr:04d}/{mo:02d}".format(TEMPO=tempo_location, collection=collection, yr=next_date_of_interest.year, mo=next_date_of_interest.month)
    file_list = os.listdir(tempo_product_location_month)

    next_date_pattern = re.compile(r'^TEMPO_NO2_L2_{collection}_{date}T\d{{6}}Z_S\d{{3}}G\d{{2}}\.nc$'.format(collection=collection, date=next_date_string))
    files_on_next_date = [f for f in file_list if next_date_pattern.match(f)]
    files_on_next_date.sort()

    next_dir_list = [tempo_product_location_month for f in files_on_next_date]
    ###

    all_possible_files = files_on_date + files_on_next_date
    all_possible_file_locations = dir_list + next_dir_list

    # Early check if any files exists, since the remaining filtering will return an exception
    # if the DataFrame is empty
    if len(all_possible_files) == 0:
        no_tempo_files = True

    else:
        file_df = pd.DataFrame({'File Name': all_possible_files, 'File Location': all_possible_file_locations})
        file_df['Date String'] = file_df['File Name'].str.extract(r'^TEMPO_NO2_L2_V\d{2}_(\d{8}T\d{6})Z_S\d{3}G\d{2}\.nc$')
        file_df['Time UTC'] = pd.to_datetime(file_df['Date String'], format='%Y%m%dT%H%M%S', utc=True)
        file_df['Time Eastern'] = file_df['Time UTC'].dt.tz_convert('US/Eastern')
        file_df['Time Pacific'] = file_df['Time UTC'].dt.tz_convert('US/Pacific')

        true_day = date_of_interest.day
        file_df_filt = file_df[ (file_df['Time Eastern'].dt.day == true_day) & (file_df['Time Pacific'].dt.day == true_day) ]

        if len(file_df_filt) == 0:
            no_tempo_files = True

        else:
            no_tempo_files = False
            list_of_pickled_files = []

            existing_pickled_files = os.listdir(intermediate_save_location)
            existing_pickled_files.sort()

            for i in file_df_filt.index:
                # Before proceeding, check the intermediate save location to see if the file has already been initialized
                nc_pat = re.compile(r'^(TEMPO_NO2_L2_V03_\d{8}T\d{6}Z_S\d{3}G\d{2})\.nc$')
                nc_name = file_df_filt.loc[i, 'File Name']
                file_id = nc_pat.match(nc_name).group(1)
                pickle_test_name = '{file_id}_{geo}_BEHR_initialized.pickle'.format(file_id=file_id, geo=geobounds_str)

                if pickle_test_name not in existing_pickled_files:
                    # Then produce the initialized dictionary
                    granule_full_file = os.path.join(file_df_filt.loc[i, 'File Location'], file_df_filt.loc[i, 'File Name'])

                    tempo_product = Dataset(granule_full_file, mode='r')

                    mirror_step = np.ma.getdata(tempo_product['mirror_step'][:])
                    xtrack = np.ma.getdata(tempo_product['xtrack'][:])

                    latitude = np.ma.getdata(tempo_product['geolocation']['latitude'][:])
                    longitude = np.ma.getdata(tempo_product['geolocation']['longitude'][:])

                    if full_FOR:
                        pickle_save_location, pickle_save_name = build_tempo_struct(tempo_product, vars_path, intermediate_save_location, lat_domain, lon_domain)

                    else:
                        # Determine scanline/ground-pixel range consistent with provided geobounds
                        ms_mesh, xt_mesh = np.meshgrid(mirror_step, xtrack, indexing='ij')
                        in_domain = ((latitude >= lat_domain[0]) & 
                                    (latitude <= lat_domain[1]) & 
                                    (longitude >= lon_domain[0]) & 
                                    (longitude <= lon_domain[1]))
                        
                        # If the dataset does not overlap with the domain at all
                        # skip the rest of this code and remove it from the list
                        if np.all(~in_domain):
                            #print('Rejected {}; all pixels outside domain'.format(file_df_filt.loc[i, 'File Name']))
                            continue

                        # Otherwise, continue with extracting data
                        
                        xt_mesh_filtered = xt_mesh[in_domain]
                        xt_min = xt_mesh_filtered.min()
                        xt_max = xt_mesh_filtered.max()
                        xt_domain = np.array([xt_min, xt_max])

                        # I can proceed with a normal definition of the minimum and maximum ground pixel values
                        ms_mesh_filtered = ms_mesh[in_domain]
                        ms_min = ms_mesh_filtered.min()
                        ms_max = ms_mesh_filtered.max()
                        ms_domain = np.array([ms_min, ms_max])

                        pickle_save_location, pickle_save_name = build_tempo_struct(tempo_product, vars_path, intermediate_save_location, lat_domain, lon_domain, ms_domain, xt_domain)

                else:
                    pickle_save_name = pickle_test_name

                list_of_pickled_files.append( pickle_save_name )

    # Saving
    file_savename = '{}/BEHR_initialized_dicts_on_{}_transient.txt'.format(script_dir, date_string)

    if os.path.exists(file_savename):
        warnings.warn("The file '{}' already exists. Overwriting existing file...".format(file_savename))
    #    overwrite = input("Overwrite existing file? (y/n): ")

    #    if overwrite.lower() == 'y':
    #        print("Overwriting existing file...")
    #    else:
    #        raise Exception("The file '{}' already exists. Remove this file or command to overwrite in future session.".format(file_savename))

    if no_tempo_files:
        print('No TEMPO files for date {}'.format(date_string))
        with open(file_savename, 'w') as file:
            file.write("9999: No TEMPO files")
        file.close()

    else:
        with open(file_savename, 'w') as file:
            for i in range(len(list_of_pickled_files)):
                file.write("{}\n".format(list_of_pickled_files[i]))
        file.close()

    return file_savename

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--date_string', type=str)
    parser.add_argument('--tempo_location', type=str)
    parser.add_argument('--collection', type=str)
    parser.add_argument('--fullfor', type=int)
    parser.add_argument('--lonmin', type=float)
    parser.add_argument('--lonmax', type=float)
    parser.add_argument('--latmin', type=float)
    parser.add_argument('--latmax', type=float)
    parser.add_argument('--vars_path', type=str)
    parser.add_argument('--intermediate_save_location', type=str)

    args = vars(parser.parse_args())

    # Controls whether retrieval is completed for entire field of regard (FOR)
    fullfor = bool(args['fullfor'])

    if fullfor:
        file_savename = TEMPO_L2_NO2_on_date(
                                            args['date_string'],
                                            args['tempo_location'],
                                            args['collection'],
                                            args['vars_path'],
                                            args['intermediate_save_location'],
                                            fullfor
                                            )

    else:
        file_savename = TEMPO_L2_NO2_on_date(
                                            args['date_string'],
                                            args['tempo_location'],
                                            args['collection'],
                                            args['vars_path'],
                                            args['intermediate_save_location'],
                                            fullfor,
                                            lonmin=args['lonmin'],
                                            lonmax=args['lonmax'],
                                            latmin=args['latmin'],
                                            latmax=args['latmax']
                                            )
    #print('Pickled datasets produced for {}'.format(date_string))
    #print('Files stored at: {}'.format(file_savename))
    
if __name__ == "__main__":
    main()