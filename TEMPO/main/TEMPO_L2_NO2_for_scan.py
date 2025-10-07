#### TEMPO_L2_NO2_for_scan.py ####

# Author: Sam Beaudry
# Last changed: 2025-10-07
# Location: Signal_Derived_Retrieval/TEMPO/main
# Contact: samuel_beaudry@berkeley.edu

###################################

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import xarray as xr
import sys

from functions.build_tempo_struct import build_tempo_struct

def TEMPO_L2_NO2_for_scan(scan_ds, tempo_head, vars_path, intermediate_save_location):
    '''
    Produces initialized structures for each granule corresponding to the provided scan_ds

    Parameters
    ----------
    scan_ds : nc.Dataset
        Existing scan_ds from a full run of amf_update_one_scan.py
    tempo_head: str
        Head of the TEMPO dataset directory
    vars_path: str
        Full file name of the .csv file containing TEMPO variable names, locations, and corresponding OMI variables
    intermediate_save_location: str
        Directory to store the temporary pickled dictionary built by this function

    Returns
    -------

    '''
    # Determine which granules were used to produce this scan_ds
    unique_granules = np.unique(scan_ds.granule).astype(int)

    list_of_pickled_files = []

    for g in unique_granules:
        lat_domain = scan_ds.attrs['LatBdy_G{:02}'.format(g)]
        lon_domain = scan_ds.attrs['LonBdy_G{:02}'.format(g)]

        tempo_id = scan_ds.attrs['TEMPO_standard_id_G{:02}'.format(g)]
        level = tempo_id[10:12]
        collection = tempo_id[13:16]
        year = tempo_id[17:21]
        month = tempo_id[21:23]
        tempo_path = '{}/NO2/{}/{}/{}/{}/{}'.format(tempo_head, level, collection, year, month, tempo_id) 

        tempo_product = Dataset(tempo_path, mode='r')
            # Get mirror_step and xtrack bounds (adapted from TEMPO_L2_NO2_on_date.py)
        mirror_step = np.ma.getdata(tempo_product['mirror_step'][:])
        xtrack = np.ma.getdata(tempo_product['xtrack'][:])

        latitude = np.ma.getdata(tempo_product['geolocation']['latitude'][:])
        longitude = np.ma.getdata(tempo_product['geolocation']['longitude'][:])

        ms_mesh, xt_mesh = np.meshgrid(mirror_step, xtrack, indexing='ij')
        in_domain = ((latitude >= lat_domain[0]) & 
                    (latitude <= lat_domain[1]) & 
                    (longitude >= lon_domain[0]) & 
                    (longitude <= lon_domain[1]))
        
        xt_mesh_filtered = xt_mesh[in_domain]
        xt_min = xt_mesh_filtered.min()
        xt_max = xt_mesh_filtered.max()
        xt_domain = np.array([xt_min, xt_max])

        ms_mesh_filtered = ms_mesh[in_domain]
        ms_min = ms_mesh_filtered.min()
        ms_max = ms_mesh_filtered.max()
        ms_domain = np.array([ms_min, ms_max])

        pickle_save_location, pickle_save_name = build_tempo_struct(tempo_product, vars_path, intermediate_save_location, lat_domain, lon_domain, ms_domain, xt_domain)
        list_of_pickled_files.append( pickle_save_name )

        tempo_product.close()    

    # Save record file for bash script
    file_savename = 'BEHR_initialized_dicts_posthoc_transient.txt'

    with open(file_savename, 'w') as file:
        for i in range(len(list_of_pickled_files)):
            file.write("{}\n".format(list_of_pickled_files[i]))
    file.close()

def main():
    scan_ds_path = sys.argv[1]
    tempo_head = sys.argv[2]
    vars_path = sys.argv[3]
    intermediate_save_location = sys.argv[4]

    scan_ds = xr.open_dataset(scan_ds_path)

    TEMPO_L2_NO2_for_scan(scan_ds, tempo_head, vars_path, intermediate_save_location)

    scan_ds.close()

if __name__ == "__main__":
    main()