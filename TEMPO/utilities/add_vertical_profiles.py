#### add_vertical_profiles.py ####

# Author: Sam Beaudry
# Last changed: 2025-10-07
# Location: Signal_Derived_Retrieval/TEMPO/utlities
# Contact: samuel_beaudry@berkeley.edu

###################################

import numpy as np
import xarray as xr
import os
import re

def add_vertical_profiles(scan_ds: xr.Dataset, tempo_dir: str, vars: list=['gas_profile']):
    '''
    scan_ds: Dataset without vertical profiles
    tempo_dir: Path to TEMPO data directory
    vars: Optional, list of variables to add
    '''
    
    vars = ['gas_profile', 'scattering_weights', 'temperature_profile']
    
    shape_2d = scan_ds.vertical_column_troposphere.shape
    
    vertical_profiles = {}
    for v in vars:
        vertical_profiles[v] = np.full((shape_2d[0], shape_2d[1], 72), np.nan, dtype=float)
    
    id_pat = re.compile(r'^TEMPO_(NO2)_(L2)_(V\d{2})_(\d{4})(\d{2})\d{2}T\d{6}Z_S\d{3}G\d{2}\.nc$')
    granules = np.unique(scan_ds.granule).astype(int)
    
    for g in granules:
        ms_g = scan_ds.mirror_step.data[scan_ds.granule.data == g]
        xt_g = scan_ds.xtrack.data
    
        assert np.all(np.diff(ms_g) == 1)
        assert np.all(np.diff(xt_g) == 1)
    
        ms_min, ms_max, xt_min, xt_max = (ms_g.min(), ms_g.max()+1, xt_g.min(), xt_g.max()+1)
        
        ims_g = scan_ds.i_mirror_step.data[scan_ds.granule.data == g]
        ixt_g = scan_ds.i_xtrack.data
    
        assert np.all(np.diff(ims_g) == 1)
        assert np.all(np.diff(ixt_g) == 1)
    
        ims_min, ims_max, ixt_min, ixt_max = (ims_g.min(), ims_g.max()+1, ixt_g.min(), ixt_g.max()+1)
        
        original_id = scan_ds.attrs['TEMPO_standard_id_G{:02d}'.format(g)]
        re_match = id_pat.match(original_id)
        gas = re_match.group(1)
        level = re_match.group(2)
        version = re_match.group(3)
        year = re_match.group(4)
        month = re_match.group(5)
    
        original_path = os.path.join(tempo_dir, gas, level, version, year, month, original_id)
    
        original_root = xr.open_dataset(original_path, group='/')
        original_support_data = xr.open_dataset(original_path, group='support_data')
        original_support_data = original_support_data.assign_coords({
                                                                    'mirror_step': original_root.mirror_step.data, 
                                                                    'xtrack': original_root.xtrack.data
        })
        
        for v in vars:
            vertical_profiles[v][ims_min:ims_max, ixt_min:ixt_max, :] = original_support_data[v].sel(mirror_step=ms_g, xtrack=xt_g).data
    
        original_root.close()
        original_support_data.close()
    
    for v in vars:
        scan_ds[v] = (['mirror_step', 'xtrack', 'swt_level'], vertical_profiles[v])

    return scan_ds