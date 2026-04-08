#### get_hrrr_coords.py ####

# Author: Sam Beaudry
# Last changed: 2026-04-07
# Location: Signal_Derived_Retrieval/contants
# Contact: samuel_beaudry@berkeley.edu
# Description: Creates the "hrrr_reanalysis_coordinates.nc" dataset

#############################

import os
import re
import warnings
import numpy as np
import xarray as xr
from herbie import Herbie

if os.path.exists('hrrr_reanalysis_coordinates.nc'):
    print('File already exists; will not replace')

else:
    H = Herbie("2024-04-01 21:00", model="hrrr", product="sfc", fxx=0)
    hrrr_pbl = H.xarray(":HPBL:surface:anl", remove_grib=True)

    hrrr_lats = hrrr_pbl.latitude.data
    hrrr_lons = hrrr_pbl.longitude.data # [0, 360)

    # Convert hrrr_lons to the TEMPO longitude convention
    hrrr_lons = np.where(hrrr_lons > 180, -1 * (360 - hrrr_lons), hrrr_lons) # (-180, 180]

    hrrr_coords = xr.Dataset({
        'latitude': (['y', 'x'], hrrr_lats),
        'longitude': (['y', 'x'], hrrr_lons),
    })

    wd = os.getcwd()
    constants_pat = re.compile(r'.*Signal_Derived_Retrieval\/constants$')

    if not constants_pat.match(wd):
        warnings.warn("File is being saved in an unexpected location. Algorithm expects to find coordinates at Signal_Derived_Retrieval/constants")

    hrrr_coords.to_netcdf('hrrr_reanalysis_coordinates.nc')
    print('File saved at {}'.format(os.path.join(wd, 'hrrr_reanalysis_coordinates.nc')))