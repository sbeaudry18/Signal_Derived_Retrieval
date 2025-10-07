#### add_pblh_hrrr.py ####

# Author: Sam Beaudry
# Last changed: 2025-05-25
# Location: Signal_Derived_Retrieval/TEMPO/main/functions
# Contact: samuel_beaudry@berkeley.edu

##########################
import numpy as np
import pandas as pd
import xarray as xr
from herbie import Herbie
from scipy.interpolate import NearestNDInterpolator
import datetime
from shapely.geometry import Point, Polygon
import pickle
                                                                                    
def add_pblh_hrrr(scan_ds: xr.Dataset, hrrr_polygon, hrrr_coordinates: xr.Dataset, hrrr_save_directory: str, constant_pblh: bool=False, pblh_value: float=750.):
    '''
    Matches pixels in scan_ds with the planetary boundary layer height from HRRR

    Parameters
    ----------
    scan_ds : xr.Dataset
        Dataset returned from with time_utc variable
    hrrr_polygon : Polygon
        Shapely polygon outlining HRRR domain
    hrrr_coordinates: xr.Dataset
        Dataset with latitude and longitude values of HRRR cells
    hrrr_save_directory: str
        Directory to look up or save HRRR data
    constant_pblh: bool, default False
        Whether to use the constant value given by pblh_value
    pblh_value: float, default 750
        PBLH (m) to apply to all pixels if constant_pblh = True

    Returns
    -------
    scan_ds : xr.Dataset
        scan_ds with the added variable 'boundary_layer_height'
    '''
    if constant_pblh:
        nearest_pblh = np.full((scan_ds['mirror_step'].size, scan_ds['xtrack'].size), pblh_value, dtype='f8')
        pblh_var_description = 'constant boundary layer height imposed on retrieval of updated AMF'

    else:
        pblh_var_description = 'boundary layer height from nearest HRRR analysis location'

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
        time_df['Nearest Hour'] = time_df['Time UTC'].apply(lambda t: (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)+datetime.timedelta(hours=t.minute//30)).hour)
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

            # Here's the basic loop I mention above
            for ms in ms_during_h.index:
                i_ms = scan_ds['i_mirror_step'].sel(mirror_step=ms).item()

                for i_xt in range(scan_ds['xtrack'].size):
                    # First, check if the geolocation flag is false so we can proceed
                    if not scan_ds.BadGeoMask.data[i_ms, i_xt]:
                        sat_lat = scan_ds.latitude.data[i_ms, i_xt]
                        sat_lon = scan_ds.longitude.data[i_ms, i_xt]

                        if Point(sat_lon, sat_lat).within(hrrr_polygon):
                            nearest_pblh[i_ms, i_xt] = nearest_hrrr_interpolator(sat_lon, sat_lat)

                        else:
                            # Use the default value of 750 m
                            nearest_pblh[i_ms, i_xt] = pblh_value

        # ---------------------------------------------------------------------------------------------------------------------------------------

    scan_ds['boundary_layer_height'] = (['mirror_step', 'xtrack'], nearest_pblh, {'units': 'm', 'description': pblh_var_description})
    return scan_ds