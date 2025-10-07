#### group_shared_priors.py ####

# Author: Sam Beaudry
# Last changed: 2025-03-11
# Location: Signal_Derived_Retrieval/TEMPO/main/functions
# Contact: samuel_beaudry@berkeley.edu

#################################
import numpy as np
import xarray as xr
                                                                                    
def group_shared_priors(scan_ds: xr.Dataset):
    '''
    Matches TEMPO pixels with nearest coordinates on a priori GEOS-CF grid

    For simplicity, I assume that the pixels with the same nearest GEOS-CF cell are sharing the same prior

    Parameters
    ----------
    scan_ds : xr.Dataset
        Dataset with variables 'latitude' and 'longitude' for cell centers

    Returns
    -------
    scan_ds : xr.Dataset
        scan_ds with the added variables 'nearest_geoscf_latitude' and 'nearest_geoscf_longitude'
    '''

    scan_ds['nearest_geoscf_latitude'] = (
                                        ['mirror_step', 'xtrack'],
                                        np.round(
                                                np.round(scan_ds['latitude'].data * 4, decimals=0) / 4, 
                                                decimals=2), # round again to avoid precision issues causing mismatches later
                                        {
                                            'units': 'degrees N',
                                            'description': 'nearest latitude on the 0.25 degree GEOS-CF grid; used to determine model domain groups'
                                        }
    )

    scan_ds['nearest_geoscf_longitude'] = (
                                        ['mirror_step', 'xtrack'],
                                        np.round(
                                                np.round(scan_ds['longitude'].data * 4, decimals=0) / 4, 
                                                decimals=2), # round again to avoid precision issues causing mismatches later
                                        {
                                            'units': 'degrees E',
                                            'description': 'nearest longitude on the 0.25 degree GEOS-CF grid; used to determine model domain groups'
                                        }
    )

    # Based on brief exploration, it looks like pixels with shared priors should have total model VCD equal within 1e13 molecules / cm^2
    # This seems reasonable enough. As for why the supplied gas_profiles are not exactly the same, I wonder if there is some degree of smoothing
    # The offending pixels seem to occur closer to the edge of the shared prior domain

    return scan_ds   