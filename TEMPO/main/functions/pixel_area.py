#### pixel_area.py ####

# Author: Sam Beaudry
# Last changed: 2025-03-10
# Location: Signal_Derived_Retrieval/TEMPO/main/functions
# Contact: samuel_beaudry@berkeley.edu

#######################

import numpy as np
import xarray as xr

def pixel_area(scan_ds: xr.Dataset):
    '''
    Calculates the area covered by each pixel

    Parameters
    ----------
    scan_ds : xr.Dataset
        Dataset returned from stack_TEMPO_datasets

    Returns
    -------
    scan_ds : xr.Dataset
        scan_ds with the added variable 'area'
    '''
    
    def great_circle_distance(origin_lat: np.ndarray, origin_lon: np.ndarray, dest_lat: np.ndarray, dest_lon: np.ndarray, units: str='km', float_mode: bool=False):
        '''
        Computes great circle distance between a pair of latitude/longitude points

        For more details, visit https://gist.github.com/rochacbruno/2883505

        origin_lat: array of latitude coordinates for the origin points
        origin_lon: array of longitude coordinates for the origin points
        dest_lat: array of latitude coordinates for the destination points
        dest_lon: array of longitude coordinates for the destination points
        '''
        #if not float_mode:
        #    if not (origin_lat.shape == origin_lon.shape) & (origin_lat.shape == dest_lat.shape) & (origin_lat.shape == dest_lon.shape):
        #        raise ValueError("input arrays must have the same dimensions")
        
        radius = 6371 # km

        dlat = np.radians(dest_lat-origin_lat)
        dlon = np.radians(dest_lon-origin_lon)
        a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(origin_lat)) \
            * np.cos(np.radians(dest_lat)) * np.sin(dlon/2) * np.sin(dlon/2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d = radius * c # km

        if units.lower() == 'm':
            d = d * 1e3 # m

        elif units.lower() != 'km':
            raise ValueError("units must be either 'm' or 'km' , not " + str(units))

        return d 

    # Following trends_a_priori_in_domain.ipynb, treat each pixel as a parallelogram

    # Distance along the xtrack direction
    xtrack_length = great_circle_distance(
                                            scan_ds['latitude_bounds'].data[..., 2], scan_ds['longitude_bounds'].data[..., 2], # top-right
                                            scan_ds['latitude_bounds'].data[..., 1], scan_ds['longitude_bounds'].data[..., 1], # bottom-right
                                            units='m' 
                                            )

    # Distance along the mirror_step direction
    mirror_step_length = great_circle_distance(
                                            scan_ds['latitude_bounds'].data[..., 0], scan_ds['longitude_bounds'].data[..., 0], # bottom-left
                                            scan_ds['latitude_bounds'].data[..., 1], scan_ds['longitude_bounds'].data[..., 1], # bottom-right
                                            units='m'
                                            )

    # Approximate the tilt of the parallelogram using angle at the bottom-right corner
    sweep_angle = np.arctan2(scan_ds['latitude_bounds'].data[..., 1] - scan_ds['latitude_bounds'].data[..., 0], scan_ds['longitude_bounds'].data[..., 1] - scan_ds['longitude_bounds'].data[..., 0])
    flight_angle = np.arctan2(scan_ds['latitude_bounds'].data[..., 2] - scan_ds['latitude_bounds'].data[..., 1], scan_ds['longitude_bounds'].data[..., 2] - scan_ds['longitude_bounds'].data[..., 1])
    bottom_right_angle = sweep_angle + np.pi - flight_angle

    # Compute pixel area
    pixel_height = xtrack_length * np.sin(bottom_right_angle)
    pixel_area = pixel_height * mirror_step_length # m^2

    scan_ds['area'] = (
                            ['mirror_step', 'xtrack'],
                            pixel_area,
                            {
                                'units': 'm^2',
                                'description': 'area of pixel footprint calculated from latitude/longitude bounds treating the pixel as a parallelogram'
                            }
    )

    return scan_ds