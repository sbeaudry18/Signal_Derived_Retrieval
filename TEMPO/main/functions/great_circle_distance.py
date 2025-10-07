# https://gist.github.com/rochacbruno/2883505
import numpy as np

def great_circle_distance(origin_lat: np.ndarray, origin_lon: np.ndarray, dest_lat: np.ndarray, dest_lon: np.ndarray, units: str='km', float_mode: bool=False):
    '''
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