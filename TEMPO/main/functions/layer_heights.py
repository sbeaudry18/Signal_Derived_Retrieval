#### layer_heights.py ####

# Author: Sam Beaudry
# Last changed: 2025-04-24
# Location: Signal_Derived_Retrieval/TEMPO/main/functions
# Contact: samuel_beaudry@berkeley.edu

##########################

import numpy as np
import xarray as xr
from scipy import constants

def layer_heights(scan_ds: xr.Dataset, eta_a, eta_b):
    '''
    Calculates interface pressures and altitudes of TEMPO vertical layers

    Parameters
    ----------
    scan_ds : xr.Dataset
    eta_a : np.ndarray
    eta_b : np.ndarray

    Returns
    -------
    scan_ds : xr.Dataset
        scan_ds with the added variables 'interface_pressures', 'midpoint_pressures', 'interface_heights', and 'vertical_layer_thickness'
    '''

    eta_a = np.expand_dims(eta_a, axis=(0, 1))
    eta_a = np.tile(eta_a, (scan_ds['mirror_step'].size, scan_ds['xtrack'].size, 1))

    eta_b = np.expand_dims(eta_b, axis=(0, 1))
    eta_b = np.tile(eta_b, (scan_ds['mirror_step'].size, scan_ds['xtrack'].size, 1))

    surface_pressure = scan_ds['surface_pressure'].data # hPa
    surface_pressure = np.expand_dims(surface_pressure, axis=2)
    surface_pressure = np.tile(surface_pressure, (1, 1, eta_a.shape[2]))

    # Convert from hybrid-eta pressure coordinates to absolute pressure
    interface_pressures = eta_a + (eta_b * surface_pressure) # hPa, length 73

    # Take midpoint pressures as the mean of interface pressures
    midpoint_pressures = (interface_pressures[:, :, 1:] + interface_pressures[:, :, :-1]) / 2 # hPa, length 72

    # Find scale heights using the temperature profiles
    Ma = 0.029 # kg / mol
    R = constants.gas_constant # J / mol * K
    g = constants.g # m / s^2

    scale_heights = (R * scan_ds['temperature_profile'].data) / (Ma * g) # m
    scale_heights = np.where(scan_ds['temperature_profile'].data < 0, 7.8e3, scale_heights)

    # Use barometric formula find the distance between interface pressures
    layer_thickness = scale_heights * np.log(interface_pressures[:, :, :-1] / interface_pressures[:, :, 1:]) # m

    # Calculate layer heights at upper interface by summing over the layer thicknesses
    interface_heights = np.zeros(layer_thickness.shape)
    interface_heights[:, :, 0] = layer_thickness[:, :, 0]

    for i in range(1, 72):
        interface_heights[:, :, i] = interface_heights[:, :, i-1] + layer_thickness[:, :, i]

    # Add to xarray dataset
    scan_ds['interface_pressures'] = (['mirror_step', 'xtrack', 'swt_level'], interface_pressures[..., 1:], {'units': 'hPa', 'description': 'pressure values at top interface between vertical layers (does not inclue surface)'})
    scan_ds['midpoint_pressures'] = (['mirror_step', 'xtrack', 'swt_level'], midpoint_pressures, {'units': 'hPa', 'description': 'pressure values at layer interfaces'})
    scan_ds['interface_heights'] = (['mirror_step', 'xtrack', 'swt_level'], interface_heights, {'units': 'm', 'description': 'altitude of layer interfaces above surface, from barometric formula'})
    scan_ds['vertical_layer_thickness'] = (['mirror_step', 'xtrack', 'swt_level'], layer_thickness, {'units': 'm', 'description': 'vertical distance between layer interfaces, from barometric formula'})
    scan_ds['eta_a'] = (['swt_level'], eta_a[0, 0, 1:], {'units': 'hPa', 'description': 'eta_a parameter for upper interfaces of vertical layers. Lowest interface value is zero (not included)'})
    scan_ds['eta_b'] = (['swt_level'], eta_b[0, 0, 1:], {'units': 1, 'description': 'eta_b parameter for upper interfaces of vertical layers. Lowest interface value is one (not included)'})

    # Add attribute for the manually set constant
    scan_ds = scan_ds.assign_attrs({'mean_molar_mass_air': '{} kg / mol'.format(Ma)})

    return scan_ds