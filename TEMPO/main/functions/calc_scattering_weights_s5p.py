#### calc_scattering_weights_s5p.py ####

# Author: Sam Beaudry
# Last changed: 2025-02-26
# Location: Signal_Derived_Retrieval/TEMPO/main/functions
# Contact: samuel_beaudry@berkeley.edu

####################################

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import time
import sys

def calc_scattering_weights_s5p(lut, pressures, sza, vza, raa, albedo, surf_pressure, linear_6d=True, verbose=False):
    '''Calculates scattering weights using the S5P LUT

    Using the NO2 air mass factor lookup table (AMF LUT) of the S5P retrieval, this function determines scattering 
    weight profiles. 

    Parameters
    ----------
    lut : xarray.DataArray or str
        DataArray object of LUT values with parameter dimensions, or path of the appropriate .nc file
    pressures : array_like
        Vertical pressures for interpolation (hPa)
    sza : array_like
        Solar zenith angle (degrees away from zenith)
    vza : array_like
        Viewing zenith angle (degrees away from zenith)
    raa : array_like
        Relative azimuth angle (degrees; 0 corresponds to forward scattering position where sun and satellite are opposed)
    albedo : array_like
        Surface albedo
    surf_pressure : surf_pressure (hPa)
        Surface pressure 
    linear_6d : bool, optional
        If True, linear interpolation in the pressure dimension. If False, log-log interpolation in the pressure dimension. Default is True.
    verbose : bool, optional
        Toggles print statements for debugging. Default is False.
    '''

    if isinstance(lut, str):
        lut = xr.open_dataset(lut)
        lut = lut.amf

    elif isinstance(lut, xr.core.dataset.Dataset):
        lut = lut.amf

    else:
        if not isinstance(lut, xr.core.dataarray.DataArray):
            raise Exception('lut must be a str, DataArray, or Dataset object')
        
    # As an aside, it is confusing that the damf file uses the terms "altitude" and "surface altitude"
    # to refer to pressures. I've kept that notation here, but note that anytime altitude appears
    # it is referring to the pressure values of the LUT
        
    # We will do linear interpolation on a 5D grid for all variables except vertical pressure
    # Afterwards, we will do log-log interpolation to put the output vector on the provided 
    # pressure levels

    # Check if we are running for a single location
    if (type(sza) == int) | (type(sza) == float):
        # Convert scalar quantities to arrays
        sza = np.array([sza])
        vza = np.array([vza])
        raa = np.array([raa])
        albedo = np.array([albedo])
        surf_pressure = np.array([surf_pressure])

        if (type(pressures) == int) | (type(pressures) == float):
            pressures = np.array([pressures])

        if len(pressures.shape) > 1:
            raise Exception('pressures cannot be more than a 1D array when running in scalar mode')
        
        pressures = np.expand_dims(pressures, axis=0)

    # Get size of inputs
    input_size = sza.size

    # All pixel-level arrays should have the same shape
    shape_sclr = sza.shape

    if (shape_sclr != vza.shape) | (shape_sclr != raa.shape) | (shape_sclr != albedo.shape) | (shape_sclr != surf_pressure.shape):
        raise Exception('sza, vza, raa, albedo, and surf_pressure must have the same shape')
    
    # Pressure array should have the same shape with an additional dimension at the last position
    if len(pressures.shape) < 2:
        raise Exception('pressures must have at least two dimensions')

    if (pressures[..., 0].shape) != shape_sclr:
        raise Exception('pressures must have same shape as other variables aside from last axis')
    
    # LUT validation ##################################################
    # The LUT should have the following shape/order of input parameters
    lut_shape = lut.shape
    expected_shape = (26, 14, 174, 10, 17, 11)
    parameter_sizes_true = {'albedo': 26, 'p_surface': 14, 'p': 174, 'dphi': 10, 'mu0': 17, 'mu': 11}

    if lut_shape != expected_shape:
        raise Exception('Shape of LUT {} does not match the expected shape {}'.format(lut_shape, expected_shape))

    parameter_sizes = {}
    for pa in lut.dims:
        parameter_sizes[pa] = lut[pa].size
        if parameter_sizes[pa] != parameter_sizes_true[pa]:
            raise Exception("Parameter '{}' has size {}; the expected size is {}".format(pa, parameter_sizes[pa], parameter_sizes_true[pa]))
    # /LUT validation ##################################################

    # Assign inputs to the parameters of the LUT, converting where necessary
    albedo = albedo
    p_surface = surf_pressure # hPa
    p = pressures
    dphi = raa
    mu0 = np.cos(np.radians(sza))
    mu = np.cos(np.radians(vza))

    # Calculate the geometric AMF values
    geometric_amfs = (1/mu) + (1/mu0)

    if linear_6d:
        # Tile the other variables to match the shape of p

        # Determine where to add the vertical axis and the tuple
        # specifying repetitions
        num_sclr_dims = len(shape_sclr)
        tile_reps = []
        for i in range(num_sclr_dims):
            tile_reps.append(1)
        tile_reps.append(p.shape[-1])
        tile_reps = tuple(tile_reps)

        # Tile
        albedo = np.tile(np.expand_dims(albedo, axis=num_sclr_dims), tile_reps)
        p_surface = np.tile(np.expand_dims(p_surface, axis=num_sclr_dims), tile_reps)
        dphi = np.tile(np.expand_dims(dphi, axis=num_sclr_dims), tile_reps)
        mu0 = np.tile(np.expand_dims(mu0, axis=num_sclr_dims), tile_reps)
        mu = np.tile(np.expand_dims(mu, axis=num_sclr_dims), tile_reps)
        geometric_amfs = np.expand_dims(geometric_amfs, axis=num_sclr_dims)

        # Define interpolator from the LUT
        lut_interpolator = RegularGridInterpolator(
                                            (lut['albedo'].data,
                                             lut['p_surface'].data,
                                             lut['p'].data,
                                             lut['dphi'].data,
                                             lut['mu0'].data,
                                             lut['mu'].data),
                                             lut.data,
                                             method='linear',
                                             fill_value=np.nan,
                                             bounds_error=False
        )

        # Call the interpolator
        interpolated_values = lut_interpolator((albedo, p_surface, p, dphi, mu0, mu))

        # interpolated_values should have the shape of pressures
        scattering_weights = interpolated_values * geometric_amfs
    #fi
    
    else:
        # Prepare the pixel-level arrays to go into the interpolator
        # Axis 0 must correspond to the pixel, axis 1 must correspond to the variable on the 5D grid

        if len(shape_sclr) > 1:
            # If this is true, we have to first flatten to 1D arrays
            sza = sza.flatten()
            vza = vza.flatten()
            raa = raa.flatten()
            albedo = albedo.flatten()
            surf_pressure = surf_pressure.flatten()

            # We also need to reduce dimensionality of pressures for later
            new_pres_shape = (sza.size, pressures.shape[-1])
            pressures = np.reshape(pressures, new_pres_shape)



        # Stack along a new axis 1 corresponding to the variable on the grid
        pixel_inputs = np.stack((albedo, p_surface, dphi, mu0, mu), axis=1)

        # Add the pressure dimension of the LUT
        horiz_intrp_shape = (parameter_sizes['p'], input_size)
        horiz_intrp_sw = np.full(horiz_intrp_shape, np.nan, dtype=float)

        for i in range(parameter_sizes['p']):
            horizontal_interpolator = RegularGridInterpolator( 
                                                            (lut.albedo.data,
                                                            lut.p_surface.data,
                                                            lut.dphi.data,
                                                            lut.mu0.data,
                                                            lut.mu.data),
                                                            lut.data[:, :, i, :, :, :], # If we passed the validation checks above, this collapses the vertical dimension p
                                                            method='linear',
                                                            bounds_error=False,
                                                            fill_value=None
                                                            )
            
            # Interpolate the values
            horiz_intrp_sw[i, :] = horizontal_interpolator(pixel_inputs)

        # horiz_intrp_sw now contains vertically resolved scattering weights on the pressure scale of 
        # the look up table.

        # We will interpolate in log-log space to the provided pressure scale
        # While the LUT pressure scale is consistent across all locations, the values of the scattering
        # weights are not. This means we have to create a unique interpolation object for each location :(

        print('Performing vertical pressure interpolation')
        ver_intrp_shape = (pressures.shape[-1], horiz_intrp_sw.shape[1])
        ver_intrp_vals = np.full(ver_intrp_shape, np.nan, dtype=float)

        if verbose:
            num_loc = horiz_intrp_sw.shape[1]
            num_completed = 0
            print("0%")
            last_printed_percent = 0
            timer_start = time.perf_counter()

            printer_step= 10

        # Loop over locations
        for j in range(horiz_intrp_sw.shape[1]):
            if verbose:
                percent_complete = (num_completed / num_loc) * 100
                if (percent_complete - last_printed_percent) > printer_step:
                    print("{:.0f}%".format(percent_complete))
                    last_printed_percent = percent_complete
                    timer_end = time.perf_counter()
                    time_block = timer_end - timer_start

                    if time_block < 60:
                        time_unit = 's'
                        time_scale = 1

                    else:
                        time_unit = 'min'
                        time_scale = 1/60
                        
                    time_remaining = (time_block / printer_step) * (100 - percent_complete) * time_scale
                    print("Estimated time remaining: {t:.2f} {u}".format(t=time_remaining, u=time_unit))
                    timer_start = time.perf_counter()
                
            
            # Vertical interpolation object in log-log
            vertical_interpolator = RegularGridInterpolator(
                                                        (np.log(lut.p.data),),
                                                        np.log(horiz_intrp_sw[:, j]),
                                                        method='linear',
                                                        bounds_error=False,
                                                        fill_value=None
            )

            # Interpolate to the input pressures
            ver_intrp_vals[:, j] = np.exp(vertical_interpolator(np.log(pressures[j, :])))

            if verbose:
                num_completed += 1

        # Reshape to the original shape, with pressure as the last axis
        # Start by putting pressure axis last
        ver_intrp_vals = np.transpose(ver_intrp_vals)

        # The LUT values are scattering weights ratioed by the geometric AMF
        # Tile the geometric AMF array from earlier to convert to scattering weights
        geometric_amfs = np.tile(geometric_amfs, (1, ver_intrp_vals.shape[-1]))
        ver_intrp_sw = ver_intrp_vals * geometric_amfs

        # Now reshape the axes if necessary
        if len(shape_sclr) > 1:    
            ver_intrp_sw = np.reshape(ver_intrp_sw, np.append(shape_sclr, ver_intrp_sw.shape[-1]))

        scattering_weights = ver_intrp_sw
    #esle

    return scattering_weights

def main():
    if len(sys.argv) != 8:
        raise Exception('To run as a script, must pass 7 variables: lut, pressures, sza, vza, raa, albedo, surf_pressure')

    ver_intrp_sw = calc_scattering_weights_s5p(str(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]))
    print(ver_intrp_sw)

if __name__ == "__main__":
    main()
