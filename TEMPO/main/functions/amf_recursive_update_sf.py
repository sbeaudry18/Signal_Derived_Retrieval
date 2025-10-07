#### amf_recursive_update_sf.py ####

# Author: Sam Beaudry
# Last changed: 2025-05-21
# Location: Signal_Derived_Retrieval/TEMPO/main/functions
# Contact: samuel_beaudry@berkeley.edu

#####################################

# Input arrays should have dimensions [pixel, level] where
# pixel indexes the pixel in the model domain and level
# indexes the vertical level
import numpy as np

def amf_calculator(no2_columns: np.ndarray, box_amfs: np.ndarray):
    if no2_columns.shape != box_amfs.shape:
        raise ValueError('no2_columns and box_amfs must have same shape. Input shapes were ' + str(no2_columns.shape) + ' and ' + str(box_amfs.shape) + 'respectively')
    amf = np.sum(no2_columns * box_amfs) / np.sum(no2_columns)
    return amf

def mismatch_check(apriori_partial_columns, retrieved_vcds, trop_index, pixel_area, Np, mismatch_itr=1):
    #################################
    #### Prior/Observed Mismatch ####
    #################################
    # 2025-05-21: This method relies on the GEOS-CF prior being correct at the model resolution
    # However, there are cases where this may not be true. For example, an isolated strong source
    # will produce very sharp gradients. If meteorology is not appopriately represented, then the 
    # model will not produce accurate priors even at native resolution. Specifically, the model
    # might place the plume in an incorrect cell. When this occurs, the model no longer offers useful
    # information on the NO2 column density that can be redistributed. 

    # One way to check for this is by comparing the SCD informed VCDs with the a prior at the model 
    # resolution. If the retrieved VCD is much greater or much smaller than the a priori VCD, flag
    # the pixels as having a potentially problematic retrieval.

    # Find the a priori tropospheric VCD
    apriori_trop_nd = 0
    apriori_area = 0

    for p in range(Np):
        apriori_trop_nd += np.sum(apriori_partial_columns[mismatch_itr, p, :(trop_index[p]+1)]) * pixel_area[p]
        apriori_area += pixel_area[p]

    apriori_trop_vcd = apriori_trop_nd / apriori_area

    # Find the retrieved tropospheric VCD
    retrieved_trop_vcd = np.sum(retrieved_vcds[mismatch_itr, :] * pixel_area) / np.sum(pixel_area)

    # Determine the mismatch
    retrieved_over_apriori = retrieved_trop_vcd / apriori_trop_vcd

    # Flags
    if retrieved_over_apriori > 3:
        # Bad: model significantly underpredicts
        retrieved_model_mismatch_flag = 1

    elif retrieved_over_apriori < (1/3):
        # Bad: model significantly overpredicts
        retrieved_model_mismatch_flag = -1

    else:
        # Good: model and retrieved VCD reasonably agree
        retrieved_model_mismatch_flag = 0

    return retrieved_over_apriori, int(retrieved_model_mismatch_flag)

def amf_recursive_update(
                         model_partial_columns: np.ndarray,
                         box_amfs: np.ndarray,
                         original_amf: np.ndarray, 
                         original_retrieved_vcd: np.ndarray, 
                         trop_index: np.ndarray,
                         boundary_layer_index: np.ndarray,
                         model_boundary_layer_vcd: np.ndarray,
                         model_tropospheric_vcd: np.ndarray,
                         pixel_area: np.ndarray,
                         good_pixels: np.ndarray,
                         Ni: int=1,
                         initial_final_only: bool=True,
                         ):
    '''
    model_partial_columns: array of partial columns from CTM output
    box_amfs: array of altitude-dependent air mass factors
    original_amf: air mass factor resulting from retrieval using provided gas_profile and scattering_weights
    original_retrieved_vcd: vertical column density from slant column density and original_amf
    trop_index: index of vertical dimension located at the tropopause (set as -1 for total column retrieval)
    boundary_layer_index: index of vertical dimension which contains the top of the boundary layer
    model_boundary_layer_vcd: vertical column density of the model profile within the boundary layer (per pixel)
    model_tropospheric_vcd: vertical column density of the model profile within the troposphere (per pixel)
    pixel_area: array of pixel areas
    good_pixels: bool array indicating whether pixels should be used in the update
    Ni: the number of times to update the AMF
    initial_final_only: optional (default = True), whether to return just the initial and final iterations rather than all
    '''

    # Column densities use SI units: mol m^-2

    if model_partial_columns.shape != box_amfs.shape:
        raise Exception('model_profile and box_amfs must have the same shape')
    
    # Pre-define array shapes
    # Ni: the number of update iterations
    Np = original_amf.size # number of pixels
    Nlev = model_partial_columns.shape[1] # number of vertical levels

    flat_shape = (Ni+1, Np)
    resolved_shape = (Ni+1, Np, Nlev)

    apriori_partial_columns = np.zeros(resolved_shape)
    amfs = np.full(flat_shape, np.nan)
    retrieved_vcds = np.full(flat_shape, np.nan)
    apriori_boundary_layer_vcd = np.full((Ni+2, Np), np.nan)

    iteration_record = np.array([], dtype=int)
    percent_difference_from_previous_iteration = np.full((Ni+1, Np), np.nan)

    # Define good pixels as those with a QA value above 0.75 (per PUM)

    model_boundary_layer_vcd = np.where(np.isnan(model_boundary_layer_vcd), 0, model_boundary_layer_vcd)

    # Find the slant column density
    scd = original_retrieved_vcd * original_amf

    # Find the amount of boundary layer NO2 available to redistribute
    model_bl_no2_total = np.sum(model_boundary_layer_vcd[good_pixels] * pixel_area[good_pixels]) # mol

    bl_spatial_sf = np.zeros((Ni+1, good_pixels.size))

    S_vertical_bl = np.full((Np, Nlev), np.nan)

    # 2025-03-11
    # Among good pixels (i.e. the pixels considered in the update) we want an estimate
    # of the portion of the tropospheric VCD that is the free troposphere
    # When we then calculate retrieved tropospheric VCD for these pixels, we can use this to estimate retrieved 
    # free-tropospheric VCD
    model_trop_no2_total = np.sum(model_tropospheric_vcd[good_pixels] * pixel_area[good_pixels])
    portion_free_trop = 1 - (model_bl_no2_total / model_trop_no2_total)

    # Record the free tropospheric VCD removed for each iteration
    removed_ft_in_practice = np.array([], dtype=float)

    ###################
    #### Main Loop ####
    ###################
        
    for i in range(Ni+1):
        iteration_record = np.append(iteration_record, i)
        if i == 0: 
            ########################
            #### Initialization ####
            ########################
            
            # Starting iteration; work with original retrieval
            # Set the prior based on the provided model_partial_columns 
            # Also, predetermine the NO2 boundary layer shape factors, S_bl
            # We don't update these, just the value of the boundary layer column    

            apriori_boundary_layer_vcd[i, :] = model_boundary_layer_vcd

            bl_prior = np.zeros((Np, Nlev))

            for p in range(Np):
                # Set the prior using the original gas profile from the product
                apriori_partial_columns[i, p, :] = model_partial_columns[p, :]

                if np.any(good_pixels == p):
                    # Find the BL shape factors
                    if boundary_layer_index[p] < 0:
                        bl_prior[p, 0] = apriori_partial_columns[i, p, 0]
                        print('boundary layer pause is in lowest model level; problems may occur')
                        #raise Exception('boundary_layer_index is at zero; problems may occur')

                    else:
                        bl_prior[p, :(boundary_layer_index[p]+1)] = apriori_partial_columns[i, p, :(boundary_layer_index[p]+1)]
                        
                    # Compute vertical shape factors for boundary layer only
                    S_vertical_bl[p, :] = bl_prior[p, :] / model_boundary_layer_vcd[p]

                #assert np.abs((np.sum(bl_prior[p, :]) - model_boundary_layer_vcd[p]) / model_boundary_layer_vcd[p]) < 1e-5, 'bl_prior used to create shape factors is inconsistent with model_bl_vcd provided to function. It should be ' + str(model_boundary_layer_vcd[p]) + ' but was calculated as ' + str(np.sum(bl_prior[p, :]))

                amfs[i, p] = amf_calculator(apriori_partial_columns[i, p, :(trop_index[p]+1)], box_amfs[p, :(trop_index[p]+1)])
                retrieved_vcds[i, p] = scd[p] / amfs[i, p]

        else:
            ########################################################
            #### Update Partial Columns, AMF, and Retrieved VCD ####
            ########################################################

            # Start by copying over the previous prior
            apriori_partial_columns[i, ...] = apriori_partial_columns[i-1, ...].copy()
            
            for p in range(Np):
                if np.any(good_pixels == p):
                    # Only do the replacement for good pixels involved in the process

                    # Retain the shape factor of the partial column profiles
                    # Multiply boundary layer shape factors by the new boundary layer column to get the boundary layer prior
                    new_bl_prior =  S_vertical_bl[p, :] * apriori_boundary_layer_vcd[i, p]
            
                    assert np.abs(np.sum(new_bl_prior) - apriori_boundary_layer_vcd[i, p]) < 1e-9, 'the sum of partial columnds in new_bl_priors is ' + str(np.sum(new_bl_prior)) + ' while the calculated prior_bl_vcd of this i is ' + str(apriori_boundary_layer_vcd[i, p])

                    apriori_partial_columns[i, p, :(boundary_layer_index[p]+1)] = new_bl_prior[:(boundary_layer_index[p]+1)]


                    assert np.all(apriori_partial_columns[i, p, (boundary_layer_index[p]+1):] == apriori_partial_columns[i-1, p, (boundary_layer_index[p]+1):]), 'values above the layer containing PBLH should not be altered, but are'

                    assert np.all(apriori_partial_columns[i, p, :] >= 0), 'partial columns are being set with negative values'

                amfs[i, p] = amf_calculator(apriori_partial_columns[i, p, :trop_index[p]+1], box_amfs[p, :trop_index[p]+1])
                retrieved_vcds[i, p] = scd[p] / amfs[i, p]

            percent_difference_from_previous_iteration[i, :] = 100 * (retrieved_vcds[i, :] - retrieved_vcds[i-1, :]) / retrieved_vcds[i-1, :]

        ###################################################################
        #### Adjust a priori Boundary Layer VCD Based on Retrieved VCD ####
        ###################################################################

        # Start by copying over existing values for a priori boundary layer vcds
        apriori_boundary_layer_vcd[i+1, :] = apriori_boundary_layer_vcd[i, :].copy()

        if len(good_pixels) > 1:
            # Find the retrieved VCD across all pixels
            retrieved_vcd_all = (1/np.sum(pixel_area[good_pixels])) * np.sum(retrieved_vcds[i, good_pixels] * pixel_area[good_pixels])

            # Compute the retrieved free tropospheric portion for all pixels
            retrieved_ft_vcd = portion_free_trop * retrieved_vcd_all # moles / m^2

            if retrieved_ft_vcd > retrieved_vcds[i, good_pixels].min():
                # If this is true, a better approximation is just to say that the
                # free tropospheric column is equal to the minimum tropospheric VCD in the domain
                retrieved_ft_vcd = retrieved_vcds[i, good_pixels].min()

            # Subtract the overall free tropospheric column from the individual retrieved vcds
            retrieved_bl_vcd = retrieved_vcds[i, good_pixels] - retrieved_ft_vcd # moles / m^2

            # Compute shape factor
            bl_spatial_sf[i, :] = (1/np.sum(retrieved_bl_vcd)) * retrieved_bl_vcd 

            #print(retrieved_bl_vcd)
            #print('')
            #print(bl_spatial_sf[i, :])
            #print(np.abs(1-np.sum(bl_spatial_sf[i, :])))

            if np.isnan(np.abs(1-np.sum(bl_spatial_sf[i, :]))):
                print(good_pixels)
                print(retrieved_vcds[i, good_pixels])
                print(retrieved_ft_vcd)
                print(retrieved_bl_vcd)

            assert np.abs(1-np.sum(bl_spatial_sf[i, :])) < 1e-10, 'shape factors do not sum to 1: {}'.format(np.sum(bl_spatial_sf[i, :]))

            if np.any(bl_spatial_sf[i, :] < 0):
                print(bl_spatial_sf[i, :].sum())
                raise Exception('some spatial shape factors are less than zero')

            proposed_boundary_layer_vcd = (bl_spatial_sf[i, :] * model_bl_no2_total) / pixel_area[good_pixels] # moles / m^2

            assert np.abs(np.sum(model_boundary_layer_vcd[good_pixels] * pixel_area[good_pixels]) - np.sum(proposed_boundary_layer_vcd * pixel_area[good_pixels])) < 1e-8, 'proposal ' + str(np.sum(proposed_boundary_layer_vcd * pixel_area[good_pixels])) + ' mol is inconsistent with initial ' + str(np.sum(model_boundary_layer_vcd[good_pixels] * pixel_area[good_pixels])) + ' mol'

            # Now that the proposed values for the boundary layer VCD:
            # (a) Conserve mass from the previous values
            # (b) Are all possitive
            # we can set them as the apriori_boundary_layer_vcd for the next iteration of the loop

            counter = 0
            for gp in good_pixels:
                apriori_boundary_layer_vcd[i+1, gp] = proposed_boundary_layer_vcd[counter]
                counter += 1

            removed_ft_in_practice = np.append(removed_ft_in_practice, retrieved_ft_vcd)

        else:
            #print('No good pixels')
            removed_ft_in_practice = np.append(removed_ft_in_practice, 0)

        if not np.all(apriori_boundary_layer_vcd[i+1, :] >= 0):
            print(Np)
            print(len(good_pixels))
            print(model_boundary_layer_vcd)
            print(apriori_boundary_layer_vcd[i+1, :])

        #assert np.all(apriori_boundary_layer_vcd[i+1, :] >= 0), "values of apriori_boundary_layer_vcd[i+1, :] will lead to negative partial column values in the next iteration's prior"

    if (initial_final_only) & (Ni > 2):
        # 2025-05-06: Even if we are not going to keep the values at all iterations, we should at minimum
        # keep the initial value (i=0), the first updated value (i=1), the second updated value (i=2), and
        # the final value (i=Ni). The difference between i=1 and i=2 reveals how important the later iterations
        # are; ideally, we would like to set Ni=2.
        
        apriori_partial_columns = np.concatenate((apriori_partial_columns[0:3, ...], apriori_partial_columns[-1:, ...]), axis=0)
        amfs = np.concatenate((amfs[0:3, ...], amfs[-1:, ...]), axis=0)
        retrieved_vcds = np.concatenate((retrieved_vcds[0:3, ...], retrieved_vcds[-1:, ...]), axis=0)
        iteration_record = np.concatenate((iteration_record[0:3], iteration_record[-1:]), axis=0)
        percent_difference_from_previous_iteration = np.concatenate((percent_difference_from_previous_iteration[0:3, ...], percent_difference_from_previous_iteration[-1:, ...]), axis=0)
        #print(removed_ft_in_practice)
        removed_ft_in_practice = np.array([removed_ft_in_practice[0], removed_ft_in_practice[1], removed_ft_in_practice[2], removed_ft_in_practice[-1]], dtype=float)

    retrieved_over_apriori, retrieved_model_mismatch_flag = mismatch_check(apriori_partial_columns, retrieved_vcds, trop_index, pixel_area, Np)

    return apriori_partial_columns, amfs, retrieved_vcds, iteration_record, percent_difference_from_previous_iteration, portion_free_trop, removed_ft_in_practice, retrieved_over_apriori, retrieved_model_mismatch_flag

def amf_recursive_update_no_good_pixels(                         
                         model_partial_columns: np.ndarray,
                         box_amfs: np.ndarray,
                         original_amf: np.ndarray, 
                         original_retrieved_vcd: np.ndarray, 
                         trop_index: np.ndarray,
                         boundary_layer_index: np.ndarray,
                         model_boundary_layer_vcd: np.ndarray,
                         model_tropospheric_vcd: np.ndarray,
                         pixel_area: np.ndarray,
                         good_pixels: np.ndarray,
                         Ni: int=1,
                         initial_final_only: bool=True
                         ):

    if model_partial_columns.shape != box_amfs.shape:
        raise Exception('model_profile and box_amfs must have the same shape')
    
    # Pre-define array shapes
    # Ni: the number of update iterations
    Np = original_amf.size # number of pixels
    Nlev = model_partial_columns.shape[1] # number of vertical levels

    flat_shape = (Np)
    resolved_shape = (Np, Nlev)

    apriori_partial_columns = model_partial_columns
    amfs = np.full(flat_shape, np.nan)
    retrieved_vcds = np.full(flat_shape, np.nan)

    # Find the slant column density
    scd = original_retrieved_vcd * original_amf

    for p in range(Np):
        # Set the prior using only the tropospheric component of the model profile\
        amfs[p] = amf_calculator(model_partial_columns[p, :(trop_index[p]+1)], box_amfs[p, :(trop_index[p]+1)])
        retrieved_vcds[p] = scd[p] / amfs[p]

    if (initial_final_only) & (Ni > 1):
        apriori_partial_columns = np.tile(np.expand_dims(apriori_partial_columns, axis=0), (3, 1, 1))
        amfs = np.tile(np.expand_dims(amfs, axis=0), (3, 1))
        retrieved_vcds = np.tile(np.expand_dims(retrieved_vcds, axis=0), (3, 1))
        iteration_record = iteration_record = np.array([0, 1, Ni], dtype=int)
        percent_difference_from_previous_iteration = np.full((3, Np), 0)
        portion_free_trop = np.nan
        removed_ft_in_practice = np.full((3), np.nan)

    retrieved_over_apriori, retrieved_model_mismatch_flag = mismatch_check(apriori_partial_columns, retrieved_vcds, trop_index, pixel_area, Np)

    return apriori_partial_columns, amfs, retrieved_vcds, iteration_record, percent_difference_from_previous_iteration, portion_free_trop, removed_ft_in_practice, retrieved_over_apriori, retrieved_model_mismatch_flag