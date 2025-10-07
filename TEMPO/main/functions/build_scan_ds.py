#### build_scan_ds.py ####

# Author: Sam Beaudry
# Last changed: 2025-03-11
# Location: Signal_Derived_Retrieval/TEMPO/main/functions
# Contact: samuel_beaudry@berkeley.edu

##########################
import numpy as np
import xarray as xr

def build_scan_ds(granule_dict: dict, scan: int):
    '''
    Constructs scan_ds for amf_update_one_scan function using one or more granule datasets

    Parameters
    ----------
    granule_dict : dict
        Dictionary containing the granules to be concatenated.
    scan : int
        TEMPO scan number
    '''
    granule_keys = list(granule_dict.keys())
    granule_keys.sort()

    if len(granule_dict) == 1:
        # We can just copy the granule dataset to the san
        scan_ds = granule_dict[granule_keys[0]].copy()

        # Add the scan global attribute
        scan_ds = scan_ds.assign_attrs(TEMPO_scan_num=scan)

    else:
        # Get the variable dimensions and attributes from one of the granule datasets
        concat_attrs = {}
        concat_dims = {}

        for v in list(granule_dict[granule_keys[0]].variables):
            concat_attrs[v] = granule_dict[granule_keys[0]][v].attrs
            concat_dims[v] = granule_dict[granule_keys[0]][v].dims

        # Get the arrays of data
        concat_data_main = {}
        concat_data_coords = {}

        # We want to concatenate arrays in the mirror_step direction
        min_xtracks = np.array([], dtype=int)
        max_xtracks = np.array([], dtype=int)

        concat_mirrorstep = np.array([], dtype=int)
        concat_granule = np.array([], dtype=int)

        first_gran = True
        for g in granule_keys:
            # Copy over granule dataset attributes
            if first_gran:
                concat_global_attrs = granule_dict[g].attrs
                first_gran = False
            else:
                concat_global_attrs.update(granule_dict[g].attrs)

            # Get xtrack bounds
            min_xtracks = np.append(min_xtracks, granule_dict[g]['xtrack'].data.min())
            max_xtracks = np.append(max_xtracks, granule_dict[g]['xtrack'].data.max())

            # Concatenate arrays in mirrorstep direction
            concat_mirrorstep = np.append(concat_mirrorstep, granule_dict[g]['mirror_step'].data)
            concat_granule = np.append(concat_granule, granule_dict[g]['granule'].data)

        # Set XTrack range which encompasses values for all granules
        min_xtrack_overall = min_xtracks.min()
        max_xtrack_overall = max_xtracks.max()
        xtrack_range_overall = np.arange(min_xtrack_overall, max_xtrack_overall + 1, dtype=int)
        xtrack_sz = (max_xtrack_overall - min_xtrack_overall) + 1

        # Store XTrack range which corresponds to a continuous rectangular dataset
        min_xtrack_rect = min_xtracks.max()
        max_xtrack_rect = max_xtracks.min()
        concat_global_attrs['Rectangular_XTrack_Bounds'] = [min_xtrack_rect, max_xtrack_rect]

        # Determine which mirrorstep and xtrack values correspond to each granule dataset
        ms_idc = np.arange(concat_mirrorstep.size)
        xt_idc = np.arange(xtrack_sz)

        ms_idc_gran = {} # For each granule, mirrorstep indices with values
        xt_idc_gran = {} # For each granule, xtrack indicies with values
        i = 0
        for g in granule_keys:
            ms_idc_gran[g] = ms_idc[concat_granule == g]

            xtrack_range_gran = np.arange(min_xtracks[i], max_xtracks[i] + 1, dtype=int)
            xt_idc_gran[g] = xt_idc[np.isin(xtrack_range_overall, xtrack_range_gran)]

            i += 1

        # Loop through variables and concatenate in the mirrorstep dimension
        for vr in list(granule_dict[granule_keys[0]].variables):
            assert isinstance(concat_dims[vr], tuple); "expected dimension values to be stored as tuple"

            if vr == 'mirror_step':
                data = concat_mirrorstep

            elif vr == 'xtrack':
                data = xtrack_range_overall

            elif vr == 'i_mirror_step':
                data = np.arange(concat_mirrorstep.size, dtype=int)

            elif vr == 'i_xtrack':
                data = np.arange(xtrack_range_overall.size, dtype=int)

            elif concat_dims[vr] == ('mirror_step',):
                data = np.full((concat_mirrorstep.size), np.nan)
                for g in granule_keys:
                    data[ms_idc_gran[g][0]:ms_idc_gran[g][-1]+1] = granule_dict[g][vr].data

            elif concat_dims[vr] == ('mirror_step', 'xtrack'):
                data = np.full((concat_mirrorstep.size, xtrack_sz), np.nan)
                for g in granule_keys:
                    data[ms_idc_gran[g][0]:ms_idc_gran[g][-1]+1, xt_idc_gran[g][0]:xt_idc_gran[g][-1]+1] = granule_dict[g][vr].data

            elif concat_dims[vr] == ('mirror_step', 'xtrack', 'corner'):
                data = np.full((concat_mirrorstep.size, xtrack_sz, 4), np.nan)
                for g in granule_keys:
                    data[ms_idc_gran[g][0]:ms_idc_gran[g][-1]+1, xt_idc_gran[g][0]:xt_idc_gran[g][-1]+1, :] = granule_dict[g][vr].data

            elif concat_dims[vr] == ('mirror_step', 'xtrack', 'swt_level'):
                swt_size = granule_dict[granule_keys[0]]['gas_profile'].shape[2]

                data = np.full((concat_mirrorstep.size, xtrack_sz, swt_size), np.nan)
                for g in granule_keys:
                    data[ms_idc_gran[g][0]:ms_idc_gran[g][-1]+1, xt_idc_gran[g][0]:xt_idc_gran[g][-1]+1, :] = granule_dict[g][vr].data
            else:
                raise Exception('Unrecognized dimensions for variable {}:'.format(vr), concat_dims[vr])
            
            # Save as coordinate or data variable
            if (vr == 'latitude') | (vr == 'longitude'):
                # Add the dimensions, data, and attributes to the coordinate map
                concat_data_coords[vr] = (concat_dims[vr], data, concat_attrs[vr])

            else:
                if vr == 'BadGeoMask':
                    # Additional work to make variable 'BadGeoMask' behave as expected
                    # We do not have geolocations on variables included in the overall xtrack range
                    # but exclude from the granule xtrack range. Ensure that these are set to bad

                    # Where there is no data, raise the BadGeoMask flag
                    data = np.where(np.isnan(data), 1, data).astype(bool)

                # Add the dimensions, data, and attributes to the main data map
                concat_data_main[vr] = (concat_dims[vr], data, concat_attrs[vr])

        concat_global_attrs['TEMPO_scan_num'] = scan
        scan_ds = xr.Dataset(concat_data_main, coords=concat_data_coords, attrs=concat_global_attrs)

    return scan_ds