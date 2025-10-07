#### build_tempo_struct.py ####

# Author: Sam Beaudry
# Last changed: 2025-04-02
# Location: Signal_Derived_Retrieval/TEMPO/main/functions
# Contact: samuel_beaudry@berkeley.edu

################################

import numpy as np
import pandas as pd
import netCDF4 as nc
import datetime as dt
import pickle
import os

from functions.build_geobounds_str import build_geobounds_str

def build_tempo_struct(tempo_ds, vars_path: str, pickle_save_location: str, lat_domain: np.ndarray, lon_domain: np.ndarray, ms_domain=np.array([0, -1]), xt_domain=np.array([0, -1]), read_main_single_only: bool=True, close_when_finished=False):
    '''
    Reads TEMPO data into a dictionary that can be interpreted by MATLAB as a structure

    The BEHR data process involves the creation and processing of structures. Rather than rewrite all of the 
    BEHR code to work with numpy objects, it is easier to transform TEMPO data into a form interpretable
    by the existing MATLAB code. This function pulls the relevant variables needed for the BEHR portion of 
    the BEHR-RED retrieval.

    Parameters
    ----------
    tempo_file : Dataset
        Opened TEMPO dataset

    vars_path : str
        Path for the csv file containing the variable names from OMI, TROPOMI, and TEMPO

    pickle_save_location : str
        Directory to store the temporary pickled dictionary built by this function

    lat_domain : ndarray
        Latitude boundaries [south, north]

    lon_domain : ndarray
        Longitude boundaries [west, east]

    ms_domain : ndarray, optional
        Mirror_step boundaries [minimum, maximum] corresponding to lat_domain and lon_domain

    xt_domain : ndarray, optional
        Xtrack boundaries [minimum, maximum] corresponding to lat_domain and lon_domain

    read_main_single_only : bool, optional
        Default True. Sets whether all BEHR and SP variables are read in, or only those necessary to run read_main_single.m

    Returns
    -------
    str
        Path to the output MATLAB structure
    '''

    # Load DataFrame indicating which TEMPO variables correspond to the BEHR variables
    vars_df = pd.read_csv(vars_path)
    # Columns: OMI Variable | TROPOMI Group | TROPOMI Name | TEMPO Group | TEMPO Name
    vars_df.set_index('OMI Variable', inplace=True)

    # Identify the variables we want
    sp_variables = [
                    "mirror_step", "xtrack",
                    "Longitude", "Latitude", "SpacecraftAltitude", "SpacecraftLatitude",
                    "SpacecraftLongitude", "Time", "ViewingZenithAngle",
                    "SolarZenithAngle", "ViewingAzimuthAngle", "SolarAzimuthAngle",
                    "AmfStrat", "AmfTrop", "CloudFraction", "CloudRadianceFraction",
                    "TerrainHeight", "TerrainPressure", "TerrainReflectivity",
                    "CloudPressure", "ColumnAmountNO2", "SlantColumnAmountNO2",
                    "ColumnAmountNO2Trop", "ColumnAmountNO2TropStd", "ColumnAmountNO2Strat",
                    "TropopausePressure", "VcdQualityFlags", "XTrackQualityFlags"
                    ]
    
    pixcor_variables = [
                        "TiledArea", "TiledCornerLongitude", "TiledCornerLatitude",
                        "FoV75Area", "FoV75CornerLongitude", "FoV75CornerLatitude"
                        ]
    
    # TEMPO doesn't separate pixel corners into a different dataset, so merge
    sp_variables = sp_variables + pixcor_variables

    behr_variables = [
                       "Date", "Grid", "LatBdy", "LonBdy", "Row", "Swath", "RelativeAzimuthAngle",
                       "MODISCloud", "MODISAlbedo", "MODISAlbedoQuality", "MODISAlbedoFillFlag", "GLOBETerrainHeight",
                       "IsZoomModeSwath", "AlbedoOceanFlag", "OMPIXCORFile", "MODISCloudFiles", "MODISAlbedoFile",
                       "GitHead_Core_Read", "GitHead_BEHRUtils_Read", "GitHead_GenUtils_Read", "OMNO2File", "BEHRRegion"]
    
    # Also create a list of the variables purely necessary to call read_main_single in MATLAB
    # These variables will enable the BEHR algorithm to match pixels with albedo, cloud, and terrain
    # height information. The remaining variables are left for later

    if read_main_single_only:
        read_main_single_vars = [
                                "mirror_step", "xtrack", "TiledCornerLongitude", "TiledCornerLatitude", "Longitude", "Latitude",
                                "MODISCloud", "MODISCloudFiles", "SolarZenithAngle", "ViewingZenithAngle", "SolarAzimuthAngle",
                                "ViewingAzimuthAngle", "RelativeAzimuthAngle", "MODISAlbedo", "MODISAlbedoQuality",
                                "MODISAlbedoFillFlag", "MODISAlbedoFile", "AlbedoOceanFlag", "GLOBETerrainHeight",
                                "Date", "LonBdy", "LatBdy", "BEHRRegion"
                                ]
        
        # Keep only the variables we absolutely need to run read_main_single.m
        sp_variables = [v for v in sp_variables if v in read_main_single_vars]
        behr_variables = [v for v in behr_variables if v in read_main_single_vars]
    
    # Create dictionary
    dict_for_struct = {}

    # Open the netCDF4 TEMPO dataset (if we haven't already)
    if type(tempo_ds) == str:
        close_when_finished = True
        tempo_ds = nc.Dataset(tempo_ds, mode='r')

    elif type(tempo_ds) == nc._netCDF4.Dataset:
        tempo_ds = tempo_ds

    else:
        raise Exception('Invalid input for tempo_ds: must be a netCDF4 dataset object or the complete path to a TEMPO dataset')
    
    # First save the name of the TEMPO Dataset
    dict_for_struct["TEMPOProductID"] = os.path.basename(tempo_ds.local_granule_id)

    try:
        # Split the datasets into each group
        tempo_ds_groups = {
                            "product": tempo_ds["product"],
                            "geolocation": tempo_ds["geolocation"],
                            "support_data": tempo_ds["support_data"],
                            "qa_statistics": tempo_ds["qa_statistics"]
        }

        # Loop through sp_variables
        for v in sp_variables:
            # SB 2025-04-02: Added to assist downstream functionality
            if (v == 'mirror_step') | (v == 'xtrack'):
                dict_for_struct[v] = tempo_ds[v][:]

            else:
                tempo_var_location = vars_df.loc[v, 'TEMPO Group']
                tempo_var_name = vars_df.loc[v, 'TEMPO Name']

                if tempo_var_name == 'none':
                    dict_for_struct[v] = float(0)

                elif (tempo_var_name == 'mirror_step') | (tempo_var_name == 'xtrack'):
                    dict_for_struct[v] = tempo_ds[v][:]

                else:
                    # Slice to remove the time axis 0
                    dict_for_struct[v] = tempo_ds_groups[tempo_var_location][tempo_var_name][:]

        # Create placeholders for the BEHR variables
        for v in behr_variables:
            dict_for_struct[v] = float(0)

        # Determine the mask for bad geolocations
        dict_for_struct["BadGeoMask"] = np.ma.getmask(tempo_ds['geolocation']['latitude'][:]).astype(float)

        # Substitute domain values for the boundary variables
        dict_for_struct["LatBdy"] = lat_domain
        dict_for_struct["LonBdy"] = lon_domain

        ms_or_xt_restrictions = False

        # Add values for scanline and groundpixel domains
        custom_ms_bounds = ~(ms_domain == np.array([0, -1]))
        custom_xt_bounds = ~(xt_domain == np.array([0, -1]))

        if np.any(custom_ms_bounds) | np.any(custom_xt_bounds):
            ms_or_xt_restrictions = True

        if not custom_ms_bounds[1]:
            # Replace -1 placeholder with true last value
            ms_domain[1] = dict_for_struct["Longitude"].shape[0] - 1

        if not custom_xt_bounds[1]:
            # Replace -1 placeholder with true last value
            xt_domain[1] = dict_for_struct["Longitude"].shape[1] - 1


        dict_for_struct["NativeDimRestrictions"] = ms_or_xt_restrictions
        dict_for_struct["MirrorStepBdy"] = ms_domain
        dict_for_struct["XTrackBdy"] = xt_domain

        # We have to switch the corner dimension to axis 0 for use with the MATLAB code. Right now it is after the scanline and ground-pixel axes
        # I don't remember why we have to us ascontiguousarray, but I do remember something not working without it
        dict_for_struct["TiledCornerLongitude"] = np.ascontiguousarray(np.moveaxis(dict_for_struct["TiledCornerLongitude"], [0, 1, 2], [1, 2, 0]))
        dict_for_struct["TiledCornerLatitude"] = np.ascontiguousarray(np.moveaxis(dict_for_struct["TiledCornerLatitude"], [0, 1, 2], [1, 2, 0]))

        if not read_main_single_only:
            raise Exception('Illegal mode; must use read_main_single_only=True')

        # Per BEHR v3 description, define relative azimuth angle with the 180 degree offset, necessary for use with the scattering weight LUT

        # 2024-10-18: I ran into issues with this method and devised an alternative one for use in the lookup table
        # Start by finding the absolute difference between saa and vaa
        raa = np.abs(dict_for_struct["SolarAzimuthAngle"] - dict_for_struct["ViewingAzimuthAngle"]) # Range: [0, 360]
        # Except, [180, 360] is really the same as [180, 0] on the circle.
        # Above 180, convert using 360 - raa
        raa = np.where(raa > 180, 360 - raa, raa) # Range: [0, 180]
        # In the LUT, raa = 0 means forward scattering position (opposite) and raa = 180 means back scattering position (aligned)
        # This is opposite to the mathematical definition above. We need to put [0, 180] onto [180, 0]
        raa = np.abs(raa - 180) # Not a satsifying way of doing this, but should accomplish the task
        dict_for_struct["RelativeAzimuthAngle"] = raa

        ds_id = tempo_ds.local_granule_id[:-3]
        geobounds_str = build_geobounds_str(lat_domain, lon_domain)
        pickle_save_full = "{pickle_save_location}/{ds_id}_{geo}_BEHR_initialized.pickle".format(pickle_save_location=pickle_save_location, ds_id=ds_id, geo=geobounds_str)

        with open(pickle_save_full, 'wb') as handle:
            pickle.dump(dict_for_struct, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

        #print("Structure saved at {}".format(pickle_save_location))

    except Exception as e:
        tempo_ds.close()
        raise e

    if close_when_finished:
        # Close the original file
        tempo_ds.close()

    return pickle_save_location, "{ds_id}_{geo}_BEHR_initialized.pickle".format(ds_id=ds_id, geo=geobounds_str)