#!/bin/bash

#### SDR-director.sh ####

# Author: Sam Beaudry
# Last changed: 2025-10-05
# Location: Signal_Derived_Retrieval/TEMPO/main
# Contact: samuel_beaudry@berkeley.edu

################################

# Number of workers to use for ipyparallel processes (1 or more)
# Enter 1 to not use ipyparallel
n_workers=1

# Start time
start=$SECONDS

# Job time limit
timelimhours=23
timelimseconds=$(( timelimhours * 3600 ))
buffertime=3600

# Algorithm version
current_commit=$(git log -1 --pretty=format:"%H")

# Parameters ####################################
# For dates, use format YYYYMMDD
startdate="20240401"
enddate="20240930"

# Option to control for either full field of regard (FOR) or partial region
fullfor=1
if [ $fullfor -eq 0 ]; then
    # Refrain from entering decimal values
    lonmin="-125"
    lonmax="-65"
    latmin="25"
    latmax="50"
fi

region="custom"

scanmin='1'
scanmax='30'

collection="V03"

# Processing options
boundary_layer="hrrr"
n_updates="2"
modis_albedo=0 # if True, will run MATLAB code to match MODIS BDRFs with TEMPO pixels
minimize_output_size=1 # if True, will remove vertically-resolved variables when able to
reprocess_if_exists=1 # if True, will process scan even if an equivalent file already exists in RESULTS

if [ $modis_albedo -eq 1 ]; then
    results_subdir="with_MODIS"
    BEHR="" # directory with necessary files from the BErkeley High Resolution (BEHR) algorithm, primarily written in MATLAB
else
    results_subdir="without_MODIS"
fi

# Instrument
instrument="TEMPO"

# Directories for permanent files
SDR=
TEMPO=
HRRR=
RESULTS="$SDR/Results/$instrument/$results_subdir"
MYD06_L2=
MCD43D=
GLOBE=

# Directories for transient (temporary) files
PY_TO_MAT_SUITCASE="$SDR/py_to_mat_suitcase/$instrument"
MAT_TO_PY_SUITCASE="$SDR/mat_to_py_suitcase/$instrument"
save_path_partial="$SDR/partially_completed_datasets"

# Constant files
CONSTANTS="$SDR/constants"
omi_tropomi_vars_path="$CONSTANTS/OMI_TROPOMI_TEMPO_vars.csv"
modis_land_mask_path=

# Job trackings
logbook="$SDR/logbook"
###################################################

# Configure Environment ###########################
# https://stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script
source ...

if [ $modis_albedo -eq 1 ]; then
    echo "Will run MATLAB for custom albedo values"
    MATLAB=...
    # Set the correct python version for the matlab session

else
    echo "Will not run MATLAB"
fi

###################################################

# Loop ############################################
python -u create_date_range.py $startdate $enddate > "$logbook/create_date_range_latest.txt"

daterange_file="daylist_transient.txt"
while read -r line; do
    current_date="$line"
    echo $current_date

    # 1st Python ##########################################
    echo "Starting 1st Python job"
    if [ $fullfor -eq 1 ]; then
        python -u TEMPO_L2_NO2_on_date.py --date_string $current_date --tempo_location $TEMPO --collection $collection --fullfor $fullfor --vars_path $omi_tropomi_vars_path --intermediate_save_location $PY_TO_MAT_SUITCASE
    else
        python -u TEMPO_L2_NO2_on_date.py --date_string $current_date --tempo_location $TEMPO --collection $collection --fullfor $fullfor --lonmin $lonmin --lonmax $lonmax --latmin $latmin --latmax $latmax --vars_path $omi_tropomi_vars_path --intermediate_save_location $PY_TO_MAT_SUITCASE
    fi
    py_to_mat_filelist="${SDR}/$instrument/main/SDR_initialized_dicts_on_${current_date}_transient.txt"
    #######################################################

    # MATLAB ##############################################
    if [ $modis_albedo -eq 1 ]; then
        echo "Starting MATLAB job"
        while read -r line_of_pickle_list; do
            pickle="$line_of_pickle_list"
            # echo "Passing $pickle to MATLAB"
            matlab -nodisplay -batch "read_main_single( '${PY_TO_MAT_SUITCASE}', '${pickle}', 'sp_mat_dir', '${MAT_TO_PY_SUITCASE}', 'modis_myd06_dir', '${MYD06_L2}', 'modis_mcd43_dir', '${MCD43D}', 'modis_land_mask_path', '${modis_land_mask_path}', 'globe_dir', '${GLOBE}', 'region', '${region}', 'behr_path', '${BEHR}', 'lonmin', '${lonmin}', 'lonmax', '${lonmax}', 'latmin', '${latmin}', 'latmax', '${latmax}' );"
        done < "$py_to_mat_filelist"
    fi
    #######################################################

    # At this point, we can do everything else via python 
    # 2nd Python ##########################################
    echo "Starting 2nd Python job"
    if [ $fullfor -eq 1 ]; then
        python -u amf_update_one_day.py --current_date $current_date --py_to_mat_textfile $py_to_mat_filelist --PY_TO_MAT_SUITCASE $PY_TO_MAT_SUITCASE --MAT_TO_PY_SUITCASE $MAT_TO_PY_SUITCASE --modis_albedo $modis_albedo --TEMPO $TEMPO --vars_path $omi_tropomi_vars_path --constants_path $CONSTANTS --save_path $RESULTS --minimize_output_size $minimize_output_size --reprocess_if_exists $reprocess_if_exists --N_updates $n_updates --first_scan_num $scanmin --last_scan_num $scanmax --fullfor $fullfor --N_workers $n_workers --pblh $boundary_layer --hrrr_grib $HRRR --save_path_partial $save_path_partial --git_commit $current_commit > "$logbook/amf_update_one_day_${current_date}.txt"
    else
        python -u amf_update_one_day.py --current_date $current_date --py_to_mat_textfile $py_to_mat_filelist --PY_TO_MAT_SUITCASE $PY_TO_MAT_SUITCASE --MAT_TO_PY_SUITCASE $MAT_TO_PY_SUITCASE --modis_albedo $modis_albedo --TEMPO $TEMPO --vars_path $omi_tropomi_vars_path --constants_path $CONSTANTS --save_path $RESULTS --minimize_output_size $minimize_output_size --reprocess_if_exists $reprocess_if_exists --N_updates $n_updates --first_scan_num $scanmin --last_scan_num $scanmax --fullfor $fullfor --N_workers $n_workers --lonmin $lonmin --lonmax $lonmax --latmin $latmin --latmax $latmax --pblh $boundary_layer --hrrr_grib $HRRR --save_path_partial $save_path_partial --git_commit $current_commit > "$logbook/amf_update_one_day_${current_date}.txt"
    fi
    #######################################################

    # Parallel only
    if [ $n_workers -gt 1 ]; then
        # 3rd Python ##########################################
        echo "Starting 3rd Python Job"

        scan_df_file_list="scan_df_file_list_transient.txt"
        while read -r entry; do
            scan_df_file="$entry"
            scan_num=${scan_df_file:5:3}
            echo "    Scan $scan_num"
            
            python -u amf_update_one_scan_par_script.py --scan_df_file $scan_df_file --PY_TO_MAT_SUITCASE $PY_TO_MAT_SUITCASE --MAT_TO_PY_SUITCASE $MAT_TO_PY_SUITCASE --tempo_dir_head $TEMPO --vars_path $omi_tropomi_vars_path --constants_path $CONSTANTS --save_path $RESULTS --minimize_output_size $minimize_output_size --full_FOR $fullfor --num_engines $n_workers --N_updates $n_updates --pblh $boundary_layer --hrrr_grib $HRRR --save_path_partial $save_path_partial --git_commit $current_commit --verbosity "5" > "$logbook/amf_update_one_scan_${current_date}_S${scan_num}.txt"

            rm $scan_df_file
        done < "$scan_df_file_list"
        rm $scan_df_file_list
        #######################################################
    fi

    rm $py_to_mat_filelist

    echo "Processing for ${current_date} is finished"
    echo " "
    runtime=$(( SECONDS - start ))
    secondsremaining=$(( timelimseconds - runtime ))
    if [ $buffertime -gt $secondsremaining ]; then
        echo "Remaining days will not be processed due to job time limit"
        break
    fi

done < "$daterange_file"
# When we are finished, remove the daterange_file use to construct the loop
rm "$daterange_file"
echo "SDR-director is finished. Results are stored at ${RESULTS}"
###################################################