#!/bin/bash

#### SDR-director-simple.sh ####

# Author: Sam Beaudry
# Last changed: 2026-03-23
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

scanmin='1'
scanmax='30'

collection="V03"
sdr_letter="A"
sdr_version="$collection-$sdr_letter"

# Processing options
boundary_layer="hrrr"
n_updates="2"
minimize_output_size=1 # if True, will remove vertically-resolved variables when able to
reprocess_if_exists=1 # if True, will process scan even if an equivalent file already exists in RESULTS

# Instrument
instrument="TEMPO"

# Load in paths in config file
# https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/../../constants/sdr_paths.config

# Directories for transient (temporary) files
save_path_partial="$SDR/partially_completed_datasets"


# Constant files
CONSTANTS="$SDR/constants"
omi_tropomi_vars_path="$CONSTANTS/OMI_TROPOMI_TEMPO_vars.csv"
RESULTS=$SDR_FILES
 
# Job trackings
logbook="$SDR/logbook"
###################################################

# Configure Environment ###########################
# https://stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script
...

###################################################

# Loop ############################################
python -u create_date_range.py $startdate $enddate > "$logbook/create_date_range_latest.txt"

daterange_file="daylist_transient.txt"
while read -r line; do
    current_date="$line"
    echo $current_date

    echo "Calling amf_update_one_day.py"
    if [ $fullfor -eq 1 ]; then
        python -u amf_update_one_day.py --current_date $current_date --TEMPO $TEMPO --collection $collection --sdr_letter $sdr_letter --vars_path $omi_tropomi_vars_path --constants_path $CONSTANTS --save_path $RESULTS --minimize_output_size $minimize_output_size --reprocess_if_exists $reprocess_if_exists --N_updates $n_updates --first_scan_num $scanmin --last_scan_num $scanmax --fullfor $fullfor --N_workers $n_workers --pblh $boundary_layer --hrrr_grib $HRRR --save_path_partial $save_path_partial --git_commit $current_commit > "$logbook/amf_update_one_day_${current_date}.txt"
    else
        python -u amf_update_one_day.py --current_date $current_date --TEMPO $TEMPO --collection $collection --sdr_letter $sdr_letter --vars_path $omi_tropomi_vars_path --constants_path $CONSTANTS --save_path $RESULTS --minimize_output_size $minimize_output_size --reprocess_if_exists $reprocess_if_exists --N_updates $n_updates --first_scan_num $scanmin --last_scan_num $scanmax --fullfor $fullfor --N_workers $n_workers --lonmin $lonmin --lonmax $lonmax --latmin $latmin --latmax $latmax --pblh $boundary_layer --hrrr_grib $HRRR --save_path_partial $save_path_partial --git_commit $current_commit > "$logbook/amf_update_one_day_${current_date}.txt"
    fi

    # Parallel only
    if [ $n_workers -gt 1 ]; then
        scan_df_file_list="scan_df_file_list_transient.txt"
        while read -r entry; do
            scan_df_file="$entry"
            scan_num=${scan_df_file:5:3}
            echo "    Scan $scan_num"
            
            python -u amf_update_one_scan_par_script.py --scan_df_file $scan_df_file --tempo_dir_head $TEMPO --vars_path $omi_tropomi_vars_path --constants_path $CONSTANTS --save_path $RESULTS --sdr_version $sdr_version --minimize_output_size $minimize_output_size --full_FOR $fullfor --num_engines $n_workers --N_updates $n_updates --pblh $boundary_layer --hrrr_grib $HRRR --save_path_partial $save_path_partial --git_commit $current_commit --verbosity "5" > "$logbook/amf_update_one_scan_${current_date}_S${scan_num}.txt"

            rm $scan_df_file
        done < "$scan_df_file_list"
        rm $scan_df_file_list
    fi

    echo "Processing for ${current_date} is finished"
    echo " "
    runtime=$(( SECONDS - start ))
    secondsremaining=$(( timelimseconds - runtime ))
    if [ $buffertime -gt $secondsremaining ]; then
        echo "Remaining days will not be processed due to job time limit"
        break
    fi

done < "$daterange_file"
# When we are finished or out of time, clean up by removing temporary files
rm "$daterange_file"

echo "SDR-director is finished. Results are stored at ${RESULTS}"
###################################################