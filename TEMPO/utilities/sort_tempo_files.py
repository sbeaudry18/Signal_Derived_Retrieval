#######################
# Signal_Derived_Retrieval/TEMPO/utilities/sort_tempo_files.py
#
# Author: Sam Beaudry
# Date: 2024-09-13
# Description: Sorts TEMPO product files to 
#              appropiate locations based on their filenames
# Contact: samuel_beaudry@berkeley.edu
#######################

import os
import re
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--unsorted', type=str, help='path to the unsorted files')
parser.add_argument('--sorted', type=str, help='path to store sorted files')

args = vars(parser.parse_args())
unsorted = args['unsorted']
sorted = args['sorted']

if not os.path.exists(unsorted):
    raise OSError("Path to unsorted files does not exists: '{}'".format(unsorted))

# Get a list of all TEMPO netCDF files in the staging directory
file_list = os.listdir(unsorted)
nc_pattern = re.compile(r"^TEMPO.*\.nc$")
file_list = [f for f in file_list if nc_pattern.match(f)]
file_list.sort()

# Loop through the files
for f in file_list:
    # Get the product information from the file name to determine where it should go
    product = re.search(r"^TEMPO_([A-Z0-9]+)_.*$", f).group(1)
    processing_level = re.search(r"^TEMPO_[A-Z0-9]+_([A-Z0-9]{2})_.*$", f).group(1)
    version_num = re.search(r"^TEMPO_[A-Z0-9]+_[A-Z0-9]{2}_([A-Z0-9]{3})_.*$", f).group(1)
    year = re.search(r"^TEMPO_[A-Z0-9]+_[A-Z0-9]{2}_[A-Z0-9]{3}_(\d{4}).*$", f).group(1)
    month = re.search(r"^TEMPO_[A-Z0-9]+_[A-Z0-9]{2}_[A-Z0-9]{3}_\d{4}(\d{2}).*$", f).group(1)

    # Assemble the full save path for the file
    # Along the way, if the previously added directory doesn't exist, create it

    # Append the next subdirectory
    save_dir = sorted + product + "/"

    # Check if the subdirectory exists
    if not os.path.isdir(save_dir):
        # If this subdirectory does not exist, create it
        os.makedirs(save_dir)

    # Repeat

    save_dir = save_dir + processing_level + "/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    save_dir = save_dir + version_num + "/"

    if not os.path.isdir(version_num):
        os.makedirs(version_num)

    save_dir = save_dir + year + "/"
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    save_dir = save_dir + month + "/"

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Finally, move the file from the staging_directory to the correct location
    if os.path.exists(save_dir + f):
        print('Warning: {} already exists at {}. File will be overwritten.'.format(f, save_dir))
        
    os.rename(os.path.join(unsorted, f), os.path.join(save_dir, f))