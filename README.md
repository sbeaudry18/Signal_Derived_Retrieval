# Signal Derived Retrieval (SDR)
## Code for updating air mass factors (AMFs) in TEMPO NO<sub>2</sub> retrieval
This repository contains the algorithm for the TEMPO signal-derived retrieval (SDR) described in Beaudry and Cohen (2026). The main processing script is **TEMPO/main/amf_update_one_scan.py**, which constructs a dataset for the TEMPO scan ("scan_ds"), adds additional variables needed for the SDR, filters pixels by quality (in **TEMPO/main/functions/prepare_for_update.py**), and then carries out the redistribution of the prior and recalculation of the AMF (in **TEMPO/main/functions/amf_recursive_update_sf.**). The **TEMPO/main/functions** folder contains most of the functions used in **amf_update_one_scan.py**. A parallel version of this process, **amf_update_one_scan_par_script.py**, writes these functions into the script for ipyparallel functionality. The output netCDF files are saved as "SDR-TEMPO..." and contain the SDR tropospheric vertical column density (VCD) as "sdr_vertical_column_troposphere". The data filtering and troubleshooting variable is stored as "update_quality_flags"; a flag of 0 indicates a good quality pixel used in the SDR update.

## Required Packages
The algorithm was written with Python version 3.12.3. The following packages are required:
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [xarray](https://docs.xarray.dev/en/stable/)
- [netCDF4](https://unidata.github.io/netcdf4-python/)
- [Herbie](https://herbie.readthedocs.io/en/stable/)
- [ipyparallel](https://ipyparallel.readthedocs.io/en/latest/) (if multiple workers are used for the parallel version of the algorithm)
- [scipy](https://scipy.org/)
- [shapely](https://shapely.readthedocs.io/en/stable/)

## Running the SDR
1. Clone this repository.
   ```
   git clone https://github.com/sbeaudry18/Signal_Derived_Retrieval.git
   cd Signal_Derived_Retrieval
   ```
2. Install the above packages (ideally in a specific environment, e.g. using conda). 
3. Identify the location of the TEMPO L2 files to be processed. If $TEMPO is the path to these files, they should be sorted as $TEMPO/NO2/L2/{processor_version}/{year}/{mo}/{file}.nc. For example:\
    \
    $TEMPO/NO2/L2/V03/2024/07/TEMPO_NO2_L2_V04_20240711T184349Z_S010G05.nc. \
    \
   To arrange unsorted files: 
   ```
   cd TEMPO/utilities
   UNSORTED="path_to_unsorted"
   SORTED="path_to_sorted"
   python sort_tempo_files.py --unsorted $UNSORTED --sorted $SORTED
   ```
4. Create the path file, "sdr_paths.config", which points to the TEMPO L2 files to be processed and sets the location SDR files will be saved. The "constants/setup_paths.py" script assists with this:
   ```
   cd constants
   python setup_paths.py
   ```
5. While in the constants folder, create the "hrrr_reanalysis_coordinates.nc" dataset. You will need to be in the environment with Herbie to do this.
   ```
   python get_hrrr_coords.py
   ```
5. The main script to run the SDR is "SDR-director-simple.sh" in the "TEMPO/main" folder. Under "Configure Environment", enter the commands needed to use the above packages. Some of the additional options for running SDR are listed below. The suggested values correspond to those used to process the data presented in Beaudry and Cohen (2026):
   - The time period to run the algorithm for
     - `startdate="20240401"`
     - `enddate="20240930"`
   - Whether to reprocess data across the full field of regard (FOR) or for a region bounded by provided latitudes/longitudes
     - `fullfor=1`
   - The range of TEMPO scan numbers to process. Set arbitrarily small and large to reprocess all scans
     - `scanmin='1'`
     - `scanmax='30'`
   - The TEMPO processor version to use
     - `collection="V03"`
   - The source of boundary layer heights
     - `boundary_layer="hrrr"`
   - Whether to process a dataset if an equivalent one already exists
     - `reprocess_if_exists=0`
   - The number of workers (e.g. CPUs) available to the algorithm
     - `n_workers=1` (Serial: will call "amf_update_one_scan" from within "amf_update_one_day")
     - `n_workers=>1` (Parallel: will call "amf_update_one_scan_par_script.py")
6. Start the algorithm:
   ```
   cd TEMPO/main
   bash SDR-director-simple.sh
   ```
