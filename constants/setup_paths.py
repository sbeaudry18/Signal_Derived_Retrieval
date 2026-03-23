#### setup_paths.py ####

# Author: Sam Beaudry
# Last changed: 2026-03-20
# Location: Signal_Derived_Retrieval/contants
# Contact: samuel_beaudry@berkeley.edu
# Description: Script which, when called, will walk the user through setting up the paths needed in the SDR algorithm

#########################

def setup_paths(BEHR_paths=False):
    import os
    import re

    constants_path = os.path.dirname(os.path.realpath(__file__))
    path_config_file = os.path.join(constants_path, 'sdr_paths.config')

    if os.path.exists(path_config_file):
        overwrite_file = input('Path config file already exists. Overwrite? (y/n): ')

        if overwrite_file.lower() == 'y':
            print('Will overwrite existing path file\n')
            create_path_file = True

        else:
            print('Will not overwrite existing path file')
            create_path_file = False

    else:
        create_path_file = True

    if create_path_file:
        path_dict = {
            'SDR': '',
            'TEMPO': '',
            'HRRR': '',
            'SDR_FILES': '',
            'MYD06_L2': '',
            'MCD43D': '',
            'GLOBE': '',
            'MODIS_LAND_MASK': ''
        }

        required_paths = ['SDR', 'TEMPO', 'HRRR', 'SDR_FILES']

        # These optional paths were necessary when we were using MODIS albedos
        # in the algorithm, but now they are unnecessary most of the time
        optional_paths = ['MYD06_L2', 'MCD43D', 'GLOBE', 'MODIS_LAND_MASK']

        if BEHR_paths:
            paths_to_set = required_paths + optional_paths
        else:
            paths_to_set = required_paths

        for path_key in paths_to_set:
            path_exists = os.path.exists(path_dict[path_key])

            loop = 0
            while not path_exists:
                if path_key == 'SDR':
                    cwd = os.getcwd()
                    sdr_pat = re.compile(r'(.*Signal_Derived_Retrieval).*')

                    if sdr_pat.match(cwd):
                        sdr_path = sdr_pat.match(cwd).group(1)

                        print(sdr_path)
                        use_inferred_path = input('Is the above path correct for the Signal_Derived_Retrieval repository? (y/n): ')

                        if use_inferred_path.lower() == 'y':
                            path_dict[path_key] = sdr_path

                        else:
                            path_dict[path_key] = input('Enter path to Signal_Derived_Retrieval: ')

                elif path_key == 'TEMPO':
                    if loop == 0:
                        print(r'TEMPO files should be sorted as TEMPO/{trace_gas}/{processing_level}/{processor_version}/{year}/{month}/{file}')
                        print('For example: TEMPO/{trace_gas}/{processing_level}/{processor_version}/{year}/{month}/{file}'.format(trace_gas='NO2', processing_level='L2', processor_version='V04', year='2024', month='07', file='TEMPO_NO2_L2_V04_20240711T184349Z_S010G05.nc'))
                    path_dict[path_key] = input('Enter path to this TEMPO directory: ')

                elif path_key == 'HRRR':
                    path_dict[path_key] = input('Enter path where HRRR grib files can be stored: ')

                elif path_key == 'SDR_FILES':
                    path_dict[path_key] = input('Enter path where the finalized SDR files will be stored: ')        

                elif path_key == 'MYD06_L2':
                    path_dict[path_key] = input('Enter path to MODIS MYD06_L2 files: ')    

                elif path_key == 'MCD43D':
                    path_dict[path_key] = input('Enter path to MODIS MCD43D files: ')   

                elif path_key == 'GLOBE':
                    path_dict[path_key] = input('Enter path to the GLOBE elevation files: ')   

                elif path_key == 'MODIS_LAND_MASK':
                    path_dict[path_key] = input('Enter path to the MODIS land mask file: ')   

                path_exists = os.path.exists(path_dict[path_key])
                if path_exists:
                    print('\x1b[6;30;42m' + 'Path exists' + '\x1b[0m\n')

                else:
                    print('Entered path \x1b[6;30;43m' + 'does not exist' + '\x1b[0m: {}\n'.format(path_dict[path_key]))
                    create_dir = input('Do you want to create this directory? (y/n): ')

                    if create_dir.lower() == 'y':
                        try:
                            os.mkdir(path_dict[path_key])
                            print('Created this directoy\n')
                            path_exists = os.path.exists(path_dict[path_key])

                        except FileNotFoundError:
                            create_recurs = input('CAUTION: One or more sub directories in this path are also missing. Do you want to recursively build this path? (y/n)')
                            
                            if create_recurs.lower() == 'y':
                                os.makedirs(path_dict[path_key])
                                print('Created this directory structure\n')
                                path_exists = os.path.exists(path_dict[path_key])

                            else:
                                print('Try entering a different path\n')

                loop += 1

        print('Please check that the following paths are accurate:')
        for path_key in paths_to_set:
            print('{}: {}'.format(path_key, path_dict[path_key]))

        print('')
        write_file = input('Write to path file? (y/n): ')

        if write_file.lower() == 'y':

            with open(path_config_file, 'w') as f:
                for path_key in list(path_dict.keys()):
                    f.write("{}={}\n".format(path_key, path_dict[path_key]))

            print('Path file saved as {}'.format(path_config_file))

        else:
            print('Path file was not saved.')


if __name__ == "__main__":
    setup_paths()