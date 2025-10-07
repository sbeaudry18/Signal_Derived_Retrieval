def create_date_range(start_date, end_date, str_format, end_inclusive=True):
    import numpy as np
    from datetime import datetime
    from datetime import timedelta
    import os
    import warnings

    script_dir = os.path.dirname(os.path.realpath(__file__))

    start_date = datetime.strptime(start_date, str_format)
    end_date = datetime.strptime(end_date, str_format)
    numdays = (end_date - start_date).days
    
    if end_inclusive:
        end_offset = 1
    else:
        end_offset = 0

    file_savename = '{}/daylist_transient.txt'.format(script_dir)

    if os.path.exists(file_savename):
        warnings.warn("The file '{}' already exists.".format(file_savename))
        #overwrite = input("Overwrite existing file? (y/n): ")
        overwrite = 'y'

        if overwrite.lower() == 'y':
            print("Overwriting existing file...")
        else:
            raise Exception("The file '{}' already exists. Remove this file or command to overwrite in future session.".format(file_savename))

    daylist = []
    with open(file_savename, 'w') as file:
        for n in range(numdays+end_offset):
            date_string = datetime.strftime(start_date + timedelta(days=n), str_format)
            daylist.append(date_string)
            file.write("{}\n".format(date_string))

    file.close()

    return file_savename

def main():
    import sys
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    str_format = '%Y%m%d'

    #range_file = create_date_range(start_date, end_date, str_format)
    create_date_range(start_date, end_date, str_format)

if __name__ == "__main__":
    main()