#### name_product_file.py ####

# Author: Sam Beaudry
# Last changed: 2026-03-26
# Location: Signal_Derived_Retrieval/TEMPO/main
# Contact: samuel_beaudry@berkeley.edu

###############################

def name_product_file(sdr_version, start_time, end_time, scan_num, geo_scope, commit=None, proc_time=False):
    import re

    # Check inputs
    sdr_version_pat = re.compile(r'V\d+-[A-Z]')
    if not sdr_version_pat.match(sdr_version):
        raise ValueError('Invalid input for sdr_version: {}'.format(sdr_version))
    
    time_pat = re.compile(r'\d{8}T\d{6}Z')
    if not time_pat.match(start_time):
        raise ValueError('Invalid input for start_time: {}'.format(start_time))
    if not time_pat.match(end_time):
        raise ValueError('Invalid input for end_time: {}'.format(end_time))
    
    scan_num_pat = re.compile(r'S\d{3}')
    if not scan_num_pat.match(scan_num):
        raise ValueError('Invalid input for scan_num: {}'.format(scan_num))

    product_name = "SDR-TEMPO_NO2_L2_{}_{}_{}_{}_{}".format(
        sdr_version,
        start_time,
        end_time,
        scan_num,
        geo_scope
    )

    if isinstance(commit, str):
        # Use only the first seven characters
        commit = commit[:7]

        product_name += "_commit_{}".format(commit)

    if isinstance(proc_time, str):
        if not time_pat.match(proc_time):
            raise ValueError('Invalid input for proc_time: {}'.format(proc_time))
        current_time_string = proc_time
        product_name += "_proc_" + current_time_string

    elif isinstance(proc_time, bool):
        if proc_time:
            import datetime

            current_time = datetime.datetime.now(tz=datetime.timezone.utc)
            current_time_string = current_time.strftime('%Y%m%dT%H%M%SZ')

            product_name += "_proc_" + current_time_string

    else:
        raise ValueError('proc_time must be bool or str, not {}'.format(type(proc_time)))

    product_name += ".nc"

    return product_name