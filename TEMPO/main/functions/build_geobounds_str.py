#### build_geobounds_str.py ####

# Author: Sam Beaudry
# Last changed: 2025-05-05
# Location: Signal_Derived_Retrieval/TEMPO/main/functions
# Contact: samuel_beaudry@berkeley.edu

################################

import numpy as np

def build_geobounds_str(lat_domain, lon_domain):
    if lat_domain[0] >= 0:
        latdir0 = 'N'
    else:
        latdir0 = 'S'

    if lat_domain[1] >= 0:
        latdir1 = 'N'
    else:
        latdir1 = 'S'

    # ---------------------------------------------------------------------

    if lon_domain[0] >= 0:
        londir0 = 'E'
    else:
        londir0 = 'W'

    if lon_domain[1] >= 0:
        londir1 = 'E'
    else:
        londir1 = 'W'

    geobounds_str = 'lat_{latdir0}{latval0:02d}-{latdir1}{latval1:02d}_lon_{londir0}{lonval0:03d}-{londir1}{lonval1:03d}'.format(
        latdir0=latdir0,
        latval0=int(np.abs(lat_domain[0])),
        latdir1=latdir1,
        latval1=int(np.abs(lat_domain[1])),
        londir0=londir0,
        lonval0=int(np.abs(lon_domain[0])),
        londir1=londir1,
        lonval1=int(np.abs(lon_domain[1])),
        )

    return geobounds_str