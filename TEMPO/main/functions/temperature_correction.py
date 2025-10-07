#### temperature_correction.py ####

# Author: Sam Beaudry
# Last changed: 2025-03-29
# Location: Signal_Derived_Retrieval/TEMPO/main/functions
# Contact: samuel_beaudry@berkeley.edu

####################################

def temperature_correction(temperature, mode='s5p'):
    allowed_modes = ['behr', 's5p', 'tempo']
    if mode not in allowed_modes:
        raise ValueError("'mode' must be one of 'behr', 's5p', or 'tempo'. Selected value was '{}'".format(mode))
    
    Tsigma = 220 # K
    
    if mode == 'behr':
        # Use the temperature correction described in Laughner et al. (2018), equation 6
        c = 1 - 0.003 * (temperature - Tsigma)

    elif mode == 's5p':
        # TROPOMI ATBD tropospheric and total NO2, equation 18
        c = 1 - 0.00316 * (temperature - Tsigma) + 3.39e-6 * (temperature - Tsigma) ** 2

    else:
        # TEMPO ATBD
        # Same as s5p
        c = 1 - 0.00316 * (temperature - Tsigma) + 3.39e-6 * (temperature - Tsigma) ** 2

    return c