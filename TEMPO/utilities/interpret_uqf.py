def interpret_uqf(uqf, scan_ds, ms=-1, xt=-1):
    '''
    Takes update_quality_flag value and prints issues that lead to non-zero values

    uqf : int
        Value of update_quality_flag
    scan_ds = xr.Dataset
    '''

    bit_meanings = scan_ds.update_quality_flags_Standard.bit_meanings

    bit_array = bin(uqf)[2:][::-1] # Convert to string, remove 0b, and reverse so earlier bits come first

    for i in range(len(bit_array)):
        if bool(int(bit_array[i])):
            minusi = -1 * (1 + i)
            issue = bit_meanings[minusi]
            print(issue)

            if (ms > 0) & (xt > 0):
                if issue == 'main_data_quality_above_0':
                    mdqf = scan_ds.main_data_quality_flag.data[ms, xt]
                    print('---> {}'.format(mdqf))
                
                elif issue == 'high_eff_cloud_fraction':
                    ecf = scan_ds.eff_cloud_fraction.data[ms, xt]
                    print('---> {:.2f}'.format(ecf))

                elif issue == 'high_sza':
                    sza = scan_ds.solar_zenith_angle.data[ms, xt]
                    print('---> {:.2f}'.format(sza))

                elif issue == 'invalid_pixel_area':
                    area = scan_ds.area.data[ms, xt]
                    print('---> {:.2f}'.format(area))

            print('')