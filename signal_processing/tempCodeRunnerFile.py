
                        low = wpe_buffer[i:i+ref_num:2, :(N_effective+1)//2] #低频间隔
                        high = wpe_buffer[i+ref_num//2:i+ref_num:1, (N_effective+1)//2:]#高频减