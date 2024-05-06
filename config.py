ms_per_bin = 500

first_second = 0
last_second = 35

step_len = ms_per_bin / 1000
bin_cnt = int((last_second - first_second) * 1000 / ms_per_bin)
def get_initial_val_list():
    return [0 for _ in range(bin_cnt)]