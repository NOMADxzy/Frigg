ms_per_bin = 500
step_len = ms_per_bin / 1000
first_second = 0
last_second = 35

def get_initial_val_list():
    return [0 for _ in range(int((last_second - first_second) * 1000 / ms_per_bin))]