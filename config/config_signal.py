import numpy as np
filter_names = ['Notch', 'ButterWorth Lowpass', 'ButterWorth Highpass', 'Butterworth Bandpass']

range_bins = 8
rd_shape = (8, 16)
rd_vmax = 800
rd_vmin = -800
ra_shape = (8, 64)
rd_controlGestureTab_display_dim = 128
rd_mmwTab_display_dim = 512
mmw_rd_rc_csr = 0.8
mmw_razi_rc_csr = 0.8
gui_mmw_rd_rc_csr_default = mmw_rd_rc_csr * 100
gui_mmw_ra_rc_csr_default = mmw_razi_rc_csr * 100
mmWave_fps = 30

indexpen_classes = ['A', 'B', 'C', 'D', 'E',
                    'F', 'G', 'H', 'I', 'J',
                    'K', 'L', 'M', 'N', 'O',
                    'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y',
                    'Z', 'Spc', 'Bspc', 'Ent', 'Act', 'Nois']

indexPen_debouncer_threshold = 70
debouncerFrameThreshold = 50
debouncerProbThreshold = 0.8
relaxPeriod = 15
inactivateClearThreshold = 10


rd_raw_range_bin_normalizer = [1400, 1000, 600, 200, 0, -200, -400, -800]
ra_raw_range_bin_normalizer = [2400, 2000, 1600, 1200, 800, 600, 200, 0]
rd_cr_range_bin_normalizer = [1000, 800, 600, 200, 0, -200, -400, -800]
ra_cr_range_bin_normalizer = [300, 250, 100, 50, 0, -50, -100, -150]

rd_raw_range_bin_normalizer = np.expand_dims(rd_raw_range_bin_normalizer, axis=-1)
ra_raw_range_bin_normalizer = np.expand_dims(ra_raw_range_bin_normalizer, axis=-1)
rd_cr_range_bin_normalizer = np.expand_dims(rd_cr_range_bin_normalizer, axis=-1)
ra_cr_range_bin_normalizer = np.expand_dims(ra_cr_range_bin_normalizer, axis=-1)

