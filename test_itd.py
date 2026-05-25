import numpy as np
import scipy.io as sio

def calculate_itd(hrir_l, hrir_r, fs=44100):
    corr = np.correlate(hrir_l, hrir_r, mode='full')
    lag_samples = np.argmax(corr) - (len(hrir_r) - 1)
    itd = lag_samples / fs
    return itd

data = sio.loadmat('hrtf_data/hrir_final.mat')
hrir_l = data['hrir_l']
hrir_r = data['hrir_r']
# Azimuths: index 24 is 80 deg, index 12 is 0 deg. Elevation index 8 is 0 deg (-45 + 5.625*8 = 0)
el_idx = 8
for az_idx, az in enumerate([-80, -45, 0, 45, 80]):
    # find actual index
    real_az_idx = [0, 3, 12, 21, 24][az_idx]
    hl = hrir_l[real_az_idx, el_idx, :]
    hr = hrir_r[real_az_idx, el_idx, :]
    itd = calculate_itd(hl, hr)
    print(f"Azimuth: {az:>3} deg, ITD: {itd * 1000:>6.2f} ms")
