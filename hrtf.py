import numpy as np
import scipy.io as sio
from scipy.signal import fftconvolve
from pathlib import Path

import matplotlib.pyplot as plt

DEBUG = 0
PROJECT_ROOT = Path(__file__).resolve().parent

class HRTFProcessor:
    def __init__(self, mat_file_path):
        """
        Load HRIR data from CIPIC database
        @param mat_file_path: Path to the .mat file containing HRTF data (str or Path)
        """
        data = sio.loadmat(str(mat_file_path))
        self.hrir_l = data['hrir_l']
        self.hrir_r = data['hrir_r']

        if DEBUG:
            # (25, 50, 200) for CIPIC, where 25 azimuths(方位角）, 50 elevations（仰角）, and 200 FIR
            print(f"DEBUG: hrir_r shape: {self.hrir_r.shape}")
            print(f"DEBUG: hrir_l shape: {self.hrir_l.shape}")

        # CIPIC grid
        self.azimuths = np.array([-80, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5, 
                                  0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65, 80])
        self.elevations = np.array([-45 + 5.625 * i for i in range(50)])

    def get_hrir_nearest(self, az, el):
        """
        @brief: 获取最近邻HRIR
        """

        az_idx = np.abs(self.azimuths - az).argmin()
        el_idx = np.abs(self.elevations - el).argmin()

        hl = self.hrir_l[az_idx, el_idx, :]
        hr = self.hrir_r[az_idx, el_idx, :]
        return hl, hr

    def get_hrir_inter(self, az, el):
        """
        @brief: 获取指定方位角和仰角的插值HRIR
        """
        az_hi = np.searchsorted(self.azimuths, az)

        if az_hi == 0:
            az_lo = az_hi
        elif az_hi >= len(self.azimuths):
            az_hi = len(self.azimuths) - 1
            az_lo = az_hi
        else:
            az_lo = az_hi - 1

        az0 = self.azimuths[az_lo]
        az1 = self.azimuths[az_hi]

        # 一维线性插值
        if az0 == az1:
            w = 0.0
        else:
            w = (az - az0) / (az1 - az0)

        el_idx = np.abs(self.elevations - el).argmin()

        hl0 = self.hrir_l[az_lo, el_idx]
        hl1 = self.hrir_l[az_hi, el_idx]

        hr0 = self.hrir_r[az_lo, el_idx]
        hr1 = self.hrir_r[az_hi, el_idx]

        hl = (1 - w) * hl0 + w * hl1
        hr = (1 - w) * hr0 + w * hr1

        return hl, hr

    def apply_hrtf(self, audio, az, el, method='nearest'):
        """
        @brief: 将输入音频应用于指定方位角和仰角的HRIR
        @param audio: 单通道输入音频信号，采样率为44100Hz
        @param az: 方位角
        @param el: 仰角
        """

        if method == 'nearest':
            hl, hr = self.get_hrir_nearest(az, el)
        elif method == 'inter':
            hl, hr = self.get_hrir_inter(az, el)
        else:
            raise ValueError("Invalid method. Use 'nearest' or 'inter'.")

        # 直接卷积，应用对应的系统，得到双耳输出
        out_l = fftconvolve(audio, hl, mode='same')
        out_r = fftconvolve(audio, hr, mode='same')
        return np.column_stack((out_l, out_r))

    def calculate_itd(self, az, el=0, method='nearest'):
        """
        @brief: 利用互相关法计算ITD

        @param az: 方位角
        @param el: 仰角
        @param method:
            'nearest' 最近邻HRIR
            'inter'   插值HRIR

        @return:
            itd_sec     ITD(秒)
            lag         ITD(samples)
        """

        if method == 'nearest':
            hl, hr = self.get_hrir_nearest(az, el)
        elif method == 'inter':
            hl, hr = self.get_hrir_inter(az, el)
        else:
            raise ValueError("Invalid method")

        corr = np.correlate(hr, hl, mode='full')
        lag = np.argmax(corr) - (len(hr) - 1)

        itd_sec = lag / 44100.0

        return itd_sec, lag

    def calculate_itd_horizon(self, method='nearest'):
        """
        @brief: 直接计算水平面上所有方位角的ITD，并绘制ITD随方位角变化的曲线
        """
        azimuths = self.azimuths if method == 'nearest' else np.arange(-80, 81, 3)
        itds = []

        for az in azimuths:
            itd_sec, _ = self.calculate_itd(az, el=0, method=method)
            itds.append(itd_sec)

        plt.figure()
        plt.plot(azimuths, itds, marker='o')
        plt.title('ITD vs Azimuth (Elevation=0), Method: ' + method)
        plt.xlabel('Azimuth (degrees)')
        plt.ylabel('ITD (seconds)')
        plt.grid()
        plt.tight_layout()
        plt.savefig(str(PROJECT_ROOT / f'itd_vs_azimuth_{method}.png'))
        plt.show()

    def rotate_sound(self, audio, block_duration=0.05, hop_ratio=0.25):
        """
        @brief: 将输入单声道音频旋转一周，输出双声道立体声
        @param audio: 输入单声道音频信号，采样率为44100Hz
        @param block_duration: 每个块的持续时间（秒）
        @param hop_ratio: 每次移动的比例（相对于块大小）
        """
        n = len(audio)
        fs = 44100
        block_size = int(fs * block_duration)
        hop = int(block_size * hop_ratio)
        n_blocks = (n + hop - 1) // hop

        out_l = np.zeros(n + block_size)
        out_r = np.zeros(n + block_size)

        # 使用Hanning窗进行平滑过渡
        window = np.hanning(block_size)

        for i in range(n_blocks):

            start = i * hop
            end = min(start + block_size, n)

            block = audio[start:end]
            if len(block) < block_size:
                block = np.pad(block, (0, block_size - len(block)))

            # 计算块角度
            angle = (i / n_blocks) * 360.0

            az = 80 * np.sin(np.radians(angle))

            el = 180 if angle > 90 and angle < 270 else 0

            # 卷积得到当前块的双耳输出
            cur = self.apply_hrtf(block, az, el, method="inter")
            cur_l = cur[:, 0] * window
            cur_r = cur[:, 1] * window

            # overlap 加到输出中
            out_start = start
            out_end = start + block_size

            out_l[out_start:out_end] += cur_l
            out_r[out_start:out_end] += cur_r

        stereo = np.column_stack([out_l, out_r])

        peak = np.max(np.abs(stereo))
        if peak > 1e-6:
            stereo = stereo / peak * 0.95

        return stereo