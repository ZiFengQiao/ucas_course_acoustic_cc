'''
Author: Wang, Qiaoyu
Date: 2026-06-07 16:14:25
LastEditors: Wang, Qiaoyu
LastEditTime: 2026-06-07 19:37:29
Description: 
'''


from pathlib import Path
from hrtf import HRTFProcessor
import soundfile as sf
from scipy.signal import resample_poly
import math

from gui import VirtualSoundGUI

PROJECT_ROOT = Path(__file__).resolve().parent
HRIR_MAT_PATH = PROJECT_ROOT / 'hrtf_data' / 'hrir_final.mat'
AUDIO_FILE_PATH = PROJECT_ROOT / 'audio_files' / 'audio_speech.flac'
OUTPUT_PATH = PROJECT_ROOT / 'output.wav'

if __name__ == '__main__':
    # load audio
    audio, fs = sf.read(str(AUDIO_FILE_PATH))
    if fs != 44100:
        g = math.gcd(fs, 44100)

        audio = resample_poly(
            audio,
            up=44100 // g,
            down=fs // g
        )

        fs = 44100
    # 初始化processor，信号处理后端
    processor = HRTFProcessor(str(HRIR_MAT_PATH))

    # 计算ITD，nearest和inter两种方法
    processor.calculate_itd_horizon()
    processor.calculate_itd_horizon(method="inter")

    # 生成旋转一周的音频
    result = processor.rotate_sound(audio)
    sf.write(str(OUTPUT_PATH), result, fs)

    # 启动交互式界面
    import tkinter as tk
    root = tk.Tk()
    app = VirtualSoundGUI(root)
    root.mainloop()