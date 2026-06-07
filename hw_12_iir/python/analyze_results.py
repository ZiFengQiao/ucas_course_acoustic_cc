'''
Author: Wang, Qiaoyu
Date: 2026-06-02 22:44:46
LastEditors: Wang, Qiaoyu
LastEditTime: 2026-06-03 22:37:13
Description: 结果分析
'''

import os

import numpy as np
from scipy import signal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
IMG_DIR  = os.path.join(PROJ_DIR, 'img')
os.makedirs(IMG_DIR, exist_ok=True)

# 任务进度
TOTAL_TASK = 9
current_task = 1

# ── 1. 加载输入 ────────────────────────────────────────────────
print(f"[{current_task}/{TOTAL_TASK}] Loading outputs…")
current_task+=1
y_py  = np.fromfile(os.path.join(DATA_DIR, 'output_python.raw'), dtype=np.float64)

# 解析系数文件获取 Q 格式
coeff_q16, coeff_q32 = 14, 30  # 默认值
with open(os.path.join(DATA_DIR, 'coeffs.txt')) as f:
    for line in f:
        if 'coeff_q_bits_16:' in line:
            coeff_q16 = int(line.split(':')[1].strip())
        elif 'coeff_q_bits_32:' in line:
            coeff_q32 = int(line.split(':')[1].strip())

# C 输出：int16_t → float / 2^Q16
y_16 = np.fromfile(os.path.join(DATA_DIR, 'output_16bit.raw'), dtype=np.int16) \
         .astype(np.float64) / float(1 << coeff_q16)

# C 输出：int32_t → float / 2^Q32
y_32 = np.fromfile(os.path.join(DATA_DIR, 'output_32bit.raw'), dtype=np.int32) \
         .astype(np.float64) / float(1 << coeff_q32)

print(f"    Q formats: 16-bit Q{coeff_q16}, 32-bit Q{coeff_q32}")

N = len(y_py)
print(f"    Samples: {N}")

# trim to shortest
N = min(len(y_py), len(y_16), len(y_32))
y_py, y_16, y_32 = y_py[:N], y_16[:N], y_32[:N]

# ── 2. 计算指标 ───────────────────────────────────────────────────
e_16 = y_py - y_16
e_32 = y_py - y_32

print(f"[{current_task}/{TOTAL_TASK}] Computing metrics…")
current_task+=1
mse16 = np.mean(e_16 ** 2)
mse32 = np.mean(e_32 ** 2)
var_py = np.var(y_py)
snr16 = 10 * np.log10(var_py / mse16) if mse16 > 0 else float('inf')
snr32 = 10 * np.log10(var_py / mse32) if mse32 > 0 else float('inf')
max16 = np.max(np.abs(e_16))
max32 = np.max(np.abs(e_32))
nmse16 = mse16 / var_py
nmse32 = mse32 / var_py

print(f"    16-bit — MSE: {mse16:.2e}, SNR: {snr16:.2f} dB, max|e|: {max16:.2e}, NMSE: {nmse16:.2e}")
print(f"    32-bit — MSE: {mse32:.2e}, SNR: {snr32:.2f} dB, max|e|: {max32:.2e}, NMSE: {nmse32:.2e}")
print(f"    Theory: 16-bit Q{coeff_q16} signal SNR ≈ {6.02*coeff_q16+1.76:.0f} dB")
print(f"    Theory: 32-bit Q{coeff_q32} signal SNR ≈ {6.02*coeff_q32+1.76:.0f} dB")

# ── 3. 时域图绘制 ───────────────────────────────────────
print(f"[{current_task}/{TOTAL_TASK}] Time-domain plot…")
current_task+=1
# 选取有代表性的两个区域：一个过渡段 + 一个稳态段
start1, length = N // 5, 400
start2 = N // 2
t1 = np.arange(start1, start1 + length)
t2 = np.arange(start2, start2 + length)

fig = plt.figure(figsize=(16, 10))

# ── 3 路输出信号叠加图 ──
ax0 = fig.add_subplot(3, 2, (1, 2))
ax0.plot(t1, y_py[start1:start1+length], 'k-', linewidth=1.2, label='Python float64 (ref)')
ax0.plot(t1, y_16[start1:start1+length], 'r--', linewidth=0.9, alpha=0.8, label=f'C 16-bit Q{coeff_q16} (SNR={snr16:.1f} dB)')
ax0.plot(t1, y_32[start1:start1+length], 'b-.', linewidth=0.7, alpha=0.9, label=f'C 32-bit Q{coeff_q32} (SNR={snr32:.1f} dB)')
ax0.set_ylabel('Amplitude')
ax0.set_title('Filtered Output — Time Domain (3-way overlay)')
ax0.legend(fontsize=8, ncol=3)
ax0.grid(True, alpha=0.3)

# ── 各信号独立子图 ──
for idx, (y, color, label) in enumerate([
    (y_py, 'black', 'Python float64 (reference)'),
    (y_16, 'red', f'C 16-bit Q{coeff_q16}'),
    (y_32, 'blue', f'C 32-bit Q{coeff_q32}'),
]):
    ax = fig.add_subplot(3, 2, 3 + idx)
    ax.plot(t1, y[start1:start1+length], color=color, linewidth=0.7)
    ax.set_ylabel(label, color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.grid(True, alpha=0.3)
    if idx == 2:
        ax.set_xlabel('Sample index')

# ── 32-bit 误差（放大尺度） ──
ax_err = fig.add_subplot(3, 2, 6)
ax_err.plot(t1, e_16[start1:start1+length], 'r-', linewidth=0.5, alpha=0.7, label=f'16-bit error')
ax_err.plot(t1, e_32[start1:start1+length], 'b-', linewidth=0.5, alpha=0.7, label=f'32-bit error')
ax_err.set_xlabel('Sample index')
ax_err.set_ylabel('Error')
ax_err.set_title('Error (Python float64 − C fixed)')
ax_err.legend(fontsize=7)
ax_err.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'time_domain_comparison.png'), dpi=150)
plt.close()

# ── 4. 误差图 ──────────────────────────────────────────
print(f"[{current_task}/{TOTAL_TASK}] Full error signal plots…")
current_task+=1
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, sharey=True)

ax1.plot(e_16, 'r-', linewidth=0.15, alpha=0.7)
ax1.set_ylabel('Error (16-bit)')
ax1.set_title(f'16-bit Error Signal, MSE={mse16:.2e}, SNR={snr16:.1f} dB')
ax1.grid(True, alpha=0.3)

ax2.plot(e_32, 'b-', linewidth=0.15, alpha=0.7)
ax2.set_xlabel('Sample index')
ax2.set_ylabel('Error (32-bit)')
ax2.set_title(f'32-bit Error Signal, MSE={mse32:.2e}, SNR={snr32:.1f} dB')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'error_full.png'), dpi=150)
plt.close()

# ── 5. 误差直方图 ─────────────────────────────────────────────────
print(f"[{current_task}/{TOTAL_TASK}] Error histogram…")
current_task+=1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 16-bit
ax1.hist(e_16, bins=80, density=True, color='red', alpha=0.6, edgecolor='darkred', linewidth=0.3)
q16_step = 1.0 / float(1 << coeff_q16)
sigma_theory_16 = q16_step / np.sqrt(12)
x_lim = 4 * sigma_theory_16 * 5
x_pdf = np.linspace(-x_lim, x_lim, 500)
pdf_theory = np.exp(-0.5 * (x_pdf / sigma_theory_16) ** 2) / (sigma_theory_16 * np.sqrt(2 * np.pi))
ax1.plot(x_pdf, pdf_theory, 'k--', linewidth=1.0, label=f'Theory Q{coeff_q16} σ={sigma_theory_16:.2e}')
ax1.set_xlabel('Error')
ax1.set_ylabel('Density')
ax1.set_title(f'16-bit Error Histogram (σ_meas={np.std(e_16):.2e})')
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)

# 32-bit
ax2.hist(e_32, bins=80, density=True, color='blue', alpha=0.6, edgecolor='darkblue', linewidth=0.3)
q32_step = 1.0 / float(1 << coeff_q32)
sigma_theory_32 = q32_step / np.sqrt(12)
x_lim32 = 4 * sigma_theory_32 * 5
x_pdf32 = np.linspace(-x_lim32, x_lim32, 500)
pdf_theory32 = np.exp(-0.5 * (x_pdf32 / sigma_theory_32) ** 2) / (sigma_theory_32 * np.sqrt(2 * np.pi))
ax2.plot(x_pdf32, pdf_theory32, 'k--', linewidth=1.0, label=f'Theory Q{coeff_q32} σ={sigma_theory_32:.2e}')
ax2.set_xlabel('Error')
ax2.set_ylabel('Density')
ax2.set_title(f'32-bit Error Histogram (σ_meas={np.std(e_32):.2e})')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'error_histogram.png'), dpi=150)
plt.close()

# ── 6. 输出频谱 ─────────────────────────────────────
print(f"[{current_task}/{TOTAL_TASK}] Output spectrum comparison…")
current_task+=1
fs = 8000
nperseg = min(4096, N // 4)

f_py, psd_py = signal.welch(y_py, fs=fs, nperseg=nperseg, return_onesided=True)
_, psd_16 = signal.welch(y_16, fs=fs, nperseg=nperseg, return_onesided=True)
_, psd_32 = signal.welch(y_32, fs=fs, nperseg=nperseg, return_onesided=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# 全频带
ax1.semilogy(f_py, psd_py, 'k-', linewidth=1.0, label='Python float64 (ref)')
ax1.semilogy(f_py, psd_16, 'r--', linewidth=0.8, alpha=0.8, label=f'C 16-bit Q{coeff_q16}')
ax1.semilogy(f_py, psd_32, 'b-.', linewidth=0.6, alpha=0.9, label=f'C 32-bit Q{coeff_q32}')
ax1.set_ylabel('Power Spectral Density')
ax1.set_title('Output Power Spectrum (Welch) — Full Band')
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, fs/2])

# 低频放大（滤波器过渡带附近，差异更明显）
ax2.semilogy(f_py, psd_py, 'k-', linewidth=1.0, label='Python float64 (ref)')
ax2.semilogy(f_py, psd_16, 'r--', linewidth=0.8, alpha=0.8, label=f'C 16-bit Q{coeff_q16}')
ax2.semilogy(f_py, psd_32, 'b-.', linewidth=0.6, alpha=0.9, label=f'C 32-bit Q{coeff_q32}')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Power Spectral Density')
ax2.set_title('Output Power Spectrum — Low Frequency Zoom (0–500 Hz)')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 500])

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'output_spectrum.png'), dpi=150)
plt.close()

# ── 7. 误差功率谱 ────────────────────────────────────────────
print(f"[{current_task}/{TOTAL_TASK}] Error power spectrum…")
current_task+=1
fs = 8000
nperseg = min(4096, N // 4)

f16, psd16 = signal.welch(e_16, fs=fs, nperseg=nperseg, return_onesided=True)
f32, psd32 = signal.welch(e_32, fs=fs, nperseg=nperseg, return_onesided=True)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.semilogy(f16, psd16, 'r-', linewidth=0.8, alpha=0.8, label='16-bit error PSD')
ax.semilogy(f32, psd32, 'b-', linewidth=0.8, alpha=0.8, label='32-bit error PSD')
# theoretical output quantization noise floor
q16_floor = (q16_step**2) / 12 / (fs/2)
q32_floor = (q32_step**2) / 12 / (fs/2)
ax.axhline(q16_floor, color='red', linestyle='--', linewidth=0.8,
           label=f'Q{coeff_q16} theory floor ({10*np.log10(q16_floor):.0f} dB/Hz)')
ax.axhline(q32_floor, color='blue', linestyle='--', linewidth=0.8,
           label=f'Q{coeff_q32} theory floor ({10*np.log10(q32_floor):.0f} dB/Hz)')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power Spectral Density')
ax.set_title('Error Power Spectrum (Welch)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, fs/2])

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'error_spectrum.png'), dpi=150)
plt.close()

# ── 8. MSE/SNR bar chart ───────────────────────────────────────────────
print(f"[{current_task}/{TOTAL_TASK}] MSE/SNR comparison…")
current_task+=1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

bits = ['16-bit', '32-bit']
mse_vals = [mse16, mse32]
snr_vals = [snr16, snr32]
colors = ['red', 'blue']

ax1.bar(bits, mse_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax1.set_ylabel('MSE')
ax1.set_title('Mean Squared Error')
for i, v in enumerate(mse_vals):
    ax1.text(i, v * 1.1, f'{v:.2e}', ha='center', fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')

ax2.bar(bits, snr_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.set_ylabel('SNR (dB)')
ax2.set_title('Signal-to-Noise Ratio')
for i, v in enumerate(snr_vals):
    ax2.text(i, v * 0.5, f'{v:.1f} dB', ha='center', fontsize=10, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'mse_comparison.png'), dpi=150)
plt.close()

# ── 9. 恢复滤波结果为wav文件 ───────────────────────────────────────────────
print(f"[{current_task}/{TOTAL_TASK}] re-generate .wav…")
current_task+=1
import wave
result = [y_py, y_16, y_32]
result_name = [
    "output_py.wav",
    "output_16.wav",
    "output_32.wav"
]
for wav, name in zip(result, result_name):
        # 转 int16
    y_int16 = np.round(wav * 32767).astype(np.int16)

    # 保存 wav
    with wave.open(os.path.join(DATA_DIR, name), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(y_int16.tobytes())
        print(f"    Generate .wav of {name}")