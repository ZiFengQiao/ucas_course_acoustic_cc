'''
Author: Wang, Qiaoyu
Date: 2026-06-02 22:19:49
LastEditors: Wang, Qiaoyu
LastEditTime: 2026-06-03 22:37:40
Description: 滤波器设计，量化
'''

import os

# .wav处理
import wave

import numpy as np
from scipy import signal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── paths ──────────────────────────────────────────────────────────────
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEECH_16K = os.path.join(PROJ_DIR, '..', 'doc', '作业4-7', 'clean_speech_16k.wav')
DATA_DIR = os.path.join(PROJ_DIR, 'data')
IMG_DIR  = os.path.join(PROJ_DIR, 'img')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ── parameters ─────────────────────────────────────────────────────────
FS   = 8000        # 目标采样率 (Hz)
FC   = 120         # 截止频率 (Hz)
ORDER = 4          # 滤波器阶数

# WAV 数据始终用 Q15（16-bit PCM）
DATA_Q_BITS = 15
DATA_SCALE = 1 << DATA_Q_BITS

# ── helper: determine Q format for coefficients ────────────────────────
def determine_coeff_q_bits(coeffs_float, total_bits):
    """根据系数最大绝对值，确定需要多少整数位，剩余给小数位。"""
    max_abs = max(abs(c) for c in coeffs_float)
    # 需要 ceil(log2(max_abs)) 个整数位（不含符号位），至少 0
    int_bits = max(0, int(np.ceil(np.log2(max_abs + 1e-16))))
    if int_bits + 1 >= total_bits:  # 至少留 1 位符号
        raise ValueError(f"Coefficient {max_abs} too large for {total_bits}-bit")
    frac_bits = total_bits - 1 - int_bits  # 1 bit sign, int_bits integer, rest fraction
    return frac_bits, 1 << frac_bits

# ── 1. load & resample speech ──────────────────────────────────────────
print("[1/6] Loading speech…")
with wave.open(SPEECH_16K, 'rb') as wf:
    assert wf.getnchannels() == 1, "mono only"
    assert wf.getsampwidth() == 2, "16-bit only"
    fs_orig = wf.getframerate()
    n_frames = wf.getnframes()
    raw = wf.readframes(n_frames)

x_16k = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
x_8k = signal.resample_poly(x_16k, up=1, down=fs_orig // FS).astype(np.float64)
print(f"    Samples: {len(x_16k)}@{fs_orig}Hz -> {len(x_8k)}@{FS}Hz")

# 保存为 Q15 WAV
x_int16 = np.round(x_8k).astype(np.int16)
with wave.open(os.path.join(DATA_DIR, 'input_8k.wav'), 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(FS)
    wf.writeframes(x_int16.tobytes())
print(f"    Saved data/input_8k.wav (int16 Q15)")

# ── 2. design IIR filter ───────────────────────────────────────────────
print("[2/6] Designing 4th-order Butterworth HPF…")
sos = signal.butter(ORDER, FC, btype='high', fs=FS, output='sos')

# plot frequency response
w, h = signal.sosfreqz(sos, worN=4096, fs=FS)
mag_db = 20 * np.log10(np.abs(h) + 1e-16)
phase = np.unwrap(np.angle(h)) * 180 / np.pi

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(w, mag_db, 'b-', linewidth=1)
ax1.axvline(FC, color='r', linestyle='--', linewidth=0.8, label=f'{FC} Hz cutoff')
ax1.set_ylabel('Magnitude (dB)')
ax1.set_title(f'{ORDER}th-Order Butterworth HPF ({FC} Hz, {FS} Hz)')
ax1.legend()
ax1.grid(True, alpha=0.4)
ax1.set_ylim([-60, 5])

ax2.plot(w, phase, 'b-', linewidth=1)
ax2.axvline(FC, color='r', linestyle='--', linewidth=0.8)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Phase (degrees)')
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'filter_response.png'), dpi=150)
plt.close()
print(f"    Saved img/filter_response.png")

# ── 3. apply filter (float reference) ──────────────────────────────────
print("[3/6] Applying float-point filter…")
x_norm = x_8k / DATA_SCALE
y_float = signal.sosfilt(sos, x_norm)
print(f"    Output range: [{y_float.min():.4f}, {y_float.max():.4f}]")
y_float.astype(np.float64).tofile(os.path.join(DATA_DIR, 'output_python.raw'))

# ── 4. extract and flatten coefficients ────────────────────────────────
print("[4/6] Extracting biquad coefficients…")
# sos shape: (2, 6) = [b0, b1, b2, a0, a1, a2], a0=1
coeffs_float = []
all_coeffs = []
for section in sos:
    b0, b1, b2, a1, a2 = float(section[0]), float(section[1]), float(section[2]), \
                         float(section[4]), float(section[5])
    coeffs_float.append((b0, b1, b2, a1, a2))
    all_coeffs.extend([b0, b1, b2, a1, a2])

# ── 5. determine Q formats & export coefficients ───────────────────────
print("[5/6] Determining coefficient Q formats…")
coeff_q16_bits, q16_scale = determine_coeff_q_bits(all_coeffs, 16)
coeff_q32_bits, q32_scale = determine_coeff_q_bits(all_coeffs, 32)

print(f"    Max |coeff| = {max(abs(c) for c in all_coeffs):.6f}")
print(f"    16-bit filter: coeff Q1.{coeff_q16_bits} (range ±2^{16-1-coeff_q16_bits:.1f})")
print(f"    32-bit filter: coeff Q1.{coeff_q32_bits} (range ±2^{32-1-coeff_q32_bits:.1f})")

with open(os.path.join(DATA_DIR, 'coeffs.txt'), 'w') as f:
    f.write(f"# Filter: Butterworth HPF, order={ORDER}, fc={FC}Hz, fs={FS}Hz\n")
    f.write(f"# coeff_q_bits_16: {coeff_q16_bits}\n")
    f.write(f"# coeff_q_bits_32: {coeff_q32_bits}\n")
    f.write(f"# sections: {len(coeffs_float)}\n")
    f.write(f"# Format per section: b0 b1 b2 a1 a2  (a0=1 implied)\n\n")
    for i, (b0, b1, b2, a1, a2) in enumerate(coeffs_float):
        q16 = [int(np.round(c * q16_scale)) for c in (b0, b1, b2, a1, a2)]
        q32 = [int(np.round(c * q32_scale)) for c in (b0, b1, b2, a1, a2)]
        f.write(f"[Section {i}]\n")
        f.write(f"float: {b0:.12f} {b1:.12f} {b2:.12f} {a1:.12f} {a2:.12f}\n")
        f.write(f"q16:   {' '.join(str(v) for v in q16)}\n")
        f.write(f"q32:   {' '.join(str(v) for v in q32)}\n\n")

print(f"    Saved data/coeffs.txt")
for i, (b0, b1, b2, a1, a2) in enumerate(coeffs_float):
    print(f"    Section {i}: b=[{b0:.6f}, {b1:.6f}, {b2:.6f}]  a=[1, {a1:.6f}, {a2:.6f}]")

# ── 5b. 量化输入数据并保存 ──────────────────────────────────────────
print(f"[5b] Quantizing input to Q{coeff_q16_bits} / Q{coeff_q32_bits}…")
x_norm = x_8k / DATA_SCALE  # float ∈ [-1, 1)

# 16-bit: Q{coeff_q16_bits} → int16_t raw
q16_int = np.round(x_norm * q16_scale).astype(np.int16)
q16_int.tofile(os.path.join(DATA_DIR, 'input_16bit.raw'))

# 32-bit: Q{coeff_q32_bits} → int32_t raw
q32_int = np.round(x_norm * q32_scale).astype(np.int32)
q32_int.tofile(os.path.join(DATA_DIR, 'input_32bit.raw'))

print(f"    Saved data/input_16bit.raw ({len(q16_int)} samples, uint16_t Q{coeff_q16_bits})")
print(f"    Saved data/input_32bit.raw ({len(q32_int)} samples, uint32_t Q{coeff_q32_bits})")

# ── 6. summary ─────────────────────────────────────────────────────────
print(f"[6/6] Done. Generated files:")
print(f"    data/input_8k.wav         — resampled speech ({len(x_int16)} samples, int16 Q15)")
print(f"    data/input_16bit.raw       — uint16_t Q{coeff_q16_bits} for C 16-bit filter")
print(f"    data/input_32bit.raw       — uint32_t Q{coeff_q32_bits} for C 32-bit filter")
print(f"    data/output_python.raw     — float64 reference output ({len(y_float)} samples)")
print(f"    data/coeffs.txt            — filter coefficients (float + Q{coeff_q16_bits}/Q{coeff_q32_bits})")
print(f"    img/filter_response.png    — frequency response")
