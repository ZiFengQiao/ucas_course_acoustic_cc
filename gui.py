'''
Author: Wang, Qiaoyu; DeepSeek-V4 Pro
Date: 2026-06-07 16:14:08
LastEditors: Wang, Qiaoyu
LastEditTime: 2026-06-07 19:38:11
Description: 
'''

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import soundfile as sf
import sounddevice as sd
import math
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from scipy.signal import resample_poly
from hrtf import HRTFProcessor

PROJECT_ROOT = Path(__file__).resolve().parent
HRIR_MAT_PATH = PROJECT_ROOT / 'hrtf_data' / 'hrir_final.mat'


def _scan_output_devices():
    """Return list of (type, identifier, display_name, extra_info) for all outputs.

    type: 'alsa' or 'pw'
    identifier: ALSA device index (int) or PipeWire sink name (str)
    display_name: short label for dropdown
    extra_info: full description for tooltip
    """
    outputs = []

    # ── ALSA hardware devices (via sounddevice) ──
    try:
        for idx, dev in enumerate(sd.query_devices()):
            name = dev['name']
            outputs.append(('alsa', idx, f"ALSA: {name}", name))
    except Exception:
        pass

    # ── PipeWire / PulseAudio sinks (via pulsectl) ──
    # for Linux systems with PipeWire
    try:
        import pulsectl
        pulse = pulsectl.Pulse('hrtf-gui')
        for s in pulse.sink_list():
            label = f"PW : {s.description}"
            if s.name.startswith('bluez'):
                label = f"BT : {s.description}"
            outputs.append(('pw', s.name, label, s.description))
        pulse.close()
    except Exception:
        pass

    # Sort: Bluetooth first, then PipeWire, then analog ALSA, then HDMI
    def _sort_key(item):
        typ, ident, label, desc = item
        if typ == 'pw' and 'bluez' in ident:
            return (0, label)
        if typ == 'pw':
            return (1, label)
        l = desc.lower()
        if 'analog' in l:
            return (2, label)
        if 'hdmi' in l or 'displayport' in l:
            return (4, label)
        return (3, label)

    outputs.sort(key=_sort_key)
    return outputs


def _is_pw_available():
    """Return True if PipeWire/PulseAudio sink control is available."""
    try:
        import pulsectl
        return True
    except ImportError:
        return False


class VirtualSoundGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HRTF Virtual Sound Source Localization")
        self.root.geometry("740x500")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use('clam')

        # Scan all output devices (ALSA + PipeWire)
        self.output_devices = _scan_output_devices()
        if self.output_devices:
            typ, ident, label, desc = self.output_devices[0]
            self._set_output_device(typ, ident)

        try:
            self.processor = HRTFProcessor(HRIR_MAT_PATH)
        except Exception as e:
            messagebox.showerror("Error",
                f"Failed to load HRTF database:\n{e}\n"
                f"Expected at: {HRIR_MAT_PATH!s}")
            self.processor = None

        self.audio_data = None
        self.fs = None
        self.audio_path = None
        self.is_playing_rotating = False
        self._playback_proc = None  # subprocess for pw-play
        self._pb_start = 0.0       # playback start time (monotonic)
        self._pb_duration = 0.0    # total duration in seconds
        self._pb_active = False
        self._pb_after_id = None   # after() id for progress updates

        self._build_ui()
        self.draw_head()

    # ── UI Construction ──────────────────────────────────────────────

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel — controls
        ctrl = ttk.LabelFrame(main_frame, text="Controls", padding=12)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))

        # File selection
        file_frame = ttk.Frame(ctrl)
        file_frame.pack(fill=tk.X, pady=(0, 8))

        self.btn_load = ttk.Button(file_frame, text="Select Audio File",
                                   command=self.load_audio)
        self.btn_load.pack(fill=tk.X)

        self.file_label = ttk.Label(ctrl, text="No file selected",
                                    foreground="gray", wraplength=240)
        self.file_label.pack(fill=tk.X, pady=(0, 12))

        # Output device selector
        ttk.Label(ctrl, text="Output device:").pack(anchor=tk.W)
        self.device_var = tk.StringVar()
        device_names = [label for _, _, label, _ in self.output_devices]
        self.device_cb = ttk.Combobox(ctrl, textvariable=self.device_var,
                                      values=device_names, state='readonly',
                                      width=32)
        if device_names:
            self.device_cb.current(0)
        self.device_cb.pack(fill=tk.X, pady=(0, 6))
        self.device_cb.bind('<<ComboboxSelected>>', self._on_device_change)

        # Azimuth / Elevation
        ttk.Label(ctrl, text="Azimuth (degrees):").pack(anchor=tk.W)
        self.azimuth_entry = ttk.Entry(ctrl, width=10)
        self.azimuth_entry.insert(0, "0")
        self.azimuth_entry.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(ctrl, text="Elevation (degrees):").pack(anchor=tk.W)
        self.elevation_entry = ttk.Entry(ctrl, width=10)
        self.elevation_entry.insert(0, "0")
        self.elevation_entry.pack(fill=tk.X, pady=(0, 5))

        self.itd_label = ttk.Label(ctrl, text="ITD: -- ms",
                                   foreground="gray")
        self.itd_label.pack(anchor=tk.W, pady=(0, 12))

        # Process / Save buttons
        btn_row1 = ttk.Frame(ctrl)
        btn_row1.pack(fill=tk.X, pady=(0, 8))
        self.btn_static = ttk.Button(btn_row1, text="Process && Play",
                                     command=self.process_play)
        self.btn_static.pack(side=tk.LEFT, fill=tk.X, expand=True,
                             padx=(0, 3))
        self.btn_save = ttk.Button(btn_row1, text="Save WAV",
                                   command=self.save_audio)
        self.btn_save.pack(side=tk.LEFT, fill=tk.X, expand=True,
                           padx=(3, 0))

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Rotate button
        self.btn_rotate = ttk.Button(ctrl,
            text="Generate && Play Rotating Audio (360)",
            command=self.play_rotating)
        self.btn_rotate.pack(fill=tk.X, pady=(0, 8))

        # Progress bar
        pb_frame = ttk.Frame(ctrl)
        pb_frame.pack(fill=tk.X, pady=(0, 0))
        self.progress_bar = ttk.Progressbar(pb_frame, mode='determinate',
                                            length=220)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True,
                               padx=(0, 6))
        self.progress_label = ttk.Label(pb_frame, text="--", width=10,
                                        anchor=tk.E)
        self.progress_label.pack(side=tk.RIGHT)

        # Right panel — head canvas
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas_size = 420
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size,
                                height=self.canvas_size, bg="white",
                                highlightthickness=1,
                                highlightbackground="#ccc")
        self.canvas.pack()

    # ── Audio I/O ────────────────────────────────────────────────────

    def load_audio(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.flac"), ("All Files", "*.*")])
        if not filepath:
            return

        try:
            audio, fs = sf.read(filepath)
            if audio.ndim > 1:
                # 取单通道
                audio = audio[:, 0]

            if fs != 44100:
                # 重采样到44100 Hz
                g = math.gcd(fs, 44100)
                audio = resample_poly(audio, up=44100 // g, down=fs // g)
                fs = 44100

            self.audio_data = audio.astype(np.float64)
            self.fs = fs
            self.audio_path = filepath

            basename = os.path.basename(filepath)
            duration = len(audio) / fs
            self.file_label.config(
                text=f"{basename}\n{duration:.1f}s, {fs} Hz, mono")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio:\n{e}")

    # ── Output device selection ──────────────────────────────────────

    def _on_device_change(self, event=None):
        sel_idx = self.device_cb.current()
        if sel_idx < 0 or sel_idx >= len(self.output_devices):
            return
        typ, ident, label, desc = self.output_devices[sel_idx]
        self._set_output_device(typ, ident)

    def _set_output_device(self, typ, ident):
        self._out_type = typ
        self._out_ident = ident
        if typ == 'alsa':
            sd.default.device = (sd.default.device[0], ident)

    def _get_selected_device(self):
        """Return (type, identifier) for current selection."""
        sel_idx = self.device_cb.current()
        if sel_idx < 0 or sel_idx >= len(self.output_devices):
            return None, None
        typ, ident, _, _ = self.output_devices[sel_idx]
        return typ, ident

    # ── Playback backends ────────────────────────────────────────────

    def _play_audio(self, binaural, fs):
        """Play binaural audio through the currently selected output device."""
        self._stop_audio()
        typ, ident = self._get_selected_device()

        self._start_progress(len(binaural) / fs)

        if typ == 'alsa':
            sd.play(binaural, fs)
        elif typ == 'pw':
            self._play_via_pwplay(binaural, fs, ident)

    def _start_progress(self, duration):
        self._pb_start = time.monotonic()
        self._pb_duration = duration
        self._pb_active = True
        self.progress_bar['maximum'] = 100
        self.progress_bar['value'] = 0
        self._schedule_progress_update()

    def _schedule_progress_update(self):
        if not self._pb_active:
            return
        self._pb_after_id = self.root.after(100, self._update_progress)

    def _update_progress(self):
        if not self._pb_active:
            return
        elapsed = time.monotonic() - self._pb_start
        pct = min(elapsed / self._pb_duration * 100, 100) if self._pb_duration > 0 else 100
        self.progress_bar['value'] = pct
        remain = max(self._pb_duration - elapsed, 0)
        self.progress_label.config(text=f"{remain:.1f}s")
        if pct >= 100 or not self._is_playback_active():
            self._stop_progress()
        else:
            self._schedule_progress_update()

    def _is_playback_active(self):
        """Check whether audio is still playing on the current device."""
        try:
            stream = sd.get_stream()
            return stream is not None and stream.active
        except Exception:
            pass
        if self._playback_proc and self._playback_proc.poll() is not None:
            return False
        return self._pb_active

    def _stop_progress(self):
        self._pb_active = False
        if self._pb_after_id:
            self.root.after_cancel(self._pb_after_id)
            self._pb_after_id = None
        self.progress_bar['value'] = 0
        self.progress_label.config(text="--")

    def _play_via_pwplay(self, binaural, fs, sink_name):
        """Write audio to temp file and play via pw-play --target."""
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_path = tmp.name
        tmp.close()
        sf.write(tmp_path, binaural, fs)

        def _run():
            try:
                proc = subprocess.Popen(
                    ['pw-play', '--target', sink_name, tmp_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._playback_proc = proc
                proc.wait()
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        threading.Thread(target=_run, daemon=True).start()

    def _stop_audio(self):
        """Stop all current playback."""
        sd.stop()
        if self._playback_proc and self._playback_proc.poll() is None:
            self._playback_proc.terminate()
            self._playback_proc = None
        self._stop_progress()

    # ── Process & Play / Save ────────────────────────────────────────

    def _get_angles(self):
        try:
            az = float(self.azimuth_entry.get())
            el = float(self.elevation_entry.get())
            return az, el
        except ValueError:
            messagebox.showerror("Error",
                                 "Invalid azimuth or elevation value.")
            return None, None

    def process_play(self):
        if self.processor is None:
            messagebox.showerror("Warning", "HRTF database not loaded.")
            return
        if self.audio_data is None:
            messagebox.showerror("Warning",
                                 "Please select an audio file first.")
            return

        az, el = self._get_angles()
        if az is None:
            return

        itd_s = self.processor.calculate_itd(az, el)[0]
        self.itd_label.config(text=f"ITD: {itd_s * 1000:.3f} ms")

        binaural = self.processor.apply_hrtf(self.audio_data, az, el)
        self._play_only(binaural, az, el)

    def save_audio(self):
        if self.processor is None:
            messagebox.showerror("Warning", "HRTF database not loaded.")
            return
        if self.audio_data is None:
            messagebox.showerror("Warning",
                                 "Please select an audio file first.")
            return

        az, el = self._get_angles()
        if az is None:
            return

        itd_s = calculate_itd(self.processor, az, el)
        self.itd_label.config(text=f"ITD: {itd_s * 1000:.3f} ms")

        binaural = self.processor.apply_hrtf(self.audio_data, az, el)
        filename = PROJECT_ROOT / f"output_az{az:.0f}_el{el:.0f}.wav"
        sf.write(str(filename), binaural, self.fs)
        self.draw_sound_source(az, el)
        messagebox.showinfo("Saved", f"Audio saved as:\n{filename.name}")

    def _play_only(self, binaural, az, el):
        self.draw_sound_source(az, el)
        self._play_audio(binaural, self.fs)

    # ── Rotate ───────────────────────────────────────────────────────

    def play_rotating(self):
        if self.processor is None or self.audio_data is None:
            messagebox.showerror("Warning",
                "Please load audio and ensure HRTF database is present.")
            return

        self.is_playing_rotating = True
        self.disable_controls()

        binaural = self.processor.rotate_sound(self.audio_data)
        sf.write(str(PROJECT_ROOT / 'output_rotating.wav'), binaural, self.fs)

        self._play_audio(binaural, self.fs)
        self._schedule_rotation_viz(binaural, 0)

    def _schedule_rotation_viz(self, binaural, step):
        if not self.is_playing_rotating:
            return

        block_dur = 0.1
        block_samples = int(self.fs * block_dur)
        n_blocks = max(
            (len(self.audio_data) + block_samples - 1) // block_samples,
            25)

        if step < n_blocks:
            angle = step * (360.0 / n_blocks)
            az = 80 * np.sin(np.radians(angle))
            el = 180 if angle >= 90 and angle < 270 else 0

            self.draw_sound_source(az, el)
            self.root.after(int(block_dur * 1000),
                            self._schedule_rotation_viz,
                            binaural, step + 1)
        else:
            self.is_playing_rotating = False
            self.enable_controls()
            self.draw_head()

    # ── Control state ────────────────────────────────────────────────

    def disable_controls(self):
        for w in [self.btn_load, self.btn_static, self.btn_save,
                  self.btn_rotate]:
            w.state(['disabled'])
        self.device_cb.state(['disabled'])

    def enable_controls(self):
        for w in [self.btn_load, self.btn_static, self.btn_save,
                  self.btn_rotate]:
            w.state(['!disabled'])
        self.device_cb.state(['!disabled'])

    # ── Canvas drawing ───────────────────────────────────────────────

    def draw_head(self):
        self.canvas.delete("all")
        c = self.canvas_size / 2
        r = 150

        self.canvas.create_oval(c - r, c - r, c + r, c + r,
                                dash=(4, 4), outline="#ddd")
        self.canvas.create_oval(c - r / 2, c - r / 2,
                                c + r / 2, c + r / 2,
                                dash=(2, 4), outline="#eee")
        self.canvas.create_line(c, c - r, c, c + r,
                                fill="#ddd", dash=(4, 4))
        self.canvas.create_line(c - r, c, c + r, c,
                                fill="#ddd", dash=(4, 4))

        self.canvas.create_text(c, c - r - 10, text="0 deg (Front)",
                                fill="gray")
        self.canvas.create_text(c + r + 25, c, text="90 deg (Right)",
                                fill="gray")
        self.canvas.create_text(c - r - 25, c, text="-90 deg (Left)",
                                fill="gray")
        self.canvas.create_text(c, c + r + 10, text="180 deg (Back)",
                                fill="gray")

        # Head (top-down view)
        self.canvas.create_oval(c - 25, c - 30, c + 25, c + 30,
                                fill="#f4d4ba", outline="#e0c0a0")
        self.canvas.create_oval(c - 30, c - 10, c - 20, c + 10,
                                fill="#f4d4ba", outline="#e0c0a0")
        self.canvas.create_oval(c + 20, c - 10, c + 30, c + 10,
                                fill="#f4d4ba", outline="#e0c0a0")
        self.canvas.create_polygon(c, c - 35, c - 5, c - 28, c + 5, c - 28,
                                   fill="#e0c0a0")

    def draw_sound_source(self, azimuth, elevation=0):
        self.draw_head()
        c = self.canvas_size / 2
        # 半径固定
        r = 150

        rad = math.radians(azimuth)
        x = c + r * math.sin(rad)


        # el映射到平面
        if elevation == 180:
            y = c + r * math.cos(rad)
        else:
            y = c - r * math.cos(rad)

        self.canvas.create_oval(x - 8, y - 8, x + 8, y + 8,
                                fill="red", outline="darkred")
        self.canvas.create_line(c, c, x, y, fill="red", dash=(2, 2))
        self.canvas.create_text(x, y - 15,
                                text=f"Az: {azimuth:.1f} deg",
                                fill="darkred",
                                font=("Arial", 9, "bold"))
        self.canvas.create_text(10, 10, anchor=tk.NW,
                                text=f"Source Position\n"
                                     f"Azimuth: {azimuth:.1f} deg\n"
                                     f"Elevation: {elevation:.1f} deg",
                                fill="black")


if __name__ == '__main__':
    root = tk.Tk()
    app = VirtualSoundGUI(root)
    root.mainloop()
