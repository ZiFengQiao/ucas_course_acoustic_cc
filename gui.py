import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import soundfile as sf
import sounddevice as sd
import math
import random

from ssl import HRTFProcessor, rotate_sound

class VirtualSoundGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HRTF Virtual Sound Source Viewer")
        self.root.geometry("700x480")
        self.root.configure(bg="#f0f0f0")
        
        style = ttk.Style()
        style.theme_use('clam')
        
        try:
            self.processor = HRTFProcessor('hrtf_data/hrir_final.mat')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load HRTF database:\n{e}\nEnsure CIPIC database is in the correct directory.")
            self.processor = None

        self.audio_data = None
        self.fs = None
        self.is_playing_random = False
        self.is_playing_rotating = False
        
        # UI Layout
        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left Panel (Controls)
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=15)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        self.btn_load = ttk.Button(control_frame, text="Select Audio (WAV)", command=self.load_audio)
        self.btn_load.pack(fill=tk.X, pady=(0, 10))
        
        self.file_label = ttk.Label(control_frame, text="No file selected", foreground="gray", wraplength=200)
        self.file_label.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(control_frame, text="Azimuth (degrees):").pack(anchor=tk.W)
        self.azimuth_entry = ttk.Entry(control_frame)
        self.azimuth_entry.insert(0, "0")
        self.azimuth_entry.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(control_frame, text="Elevation (degrees):").pack(anchor=tk.W)
        self.elevation_entry = ttk.Entry(control_frame)
        self.elevation_entry.insert(0, "0")
        self.elevation_entry.pack(fill=tk.X, pady=(0, 20))
        
        self.btn_static = ttk.Button(control_frame, text="Process & Play Static", command=self.process_play)
        self.btn_static.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        self.enable_ola_var = tk.BooleanVar(value=True)
        self.chk_ola = ttk.Checkbutton(control_frame, text="Enable Overlap-Add Smoothing", variable=self.enable_ola_var)
        self.chk_ola.pack(fill=tk.X, pady=(0, 10))
        
        self.btn_rotate = ttk.Button(control_frame, text="Generate & Play Rotating Audio", command=self.play_rotating)
        self.btn_rotate.pack(fill=tk.X, pady=(0, 10))
        
        self.btn_random = ttk.Button(control_frame, text="Start Random Continuous Audio", command=self.toggle_random)
        self.btn_random.pack(fill=tk.X, pady=(0, 10))

        # Right Panel (Canvas for drawing)
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas_size = 400
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size, bg="white", highlightthickness=1, highlightbackground="#ccc")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Draw Initial Head and coordinate system
        self.draw_head()
        
    def draw_head(self):
        self.canvas.delete("all")
        c = self.canvas_size / 2
        r = 150
        
        # Grid circles
        self.canvas.create_oval(c-r, c-r, c+r, c+r, dash=(4, 4), outline="#ddd")
        self.canvas.create_oval(c-r/2, c-r/2, c+r/2, c+r/2, dash=(2, 4), outline="#eee")
        
        # Grid lines
        self.canvas.create_line(c, c-r, c, c+r, fill="#ddd", dash=(4, 4))
        self.canvas.create_line(c-r, c, c+r, c, fill="#ddd", dash=(4, 4))
        self.canvas.create_text(c, c-r-10, text="0° (Front)", fill="gray")
        self.canvas.create_text(c+r+25, c, text="90° (Right)", fill="gray")
        self.canvas.create_text(c-r-25, c, text="-90° (Left)", fill="gray")
        self.canvas.create_text(c, c+r+10, text="180° (Back)", fill="gray")

        # Head (Top-down view)
        self.canvas.create_oval(c-25, c-30, c+25, c+30, fill="#f4d4ba", outline="#e0c0a0") # Head
        self.canvas.create_oval(c-30, c-10, c-20, c+10, fill="#f4d4ba", outline="#e0c0a0") # Left Ear
        self.canvas.create_oval(c+20, c-10, c+30, c+10, fill="#f4d4ba", outline="#e0c0a0") # Right Ear
        self.canvas.create_polygon(c, c-35, c-5, c-28, c+5, c-28, fill="#e0c0a0") # Nose indicator

    def draw_sound_source(self, azimuth, elevation=0):
        self.draw_head()
        c = self.canvas_size / 2
        r = 150
        
        # Convert azimuth to top-down radians (0 is top, positive is right)
        # Azimuth 0 -> (0, -r)
        # Azimuth 90 -> (r, 0)
        # We need to adapt CIPIC azimuth, which is interaural polar (lateral angle). 
        # But roughly: x = r * sin(az), y = -r * cos(az)
        
        rad = math.radians(azimuth)
        x = c + r * math.sin(rad)
        y = c - r * math.cos(rad)
        
        # Draw sound source point
        self.source_id = self.canvas.create_oval(x-8, y-8, x+8, y+8, fill="red", outline="darkred")
        
        # Draw line to head
        self.canvas.create_line(c, c, x, y, fill="red", dash=(2, 2))
        
        # Display Text
        self.canvas.create_text(x, y-15, text=f"Az: {azimuth:.1f}°", fill="darkred", font=("Arial", 9, "bold"))
        self.canvas.create_text(10, 10, anchor=tk.NW, text=f"Current Audio Source Position:\nAzimuth: {azimuth:.1f}°\nElevation: {elevation:.1f}°", fill="black")

    def load_audio(self):
        filepath = filedialog.askopenfilename(filetypes=[("WAV Audio", "*.wav"), ("All Files", "*.*")])
        if filepath:
            try:
                self.audio_data, self.fs = sf.read(filepath)
                if self.audio_data.ndim > 1:
                    self.audio_data = self.audio_data[:, 0]  # Mono
                self.file_label.config(text=f"...{filepath[-25:]}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load audio:\n{e}")

    def process_play(self):
        if self.processor is None:
            messagebox.showerror("Warning", "HRTF database not loaded.")
            return
        if self.audio_data is None:
            messagebox.showerror("Warning", "Please select an audio file first!")
            return
        
        try:
            az = float(self.azimuth_entry.get())
            el = float(self.elevation_entry.get())
            
            binaural_sig = self.processor.apply_hrtf(self.audio_data, az, el)
            
            self.draw_sound_source(az, el)
            
            sf.write('output_static.wav', binaural_sig, self.fs)
            sd.stop()
            sd.play(binaural_sig, self.fs)
            
        except ValueError:
            messagebox.showerror("Error", "Invalid Azimuth or Elevation values.")

    def play_rotating(self):
        self.is_playing_rotating = True
        if self.processor is None or self.audio_data is None:
            messagebox.showerror("Warning", "Please load audio file and ensure HRTF database is present.")
            return
        
        enable_ola = self.enable_ola_var.get()
        binaural_sig = rotate_sound(self.processor, self.audio_data, self.fs, enable_smoothing=enable_ola)
        sf.write('output_rotating.wav', binaural_sig, self.fs)
        
        sd.stop()
        sd.play(binaural_sig, self.fs)
        
        # Start visualization sync
        self.visualize_rotation()

    def visualization_loop(self, step, chunk_size_ms):
        if step < len(self.processor.azimuths) and self.is_playing_rotating:
            az = self.processor.azimuths[step]
            self.draw_sound_source(az, 0)
            self.root.after(chunk_size_ms, self.visualization_loop, step + 1, chunk_size_ms)
        else:
            self.draw_head()
            self.is_playing_rotating = False

    def visualize_rotation(self):
        chunk_size_ms = 100 # In rotate_sound, chunk size is fs // 10 -> 100ms
        self.root.after(0, self.visualization_loop, 0, chunk_size_ms)

    def toggle_random(self):
        if self.processor is None or self.audio_data is None:
            messagebox.showerror("Warning", "Please load audio file and ensure HRTF database is present.")
            return

        self.is_playing_random = not self.is_playing_random
        
        if self.is_playing_random:
            self.btn_random.config(text="Stop Random Audio")
            self.disable_controls()
            self.play_next_random_chunk()
        else:
            self.btn_random.config(text="Start Random Continuous Audio")
            self.enable_controls()
            sd.stop()
            self.draw_head()

    def play_next_random_chunk(self):
        if not self.is_playing_random:
            return
            
        # Select random azimuth and elevation
        az = random.choice(self.processor.azimuths)
        el = random.choice([e for e in self.processor.elevations if -45 <= e <= 45])
        
        # Process a 1.5-second chunk
        chunk_len = int(self.fs * 1.5)
        if len(self.audio_data) < chunk_len:
            audio_chunk = np.pad(self.audio_data, (0, chunk_len - len(self.audio_data)), 'wrap')
        else:
            start_idx = random.randint(0, len(self.audio_data) - chunk_len)
            audio_chunk = self.audio_data[start_idx:start_idx+chunk_len]
            
        binaural_sig = self.processor.apply_hrtf(audio_chunk, az, el)
        
        self.draw_sound_source(az, el)
        
        sd.stop()
        sd.play(binaural_sig, self.fs)
        
        # Schedule the next chunk after 1.5 seconds (1500 ms)
        self.root.after(1500, self.play_next_random_chunk)
        
    def disable_controls(self):
        self.btn_load.state(['disabled'])
        self.btn_static.state(['disabled'])
        self.btn_rotate.state(['disabled'])

    def enable_controls(self):
        self.btn_load.state(['!disabled'])
        self.btn_static.state(['!disabled'])
        self.btn_rotate.state(['!disabled'])

