import numpy as np
import scipy.io as sio
from scipy.signal import fftconvolve

class HRTFProcessor:
    def __init__(self, mat_file_path):
        """
        Load HRIR data from CIPIC database (subject_003).
        hrtf_data/hrir_final.mat
        Contains:
          - hrir_l: (25, 50, 200) -> (azimuths, elevations, time_samples)
          - hrir_r: (25, 50, 200)
        """
        data = sio.loadmat(mat_file_path)
        self.hrir_l = data['hrir_l']
        self.hrir_r = data['hrir_r']
        
        # CIPIC grid
        self.azimuths = np.array([-80, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5, 
                                  0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65, 80])
        self.elevations = np.array([-45 + 5.625 * i for i in range(50)])
        
    def get_closest_indices(self, az, el):
        az_idx = np.abs(self.azimuths - az).argmin()
        el_idx = np.abs(self.elevations - el).argmin()
        return az_idx, el_idx
    
    def apply_hrtf(self, audio, az, el):
        az_idx, el_idx = self.get_closest_indices(az, el)
        hl = self.hrir_l[az_idx, el_idx, :]
        hr = self.hrir_r[az_idx, el_idx, :]
        
        out_l = fftconvolve(audio, hl, mode='full')
        out_r = fftconvolve(audio, hr, mode='full')
        return np.column_stack((out_l, out_r))

def calculate_itd(processor, az, el=0):
    """
    Calculate ITD using cross-correlation for horizontal plane HRIR (el=0).
    """
    az_idx, el_idx = processor.get_closest_indices(az, el)
    hl = processor.hrir_l[az_idx, el_idx, :]
    hr = processor.hrir_r[az_idx, el_idx, :]
    
    corr = np.correlate(hl, hr, mode='full')
    lag = np.argmax(corr) - (len(hr) - 1)
    # The ITD in samples, converted to time you would mult by 1/fs (where fs is usually 44100 form CIPIC)
    return lag / 44100.0

def rotate_sound(processor, audio, fs, enable_smoothing=True):
    """
    Simulate a sound source rotating around the horizontal plane.
    Uses Overlap-Add with crossfading to prevent clicks when smoothing is enabled.
    """
    hop_size = fs // 10  # 100ms per step
    
    if not enable_smoothing:
        binaural_frames = []
        # Rotate around the horizontal plane
        for i, az in enumerate(processor.azimuths):
            start = i * hop_size
            end = start + hop_size
            if start >= len(audio):
                break
            chunk = audio[start:end]
            binaural_chunk = processor.apply_hrtf(chunk, az, 0)
            binaural_frames.append(binaural_chunk[:len(chunk)])
        
        return np.concatenate(binaural_frames, axis=0)

    # Overlap-Add Method for smooth transition
    overlap = fs // 50  # 20ms overlapping crossfade window
    chunk_size = hop_size + overlap
    
    out_binaural = np.zeros((len(audio) + max(fs, 44100), 2)) # safety padded buffer
    
    for i, az in enumerate(processor.azimuths):
        start = i * hop_size
        end = start + chunk_size
        if start >= len(audio):
            break
            
        chunk = audio[start:end]
        if len(chunk) == 0:
            break
            
        binaural_chunk = processor.apply_hrtf(chunk, az, 0)
        n_out = len(binaural_chunk)
        
        # Apply crossfade windowing on the outputs
        win = np.ones(n_out)
        
        if i > 0:
            # Linear Fade-in for the overlap part
            fade_in = min(overlap, n_out)
            win[:fade_in] = np.linspace(0, 1, fade_in)
            
        if i < len(processor.azimuths) - 1 and end <= len(audio):
            # Linear Fade-out starting from hop_size
            fade_start = hop_size
            fade_len = min(overlap, n_out - fade_start)
            if fade_len > 0:
                win[fade_start:fade_start+fade_len] = np.linspace(1, 0, fade_len)
                win[fade_start+fade_len:] = 0 # zero out the tail beyond overlap
                
        binaural_chunk *= win[:, np.newaxis]
        
        # Add to global output
        add_end = start + n_out
        if add_end > len(out_binaural):
            out_binaural = np.pad(out_binaural, ((0, add_end - len(out_binaural)), (0, 0)), 'constant')
        out_binaural[start:add_end] += binaural_chunk
        
    return out_binaural[:len(audio)]

def generate_random_positions(processor, audio, fs, chunk_sec=1.5, num_chunks=10, enable_smoothing=True):
    """
    Generate audio combining random spatial orientations continuously using OLA.
    """
    import random
    
    hop_size = int(fs * chunk_sec)
    total_len = hop_size * num_chunks
    
    # We will tile or loop the input audio if it's too short
    while len(audio) < total_len + fs:
        audio = np.concatenate((audio, audio))
        
    overlap = fs // 50  # 20ms overlapping crossfade window
    chunk_size = hop_size + overlap
    
    out_binaural = np.zeros((total_len + max(fs, 44100), 2))
    positions = []
    
    for i in range(num_chunks):
        az = random.choice(processor.azimuths)
        el = random.choice([e for e in processor.elevations if -45 <= e <= 45])
        positions.append((az, el))
        
        start = i * hop_size
        end = start + chunk_size
        if start >= len(audio):
            break
            
        chunk = audio[start:end]
        if len(chunk) == 0:
            break
            
        binaural_chunk = processor.apply_hrtf(chunk, az, el)
        
        if enable_smoothing:
            n_out = len(binaural_chunk)
            win = np.ones(n_out)
            if i > 0:
                fade_in = min(overlap, n_out)
                win[:fade_in] = np.linspace(0, 1, fade_in)
            if i < num_chunks - 1:
                fade_start = hop_size
                fade_len = min(overlap, n_out - fade_start)
                if fade_len > 0:
                    win[fade_start:fade_start+fade_len] = np.linspace(1, 0, fade_len)
                    win[fade_start+fade_len:] = 0
            binaural_chunk *= win[:, np.newaxis]
        else:
            # truncate cleanly to avoid overlaps accumulating without Window
            binaural_chunk = binaural_chunk[:hop_size]

        add_end = start + len(binaural_chunk)
        if add_end > len(out_binaural):
            out_binaural = np.pad(out_binaural, ((0, add_end - len(out_binaural)), (0, 0)), 'constant')
        out_binaural[start:add_end] += binaural_chunk
        
    return out_binaural[:total_len], positions
