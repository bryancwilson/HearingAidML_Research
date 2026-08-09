import matplotlib.pyplot as plt
import numpy as np
import librosa
import tensorflow as tf
import pandas as pd

from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfilt
from scipy.fft import rfft, rfftfreq, fft, fftfreq

from params import *
from helper import *
from helper import wavfile_read
from stim import *

def db_spl_to_amplitude(db_spl: float) -> float:
    """Converts dB SPL to linear physical pressure amplitude (Pascals)."""
    return 2e-5 * (10 ** (db_spl / 20.0))


def apply_tukey_ramp(tone: np.ndarray, ramp_ms: float, sr: int) -> np.ndarray:
    """Applies a raised cosine window to prevent spectral splatter."""
    num_ramp_samples = int((ramp_ms / 1000) * sr)
    if num_ramp_samples > 0 and num_ramp_samples * 2 < len(tone):
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(num_ramp_samples) / num_ramp_samples))
        tone[:num_ramp_samples] *= ramp
        tone[-num_ramp_samples:] *= ramp[::-1]
    return tone


def _gen_pure_tone(t: np.ndarray, freq: float, db_spl: float, **_) -> np.ndarray:
    amplitude = db_spl_to_amplitude(db_spl)
    return amplitude * np.sin(2 * np.pi * freq * t)


def _gen_dpoae(
    t: np.ndarray,
    freq: float,
    db_spl: float,
    use_suppressor: bool = False,
    suppressor_db_spl: float = 40.0,
    suppressor_freq: float | None = None,
    **_,
) -> np.ndarray:
    f2 = freq
    f1 = freq / 1.2
    amp_f1 = db_spl_to_amplitude(39 + 0.4 * db_spl)
    amp_f2 = db_spl_to_amplitude(db_spl)

    # Primary tones (f1, f2)
    signal = amp_f1 * np.sin(2 * np.pi * f1 * t) + amp_f2 * np.sin(2 * np.pi * f2 * t)

    # Conditional Suppressor Tone
    if use_suppressor:
        # Default to the 2f1 - f2 characteristic place if f3 is not explicitly set
        f3 = suppressor_freq if suppressor_freq is not None else (2 * f1 - f2) + 10
        amp_f3 = db_spl_to_amplitude(suppressor_db_spl)
        signal += amp_f3 * np.sin(2 * np.pi * f3 * t)

    return signal


def _gen_white_noise(t: np.ndarray, freq: float, db_spl: float, sr: int, process: str = None, **_) -> np.ndarray:
    amplitude = db_spl_to_amplitude(db_spl)
    tone = amplitude * np.random.normal(size=t.shape)
    
    if process == 'bandpass':
        nyq = 0.5 * sr
        sos = butter(N=8, Wn=[2000.0 / nyq, 3000.0 / nyq], btype='band', output='sos')
        tone = sosfilt(sos, tone)
        
    elif process == 'amplitude_modulation':
        db_envelope = np.linspace(20.0, db_spl, len(t))
        peak_amplitude_envelope = db_spl_to_amplitude(db_envelope) * np.sqrt(2)
        tone = peak_amplitude_envelope * np.sin(2 * np.pi * 3000 * t)
        
    return tone


def _gen_chirp(t: np.ndarray, freq: float, db_spl: float, sr: int, duration: float, **_) -> np.ndarray:
    amplitude = db_spl_to_amplitude(db_spl)
    return amplitude * librosa.chirp(fmin=100, fmax=9000, sr=sr, duration=duration)


def _gen_click(t: np.ndarray, db_spl: float, sr: int, **kwargs) -> np.ndarray:
    """
    Generates a monophasic rectangular click transient.
    Default duration is 100 microseconds (0.1 ms), standard for ABR/OAE testing.
    """
    amplitude = db_spl_to_amplitude(db_spl)
    click_duration_ms = kwargs.get('click_duration_ms', 0.1)
    
    # Initialize a silent timeline
    tone = np.zeros_like(t)
    
    # Calculate how many samples the pulse lasts based on the sampling rate
    pulse_samples = int(np.round((click_duration_ms / 1000.0) * sr))
    
    # Fire the rectangular pulse starting at the very beginning of the active window
    if pulse_samples > 0:
        tone[:pulse_samples] = amplitude
        
    return tone


# Updated strategy registry mapping names to functions
STIMULUS_REGISTRY = {
    'pure_tone': _gen_pure_tone,
    'dpoae': _gen_dpoae,
    'white_noise': _gen_white_noise,
    'chirp': _gen_chirp,
    'click': _gen_click
}

def create_stim(freq: float, db_spl: float, stim_type: str = 'pure_tone', plotfig: bool = False, **kwargs) -> np.ndarray:
    """
    Main factory interface for generating deep-learning compatible audiometric inputs.

    This function calculates a phase-consistent time axis based on the targeted frequency 
    to prevent fractional cycle endpoints, dispatches the signal matrix generation to 
    specialized internal sub-generators, handles onset/offset windowing ramps, and packs 
    the final waveform into the Keras-compliant tensor format shape [Batch, Time, Channels].

    Args:
        freq (float): 
            The target baseline physical frequency in Hertz (Hz). For pure tones, this is the 
            oscillation frequency. For DPOAEs, this functions explicitly as the f2 primary tone.
            (Ignored when stim_type='click').
        db_spl (float): 
            The target Sound Pressure Level in decibels relative to human hearing threshold 
            reference (20 microPascals). Controls global volume output or peak envelope bounds.
        stim_type (str, optional): 
            The structural generation profile to select. Must be one of:
            - 'pure_tone': Single sinusoidal burst.
            - 'dpoae': Simultaneous f1 and f2 primary tones for distortion analysis.
            - 'white_noise': Normally distributed random noise array.
            - 'chirp': Frequency swept signal.
            - 'click': Sharp 100-microsecond transient pulse for waveguide verification.
            Defaults to 'pure_tone'.

    Keyword Args (Passed via **kwargs):
        sr (int, optional): 
            Sampling rate of the digital audio pipeline in samples per second. Defaults to 100000.
        duration (float, optional): 
            Total primary timeline window for the active stimulus segment in seconds. Defaults to 0.1.
        ramp_ms (float, optional): 
            Onset and offset transition window duration in milliseconds. Applies a raised cosine 
            (Tukey) window profile to suppress edge splatter artifacts. Defaults to 500 for 
            pure tones and 0 for alternative types/clicks.
        padding (int, optional): 
            Number of explicit zero samples to symmetrically wrap around both sides of the 
            final compiled audio vector. Defaults to 256.
        click_duration_ms (float, optional):
            The active width of the monophasic click pulse in milliseconds. Defaults to 0.1 (100 us).
        process (str, optional): 
            Secondary signal processing toggle used exclusively when stim_type='white_noise'.

    Returns:
        np.ndarray: A 3D float32 NumPy tensor representing the processed acoustic input. 
                    Formatted matching shape: [1, Total_Samples, 1], satisfying 
                    Keras/TensorFlow convolutional layer input expectations.
    """
    # Pull optional settings out of kwargs or assign explicit defaults
    sr       = kwargs.get('sr', 20000)
    duration = kwargs.get('duration', 0.1)
    
    # FORCE ramp_ms to 0 for clicks to prevent windowing out our transient pulse!
    default_ramp = 500 if stim_type == 'pure_tone' else 0
    ramp_ms  = kwargs.get('ramp_ms', default_ramp)
    if stim_type == 'click':
        ramp_ms = 0
        
    padding  = kwargs.get('padding', 256)
    left_an_padding = kwargs.get('left_an_padding', 0)
    right_an_padding = kwargs.get('right_an_padding', 0)

    # Calculate phase-consistent time axis
    num_samples = int(np.round(sr * duration))  # e.g., exactly 2000
    t = np.arange(num_samples) / sr

    # Look up the designated strategy and generate raw wave
    if stim_type not in STIMULUS_REGISTRY:
        raise ValueError(f"Unknown stimulus type '{stim_type}'. Choose from: {list(STIMULUS_REGISTRY.keys())}")
        
    generator = STIMULUS_REGISTRY[stim_type]
    
    # Call generator safely handling parameters
    if stim_type == 'click':
        tone = generator(t=t, db_spl=db_spl, **kwargs)
    else:
        tone = generator(t=t, freq=freq, db_spl=db_spl, **kwargs)

    # Post-processing pipelines (Ramps, post-padding, tensor packaging)
    if ramp_ms > 0:
        tone = apply_tukey_ramp(tone, ramp_ms, sr)

    if plotfig:
        _plot_stimulus(t, tone, freq if stim_type != 'click' else 0, db_spl, sr)

    tone_padded = np.pad(tone, (padding + left_an_padding, padding + right_an_padding), mode='constant')

    # Format tensor to meet Keras expected layout: [Batch, Time, Channels]
    # return tone_padded
    return tone_padded[np.newaxis, :, np.newaxis].astype(np.float32)


def prepare_speech_stim(wavfile="TIMIT/TIMIT_DATA/lisa/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV", 
                        ramp_ms=20,
                        padding=256,
                        connear_fs=20e3,
                        sec=1,
                        plot_fig=False,
                        dBSPL=30):

    # Load Phoneme Timestamps
    phoneme_timestamps = pd.read_csv(
        "TIMIT/TIMIT_DATA/lisa/TIMIT/TRAIN/DR1/FCJF0/SA1.PHN", 
        sep="\s+", 
        header=None, 
        names=["start", "end", "phoneme"]
    ).to_numpy()
    
    # Extract TIMIT Data
    speech_signal, sr = wavfile_read(wavfile) # Audio File
    signalr = sp_sig.resample_poly(speech_signal, int(connear_fs), sr) # Resample to 20 kHz

    # Convert to dB SPL
    L = np.array([dBSPL]) #sound-level in dB SPL
    stim = np.zeros((len(L), signalr.size))
    for j in range(len(L)):
        stim[j, :] = p0 * 10**(L[j]/20) * signalr/rms(signalr)

    # Chop Time
    stim = stim[:, :int(connear_fs * sec)]

    # Apply Cosine Ramp
    num_ramp_samples = int((ramp_ms / 1000) * connear_fs)
    if num_ramp_samples * 2 < len(stim[0]):
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(num_ramp_samples) / num_ramp_samples))
        stim[:, :num_ramp_samples] *= ramp
        stim[:, -num_ramp_samples:] *= ramp[::-1]

    stim = np.squeeze(stim)
    stim = np.pad(stim, (padding, padding), mode='constant')
    stim = stim[np.newaxis, :, np.newaxis].astype(np.float32)
    # Plot Stimulis
    if plot_fig:
        plt.figure(figsize=(12, 6))
        plt.plot(stim)
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.title("Speech Stimulus")
        plt.grid()
        plt.show()

    # sd.play(stim.flatten(), connear_fs)
    # sd.wait()

    signal_tf = tf.cast(stim, tf.float32)

    return signal_tf

def _plot_stimulus(t: np.ndarray, tone: np.ndarray, freq: float, db_spl: float, sr: int):
    """Generates time-domain and frequency-domain plots for diagnostic verification."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time Domain Plot
    ax1.plot(t, tone, color='tab:blue', lw=1.2, label=f"{freq} Hz @ {db_spl} dB SPL")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (Pa)")
    ax1.set_title("Stimulus - Time Domain Waveform")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # Frequency Domain Plot
    freqs, spectrum = compute_spectrum(tone, sr)
    ax2.plot(freqs, spectrum, color='tab:red', lw=1.2)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_title("Acoustic Spectrum Profile")
    ax2.grid(True, alpha=0.3)
    
    # Focus frequency axis layout to avoid wasting space on empty spectra
    # if freq > 0:
    #     ax2.set_xlim(0, max(freq * 2.5, 4000))
        
    plt.tight_layout()
    plt.show()