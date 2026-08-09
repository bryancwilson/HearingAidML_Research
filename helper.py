from tqdm import tqdm
from keras.models import model_from_json, Model
from keras.utils import CustomObjectScope
from keras.initializers import GlorotUniform
from pathlib import Path

import librosa
import tensorflow as tf
import numpy as np
import scipy.signal as sp_sig
import scipy.io.wavfile
import matplotlib.pyplot as plt 

def load_connear_model(modeldir,json_name="/Gmodel.json",weights_name="/Gmodel.h5",crop=True,name=[]):
    # Function to load each CoNNear model using tensorflow and keras
    #print ("loading model from " + modeldir )
    json_file = open (modeldir + json_name, "r")
    loaded_model_json = json_file.read()
    json_file.close()

    custom_objects = {
        'Model': Model,
        'GlorotUniform': GlorotUniform
    }
    with CustomObjectScope(custom_objects):
        model = model_from_json(loaded_model_json, custom_objects={'tf': tf})

    if name:
        try:
            model.name = name
        except: # fix tensorflow 2 compatibility
            model._name = name
    model.load_weights(modeldir + weights_name)
    
    # if not crop: # for connecting the different modules
    #     model=model.layers[1]
    #     if name:
    #         model=Model(model.layers[0].input, model.layers[-2].output,name=name)
    #     else:
    #         model=Model(model.layers[0].input, model.layers[-2].output) # get uncropped output
    #else:
    #    model=Model(model.layers[0].input, model.layers[-1].output) # get cropped output

    return model

def compute_spectrum(signal, sr=20000):
    """Computes the single-sided amplitude spectrum of a signal in dB."""
    n = len(signal)
    # Compute the FFT
    fft_vals = np.fft.fft(signal)
    # Get the single-sided magnitudes
    fft_mag = np.abs(fft_vals[:n // 2]) / n
    # Convert to decibels (normalized to a peak of 0 dB for easy comparison)
    fft_db = 20 * np.log10(fft_mag / (np.max(fft_mag) + 1e-12))
    # Construct the frequency axis
    freqs = np.fft.fftfreq(n, 1/sr)[:n // 2]
    return freqs, fft_db

def get_training_chunks(stim, model):
    """
    Runs the model and extracts the input/output pair.
    (This wraps your existing generate_OAEs logic)
    """
    # verify_damage_plot(model, PARAMS)

    # Run Model
    prediction = model.predict(stim, verbose=0)
    
    # Return as chunks (Assuming you want raw waveforms)
    return stim, prediction

def get_dpoae(tl_bmm, cf_location=0, sig_start=0):
    # get the fft of last channel to predict the dpoae
    oae_sig = tl_bmm[0, sig_start: ,cf_location] # pick a CF
    oae_fft = np.fft.fft(oae_sig)
    nfft = oae_fft.shape[0]
    return np.absolute(oae_fft[:int(nfft/2)]), nfft

def slice_1dsignal_tf(signal, window_size, winshift, left_context=256, right_context=256):
    n_samples = tf.shape(signal)[0]
    total_window = window_size + left_context + right_context
    
    # 1. Calculate exactly how many slices the NumPy loop would produce
    # This mimics: len(list(range(0, n_samples, winshift)))
    num_slices = tf.cast(tf.math.ceil(tf.cast(n_samples, tf.float32) / tf.cast(winshift, tf.float32)), tf.int32)

    # 2. Pad enough to ensure we don't drop the last intended frame
    # We need at least (num_slices - 1) * winshift + total_window total length
    required_total_len = (num_slices - 1) * winshift + total_window
    current_padded_len = n_samples + left_context + right_context
    
    pad_at_very_end = tf.maximum(0, required_total_len - current_padded_len)
    
    padded_signal = tf.pad(signal, [[left_context, right_context + pad_at_very_end]], mode='CONSTANT')
    
    # 3. Frame it
    slices = tf.signal.frame(padded_signal, frame_length=total_window, frame_step=winshift)
    
    # 4. Force the count to match the NumPy loop count
    slices = slices[:num_slices]
    
    return tf.expand_dims(slices, axis=-1)

def slice_1dsignal(signal, window_size, winshift, minlength, left_context=256, right_context=256):
    """ 
    Return windows of the given signal by sweeping in stride fractions
    of window
    Slices that are less than minlength are omitted
    """
    # concatenate zeros to beginning for adding context
    n_samples = signal.shape[0]
    num_slices = (n_samples)
    slices = [] # initialize empty array 

    for beg_i in range(0, n_samples, winshift):
        beg_i_context = beg_i - left_context
        end_i = beg_i + window_size + right_context
        if n_samples - beg_i < minlength :
            break
        if beg_i_context < 0 :
            slice_ = np.concatenate((np.zeros((1, left_context - beg_i)),np.array([signal[:end_i]])), axis=1)
        elif end_i <= n_samples :
            slice_ = np.array([signal[beg_i_context:end_i]])
        else :
            slice_ = np.concatenate((np.array([signal[beg_i_context:]]), np.zeros((1, end_i - n_samples))), axis=1)
        # print(slice_.shape)
        slices.append(slice_)
    slices = np.vstack(slices)
    slices = np.expand_dims(slices, axis=2) # the CNN will need 3D data
    return slices

def undo_window_tf(tl_pr, window_size, winshift):
    """
    tl_pr: (num_frames, window_size, channels)
    This replaces the manual overlap-add loop.
    """
    # Remove the context if it was added during slicing before calling this, 
    # or ensure tl_pr only contains the central window_size.
    
    # Transpose to (channels, num_frames, window_size) for overlap_add
    tl_pr_t = tf.transpose(tl_pr, perm=[2, 0, 1])
    
    # Perform the overlap-add
    reconstructed = tf.signal.overlap_and_add(tl_pr_t, frame_step=winshift)
    
    # Transpose back to (time, channels)
    reconstructed = tf.transpose(reconstructed, perm=[1, 0])
    
    # To handle the "averaging" (the scale_ logic in your original code),
    # we generate the window weights and overlap-add them as well.
    ones = tf.ones_like(tl_pr_t)
    weights = tf.signal.overlap_and_add(ones, frame_step=winshift)
    
    # Avoid division by zero
    final_signal = reconstructed / (tf.transpose(weights) + 1e-8)
    
    return tf.expand_dims(final_signal, axis=0) # (1, time, channels)

def undo_window(tl_pr, winlength, winshift, ignore_first_set=0, fs = 20e3):
    trailing_silence = 0.
    nframes = tl_pr.shape[0]
    slength = ((nframes - 1)) * winshift + winlength
    tl_2d = np.zeros((slength, tl_pr.shape[2]))
    scale_ = np.zeros((slength,1))
    dummyones = np.ones((tl_pr.shape[0], tl_pr.shape[1]))
    trailing_zeros = int(trailing_silence * fs)
    sigrange = range (winlength)
    tl_2d [sigrange, :] = tl_2d [sigrange, :] + tl_pr[0]
    scale_[sigrange,0] = scale_[sigrange,0] + dummyones[0]
    for i in range(1,nframes):
        sigrange = range (i * winshift + ignore_first_set, (i*winshift) + winlength)
        tl_2d [sigrange, :] = tl_2d [sigrange, :] + tl_pr[i,ignore_first_set:,:]
        scale_[sigrange,0] = scale_[sigrange,0] + dummyones[i,ignore_first_set:]
    
    tl_2d /= scale_
    return np.expand_dims(tl_2d[trailing_zeros:,:], axis=0)

def rms (x, axis=0):
    # compute rms of a matrix
    sq = np.mean(np.square(x), axis = axis)
    return np.sqrt(sq)

def wavfile_read(wavfile,fs=[]):
    # read a wavfile and normalize it
    # if fs is given the signal is resampled to the given sampling frequency
    speech, fs_signal = librosa.load(wavfile, sr=None)
    if not fs:
        fs=fs_signal

    if speech.dtype != 'float32':
        if speech.dtype == 'int16':
            nb_bits = 16 # -> 16-bit wav files
        elif speech.dtype == 'int32':
            nb_bits = 32 # -> 32-bit wav files
        max_nb_bit = float(2 ** (nb_bits - 1))
        speech = speech / (max_nb_bit + 1.0) # scale the signal to [-1.0,1.0]

    if fs_signal != fs :
        signalr = sp_sig.resample_poly(speech, fs, fs_signal)
    else:
        signalr = speech

    return signalr, fs

def get_all_files_pathlib(directory_path):
    # Convert the string path to a Path object
    base_dir = Path(directory_path)
    
    # .rglob('*') matches all files and folders recursively
    # 'is_file()' ensures we only get files, skipping folder names
    file_names = [file.name for file in base_dir.rglob('*') if file.is_file()]
    
    return file_names

