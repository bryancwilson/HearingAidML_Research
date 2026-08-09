import numpy as np

from keras.models import Model, model_from_json
from keras.utils import CustomObjectScope
from keras.initializers import GlorotUniform
from helper import load_connear_model

MODEL = "COCHLEA"
model_dir = "connear_/"

# Define model specific variables
down_rate = 2
fs = 20e3 # sample rate of the CoNNear Model
fs_tl = 100e3 # sample rate of the transmission line
factor_fs = int(fs_tl / fs) # factor for downsampling
p0 = 2e-5 # reference pressure
right_context = 256 # right context for windowing
left_context = 256 # left context for windowing

# load CFs 
CF = np.loadtxt('cf.txt') # characteristic frequencies
channels = CF.size # number of channels
audiogram_freqs = np.array([250, 500, 1000, 2000, 4000, 8000]) # standard audiogram frequencies
levels_dbSPL = np.arange(0., 91., 10.) # levels from 0 to 90dB SPL
# =================================== CoNNear Cochlea Model Parameters ===================================
if MODEL == "COCHLEA":
    # Load the 201-channel CoNNear model for normal hearing
    cochlea_NH = load_connear_model(model_dir,json_name="/Gmodel.json",weights_name="/Gmodel.h5",name="cochlea_model", crop=False)
    cochlea_NH.summary()
    # Load the 201-channel CoNNear model for hearing impaired
    cochlea_HI = load_connear_model(model_dir,json_name="/Gmodel.json",weights_name="/Gmodel.h5",name="cochlea_model", crop=False)
    cochlea_HI.summary()
# =================================== CoNNear IHC Model Parameters ===================================
elif MODEL == "IHC":
    # Define the IHC model hyperparameters
    context_left = 256
    context_right = 256
    Nenc = 3 # number of layers in the encoder - check for the input size
    # Load the 201-channel IHC model to simulate for all CFs
    N_cf = 201

    cochlea_NH = load_connear_model(model_dir,json_name="/Gmodel.json",weights_name="/Gmodel.h5",name="cochlea_model", crop=False)
    cochlea_NH.summary()
    cochlea_HI = load_connear_model(model_dir,json_name="/Gmodel.json",weights_name="/Gmodel.h5",name="cochlea_model", crop=False)
    ihc_NH = load_connear_model(model_dir,json_name="/ihc.json",weights_name="/ihc.h5",name="ihc_model")
    ihc_NH.summary()
    ihc_HI = load_connear_model(model_dir,json_name="/ihc.json",weights_name="/ihc.h5",name="ihc_model")
# =================================== CoNNear IHC Model Parameters ===================================
elif MODEL == "AN":
    # Define the IHC model hyperparameters
    context_left = 256
    context_right = 256
    Nenc = 3 # number of layers in the encoder - check for the input size
    # Load the 201-channel IHC model to simulate for all CFs
    N_cf = 201

    cochlea_NH = load_connear_model(model_dir,json_name="/Gmodel.json",weights_name="/Gmodel.h5",name="cochlea_model", crop=False)
    cochlea_NH.summary()
    cochlea_HI = load_connear_model(model_dir,json_name="/Gmodel.json",weights_name="/Gmodel.h5",name="cochlea_model", crop=False)
    ihc_NH = load_connear_model(model_dir,json_name="/ihc.json",weights_name="/ihc.h5",name="ihc_model")
    ihc_NH.summary()
    ihc_HI = load_connear_model(model_dir,json_name="/ihc.json",weights_name="/ihc.h5",name="ihc_model")
    anfh = load_connear_model(model_dir,json_name="/anfh_1cf.json",weights_name="/anfh.h5",name="anfh_model")
    anfh.summary()
    anfm = load_connear_model(model_dir,json_name="/anfm_1cf.json",weights_name="/anfm.h5",name="anfm_model")
    anfm.summary()
    anfl = load_connear_model(model_dir,json_name="/anfl_1cf.json",weights_name="/anfl.h5",name="anfl_model")
    anfl.summary()