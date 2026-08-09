import numpy as np
import matplotlib.pyplot as plt

from params import *
from stim import *
from params import MODEL

# Parameters
dBSPL = 40 # intensity of stimulus in dB SPL
frequency = 1000 # frequency of stimulus in Hz
dur = .1 # duration of stimulus in seconds

# Stimulus
stim = create_stim(freq=frequency, 
                db_spl=dBSPL, 
                stim_type="pure_tone",
                use_suppressor=False, # only used when stim_type = "dpoae"
                padding=256, # required padding for CoNNear
                duration=dur, 
                ramp_ms=0, # for gradual ramp to maximum amplitude
                sr=20000, # required for CoNNear (don't change for CoNNear)
                plotfig=False # visualize stimulus flag
                )

# Generate basilar membrane displacement
bm_displacement = cochlea_NH.predict(stim)  # "Healthy" CoNNear
bm_displacement = bm_displacement.squeeze() 

# Ensure correct orientation: (channels, time) for imshow
if bm_displacement.shape[0] != len(CF):
    bm_displacement = bm_displacement.T

# Determine symmetric color limits for zero-centered displacement
d_max = float(np.max(np.abs(bm_displacement)))

# Visualize basilar membrane displacement
plt.figure(figsize=(10, 6))
plt.imshow(
    bm_displacement, 
    aspect='auto', 
    origin='lower', 
    extent=[0, dur, 0, 201], 
    cmap='seismic',
    vmin=-d_max,
    vmax=d_max
)
plt.colorbar(label='BM Displacement (µm)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency Channel')
plt.title('Basilar Membrane Displacement (Normal Hearing)', fontweight='bold')
plt.tight_layout()
plt.show()
