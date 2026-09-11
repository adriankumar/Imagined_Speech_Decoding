SEED = 24573471
EPSILON = 1e-8 #avoid divisions by zero

#physiological or hardware auxiliary names, not scalp eeg, matched case-insensitively
DEFAULT_EXCLUDE = {"veo", "heo", "eog", "ecg", "ekg", "emg", "trigger", "sti", "status"}

def _get_built_in_channels():
    import mne 
    return mne.channels.get_builtin_montages() #list of strings

MNE_MONTAGES = _get_built_in_channels()

#canonical eeg bands in hz, low inclusive and high exclusive except the last
FREQ_BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
              "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}

#band features share a computation path and each needs sfreq to build its bin mask
BAND_FEATURES = ("delta", "theta", "alpha", "beta", "gamma")

#everything derived from the periodogram, entropy needs no frequency labels but shares the fft
SPECTRAL_FEATURES = BAND_FEATURES + ("entropy",)

#order of F, time-domain first then spectral
FEATURE_NAMES = ("mean", "median", "iqr", "mobility", "complexity") + SPECTRAL_FEATURES

SOLVER_TYPES = ["B=I", "B=diag"]

WINDOW_SIZE = 0.5
MONTAGE = "standard_1005" #densest reference dictionary; 10-20 and 10-10 names are position-identical subsets

#iqr and median dominate in scale making learning harder, 
#they may need to be scaled down rather than mobility and complexity
FEATURE_TOGGLES = {"mean": False, "median": False, "iqr": False, 
                   "mobility": True, "complexity": True, 
                   #dimless spectral-based
                   "entropy": False, "delta": False, "theta": False, 
                   "alpha": False, "beta": False, "gamma": False} 

L = 9 
NUM_SCHNS = (L+1)**2 #number of simulated channels
SFREQ_MAX = 1000 #for model to normalise against arbitrary eeg inputs
IMG_DIMS = (64, 64) #img size for eeg inputs and outputs
MARGIN = 0.95 #for img interpol

DROPOUT = 0.2

DEFAULT_CMAP = "inferno"
DELTA_CMAP = "RdBu"