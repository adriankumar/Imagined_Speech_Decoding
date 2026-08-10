SEED = 24573471
EPSILON = 1e-8 #avoid divisions by zero

#physiological or hardware auxiliary names, not scalp eeg, matched case-insensitively
DEFAULT_EXCLUDE = {"veo", "heo", "eog", "ecg", "ekg", "emg", "trigger", "sti", "status"}

def _get_built_in_channels():
    import mne 
    return mne.channels.get_builtin_montages() #list of strings

MNE_MONTAGES = _get_built_in_channels()

#current existing feature name
FEATURE_NAMES = ["mean", "median", "iqr", "mobility", "complexity"]

SOLVER_TYPES = ["B=I", "B=diag"]

#DEFAULT-ARGS for models, data-caching etc
WINDOW_SIZE = 0.5
MONTAGE = "standard_1005" #densest reference dictionary; 10-20 and 10-10 names are position-identical subsets

#iqr and median dominate in scale making learning harder, 
#they may need to be scaled down rather than mobility and complexity
FEATURE_TOGGLES = {"mean": False, "median": False, "iqr": False, 
                   "mobility": True, "complexity": True} 

L = 9 
NUM_SCHNS = (L+1)**2 #number of simulated channels
SFREQ_MAX = 1000
IMG_DIMS = (64, 64) #img size for eeg inputs and outputs
MARGIN = 0.95 #for img interpol

DROPOUT = 0.2

DEFAULT_CMAP = "inferno"
DELTA_CMAP = "RdBu_r"