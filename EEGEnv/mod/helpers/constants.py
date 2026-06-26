#----------------
#Channels
#----------------
def _get_built_in_channels():
    import mne 
    return mne.channels.get_builtin_montages() #list of strings

MNE_MONTAGES = _get_built_in_channels()

#channel names that are physiological or hardware auxilllary; not scalp eeg, matched case-insensitively
DEFAULT_EXCLUDE = {"veo", "heo", "eog", "ecg", "ekg", "emg", "trigger", "sti", "status"} #set

#----------------
#Image Stack
#----------------
IMG_MIN, IMG_MAX = 8, 256 #pixel bounds per axis, keeps the M build and transfer cheap
MARGIN_MIN, MARGIN_MAX = 0.1, 1.0 #fraction of the half-grid the furthest electrode may reach

#----------------
#Features
#----------------
FEATURE_NAMES = [
    "median", "iqr", "mobility", "complexity", #standard features

    #lag features of above features; basically previous window to current window difference
    "median_lag", "iqr_lag", "mobility_lag", "complexity_lag", 
    ]

DEFAULT_FEATURE_WEIGHTS = {name: 1.0 for name in FEATURE_NAMES}