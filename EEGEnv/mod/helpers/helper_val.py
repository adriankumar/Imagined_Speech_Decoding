from .constants import (MNE_MONTAGES, IMG_MIN, IMG_MAX, MARGIN_MIN, MARGIN_MAX)
from .helper_functions import (_lowercase_key,
                               )

#----------------
#Spherical Harmonics
#----------------
def check_harmonic_capacity(L, n_chns):
    if not isinstance(L, int) or L < 0:
        raise ValueError(f"L must be a non-negative integer; got {L}")
    
    if (L + 1) ** 2 > n_chns:
        raise ValueError(f"L={L} with {(L+1)**2} bases/harmonic modes cannot reliably represent reconstruction with {n_chns} channels; {n_chns} must be >= {(L+1)**2};",
                         "Lower the value of L, or instantiate a new EEGEnv with lower L to make up for electrode ranges")

#----------------
#Feature Stack
#----------------
#check if any lag feature is enabled without its base, and if all features are toggled to false
def validate_toggles(feature_toggles, feature_names):
        unknown = [k for k in feature_toggles if k not in feature_names]
        if unknown:
            raise ValueError(f"unknown feature toggles {unknown}; valid names are {feature_names}")
      
        if not all(isinstance(v, bool) for v in feature_toggles.values()):
            raise ValueError("toggle values must all be bool")
       
        if not any(feature_toggles.values()):
            raise ValueError("At least one feature must be enabled.")

        #each lag pairs with the base at the same position in the first half, a lag needs its base
        for base, lag in zip(feature_names[:4], feature_names[4:]):
            if feature_toggles[lag] and not feature_toggles[base]:
                raise ValueError(f"Lag feature '{lag}' is enabled but its base feature '{base}' is disabled.")
#----------------
#Image Stack
#----------------
#image resolution must be a height-width pair within the pixel bounds
def check_img_res(img_res):
    if not (isinstance(img_res, (list, tuple)) and len(img_res) == 2):
        raise ValueError(f"img_res must be a (H, W) pair; got {img_res}")
    
    H, W = img_res
    
    if not (IMG_MIN <= H <= IMG_MAX and IMG_MIN <= W <= IMG_MAX):
        raise ValueError(f"img_res {img_res} out of bounds, each axis must be within [{IMG_MIN}, {IMG_MAX}]")
    
def check_margin(margin):
    if not (MARGIN_MIN <= margin <= MARGIN_MAX):
        raise ValueError(f"margin {margin} out of bounds, must be within [{MARGIN_MIN}, {MARGIN_MAX}]")

#feature scale or weight must map known feature names to positive finite multipliers
def check_feature_scale(scale, feature_names):
    import numpy as np

    for name, value in scale.items():
        if name not in feature_names:
            raise ValueError(f"Unknown feature in scale: {name}; valid names: {feature_names}")
        
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Feature scale for {name} must be a positive finite number; got {value}")


#----------------
#EEG ENV
#----------------
def _validate_montage(montage_input):
    #check its not none
    if montage_input is None:
        raise ValueError(f"montage cannot be None")
    
    #if its a listed input i.e ['montage'] check that only one is specified
    if isinstance(montage_input, list):
        if len(montage_input) > 1:
            print(f"Warning: {len(montage_input)} args were given; only using {montage_input[0]} from input")
        
        m_name = montage_input[0] #get montage name regardless
    
    #otherwise if it is a string
    elif isinstance(montage_input, str):
        m_name = montage_input
    
    else:
        raise ValueError(f"Unrecognised montage input; got {type(montage_input)}, must be str or ['str']")

    #check specified montage exists
    if _lowercase_key(m_name) not in map(str.lower, MNE_MONTAGES): #map(function, iterable)
        raise ValueError(f"montage {m_name} not recognised")

def validate_explicit_args(src, montage):
    if src is None or not src.endswith(".edf"):
        raise ValueError(f"Source must be specified or be an .edf file; got: {src}")
    
    _validate_montage(montage) 

def is_string(string):
    if string is None or not isinstance(string, str):
        raise ValueError(f"Input must be specified as a string; got {string} | type: {type(string)}")

def validate_sampling_rate(specified, inferred):
    if specified is not None and specified != inferred:
        raise ValueError(f"Specified sampling rate: {specified} does not match inferred sampling rate: {inferred}; leave as None or correct argument")

#window must sit fully inside the recording, checked from scalars before any data read
def check_window_range(start, length, timepoints):
    if start < 0:
        raise ValueError(f"window start {start} must be non-negative")
    
    if start + length > timepoints:
        raise ValueError(f"window [{start}, {start + length}) exceeds the recording length {timepoints} samples")