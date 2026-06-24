from .helper_functions import (MNE_MONTAGES, DEFAULT_EXCLUDE, lowercase_key,
                               get_chns_clssf)
from .helper_maths import l_max
import numpy as np

#configurable bounds for the changeable arguments, used by the changer validation
IMG_MIN, IMG_MAX = 8, 256 #pixel bounds per axis, keeps the M build and transfer cheap
MARGIN_MIN, MARGIN_MAX = 0.1, 1.0 #fraction of the half-grid the furthest electrode may reach

#image resolution must be a height-width pair within the pixel bounds
def check_img_res(img_res):
    if not (isinstance(img_res, (list, tuple)) and len(img_res) == 2):
        raise ValueError(f"img_res must be a (H, W) pair; got {img_res}")
    H, W = img_res
    if not (IMG_MIN <= H <= IMG_MAX and IMG_MIN <= W <= IMG_MAX):
        raise ValueError(f"img_res {img_res} out of bounds, each axis must be within [{IMG_MIN}, {IMG_MAX}]")

#projection margin must sit within the disk-fill bounds
def check_margin(margin):
    if not (MARGIN_MIN <= margin <= MARGIN_MAX):
        raise ValueError(f"margin {margin} out of bounds, must be within [{MARGIN_MIN}, {MARGIN_MAX}]")

#confirm every excluded channel exists in the source, nothing to resolve if the list is empty
def check_excluded_chns_exist(excluded, specified):
    if not excluded:
        return #empty or none, nothing to resolve

    if not isinstance(excluded, list):
        raise ValueError(f"excluded chns must be in a list format: ['name',...]")

    keys = {lowercase_key(s) for s in specified} #source names as a lowercase set
    missing = [e for e in excluded if lowercase_key(e) not in keys]
    if missing:
        raise ValueError(f"some excluded electrodes do not exist in the current source; please remove from list: {missing}")
    
#check the target reference is a form the re-reference step can apply, average or a subset of resolved channels
def check_re_ref_validity(target_ref, channel_names):
    if target_ref == "average":
        return
    keys = {lowercase_key(c) for c in channel_names}
    if isinstance(target_ref, (list, tuple)) and all(lowercase_key(c) in keys for c in target_ref):
        return
    raise ValueError(f"target_ref {target_ref} must be 'average' or a list of channels within the resolved set")

def check_window_size(segment_length, timepoints):
    floor = 1 #change here for lower bound on window size
    if segment_length < floor:
        raise ValueError(f"segment_length {segment_length} is below the window floor {floor}")
    if segment_length > timepoints:
        raise ValueError(f"segment_length {segment_length} exceeds the recording length {timepoints} samples")

#the harmonic basis cannot carry more modes than the electrode count can resolve
def check_harmonic_capacity(L, n_channels):
    if not isinstance(L, int) or L < 0:
        raise ValueError(f"L must be a non-negative integer; got {L}")
    if (L + 1) ** 2 > n_channels:
        raise ValueError(f"L={L} needs {(L + 1) ** 2} modes but only {n_channels} channels resolved, lower the value of L")

#advisory helper for picking a fixed model-level L from the electrode counts expected at deployment
#electrode_counts is a single expected count or a list of them e.g. [6, 64, 128]
#the largest L every count resolves is bound by the smallest, since more electrodes only ever resolve finer modes
#target_L checks an already-chosen model L against the same counts, none reports against the recommendation
#returns the recommendation, the L the report is against, the modes it needs, and a per-count resolvability flag
def recommend_harmonic_order(electrode_counts, target_L=None):
    counts = [electrode_counts] if isinstance(electrode_counts, int) else list(electrode_counts)
    if not counts or not all(isinstance(n, int) and n > 0 for n in counts):
        raise ValueError(f"electrode_counts must be a positive int or a list of positive ints; got {electrode_counts}")
    if target_L is not None and (not isinstance(target_L, int) or target_L < 0):
        raise ValueError(f"target_L must be a non-negative integer or None; got {target_L}")

    recommended_L = l_max(min(counts))  #the smallest count is the binding constraint on a joint L
    L_report = recommended_L if target_L is None else target_L
    modes = (L_report + 1) ** 2  #channels each count must carry to resolve the reported L

    per_count = [{"n_channels": n, "ceiling_L": l_max(n), "resolves": n >= modes} for n in sorted(set(counts))]
    unresolved = [r["n_channels"] for r in per_count if not r["resolves"]]

    return {"recommended_L": recommended_L, "L_report": L_report, "modes_required": modes,
            "per_count": per_count, "unresolved": unresolved}

#window must sit fully inside the recording, validated from scalars before any data read
def check_window_range(start, length, timepoints):
    if start < 0:
        raise ValueError(f"window start {start} must be non-negative")
    if length < 1:
        raise ValueError(f"window length {length} must be at least 1 sample")
    if start + length > timepoints:
        raise ValueError(f"window [{start}, {start + length}) exceeds the recording length {timepoints} samples")

#return type must be one the env knows how to cast to
def check_tensor_type(tensor_type):
    if tensor_type not in ("numpy", "torch"):
        raise ValueError(f"tensor_type {tensor_type} must be 'numpy' or 'torch'")

#a flags override must use known feature names and boolean values
def check_feature_flags(flags, valid_names):
    if not isinstance(flags, dict):
        raise ValueError(f"flags must be a dict of feature name to bool; got {type(flags)}")
    unknown = [k for k in flags if k not in valid_names]
    if unknown:
        raise ValueError(f"unknown feature flags {unknown}; valid names are {list(valid_names)}")
    if not all(isinstance(v, bool) for v in flags.values()):
        raise ValueError("flags values must all be bool")

#feature scale or weight must map known feature names to positive finite multipliers
def check_feature_scale(scale, feature_names):
    for name, value in scale.items():
        if name not in feature_names:
            raise ValueError(f"Unknown feature in scale: {name}; valid names: {feature_names}")
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Feature scale for {name} must be a positive finite number; got {value}")

def validate_montage(montage_input):
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
    if lowercase_key(m_name) not in map(str.lower, MNE_MONTAGES): #map(function, iterable)
        raise ValueError(f"montage {m_name} not recognised")

#raised when scalp-side channels cannot resolve against the montage, carries the names so the caller can apply its own policy
class UnresolvedChannelsError(ValueError):
    def __init__(self, unresolved, montage):
        self.unresolved = unresolved
        self.montage = montage
        super().__init__(f"channels {unresolved} are not recognised in {montage}; "
                         f"auto-exclude them, add them to exclude_chns, or change the montage")

#classify the working channel set against the montage and return the breakdown, applying no unresolved policy
#a specified set missing from the source still raises, that is a hard mismatch the caller cannot recover from
#default_dropped are auxiliary names matched by DEFAULT_EXCLUDE, dropped silently as known non-scalp channels
#unresolved are scalp-side names with no montage entry, left for the caller to auto-exclude, exclude, or reject
def classify_channels(specified_chns, inferred_chns, chns_in_montage, exclude_chns=None):
    if specified_chns is not None:
        if not isinstance(specified_chns, list):
            raise ValueError(f"specified channel names should be a list of strings; got {specified_chns} | type: {type(specified_chns)}")
        keys = {lowercase_key(name) for name in inferred_chns}
        missing = [name for name in specified_chns if lowercase_key(name) not in keys]
        if missing:
            raise ValueError(f"Specified channel names: {specified_chns} are missing {missing} from what was inferred from the source; set specified to None or include all the channel names")

    working_chns = inferred_chns if specified_chns is None else specified_chns

    #auxiliary names are known non-scalp hardware channels, dropped silently and reported in their own bucket
    default_dropped = [name for name in working_chns if lowercase_key(name) in DEFAULT_EXCLUDE]

    drop = DEFAULT_EXCLUDE | {lowercase_key(name) for name in (exclude_chns or [])}
    kept = [name for name in working_chns if lowercase_key(name) not in drop]

    resolved, unresolved = get_chns_clssf(kept, chns_in_montage)

    return {"resolved": resolved, "unresolved": unresolved, "default_dropped": default_dropped}