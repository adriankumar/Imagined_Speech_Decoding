from .constants import DEFAULT_EXCLUDE

#lowercase key used only for comparison so string differences never block a match
def _lowercase_key(name):
    return name.lower()

#------------
#resolving channels
#------------
#excluded chans (provided_list) that are specified but dont exist in the inferred channels are dropped
def remove_nonex_chns(provided_list, true_list):
    if not provided_list:  #none or empty, nothing specified to reconcile
        return []
    
    true_chns = {_lowercase_key(s) for s in true_list}
    exists = [c for c in provided_list if _lowercase_key(c) in true_chns]
    dropped = [c for c in provided_list if c not in exists]

    if dropped:
        print(f"Warning: excluded channels {provided_list} not found in the current source were dropped: {dropped}")
    
    return exists #lowercase dictionary as key of electrode name, with canoninical name as value

#get the channel names used in montage standard to compare against the meta data of the recording
def _get_montage_channels(montage_name):
    import mne
    return mne.channels.make_standard_montage(montage_name).ch_names

def _get_montage_lookup(montage_name):
    return {_lowercase_key(name): name for name in _get_montage_channels(montage_name)}

#get channels classification between specified list, and channels in a monetage standard
#considering that  electrode count != standard
def _get_chns_clssf(specified, montage_lookup):
    matched, unresolved = [], []

    for name in specified:
        key = _lowercase_key(name)
        if key in montage_lookup:
            matched.append(montage_lookup[key])
        else:
            unresolved.append(name) #no montage entry to look up, keep the source name
    
    return matched, unresolved

#classify the working channel set against the montage and return the breakdown, applying no unresolved policy
#default_dropped are auxiliary names matched by DEFAULT_EXCLUDE, dropped silently as known non-scalp channels
#unresolved are scalp-side names with no montage entry, left for the caller to auto-exclude, manual-exclude, or reject
def _classify_channels(inferred_chns, chns_in_montage, exclude_chns=None):
    #auxiliary names are known non-scalp hardware channels, dropped silently and reported in their own bucket
    default_dropped = [name for name in inferred_chns if _lowercase_key(name) in DEFAULT_EXCLUDE]

    #DEFAULT_EXCLUDE merged with any additional channels from exclude_chns, with no duplicates
    drop = DEFAULT_EXCLUDE | {_lowercase_key(name) for name in (exclude_chns or [])} #union() operation
    kept = [name for name in inferred_chns if _lowercase_key(name) not in drop]

    resolved, unresolved = _get_chns_clssf(kept, chns_in_montage)

    return {"resolved": resolved, "unresolved": unresolved, "default_dropped": default_dropped}

def classify_chns_w_montage(inferred_chns, montage, excluded_chns):
        lookup_dict = _get_montage_lookup(montage) #lookup where key is just lowercase for string comparison but value is true name casing
        return _classify_channels(inferred_chns, chns_in_montage=lookup_dict, exclude_chns=excluded_chns)

#raw row indices in the reconciled channel order, precomputed once so window reads need no reorder
def build_pick_order(inferred_chns, resolved_chns):
    lower_to_idx = {_lowercase_key(name): i for i, name in enumerate(inferred_chns)}
    return [lower_to_idx[_lowercase_key(name)] for name in resolved_chns] #essentially another look-up table helper

#------------
#target reference
#------------
#if target reference is a list of channels and has any disrepancies
#with the actual resolved channels in the current recording; fall-back to "average"
#and surface the warning; user can re-change target reference list 
def reconcile_target_ref(target_ref, resolved_chns):
    #If it's a list/tuple
    #validate all channels in list/tuple exist in resolved_chns, warn and fall back if not
    if isinstance(target_ref, (list, tuple)):
        keys = {_lowercase_key(c) for c in resolved_chns}
        
        if not all(_lowercase_key(c) in keys for c in target_ref):
            print("Warning: target reference channels do not fully resolve against the current recording, reverting to average.")
            return "average"
        
        return target_ref

    #If it's a string but not "average", it's an unrecognised value, warn and fall back
    elif target_ref != "average":
        print(f"Warning: unrecognised target reference '{target_ref}', reverting to average.")
        return "average"

    return "average" #otherwise it's already "average" just return it

#------------
#windows
#------------
#window size cannot be 0 or larger than the number of timepoints in current recording
def reconcile_window_size(window_size, timepoints):
    if window_size > timepoints: 
        print(f"Warning: window size {window_size} is too large; reduced to {timepoints} to fit recording length")

    if window_size < 1:
        print(f"Warning: window size cannot be {window_size}; defaulting to 1")
        window_size = 1

    return min(window_size, timepoints)

#read one window from the recording, picks already in the reconciled order, columns are time
def get_window_data(raw, pick_order, start, stop):
    return raw.get_data(picks=pick_order, start=start, stop=stop)

#------------
#electrode position on idealised human head (one step of compression from true recording)
#------------
#get 3d positions of electrode channels from mne's montage
def get_channel_positions(montage, chns_list):
    import mne, numpy as np
    chn_pos = mne.channels.make_standard_montage(montage).get_positions()["ch_pos"] #idealised coordinates
    pos = np.array([chn_pos[name] for name in chns_list]) #row i is chn_list[i]
    
    if np.isnan(pos).any():
        raise ValueError("montage returned NaN positions for resolved channels")
    
    return pos #array of channel positions; ensure compositions remember pos[i] is chn_list[i]
