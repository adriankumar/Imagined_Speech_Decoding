import mne 
import numpy as np 
from .helper_maths import (compute_sphere_params, compute_spherical_angles,
                           azimuthal_2d, build_interpolation_operator, build_sh_basis)

MNE_MONTAGES = mne.channels.get_builtin_montages() #list of strings
#channel names that are physiological or hardware auxilllary; not scalp eeg, matched case-insensitively
DEFAULT_EXCLUDE = {"veo", "heo", "eog", "ecg", "ekg", "emg", "trigger", "sti", "status"} #set

#lowercase key used only for comparison so string differences never block a match
def lowercase_key(name):
    return name.lower()

#returns only the header of source file that other functions can extract from
def bind_source(source, preload=False, verbose=False):
    return mne.io.read_raw_edf(source, preload=preload, verbose=verbose)

def get_channel_names(raw):
    return raw.ch_names 

def get_sampling_rate(raw):
    return raw.info['sfreq']

def get_timepoints(raw):
    return raw.n_times

#get the channel names used in montage standard
def get_montage_channels(montage_name):
    return mne.channels.make_standard_montage(montage_name).ch_names

#lowercase dictionary as key of electrode name, with canoninical name as value
def get_montage_lookup(montage_name):
    return {lowercase_key(name): name for name in get_montage_channels(montage_name)}

#get channels classification between specified list, and channels in a monetage standard
#considering that  electrode count != standard
def get_chns_clssf(specified, montage_lookup):
    matched, unresolved = [], []

    for name in specified:
        key = lowercase_key(name)
        if key in montage_lookup:
            matched.append(montage_lookup[key])
        else:
            unresolved.append(name) #no montage entry to look up, keep the source name
    
    return matched, unresolved

#get 3d positions of electrode channels from mne's montage
def get_channel_positions(montage, chns_list):
    chn_pos = mne.channels.make_standard_montage(montage).get_positions()["ch_pos"] #idealised coordinates
    pos = np.array([chn_pos[name] for name in chns_list]) #row i is chn_list[i]
    
    if np.isnan(pos).any():
        raise ValueError("montage returned NaN positions for resolved channels")
    
    return pos #array of channel positions; ensure compositions remember pos[i] is chn_list[i]

def get_interpolation_operator(pos_3d, img_res, margin=0.9):
    
    sphere_centre, sphere_radius = compute_sphere_params(pos_3d)
    theta, phi = compute_spherical_angles(pos_3d, sphere_centre)

    pos_2d = azimuthal_2d(theta, phi)
    M = build_interpolation_operator(pos_2d, img_res, margin)
    
    return {'interpol_operator': M,
            'theta': theta, 'phi': phi, 
            'pos_2d': pos_2d, 'pos_3d': pos_3d, 
            'centre': sphere_centre, 'radius': sphere_radius}


def get_sh_basis(theta, phi, L):
    Y = build_sh_basis(theta, phi, L)
    return {'SH_basis': Y}


#reduce the window to the recording length so a shorter source never invalidates it
def fit_window_size(segment_length, timepoints):
    return min(segment_length, timepoints)

#fall back a channel-list reference to average when it no longer fully resolves against the channel set
def fit_target_ref(target_ref, channel_names):
    if target_ref == "average":
        return "average"
    keys = {lowercase_key(c) for c in channel_names}
    if isinstance(target_ref, (list, tuple)) and all(lowercase_key(c) in keys for c in target_ref):
        return target_ref
    return "average"

#drop excluded names that do not exist in the source, returns the surviving list
def fit_excludes(excluded, source_chns):
    keys = {lowercase_key(s) for s in source_chns}
    return [e for e in excluded if lowercase_key(e) in keys]

#raw row indices in the reconciled channel order, precomputed once so window reads need no reorder
def build_pick_order(raw_ch_names, chn_list):
    lower_to_idx = {lowercase_key(name): i for i, name in enumerate(raw_ch_names)}
    return [lower_to_idx[lowercase_key(name)] for name in chn_list]

#read only one window from the source, picks already in the reconciled order
def get_window_data(raw, pick_order, start, stop):
    return raw.get_data(picks=pick_order, start=start, stop=stop)

#cast a numpy array to the requested return type and dtype, torch imported only when asked
def to_tensor(array, tensor_type, dtype):
    array = np.ascontiguousarray(array, dtype=dtype)
    if tensor_type == "numpy":
        return array
    
    import torch #lazy import 

    return torch.from_numpy(array)