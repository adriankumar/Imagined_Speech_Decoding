from ..constants import DEFAULT_EXCLUDE
import mne, numpy as np

#------------
#Helpers for resolving electrode channel names and positions against MNE
#------------
#channel names are canse-sensitive, lowercase only for comparison
def lowercase_key(name):
    return name.lower()

#the montage's channel names, lowercase key -> canonical name
def lookup_montage(montage):
    names = mne.channels.make_standard_montage(montage).ch_names
    return {lowercase_key(n): n for n in names}

#row indices of resolved chns within the original channel order, 
#so a window can be sliced to montage order; note used with model because 
#want the n_schns to be arbitrary
def build_chn_name_order(original_chns, resolved_chns):
    lower_to_idx = {name.lower(): i for i, name in enumerate(original_chns)}
    return [lower_to_idx[name.lower()] for name in resolved_chns]

#classify source channels against the montage; aux names and any the montage lacks are auto-excluded
def classify_chns(montage, original_chns, printout=False):
    #montage names keyed by lowercase for casing-insensitive match, value keeps canonical casing
    montage_lookup = lookup_montage(montage=montage)

    resolved, excluded = [], []
    for name in original_chns:  #source order retaied
        key = name.lower()
        #drop known non-scalp aux channels and any name the montage doesn't carry
        if key in DEFAULT_EXCLUDE or key not in montage_lookup:
            excluded.append(name)
        else:
            resolved.append(montage_lookup[key])  #montage casing so position lookups match later
    if printout:
        print(f"[{montage}] resolved {len(resolved)}/{len(original_chns)}, excluded {excluded}")

    return resolved, excluded #list

def _lookup_positions(chn_pos, names):
    return np.array([chn_pos[name] for name in names])  #(len(names), 3)

#get 3d positions of electrode channels from mne's montage
#positions on the idealised head
def get_3d_pos(montage, chns_list=None, num_chns=None):
    #idealised head positions from mne's standard montage: {name: (x, y, z)}
    chn_pos = mne.channels.make_standard_montage(montage).get_positions()["ch_pos"]

    if chns_list is not None:
        pos = _lookup_positions(chn_pos, chns_list) #(len(chns_list), 3)
        if np.isnan(pos).any():
            raise ValueError("montage returned NaN positions for resolved channels")
        return pos

    if num_chns is not None:
        names = list(chn_pos.keys())
        pos = _lookup_positions(chn_pos, names) #(m, 3)
        pos = pos[~np.isnan(pos).any(axis=1)] #drop undefined template points

        m = len(pos)
        if num_chns > m:
            raise ValueError(f"requested {num_chns} exceeds {m} available montage points")
        if num_chns == m:
            return pos #(m, 3)

        idx = np.linspace(0, m - 1, num_chns).round().astype(int) #even stride over available points
        return pos[idx] #(num_chns, 3)

    raise ValueError("must provide either chns_list or num_chns")

def get_2d_pos(theta, phi): #azimuthual projection
    x = theta * np.cos(phi)
    y = theta * np.sin(phi)
    return np.column_stack([x, y])

