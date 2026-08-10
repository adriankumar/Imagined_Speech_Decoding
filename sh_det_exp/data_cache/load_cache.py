import os
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

#load one cached example back into {chns, sfreq, windows}
def load_example(npz_path):
    import numpy as np

    with np.load(npz_path) as f:
        windows = list(f["windows"])  #list of (C, W)
        chns = f["chns"].tolist()
        sfreq = float(f["sfreq"])

    return {"chns": chns, "sfreq": sfreq, "windows": windows}

#load the whole cache as {'1': {...}, '2': {...}, '3': {...}}
def load_caches(out_dir=CACHE_DIR, keys=("1", "2", "3")):
    return {k: load_example(os.path.join(out_dir, f"{k}.npz")) for k in keys}

#example use case
# if __name__ == "__main__":
#     cache = load_cache() #dictionary wiht keys 1 2 3
#     for k, e in cache.items(): #each k has another dict of values with the following keys
#         w = e["windows"][0]
#         print(f"{k}: sfreq={e['sfreq']} chns={len(e['chns'])} n_windows={len(e['windows'])} window_shape={w.shape}")

#===data cache details====
# 1: sfreq=500.0 chns=125 n_windows=6 window_shape=(125, 250)
# 2: sfreq=256.0 chns=128 n_windows=9 window_shape=(128, 128)
# 3: sfreq=250.0 chns=22 n_windows=8 window_shape=(22, 125)