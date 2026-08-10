import numpy as np 

#------------
#Features to compute on raw voltage window or individual time point
#------------
#get timepoints from seconds
def window_size_from_seconds(window_seconds, sfreq):
    assert 0.1 <= window_seconds <= 0.9, f"window_seconds must be in [0.1, 0.9], got {window_seconds}"
    w = round(window_seconds * sfreq)

    if w < 8: #arbitrary to ensure stable features, actual value would be 3 but still allows poorly chosen statistics
        raise ValueError(f"window {w} samples too small for stable features at sfreq {sfreq}")

    return int(w)

#median and mean will produce negative values on the heatmap img, so
#applying the topographic masks in terms of visualisation is weird because
#0 value becomes centred instead of lowest;
def compute_mean(x):
    return np.mean(x, axis=-1)

def compute_median(x):
    return np.median(x, axis=-1)

def compute_iqr(x):
    q75, q25 = np.percentile(x, [75, 25], axis=-1)
    return q75 - q25

def _window_variance(x):
    if x.shape[-1] == 0:
        return np.zeros(x.shape[:-1])
    return np.var(x, axis=-1)

#hjorth mobility, time-domain proxy for mean frequency, zero where the signal does not vary
def hjorth_mobility(x):
    var_x = _window_variance(x)
    var_dx = _window_variance(np.diff(x, axis=-1))
    return np.sqrt(np.divide(var_dx, var_x, out=np.zeros_like(var_x), where=var_x > 0))

#hjorth complexity, time-domain proxy for bandwidth, zero where the first derivative does not vary
def hjorth_complexity(x):
    dx = np.diff(x, axis=-1)
    mob_x = hjorth_mobility(x)
    mob_dx = hjorth_mobility(dx)
    return np.divide(mob_dx, mob_x, out=np.zeros_like(mob_x), where=mob_x > 0)