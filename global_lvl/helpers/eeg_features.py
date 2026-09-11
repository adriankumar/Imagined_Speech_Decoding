import numpy as np
from ..constants import FREQ_BANDS

#------------
#Features to compute on raw voltage window or individual time point
#------------
#get timepoints from seconds
def window_size_from_seconds(window_seconds, sfreq):
    w = round(window_seconds * sfreq)

    #complexity needs a second derivative and the periodogram needs bins
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

#x is nchns x timepoints
def _window_variance(x):
    if x.shape[-1] == 0:
        return np.zeros(x.shape[:-1])
    return np.var(x, axis=-1) #variance across timepoints axis

#hjorth mobility, time-domain proxy for mean frequency, zero where the signal does not vary
def hjorth_mobility(x):
    var_x = _window_variance(x)
    var_dx = _window_variance(np.diff(x, axis=-1)) #np.diff computes n-th discrete difference
    return np.sqrt(np.divide(var_dx, var_x, out=np.zeros_like(var_x), where=var_x > 0))

#hjorth complexity, time-domain proxy for bandwidth, zero where the first derivative does not vary
def hjorth_complexity(x):
    dx = np.diff(x, axis=-1)
    mob_x = hjorth_mobility(x)
    mob_dx = hjorth_mobility(dx)
    return np.divide(mob_dx, mob_x, out=np.zeros_like(mob_x), where=mob_x > 0)


#------------
#Spectral features; all computed from one periodogram per window so the fft is paid once
#------------
#x is nchns x timepoints, returns power nchns x bins and the frequency of each bin
#mean removed and hann tapered so leakage does not smear power across bands
def power_spectrum(x, sfreq):
    n_time = x.shape[-1]
    tapered = (x - x.mean(axis=-1, keepdims=True)) * np.hanning(n_time)

    spectrum = np.fft.rfft(tapered, axis=-1)
    power = (spectrum.real**2 + spectrum.imag**2) #nchns x bins

    return power, np.fft.rfftfreq(n_time, d=1.0/sfreq)

#power inside a band as a share of power across all declared bands
#the ratio is what makes it dimensionless, raw band power carries amplitude units
def relative_band_power(power, freqs, band):
    low, high = FREQ_BANDS[band]
    band_mask = (freqs >= low) & (freqs < high) #bins falling inside this band

    #total is over the declared bands only, not the full spectrum, so the shares sum to one
    declared_mask = (freqs >= FREQ_BANDS["delta"][0]) & (freqs < FREQ_BANDS["gamma"][1])

    in_band = power[..., band_mask].sum(axis=-1)
    total = power[..., declared_mask].sum(axis=-1)

    return np.divide(in_band, total, out=np.zeros_like(total), where=total > 0)

#shannon entropy of the normalised spectrum, how flat the spectrum is
#divided by log(bins) so it lands in 0-1 regardless of window length
def spectral_entropy(power, freqs):
    declared_mask = (freqs >= FREQ_BANDS["delta"][0]) & (freqs < FREQ_BANDS["gamma"][1])
    band_power = power[..., declared_mask]

    total = band_power.sum(axis=-1, keepdims=True)
    p = np.divide(band_power, total, out=np.zeros_like(band_power), where=total > 0)

    #zero bins contribute nothing, log is only taken where p is positive
    entropy = -(p * np.log(p, out=np.zeros_like(p), where=p > 0)).sum(axis=-1)

    return entropy / np.log(band_power.shape[-1])