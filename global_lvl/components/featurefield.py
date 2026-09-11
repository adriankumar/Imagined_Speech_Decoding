from ..constants import FREQ_BANDS, FEATURE_NAMES, SPECTRAL_FEATURES, BAND_FEATURES
from ..helpers import (compute_mean, compute_median, compute_iqr,
                       hjorth_complexity, hjorth_mobility,
                       power_spectrum, relative_band_power, spectral_entropy)

import numpy as np

class FeatureField:
    def __init__(self, channels_order=None, feature_toggles=None, reference=None, sfreq=None):
        self._feat_fns = {"mean": compute_mean, "median": compute_median, "iqr": compute_iqr,
                          "mobility": hjorth_mobility, "complexity": hjorth_complexity,
                          "entropy": spectral_entropy}

        #band features share one function, the band name selects the mask
        for band in FREQ_BANDS:
            self._feat_fns[band] = relative_band_power

        assert channels_order is not None, "FeatureField requires a channel order list from the electrode-sim"
        self._chns_order = channels_order

        #declared toggles fix F for this field and never mutate after construction
        self._declared = {n: True for n in FEATURE_NAMES}
        if feature_toggles is not None:
            self._declared.update(self._checked(feature_toggles))

        assert len(self.toggled_features) >= 1, "Must have at least one feature toggled"

        #band masks are built from the frequency of each fft bin, so cant exist without sfreq
        #declared at construction
        assert sfreq is None or sfreq > 0, f"sfreq must be positive, got {sfreq}"
        assert sfreq is not None or not self._toggled_bands, f"band features {self._toggled_bands} require sfreq"
        self._sfreq = sfreq

        #re-referencing is applied after the channel slice,
        assert reference in (None, "average"), f"reference must be None or 'average', got {reference}"
        self._reference = reference

    def _checked(self, ft_toggles):
        unknown = set(ft_toggles) - set(FEATURE_NAMES)
        assert not unknown, f"unrecognised feature names: {sorted(unknown)}"
        return ft_toggles

    #overrides into a copy so the declared set survives the call
    def _resolve_toggles(self, ft_toggles):
        if not ft_toggles:
            return self._declared

        toggles = dict(self._declared) #copy
        toggles.update(self._checked(ft_toggles))
        return toggles

    #a band needs at least one fft bin inside it, resolution is sfreq / timepoints
    def _assert_band_resolution(self, n_time, fn_names):
        resolution = self._sfreq / n_time

        for name in fn_names:
            if name in BAND_FEATURES:
                low, high = FREQ_BANDS[name]
                assert resolution <= (high - low), (
                    f"{name} spans {high - low}hz but resolution is {resolution:.2f}hz, "
                    f"window of {n_time} samples at {self._sfreq}hz is too short")

    #spectral features share one periodogram, computed only when one of them is toggled
    def _compute_feature(self, name, window, power, freqs):
        if name in BAND_FEATURES:
            return self._feat_fns[name](power, freqs, name)

        if name in SPECTRAL_FEATURES:
            return self._feat_fns[name](power, freqs)

        return self._feat_fns[name](window)

    #takes window of nchns x timepoints and returns nchns x F, where each F is a vector
    #ft_toggles is for diagnostics only- F will not match num_features when it is used
    def window_to_vec(self, window, ft_toggles=None):
        toggles = self._resolve_toggles(ft_toggles)
        fn_names = [n for n in FEATURE_NAMES if toggles[n]]

        assert len(fn_names) >= 1, "Must have at least one feature toggled"
        assert window.ndim >= 2, f"expected window shape chns x window_size, got: {window.ndim}"

        window = window[..., self._chns_order, :] #keep only resolved chns in resolved order so M and Y line up

        if self._reference == "average":
            window = window - window.mean(axis=-2, keepdims=True)

        #window.shape[-1] >= 3 asserted because complexity needs enough time samples
        #to take a second derivative
        if "complexity" in fn_names:
            assert window.shape[-1] >= 3, f"window needs >= 3 samples for hjorth complexity, got {window.shape[-1]}"

        #diagnostic toggles can turn a band on for a field built without sfreq
        spectral = [n for n in fn_names if n in SPECTRAL_FEATURES]
        assert not spectral or self._sfreq is not None, f"spectral features {spectral} require sfreq"

        power, freqs = (None, None)
        if spectral:
            self._assert_band_resolution(window.shape[-1], fn_names)
            power, freqs = power_spectrum(window, self._sfreq) #once for every spectral feature

        return np.stack([self._compute_feature(n, window, power, freqs) for n in fn_names], axis=-1) #n_chns x F

    #bands declared on this field, checked against sfreq at construction
    @property
    def _toggled_bands(self):
        return [n for n in BAND_FEATURES if self._declared[n]]

    @property
    def toggled_features(self):
        return [n for n in FEATURE_NAMES if self._declared[n]]

    #F produced by this field under its declared toggles
    @property
    def num_features(self):
        return len(self.toggled_features)

    @property
    def declared_toggles(self):
        return dict(self._declared)

    @property
    def reference(self):
        return self._reference

    @property
    def sfreq(self):
        return self._sfreq