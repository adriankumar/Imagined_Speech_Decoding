from ..constants import FEATURE_NAMES
from ..helpers import (compute_mean, compute_median, compute_iqr, 
                       hjorth_complexity, hjorth_mobility)

import numpy as np

class FeatureField:
    def __init__(self, channels_order=None, feature_toggles=None):
        self._feat_fns = {"mean": compute_mean, "median": compute_median, "iqr": compute_iqr,
                          "mobility": hjorth_mobility, "complexity": hjorth_complexity}

        assert channels_order is not None, "FeatureField requires a channel order list from the electrode-sim"
        self._chns_order = channels_order 

        #declared toggles fix F for this field and never mutate after construction
        self._declared = {n: True for n in FEATURE_NAMES}
        if feature_toggles is not None:
            self._declared.update(self._checked(feature_toggles))

        assert len(self.toggled_features) >= 1, "Must have at least one feature toggled"

    def _checked(self, ft_toggles):
        unknown = set(ft_toggles) - set(FEATURE_NAMES)
        assert not unknown, f"unrecognised feature names: {sorted(unknown)}"
        return ft_toggles

    #overrides into a copy so the declared set survives the call
    def _resolve_toggles(self, ft_toggles):
        if not ft_toggles:
            return self._declared

        toggles = dict(self._declared)
        toggles.update(self._checked(ft_toggles))
        return toggles

    #takes window of nchns x timepoints and returns nchns x F, where each F is a vector
    #ft_toggles is for diagnostics only- F will not match num_features when it is used
    def window_to_vec(self, window, ft_toggles=None):
        toggles = self._resolve_toggles(ft_toggles)
        fn_names = [n for n in FEATURE_NAMES if toggles[n]]

        assert len(fn_names) >= 1, "Must have at least one feature toggled"
        assert window.ndim >= 2, f"expected window shape chns x window_size, got: {window.ndim}"

        window = window[..., self._chns_order, :] #keep only resolved chns in resolved order so M and Y line up

        if "complexity" in fn_names:
            assert window.shape[-1] >= 3, f"window needs >= 3 samples for hjorth complexity, got {window.shape[-1]}"

        return np.stack([self._feat_fns[n](window) for n in fn_names], axis=-1) #n_chns x F

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