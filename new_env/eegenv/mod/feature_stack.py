import numpy as np 
from .helpers import (DEFAULT_FEATURE_WEIGHTS, FEATURE_NAMES,
                      validate_toggles, check_feature_scale,
                      compute_median, compute_iqr, hjorth_mobility, hjorth_complexity,
                      )

#optional accumulation features
# _ACCUM_FEATURES = FEATURE_NAMES[:4] #features to accumulate; with window resetting; 

#computes and stores features for current window in recording
class FeatureStack:
    def __init__(self, median=True, iqr=True, mobility=True, complexity=True, 
                 median_lag=False, iqr_lag=False, mobility_lag=False, complexity_lag=False,
                 weights=None):

        #on init
        self.feature_toggles = self._build_feature_dict([median, iqr, mobility, complexity, 
                                                          median_lag, iqr_lag, 
                                                          mobility_lag, complexity_lag])
        
        #copy so the per-instance weights never mutate the shared module default
        self.feature_weights = dict(DEFAULT_FEATURE_WEIGHTS)
        if weights is not None:
            self.change_weights(weights) #validates and merges by name

        self.reset() #clean init

    #toggles for feature computation; lag features can't be computed
    #without their previous part
    def _build_feature_dict(self, feature_toggles):
        f_dict = {}

        for i, f in enumerate(FEATURE_NAMES):
            f_dict[f] = feature_toggles[i] #create dict

        validate_toggles(f_dict, FEATURE_NAMES) #surface any errors

        return f_dict

    #set prev variables to current 
    def _update_lag_cache(self, med, iqr, mob, comp):
        self.prev_median = med 
        self.prev_iqr = iqr 
        self.prev_mobility = mob
        self.prev_complexity = comp
        self.zero_iqr_channels = np.where(iqr == 0)[0] if iqr is not None else []

    #resets previous and accumulative features
    def reset(self):
        self.prev_median = None 
        self.prev_iqr = None 
        self.prev_mobility = None 
        self.prev_complexity = None 
        self.zero_iqr_channels = [] #indices of channels that have zero iqr from the previous window
        self.lag_cache_ready = False  # flag

    #computes current n_electrodes, F
    def compute_features(self, window, feature_toggles=None, advance_lag=True):
        n_electrodes = window.shape[0] #n_chans x window_size

        #dict of active features, or can re-pass for current window diagnostics
        active_features = self.feature_toggles if feature_toggles is None else {**self.feature_toggles, **feature_toggles}

        #default features
        median_t = compute_median(window) if active_features["median"] else None
        iqr_t = compute_iqr(window) if active_features["iqr"] else None
        mobility_t = hjorth_mobility(window) if active_features["mobility"] else None
        complexity_t = hjorth_complexity(window) if active_features["complexity"] else None

        #base values keyed by name, none where the toggle is off, validate_toggles guarantees a lag never outlives its base
        base = {"median": median_t, "iqr": iqr_t, "mobility": mobility_t, "complexity": complexity_t}
        prev = {"median": self.prev_median, "iqr": self.prev_iqr,
                "mobility": self.prev_mobility, "complexity": self.prev_complexity}

        #computing lag if its active; insantiating if the cache is None; and updating if cache is present
        lag = {}
        for name in ("median", "iqr", "mobility", "complexity"):
            lag_name = f"{name}_lag"
            if not active_features[lag_name]:
                lag[lag_name] = None
            elif not self.lag_cache_ready:
                lag[lag_name] = np.zeros(n_electrodes)
            else:
                lag[lag_name] = base[name] - prev[name]

        values = {**base, **lag}

        #select active features in fixed order, weight each by name
        names = [name for name in FEATURE_NAMES if active_features[name]]
        stack = [values[name] * self.feature_weights[name] for name in names]

        #advance the lag cache
        if advance_lag:
            self._update_lag_cache(median_t, iqr_t, mobility_t, complexity_t)
            self.lag_cache_ready = True

        return np.stack(stack, axis=1) #stack along columns (F); shape (n_electrodes, F)
    
    #detached copy carrying the same toggles, weights and lag cache, for a state-free preview off the live cursor
    #the cache is preserved not cleared so the copy reproduces the live stream's current window exactly
    def copy(self):
        clone = FeatureStack()
        clone.feature_toggles = dict(self.feature_toggles)
        clone.feature_weights = dict(self.feature_weights)
        clone.prev_median = self.prev_median
        clone.prev_iqr = self.prev_iqr
        clone.prev_mobility = self.prev_mobility
        clone.prev_complexity = self.prev_complexity
        clone.zero_iqr_channels = self.zero_iqr_channels
        clone.lag_cache_ready = self.lag_cache_ready
        return clone

    #=====
    #changer functions
    #====   
    def change_weights(self, weights):
        check_feature_scale(weights, FEATURE_NAMES)
        self.feature_weights.update(weights) #update weights

    def change_toggles(self, toggles):
        candidate = {**self.feature_toggles, **toggles}
        validate_toggles(candidate, FEATURE_NAMES)
        self.feature_toggles = candidate
        self.reset() #base set may have changed, clear the stale lag cache

    @property
    def flat_iqr_channels(self):
        return self.zero_iqr_channels 
    
    @property 
    def lag_ready(self):
        return self.lag_cache_ready