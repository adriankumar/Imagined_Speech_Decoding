import numpy as np
from .helper_maths import (feature_mean, feature_median, feature_iqr,
                           hjorth_complexity, hjorth_mobility, ema_update)

from .helper_val_functions import check_feature_scale

#fixed feature order, also the set of valid names for a feature_toggle, scale or weight override
#ema suffix is the smoothed level, ema_res suffix is the current minus smoothed residual
FEATURE_NAMES = (
    "raw", "median", "iqr", "mobility", "complexity",
    "raw_lag", "median_lag", "iqr_lag",
    "raw_ema", "raw_ema_res",
    "median_ema", "median_ema_res",
    "iqr_ema", "iqr_ema_res",
    "mobility_ema", "mobility_ema_res",
    "complexity_ema", "complexity_ema_res",
)

#base features that support ema accumulation, used to initialise the accum dicts and ema state
_ACCUM_BASES = ("raw", "median", "iqr", "mobility", "complexity")

#per-window temporal feature stack, lag-1 cache persists until reset, scale times weight balances the output
#accum enables ema accumulation per base feature, alpha sets the per-feature ema decay rate
class FeatureStack:
    #feature_toggle fix which features land in the output, scale carries the calibration equalisation, weight carries manual emphasis
    #accum enables ema accumulation per base feature, alpha sets the ema decay rate (higher alpha = faster decay = shorter memory)
    def __init__(self, raw=True, median=True, iqr=True, mobility=True, complexity=True,
                 raw_lag=True, median_lag=True, iqr_lag=True, scale=None, weight=None,
                 accum=None, alpha=None):

        self.feature_toggle = {"raw": raw, "median": median, "iqr": iqr, "mobility": mobility,
                               "complexity": complexity,
                               "raw_lag": raw_lag, "median_lag": median_lag, "iqr_lag": iqr_lag,
                               "raw_ema": False, "raw_ema_res": False,
                               "median_ema": False, "median_ema_res": False,
                               "iqr_ema": False, "iqr_ema_res": False,
                               "mobility_ema": False, "mobility_ema_res": False,
                               "complexity_ema": False, "complexity_ema_res": False}

        self.scale = {name: 1.0 for name in FEATURE_NAMES}   #calibration equalisation, unit by default
        self.weight = {name: 1.0 for name in FEATURE_NAMES}  #manual emphasis on top, unit by default
        self.accum = {name: False for name in _ACCUM_BASES}   #ema accumulation off by default per base feature
        self.alpha = {name: 0.1 for name in _ACCUM_BASES}     #default decay rate, effective memory ~10 windows

        if scale is not None:
            self.set_scale(scale)
        if weight is not None:
            self.set_weight(weight)
        if accum is not None:
            self.set_accum(accum)
        if alpha is not None:
            self.set_alpha(alpha)
        self.reset()

    #clear the lag cache and ema state so the next forward window reinitialises both from scratch
    def reset(self):
        self.prev_raw = None     #cached previous window mean, unset means start of a trial
        self.prev_median = None  #cached previous window median, unset means start of a trial
        self.prev_iqr = None     #cached previous window iqr, unset means start of a trial
        self.last_flat = []      #indices of channels with zero iqr in the last forward window
        self.ema = {name: None for name in _ACCUM_BASES}  #ema state per base feature, none until first forward pass

    #prime the lag cache from a window without initialising the ema state, for an isolated read-only peek
    def prime_lag(self, window):
        self.prev_raw = feature_mean(window)
        self.prev_median = feature_median(window)
        self.prev_iqr = feature_iqr(window)

    #merge a partial calibration scale, survives a stream reset
    def set_scale(self, scale):
        check_feature_scale(scale, FEATURE_NAMES)
        self.scale.update(scale)

    #merge a partial manual weight, survives a stream reset
    def set_weight(self, weight):
        check_feature_scale(weight, FEATURE_NAMES)
        self.weight.update(weight)

    #merge a partial accumulation toggle, survives a stream reset, also enables the corresponding output features
    def set_accum(self, accum):
        if not isinstance(accum, dict):
            raise ValueError(f"accum must be a dict of base feature name to bool; got {type(accum)}")
        unknown = [k for k in accum if k not in _ACCUM_BASES]
        if unknown:
            raise ValueError(f"unknown accum keys {unknown}; valid base features are {list(_ACCUM_BASES)}")
        if not all(isinstance(v, bool) for v in accum.values()):
            raise ValueError("accum values must all be bool")
        self.accum.update(accum)
        #enabling accumulation for a base feature also enables its ema output features in the toggle
        for name, enabled in accum.items():
            self.feature_toggle[f"{name}_ema"] = enabled
            self.feature_toggle[f"{name}_ema_res"] = enabled

    #merge a partial alpha dict, survives a stream reset, alpha must be in (0, 1] for a valid convex combination
    def set_alpha(self, alpha):
        if not isinstance(alpha, dict):
            raise ValueError(f"alpha must be a dict of base feature name to float in (0, 1]; got {type(alpha)}")
        unknown = [k for k in alpha if k not in _ACCUM_BASES]
        if unknown:
            raise ValueError(f"unknown alpha keys {unknown}; valid base features are {list(_ACCUM_BASES)}")
        for name, value in alpha.items():
            if not (0 < value <= 1):
                raise ValueError(f"alpha for {name} must be in (0, 1]; got {value}")
        self.alpha.update(alpha)

    #the enabled feature names in fixed order, with an optional per-call feature_toggle override
    def enabled_names(self, feature_toggle=None):
        active = self.feature_toggle if feature_toggle is None else {**self.feature_toggle, **feature_toggle}
        return [name for name in FEATURE_NAMES if active[name]]

    #collapse one window (n_channels, segment_length) into the enabled feature stack (n_channels, F)
    #feature_toggle overrides the output selection for this call only, update advances the lag cache (False for a read-only peek)
    def compute(self, window, feature_toggle=None, update=True):
        n = window.shape[0]
        active = self.feature_toggle if feature_toggle is None else {**self.feature_toggle, **feature_toggle}  #per-call output selection

        raw_t = feature_mean(window)      #always computed, the raw lag needs it cached
        median_t = feature_median(window) #always computed, the cache and the lag terms need them
        iqr_t = feature_iqr(window)

        #mobility and complexity computed only when active, guards prevent wasted work on disabled features
        mobility_t = hjorth_mobility(window) if (active["mobility"] or active["mobility_ema"] or active["mobility_ema_res"]) else None
        complexity_t = hjorth_complexity(window) if (active["complexity"] or active["complexity_ema"] or active["complexity_ema_res"]) else None

        #lag-1 terms are the raw change against the cached previous window, zero before any forward pass
        if self.prev_median is None:
            raw_lag = np.zeros(n)
            median_lag = np.zeros(n)
            iqr_lag = np.zeros(n)
        else:
            raw_lag = raw_t - self.prev_raw
            median_lag = median_t - self.prev_median
            iqr_lag = iqr_t - self.prev_iqr

        #ema levels use the current ema state, initialised to current on cold start via ema_update
        #residual is current minus the smoothed level, capturing fast deviation from the running baseline
        base_vals = {"raw": raw_t, "median": median_t, "iqr": iqr_t,
                     "mobility": mobility_t, "complexity": complexity_t}

        ema_levels = {}
        ema_residuals = {}
        for name in _ACCUM_BASES:
            if base_vals[name] is not None:
                level = ema_update(base_vals[name], self.ema[name], self.alpha[name])
                ema_levels[name] = level
                ema_residuals[name] = base_vals[name] - level
            else:
                ema_levels[name] = None
                ema_residuals[name] = None

        values = {"raw": raw_t, "median": median_t, "iqr": iqr_t,
                  "mobility": mobility_t,
                  "complexity": complexity_t,
                  "raw_lag": raw_lag, "median_lag": median_lag, "iqr_lag": iqr_lag,
                  "raw_ema": ema_levels["raw"],         "raw_ema_res": ema_residuals["raw"],
                  "median_ema": ema_levels["median"],   "median_ema_res": ema_residuals["median"],
                  "iqr_ema": ema_levels["iqr"],         "iqr_ema_res": ema_residuals["iqr"],
                  "mobility_ema": ema_levels["mobility"],     "mobility_ema_res": ema_residuals["mobility"],
                  "complexity_ema": ema_levels["complexity"], "complexity_ema_res": ema_residuals["complexity"]}

        #select the active features in fixed order and apply the frozen scale times weight to each
        names = [name for name in FEATURE_NAMES if active[name]]
        stack = [values[name] * self.scale[name] * self.weight[name] for name in names]

        #advance the lag cache and ema state only on a forward pass, a read-only peek leaves both untouched
        if update:
            self.prev_raw = raw_t
            self.prev_median = median_t
            self.prev_iqr = iqr_t
            self.last_flat = np.where(iqr_t == 0)[0]
            for name in _ACCUM_BASES:
                if ema_levels[name] is not None:
                    self.ema[name] = ema_levels[name]

        return np.stack(stack, axis=1)  #(n_channels, F)