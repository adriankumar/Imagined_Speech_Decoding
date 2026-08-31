# Spherical Harmonics - deterministic experiments

# How to use
1. Ensure `requirements.txt` dependencies are installed
2. Run an example script from outside this directory as `python -m sh_det_exp.<script>`; i.e:

```
python -m sh_det_exp.api_example
```

---
# Dataset info
3 EEG recordings have been cached in `data_cache/` that can be loaded by `load_caches()`; but they do not have any labels and have already been split into windows with a shared window size of 0.5s relative to its own sampling frequency

The 3 samples are from:

- "1": Chisco
- "2": Thinking Out Loud
- "3": Motor BCICIV2a
---