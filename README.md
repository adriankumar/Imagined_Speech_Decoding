# How to run:
- Ensure to check `requirements.txt` for current dependencies
- Run at Root directory: `python -m eegenv.gui.app`

---
---
### TODO
- Decode simulation, measure losses as raw pixels and other alternative heatmap specific losses and have the loss computed live as a value to certify the accuracy of the metric(s) itself

- Allow upload of other EEG file types (pickle, etc., but only ones that fit the meta-data extraction process of the current module; will extend file reading types but keep minimal compute)

- Extend module as a `DataLoader` for batched files for training

- github.io considerations? like a demo
---