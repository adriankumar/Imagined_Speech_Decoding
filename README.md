How to run:
- Run at Root directory: `python -m EEGEnv.gui.app`

---

Ensure to check `requirements.txt` for current dependencies

---
TODO
- move geomtry controls and feature stack as one widget for saving space so that there isnt such a huge gap or space taken for the controls; perhaps dynamic placement mechancisms?

- Add spherical harmonics visualisation (not sure if bases are relevant here; can use the sh sandbox for that?), Y matrix visualisation, and reconstruction from electrode count; Fix the module's spherical harmonic computation

- Decode simulation, measure losses as raw pixels and other alternative heatmap specific losses, compute forward window's sh coeffs, subtract from current window to get true delta, and have the loss computed live as a value to certify the accuracy of the metric itself

- need massive renaming and cleaning of functions and variables

- github.io considerations? like a demo
---