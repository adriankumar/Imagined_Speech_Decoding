# Motor Decoding Experiment;
## Deterministic SH Coefficients vs Raw Electrodes

This directory contains the scripts for experimenting the recovery of Motor classification labels using the BCICIV2a (Motor2a) dataset.

The experiment has two parts:

1. Measure how much of the original electrode field's mean and variane is retained by the coefficients at different L's; if a low L retains already 99% of the mean and variance, then the compression and 22-channel recordings are the same in every respect a decoder can recieve; the raw electrode input can serve as a 'control' in the training part.

2. Ablate how labels are organised in the readout for recovery with:
-   Hierarchical, gated & specialised
-   One readoutt axis
~comparing results between the compressed and electrode inputs, as any difference between them becomes attributable to the decoding structure instead of what the compression lost; if they converge then the experiment answers what is recoverable given the limited information Motor2a's recordings actually contained.