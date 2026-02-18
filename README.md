# Imagined_Speech_Decoding
building a lightweight world model to decode imagined sentences from EEG signals in real-time


Current Progress in this repo:
- Added nested multi head-attetnion for 'sensory' input
- concatenatted attention output from sensory is what we use as the 'state_t' with action outputs
- Updated architecture to include world model
- changed semantic predictions to be k principle components via svd decomp instead of norm of raw embeddings


What needs to be done next:
- Move all previous pre-training and soft actor-critic to this repo
- Move prototype evaluation GUI here
- Add a sentence reconstruction module to model
- Possibly change model output parameters if we adopt 'Hindsight Credit Assignment (HCA)' to replace entropy-regularisation in traditional SAC
- modify and make reward and objective functions more robust/related to experience and behaviour rather than human intuition and intent