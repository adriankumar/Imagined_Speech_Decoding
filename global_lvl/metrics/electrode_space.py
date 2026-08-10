import numpy as np
#losses -> lower is better, 0 is perfect | scores -> higher is better, 1 is perfect
#------------
#electrode space, inputs are nchns x F: nchns is resolved channels, F active features
#every metric reduces over nchns and keeps F, so each feature gets its own number
#------------
#signed per-channel miss, the raw error; loss
def reconstruction_loss(true, recon):
    return true - recon  #(nchns x F)

#error as a fraction of the feature's own size, per feature; loss
#scale-free, so features with different units sit on the same 0-1
def relative_error_loss(true, recon):
    err_sq = ((true - recon) ** 2).sum(axis=0) #(F,)
    sig_sq = (true ** 2).sum(axis=0) #(F,)
    return err_sq / sig_sq  #(F,)

#fraction of the across-channel pattern the reconstruction keeps, per feature; score
#(spatial - differences from one electrode position to the next, not across time)
def recovered_detail_score(true, recon):
    err_sq = ((true - recon) ** 2).sum(axis=0) #(F,)
    var_sq = ((true - true.mean(axis=0)) ** 2).sum(axis=0) #(F,)
    return 1 - err_sq / var_sq  #(F,)

#whether the reconstructed map has the same shape as the true one, ignoring overall size; score
#near 1 means same peaks and troughs in the same places even if the heights are off
def shape_match_score(true, recon):
    dot = (true * recon).sum(axis=0) #(F,)
    norms = np.linalg.norm(true, axis=0) * np.linalg.norm(recon, axis=0) #(F,)
    return dot / norms  #(F,)