import numpy as np
#losses -> lower is better, 0 is perfect | scores -> higher is better, 1 is perfect
#------------
#electrode space, inputs are nchns x F: nchns is resolved channels, F active features
#every metric reduces over nchns and keeps F, so each feature gets its own number
#------------
#the raw error; 
def difference(true, recon):
    return true - recon  #original shape; technically works for img too but can just compute from electrode and transform this result into an image and its the same thing

#loss type
def mean_error(true, recon, err_type="mse"):
    if err_type not in ["mse", "mae", "rmse"]:
        raise ValueError(f"mean error type: {err_type} unrecognised from mse, mae, and rmse")

    diff = difference(true, recon) #nchns x F

    if err_type == "mae":
        return np.abs(diff).mean(axis=0) #F,

    mse = (diff ** 2).mean(axis=0) #F,

    if err_type == "rmse":
        return np.sqrt(mse) #F,

    return mse

#error as a fraction of the feature's own size, per feature; loss
#scale-free, so features with different units are on the same 0-1
def sqr_diff_ratio(true, recon):
    err_sq = ((true - recon) ** 2).sum(axis=0) #(F,)
    sig_sq = (true ** 2).sum(axis=0) #(F,)
    return err_sq / sig_sq  #(F,)

#squared error / true variance is fraction of the signal's variance that is wrong; 
#1 - changes that to what we got right
def recovered_variance(true, recon):
    err_sq = ((true - recon) ** 2).sum(axis=0) #(F,)
    var_sq = ((true - true.mean(axis=0)) ** 2).sum(axis=0) #(F,)
    return 1 - err_sq / var_sq  #(F,)

def cosine_sim(true, recon):
    #a * b / ||a|||b||
    dot = (true * recon).sum(axis=0) #(F,)
    mag = np.linalg.norm(true, axis=0) * np.linalg.norm(recon, axis=0) #(F,)
    return dot / np.maximum(mag, 1e-8)
