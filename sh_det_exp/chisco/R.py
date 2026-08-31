from global_lvl import (EEGEnv, SOLVER_TYPES,
                        difference, sqr_diff_ratio, mean_error, recovered_variance, cosine_sim)

from ..data_cache import load_caches
import numpy as np

#================loading dataset================
ds = load_caches() 

ds_num = 1 #key 1 is chisco sample (subject 1, preprocessed)
sample = ds[str(ds_num)] #the dict for the current sample

sequence = sample["windows"] #returns list of windows in shape nchns x window_size
num_windows = len(sequence)
chns = sample["chns"] #list of channel names this sample uses
sfreq = sample["sfreq"] #sfreq of recording (not used in deterministic compression and decoding)

print(f"Chisco dataset | Subject 1 | Preprocessed (imagined) | Run 1 | Trial 1 |")
print(f"Number of windows: {num_windows} with each window size being a duration of {(sequence[0].shape[-1] / sfreq):.3f}s")
print(f"sampling frequency: {sfreq}, with {sequence[0].shape[-1]} datapoints (N)")
print(len(chns))

#================================================

#================EEG Environment for deterministic solver================
montage = "standard_1005" #resolves the most channels against MNE's expected channels for this dataset

#different L sizes
high_L = 9 
med_L = 7
low_L = 4 

img_size = (64, 64)
margin = 0.80 

#Using all current features
declared_features = {"mean": True, "median": True, "iqr": True, "mobility": True, "complexity": True}
num_features = sum(declared_features.values())
print(f"Using {num_features} features") 

def _build_env(L):
    return EEGEnv(src_chn_names=chns, L_degree=L, feature_toggles=declared_features, 
                  montage=montage, img_dims=img_size, img_margin=margin)

#================================================

#================Stand-Alone performance================
def _solver(index=0):
    return SOLVER_TYPES[index] #0 is B=I (ridge regression); 1 is B=diag

#returns (F,), averaging per-window scores over all windows
def _average_metrics(result_list, num_windows):
    return np.stack(result_list, axis=0).sum(axis=0) / num_windows #(W, F) -> (F,)

#prints averaged metrics as an aligned table for transcription
def _print_table(results):
    feature_names = results["features"]

    bounded = ["sqr_diff_ratio", "recovered_var", "cos_sim"] #read as :.4f
    errors = ["mse", "mae", "rmse"] #span orders of magnitude, read as :.3e
    columns = ["sqr_diff_ratio", "mse", "mae", "rmse", "recovered_var", "cos_sim"]

    header = f"{'Feature':<18}" + "".join(f"{col:>15}" for col in columns)

    print(f"\nL = {results['L']} | solver = {results['solver_idx']} | averaged over {results['num_windows']} windows")
    print(header)
    print("-" * len(header))

    for i, name in enumerate(feature_names):
        row = f"{name:<18}"
        for col in columns:
            val = results[col][i]
            row += f"{val:>15.4f}" if col in bounded else f"{val:>15.3e}"
        print(row)
    print()

def run_experiment(L, solver_idx=0):
    #build env 
    detenv = _build_env(L=L)

    #storing per window metrics
    sqr_diffs = [] 
    mse, mae = [], []
    recovered_var = []
    cos_sim = []

    #storing per window vectors for imgs
    true_vecs = []
    recon_vecs = []

    #forward over windows
    for window in sequence:
        true_vec = detenv.window_to_features(window=window) #nchns x F
        coeffs = detenv.deterministic_compress(feature_vectors=true_vec, solver_type=_solver(solver_idx)) #coeffs x F
        recon_vec = detenv.decode_coeffs(coeffs=coeffs) #nchns x F

        #storing
        true_vecs.append(true_vec)
        recon_vecs.append(recon_vec)

        #compute electrode-space metrics only and store; all a list of #list of (F,)
        sqr_diffs.append(sqr_diff_ratio(true=true_vec, recon=recon_vec)) 

        mse.append(mean_error(true=true_vec, recon=recon_vec, err_type="mse"))
        mae.append(mean_error(true=true_vec, recon=recon_vec, err_type="mae"))

        recovered_var.append(recovered_variance(true=true_vec, recon=recon_vec))
        cos_sim.append(cosine_sim(true=true_vec, recon=recon_vec))

    #compute averages; all shape F,
    sdr_avg = _average_metrics(sqr_diffs, num_windows)
    mae_avg = _average_metrics(mae, num_windows)
    mse_avg =_average_metrics(mse, num_windows)
    rsme = np.sqrt(mse_avg)
    rvar_avg = _average_metrics(recovered_var, num_windows)
    vec_sim_avg = _average_metrics(cos_sim, num_windows)

    #view and save metrics
    bar_path = f"det_exp/chisco/{L}/solver_{solver_idx}/results_bgraph" #for bar graph

    #bar plots
    detenv.view_metric_bar(values=sdr_avg, metric_name="sqaure difference ratio", 
                           feature_names=detenv.toggled_features, 
                           subtitle=f"Square Difference Ratio averaged over {num_windows} windows | ↓ is better", 
                           save=True, save_path=bar_path, file_name="sdr.png")

    detenv.view_metric_bar(values=mae_avg, metric_name="mean absolute error", 
                           feature_names=detenv.toggled_features, 
                           subtitle=f"Mean Absolute Error averaged over {num_windows} windows | ↓ is better ", 
                           save=True, save_path=bar_path, file_name="mae.png")

    detenv.view_metric_bar(values=mse_avg, metric_name="mean squared error", 
                           feature_names=detenv.toggled_features, 
                           subtitle=f"Mean Squared Error averaged over {num_windows} windows | ↓ is better", 
                           save=True, save_path=bar_path, file_name="mse.png")

    detenv.view_metric_bar(values=rsme, metric_name="root mean squared error", 
                           feature_names=detenv.toggled_features, 
                           subtitle=f"Root Mean Squared Error averaged over {num_windows} windows | ↓ is better", 
                           save=True, save_path=bar_path, file_name="rmse.png")

    detenv.view_metric_bar(values=rvar_avg, metric_name="recovered variance", 
                           feature_names=detenv.toggled_features, 
                           subtitle=f"Recovered variance averaged over {num_windows} windows | ↑ is better", 
                           save=True, save_path=bar_path, file_name="rvar.png")

    detenv.view_metric_bar(values=vec_sim_avg, metric_name="cosine similarity", 
                           feature_names=detenv.toggled_features, 
                           subtitle=f"Cosine Similarity averaged over {num_windows} windows | ↑ is better", 
                           save=True, save_path=bar_path, file_name="cos.png")

    print(f"saved all bar plots in {bar_path}")

    #viewing and saving images
    recon_path = f"det_exp/chisco/{L}/solver_{solver_idx}/results_recon" #for electrode recon

    for i in range(num_windows):

        detenv.view_image_fields(true_field=true_vecs[i], recon_field=recon_vecs[i], apply_mask=True,
                                 subtitle=f"Electrode-to-Img Feature(s) Reconstruction for window {i+1}",
                                 save=True, save_path=recon_path, file_name=f"field_window_{i+1}")


    print(f"saved all electrode fields in {recon_path}")

    return {"L": L, "solver_idx": solver_idx, "num_windows": num_windows,
            "features": detenv.toggled_features,
            "sqr_diff_ratio": sdr_avg, "mse": mse_avg, "mae": mae_avg, "rmse": rsme,
            "recovered_var": rvar_avg, "cos_sim": vec_sim_avg}

#================================================


#================main run================
if __name__ == "__main__":
    Ls = [high_L, med_L, low_L] #for each L 
    all_results = []

    for deg in Ls:
        for j in range(2): #for each solver
            all_results.append(run_experiment(L=deg, solver_idx=j))

    for results in all_results:
        _print_table(results)