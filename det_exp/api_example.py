from global_lvl import (EEGEnv, MNE_MONTAGES, SOLVER_TYPES,
                        sqr_diff_ratio, recovered_variance, cosine_sim,
                        pixel_loss, sobel_stack, sobel_loss)

from .data_cache import load_caches
import numpy as np

#================loading dataset================
ds = load_caches() #returns a dict; with keys "1", "2", 3"

ds_num = 3 #choose a dataset from 1-3
sample = ds[str(ds_num)] #the dict for the current sample

print(f"sample keys: {sample.keys()}\n")

sequence = sample["windows"] #returns list of windows in shape nchns x window_size
chns = sample["chns"] #list of channel names this sample uses

#note: the data currently cached is already split into windows with a specified
#window size of 0.5 seconds each; so window_size will be different because each sample
#had a different sfreq
sfreq = sample["sfreq"] 

print(f"Dataset {ds_num}, has {len(sequence)} windows representing 0.5s from a sampling frequency of {sfreq}hz\n")

#montages that each dataset uses
ds_montages = {"1": "standard_1005", #from chisco, inferred because it resolves the most channels
               "2": "biosemi128", #thinking out loud, from paper: https://www.nature.com/articles/s41597-022-01147-2
               "3": "standard_1005"} #motor2a, from paper: https://www.bbci.de/competition/iv/desc_2a.pdf

print(f"There are {len(MNE_MONTAGES)} montages to experiment with; use the dictionary for",
      "the actual montages  per-dataset, or index one from MNE_MONTAGES variable\n")


#================================================

#================EEG Environment for deterministic solver================
# m_selection = MNE_MONTAGES[-1] #uncomment or use selection below
m_selection = ds_montages[str(ds_num)]
L_degree = 9 if ds_num != 3 else 3 #dataset 3 only has 22 electrodes, so highest L supported is 3
img_size = (64, 64)
margin = 0.75 

declared_features = {"mean": False, "median": False, "iqr": False, "mobility": True, "complexity": True}
num_features = sum(declared_features.values())
print(f"Using {num_features} features") #F dim in n_chns x F

#no need to pass sfreq or window_seconds, they're not used 
DetEnv = EEGEnv(src_chn_names=chns, L_degree=L_degree, feature_toggles=declared_features,
                   montage=m_selection, img_dims=img_size, img_margin=margin)

DetEnv.view_img_transform()

#================================================

#================Forward pass (compression -> reconstruction)================
solver_type = SOLVER_TYPES[0] #0 is B=I (ridge regression); 1 is B=diag
for window in sequence: #shape nchns x window_size

      #1. compute window features; nchns x F
      vec_field = DetEnv.window_to_features(window=window)

      #2. compress vector into coefficients; coeffs x F
      coeffs = DetEnv.deterministic_compress(feature_vectors=vec_field,
                                             solver_type=solver_type)

      #3. reconstruct vector field from coeffs; nchns x F
      recon_field = DetEnv.decode_coeffs(coeffs=coeffs)

      #nchns x F
      residual = vec_field - recon_field #also can be computed from detenv.compression_loss(feature_vectors=vec_field)

      true_img = DetEnv.to_img(feature_vectors=vec_field, apply_mask=True)
      recon_img = DetEnv.to_img(feature_vectors=recon_field, apply_mask=True)

      imgdx, imgdy = sobel_stack(img=true_img, per_axis=True)
      rimgdx, rimgdy = sobel_stack(img=recon_img, per_axis=True)

print(f"true vector shape: {vec_field.shape}\n",
      f"coefficient vector shape: {coeffs.shape}\n",
      f"recon vector shape: {recon_field.shape}\n",
      f"delta shape: {residual.shape}\n",
      f"img shape: {true_img.shape}\n",
      f"sobel/derivative img shape: dx={imgdx.shape} | dy={imgdy.shape}\n")

#================================================

#================Metrics================
single_window = sequence[0] #nchns x window_size

true_field = DetEnv.window_to_features(window=single_window)
recon_field = DetEnv.decode_coeffs(coeffs=DetEnv.deterministic_compress(feature_vectors=true_field, solver_type=solver_type))

#no mask
true_img = DetEnv.to_img(feature_vectors=true_field)
recon_img = DetEnv.to_img(feature_vectors=recon_field)

#electrode-space metrics
sqr_diff = sqr_diff_ratio(true=true_field, recon=recon_field) #square difference, norm against true squared, per feature; (F,)
detail_score = recovered_variance(true=true_field, recon=recon_field) #F,
cos_score = cosine_sim(true=true_field, recon=recon_field)

# image-space metrics
pixel_mse = pixel_loss(true_img=true_img, recon_img=recon_img, mask=DetEnv.topo_mask) # F
sobel_mse = sobel_loss(true_img=true_img, recon_img=recon_img, mask=DetEnv.topo_mask) #F

for ft, using in declared_features.items():
      i = 0
      if using: #if using feature
            print(f"{ft}:")
            print(f"squared difference divided by true sqaure: {sqr_diff[i]:.3f}") #lower is better
            print(f"recovered variance: {detail_score[i]:.3f}") #higher is better
            print(f"cosine similarity: {cos_score[i]:.3f}") #higher is better 
            print(f"pixel mse: {pixel_mse[i]:.3f}")
            print(f"sobel mse: {sobel_mse[i]:.3f}")
            print("\n")
            i += 1

      else:
            continue
#================================================

#================Visuals================
DetEnv.view_coeff_decoder()
DetEnv.view_img_transform()
# DetEnv.view_basis_sphere()
#================================================


#================Metric Visuals================
normalise = False

#metric bar expects shape of (F,)
DetEnv.view_metric_bar(values=sqr_diff, metric_name="square diff", 
                       feature_names=DetEnv.toggled_features, subtitle="Sqaure Difference Ratio per feature for window 0, subject 1 | Lower is better", 
                       norm=normalise, save=False)

DetEnv.view_metric_bar(values=detail_score, metric_name="variance recovered", 
                       feature_names=DetEnv.toggled_features, subtitle="Recovered Variance per feature for window 0, subject 1 | Higher is better", 
                       norm=normalise, save=False)

DetEnv.view_metric_bar(values=cos_score, metric_name="cosine similarity", 
                       feature_names=DetEnv.toggled_features, subtitle="Cosine Similarity per feature for window 0, subject 1 | Higher is better", 
                       norm=normalise, save=False)

DetEnv.view_metric_bar(values=pixel_mse, metric_name="Pixel loss", 
                       feature_names=DetEnv.toggled_features, subtitle="Pixel Loss per feature for window 0, subject 1 | Lower is better", 
                       norm=normalise, save=False)

DetEnv.view_metric_bar(values=sobel_mse, metric_name="Sobel loss", 
                       feature_names=DetEnv.toggled_features, subtitle="Sobel Loss per feature for window 0, subject 1 | Lower is better", 
                       norm=normalise, save=False)

DetEnv.view_electrode_fields(true_field=true_field, recon_field=recon_field, subtitle="Per Feature Reconstruction for window 0, Subject 1 | wz=0.5", 
                             norm=normalise, save=False)

DetEnv.view_image_fields(true_field=true_field, recon_field=recon_field, apply_mask=True,
                         subtitle="Per Feature Reconstruction for window 0, Subject 1 | wz=0.5")


DetEnv.view_sobel_fields(true_field=true_field, recon_field=recon_field, apply_mask=True,
                         subtitle="Sobel Filters/Derivative of Heatmap per feature (scale=0.01)", scale=0.01, save=False)
#================================================

#================Gif Visuals================
# true_fields = np.stack([DetEnv.window_to_features(window=w) for w in sequence]) #num_windows x nchns x F
# recon_fields = np.stack([DetEnv.decode_coeffs(DetEnv.deterministic_compress(t, solver_type))
#                          for t in true_fields]) 


# DetEnv.view_electrode_fields_gif(true_fields=true_fields, recon_fields=recon_fields,
#                                  feature="mobility", subtitle=f"mobility electrode fields at L={L_degree}",
#                                  save_path="temp_saves", file_name="mobility_e_field.gif")

# DetEnv.view_image_fields_gif(true_fields=true_fields, recon_fields=recon_fields,
#                              feature="mobility", subtitle=f"mobility image fields at L={L_degree}",
#                              save_path="temp_saves", file_name="mobility_m_field.gif")
