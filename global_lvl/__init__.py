from .constants import (DEFAULT_EXCLUDE, MNE_MONTAGES, SEED, EPSILON, 
                        FEATURE_NAMES, WINDOW_SIZE, SOLVER_TYPES, MONTAGE, FEATURE_TOGGLES, 
                        L, NUM_SCHNS, SFREQ_MAX, IMG_DIMS, MARGIN, DROPOUT, 
                        DEFAULT_CMAP, DELTA_CMAP)

from .helpers import (
    #for eeg features
    window_size_from_seconds, compute_mean, compute_median, compute_iqr, hjorth_mobility, hjorth_complexity,

    #for electrode handling
    lowercase_key, lookup_montage, build_chn_name_order, classify_chns, get_3d_pos, get_2d_pos,

    #image transform handler for shared representation; only for indifference to electrode count
    build_img_interpolation, build_topo_mask, apply_topo_mask,

    #for spherical harmonics
    compute_spherical_angles, build_sh_basis, solve_sh_coefficients, decoded_coeffs, max_L_for_chns, is_compatible,          

)

#makes up eeg-env
from .components import (ElectrodeSim, FeatureField, SphericalHarmonics)

from .EEGENV import EEGEnv

from .metrics import (difference, sqr_diff_ratio, mean_error,
                      recovered_variance, cosine_sim,
                      pixel_loss, sobel_stack, sobel_loss)

from .visuals import (basis_matrix_fig, img_transform_fig, view_basis_sphere,
                      save_fig, metric_bar_fig, reconstruction_fig,
                      sobel_feature_fig, sobel_cycle_fig)


