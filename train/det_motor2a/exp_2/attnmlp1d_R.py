from models.det_motor2a import Motor2aMLP1D
from global_lvl import load_eegenv
from ..ds.loaders import build_window_loader
from ..ds.stats import compute_feature_clips, compute_electrode_feature_means
from ..exp_1.compression_stats import mean_through_operator
from .train_helpers.for_mlp1d import train_decoder
import torch 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#===load dataset===
wsizes = [0.2, 0.5, 0.7] #window sizes of cached motor2a features
paths = [f"F:/EEG_datasets/cached/deterministic_motor/wz_{str(wz)}" for wz in wsizes]
thresh = 1.0 #for labels existing in partial windows
batch_size = 64 
shuffle = True 
drop_irregular_sequence = False 

classes = ["hand", "feet", "tongue", "left", "right"]

#decoder classes x motor2a labels
class_encodings = {
    0: (0, 0, 0, 0, 0), #rest implicit — no positive head, derived later as 1-max(task)
    1: (1, 0, 0, 1, 0), #left hand
    2: (1, 0, 0, 0, 1), #right hand
    3: (0, 1, 0, 1, 1), #both feet
    4: (0, 0, 1, 0, 0)  #tongue
}


#cache srcs is a list of Motor2aCache classes, there should be 3
cache_srcs, loader = build_window_loader(cache_paths=paths, class_encoding=class_encodings, 
                                         classes=classes, batch_size=batch_size, 
                                         threshold=thresh, shuffle=shuffle, 
                                         drop_last=drop_irregular_sequence)

#===env loading===
eegenv_pth = "train/det_motor2a/ds/motor2a_env.json"
eegenv = load_eegenv(config_path=eegenv_pth, print_channel_resolve=True) #motor2a has 25 channels but only 22 EEG ones are used
L_degree = 2 #9 coefficients for 22 electrodes
eegenv.change_L(L_degree=L_degree)

#==training dist preprocessing for model==
percentile = 99.9 #what value does 99.9% of the training set fall below
clip = compute_feature_clips(sources=cache_srcs, percentile=percentile) #F,
chn_mean = compute_electrode_feature_means(sources=cache_srcs, clip=clip) #nchns x F;   
#indexxed as 0 because it returns a tuple (solver, identity)
coeff_mean = mean_through_operator(chn_mean, eegenv.extract_solver_operator(is_torch=False)[0])  #coeffs x F


#===MLP 1 DECODER===
input_dim = eegenv.total_coeffs #using coefficients as input
# input_dim = eegenv.num_channels
num_feat = eegenv.num_features 
query_dim = 20
attn_heads = 2
attn_name = "coeff_attn" #only one attn head so can call it coeffs instead of decoder name
# attn_name = "electrode_attn"
hidden_dim = 30 
n_layers = 2 
out_dim = len(classes)
dr = 0.2
reduce_layers = True 

decoder = Motor2aMLP1D(input_dim=input_dim, num_features=num_feat, query_dim=query_dim, num_attn_heads=attn_heads, 
                       hidden_dim=hidden_dim, output_dim=out_dim, n_layers=n_layers, dr=dr, reduce_layers=reduce_layers, 
                       attn_label=attn_name, input_clip=clip, input_mean=coeff_mean)

decoder.print_param_count()

#===Training===
lr = 1e-4
grad_clip = 3.0
epochs = 700
epochs_till_break = 100 #if loss remains stuck for at least these amount of epcohs then stop the training and save
save_freq = 100
input_mode="coeffs"
# input_mode = "electrodes"
mlp_path = "train/det_motor2a/exp_2/saves/one_decoder"
save_name = f"mlp_one_decoder_coeff_L{L_degree}"
# save_name = "mlp_one_decoder_electrodes"

trained, metrics = train_decoder(eegenv=eegenv, model=decoder, ds_loader=loader, device=device,
                                 lr=lr, n_epochs=epochs, epoch_tll_terminate=epochs_till_break, 
                                 input_mode=input_mode, grad_clip=grad_clip, save_freq=save_freq,
                                 save_pth=mlp_path, save_name=save_name)