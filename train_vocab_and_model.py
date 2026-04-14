import torch, numpy as np
from model_architecture import NWMv0, NWMv1
from utilities import (
    load_all_data, organise_dataset, get_train_list, 
    load_vocab_embedding, train_vocab_compression, run_vocab_diagnostics,
    load_model_checkpoint, training_V2, training_V3, training_V4, training_V4_1
    )


#globals
dataset = organise_dataset(load_all_data())
num_sentences = 3 #895, num sentences builds the exact vocab list needed to produce these num of sentences so vocab embedding will be made
embedding_dim = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device {device}")

vocab_path = r"demo_weights_metrics\vocab_and_model\vocab_embedding.pt"
save_path = r"demo_weights_metrics\vocab_and_model"

train_tensor, sentence_id_tensor, sentence_mapping, vocab_list, vocab_dict = get_train_list(
    train_size=num_sentences, dataset=dataset
)

#train vocab embedding:
compressed_dim = embedding_dim
lr = 1e-2
epochs = 2000

#train compression
train_vocab_compression(
    compressed_dim=compressed_dim, vocab_list=vocab_list, 
    device=device, lr=lr, epochs=epochs, 
    save_path=vocab_path
    )


#load trained vocab embedding
vocab_embedding, _ = load_vocab_embedding(vocab_path, device, normalise=False)

world_modelv2 = NWMv0(
    vocab_list=vocab_list,
    vocab_embedding=vocab_embedding,
    embedding_size=embedding_dim #esnure same embedding dim
)

#training hyperparams (experiemnent model hyperparams in kwargs)
#globals
n_epochs = 5000
n_eps = train_tensor.shape[0] #sourced from same number of sentences 
segment_length = 500 
batch_size = 32 #or n_eps * (1 + augment_factor)
lr_q, lr_fe, lr_policy, lr_world = 5e-4, 5e-5, 1e-4, 5e-4

print(f"number of episodes: {train_tensor.shape[0]} using {n_eps} episodes (excluding augmented samples)")

#scheduler hyperparams
adaptive_lr = True
qmode, fmode, pmode, wmode = 'min', 'min', 'min', 'min'
qfac, ffac, pfac, wfac = 0.5, 0.5, 0.5 , 0.5
qpat, fpat, ppat, wpat = 30, 30, 30, 30
qmin_lr, fmin_lr, pmin_lr, wmin_lr = 1e-6, 1e-6, 1e-6, 1e-6 

#objective function hyperparams 
gamma = 0.99 
b_alpha = -2.0 
alpha_lr = 1e-3
tau = 5e-3 #dont make too low 
critic_grad_clip = 5.0
actor_grad_clip = 7.0
world_grad_clip = 7.0 
world_temp = 1.0 #for model.predict_next_state(), ensure this is the same during inference
save_freq = 50 
world_model_threshold = 0.70
fe_thresh = 0.60
augment_factor = 2
augmentation_settings = {
    'noise_prob': 0.7, #higher, the more relaxed
    'noise_std_range': (0.02, 0.15), #range
    'scale_prob': 0.7,
    'scale_range': (0.75, 1.25),
    'dropout_prob': 0.4,
    'dropout_count_range': (2, 8)
}

#train world model + policy
training_V4(
    model=world_modelv2,
    n_epochs=n_epochs,
    n_eps=n_eps,
    segment_length=segment_length,
    train_tensor=train_tensor,
    label_id_tensor=sentence_id_tensor,
    label_mapping=sentence_mapping,
    batch_size=batch_size,
    device=device,
    lr_q=lr_q,
    lr_fe=lr_fe,
    lr_policy=lr_policy,
    lr_world=lr_world,
    qmode=qmode,
    qfac=qfac,
    qpat=qpat,
    qmin_lr=qmin_lr,
    fmode=fmode,
    ffac=ffac,
    fpat=fpat,
    fmin_lr=fmin_lr,
    pmode=pmode,
    pfac=pfac,
    ppat=ppat,
    pmin_lr=pmin_lr,
    wmode=wmode,
    wfac=wfac,
    wpat=wpat,
    wmin_lr=wmin_lr,
    adaptive_lr=adaptive_lr,
    gamma=gamma,
    tau=tau,
    critic_grad_clip=critic_grad_clip,
    actor_grad_clip=actor_grad_clip,
    base_alpha=b_alpha,
    alpha_lr=alpha_lr,
    world_grad_clip=world_grad_clip,
    world_temp=world_temp,
    save_freq=save_freq,
    world_model_threshold=world_model_threshold,
    fe_threshold=fe_thresh,
    aug_fac=augment_factor,
    aug_config=augmentation_settings,
    save_path=save_path
)