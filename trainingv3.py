#includes pre-training feature extractors
import torch, numpy as np
from model_architecture import NWMv1
from utilities import (
    load_all_data, organise_dataset, get_train_list, segment_eeg_tensor,
    load_vocab_embedding, train_vocab_compression, 
    load_model_checkpoint, load_pretrained_model, training_V8, pretrain_feature_extractor, reinit_variance
    )


#globals
dataset = organise_dataset(load_all_data())
num_sentences = 5 #895, num sentences builds the exact vocab list needed to produce these num of sentences so vocab embedding will be made
num_sentences_in_vocab_list = 500 #500
embedding_dim = 150
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device {device}")
print(f"embedding dim: {embedding_dim}")

vocab_path = r"demo_weights_metrics\vocab_and_model\vocab_embedding.pt"
save_path = r"demo_weights_metrics\vocab_and_model"

#get vocab list for vocab compression
_, _, _, vocab_list, vocab_dict = get_train_list(
    train_size=num_sentences_in_vocab_list, dataset=dataset
)


#train vocab embedding:
compressed_dim = embedding_dim
lr = 1e-2
epochs = 8000
# train compression
print("compressing vocab embedding...")
train_vocab_compression(
    compressed_dim=compressed_dim, vocab_list=vocab_list, 
    device=device, lr=lr, epochs=epochs, 
    save_path=vocab_path
    )


#load trained vocab embedding
vocab_embedding, _ = load_vocab_embedding(vocab_path, device, normalise=False)

world_modelv1 = NWMv1(
    vocab_list=vocab_list,
    vocab_embedding=vocab_embedding,
    embedding_size=embedding_dim
)

segment_length = 500 

#get samples for pre-training and policy
train_tensor, sentence_id_tensor, sentence_mapping, _, _ = get_train_list(
    train_size=num_sentences, dataset=dataset
)

#pretraining hyperparams & settings
# pretrained_path = r"demo_weights_metrics\pretrained"
# pt_epochs = 320
# pt_batch_size = 4 #train_tensor.shape[0] // 2
# print(f"pretraining batch size: {pt_batch_size}")
# fe_lr = 1e-4
# diversity_weight = 0.2
# fe_clip = 7.0
# pt_save_freq = 50
# pt_aug_config = {
#     'noise_prob': 0.7,
#     'noise_std_range': (0.02, 0.15),
#     'scale_prob': 0.7,
#     'scale_range': (0.75, 1.25),
#     'dropout_prob': 0.4,
#     'dropout_count_range': (2, 8)
# }
# aug_fac = 5

# best_loss = pretrain_feature_extractor(
#     model=world_modelv1,
#     train_tensor=train_tensor,
#     label_id_tensor=sentence_id_tensor,
#     label_mapping=sentence_mapping,
#     n_epochs=pt_epochs,
#     batch_size=pt_batch_size,
#     seg_len=segment_length,
#     lr=fe_lr,
#     diversity_weight=diversity_weight,
#     grad_clip=fe_clip,
#     save_freq=pt_save_freq,
#     augment=True,
#     augment_config=pt_aug_config,
#     augment_factor=aug_fac,
#     device=device,
#     save_path=pretrained_path
# )

# print(f"Pretraining complete with best loss achieved {best_loss:.4f}")

pretrained_model, _ = load_pretrained_model(world_modelv1, 
                                         r"demo_weights_metrics\pretrained\pretmodel_best.pt", 
                                         device)

reinit_variance(pretrained_model, bias_init=-2.3) #reset variance bias

# print(f"Variance projector bias after reset: {pretrained_model.variance_projector.bias.mean().item()}")
# print(f"Expected: ~0.0")

#test this before training to see log prob
# cog_state, motor_state, prev = None, None, None
# with torch.no_grad():
#     test_idx = 0
#     test_eeg = train_tensor[test_idx, :, :]
#     windows = segment_eeg_tensor(test_eeg, window_size=segment_length)
#     for window in windows:
#         window = window.unsqueeze(0)
#         state = pretrained_model.extract_features(window)
#         signals, cog_state = pretrained_model.think(state, cog_state, prev_output=prev)
#         mu, log_sigma, motor_state = pretrained_model.propagate_action(signals, motor_state)
#         prev = mu
#         _, test_log_prob = pretrained_model.sample_action(mu, log_sigma)
#         print(f"Test log_prob: {test_log_prob.item():.1f} (should be around -2400)")


# model = load_model_checkpoint(path=r"demo_weights_metrics\vocab_and_model\policy_best.pt",
#                       model=world_modelv1,
#                       device=device)
#training hyperparams (experiemnent model hyperparams in kwargs)
#globals
n_epochs = 5000
n_eps = train_tensor.shape[0] #sourced from same number of sentences 
batch_size = 32 #or n_eps * (1 + augment_factor)
lr_q, lr_policy = 5e-4, 5e-4

print(f"number of episodes: {train_tensor.shape[0]} using {n_eps} episodes (excluding augmented samples)")

#scheduler hyperparams
adaptive_lr = True
qmode, pmode = 'min', 'min'
qfac, pfac = 0.5, 0.5
qpat, ppat = 30, 30
qmin_lr, pmin_lr = 1e-6, 1e-6

#objective function hyperparams 
gamma = 0.99 
# b_alpha = 2.0 / (embedding_dim * 16) #action dim aware coefficient 
# b_alpha = -3.5  #decrease -> -1.5 or increase -> -2.5 because base alpha is exponentiated to be alpha
learnable_alpha = False
tau = 5e-3 #dont make too low 
critic_grad_clip = 5.0
actor_grad_clip = 7.0
save_freq = 50 
q_clamps = [-1, 6]
augment_factor = 3
augmentation_settings = {
    'noise_prob': 0.7, #higher, the more relaxed
    'noise_std_range': (0.02, 0.15), #range
    'scale_prob': 0.7,
    'scale_range': (0.75, 1.25),
    'dropout_prob': 0.4,
    'dropout_count_range': (2, 8)
}

#train policy - now using state t from feature extractor
training_V8(
    model=pretrained_model,
    n_epochs=n_epochs,
    n_eps=n_eps,
    segment_length=segment_length,
    train_tensor=train_tensor,
    label_id_tensor=sentence_id_tensor,
    label_mapping=sentence_mapping,
    batch_size=batch_size,
    device=device,
    lr_q=lr_q,
    lr_policy=lr_policy,
    qmode=qmode,
    qfac=qfac,
    qpat=qpat,
    qmin_lr=qmin_lr,
    pmode=pmode,
    pfac=pfac,
    ppat=ppat,
    pmin_lr=pmin_lr,
    adaptive_lr=adaptive_lr,
    gamma=gamma,
    tau=tau,
    critic_grad_clip=critic_grad_clip,
    actor_grad_clip=actor_grad_clip,
    learnable_alpha=learnable_alpha,
    save_freq=save_freq,
    q_clamps=q_clamps,
    aug_fac=augment_factor,
    aug_config=augmentation_settings,
    save_path=save_path
)