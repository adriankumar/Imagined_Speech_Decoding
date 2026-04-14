from utilities.model_utils.inspection_v2 import *
from utilities import (
    load_all_data, organise_dataset, get_train_list, load_model_checkpoint,
    load_vocab_embedding, load_pretrained_model, run_phase3_diagnostics, run_phase1_diagnostics, segment_eeg_tensor
    )
import torch
from model_architecture import NWMv1
from sentence_transformers import SentenceTransformer
import time

dataset = organise_dataset(load_all_data())
num_sentences = 5 #895
sentences_for_vocab_list = 500
sentence_encoder = SentenceTransformer('all-mpnet-base-v2')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f"using device {device}")

vocab_p = r"demo_weights_metrics\vocab_and_model\vocab_embedding.pt"

train_tensor, sentence_id_tensor, sentence_mapping, _, _ = get_train_list(
    train_size=num_sentences, dataset=dataset
)

#get correct voab list separately from training data
_, _, _, vocab_list, _ = get_train_list(
    train_size=sentences_for_vocab_list, dataset=dataset
)

vocab_embedding, _ = load_vocab_embedding(vocab_p, device, normalise=False)


model = NWMv1(
    vocab_list=vocab_list, vocab_embedding=vocab_embedding, embedding_size=vocab_embedding.shape[-1]
)

# model_path = r"demo_weights_metrics\pretrained\pretmodel_best.pt" #inspect pre-trained
model, _ = load_pretrained_model(model, r"current_best.pt", device)

# model_path = r"demo_weights_metrics\vocab_and_model\policy_best.pt" #inspect policy best
# model_path = r"demo_weights_metrics\vocab_and_model\model_epoch_200.pt" #inspect latest policy

# model = load_model_checkpoint(model_path, model, device)

model.to(device)

model.print_parameter_count()



# sample = train_tensor[0, :, :] #channels x timepoints
# windows = segment_eeg_tensor(sample, window_size=500)
# c_state, p_state, prev = None, None, None


# start = time.perf_counter()
# for window in windows:
#     curr_win = window.unsqueeze(0).to(device)
#     state_t = model.extract_features(curr_win)
#     c_signals, c_state = model.think(state_t, c_state, prev_output=prev)
#     mu, log_sig, p_state = model.propagate_action(c_signals, p_state)
#     prev = mu
#     sampled, log_prob = model.sample_action(mu, log_sig)
#     # print(f"log_prob: {log_prob.squeeze(0).mean():.2f}, min: {log_prob.squeeze(0).min():.2f}, max: {log_prob.squeeze(0).max():.2f}")
#     det_sen, det_confs, det_conf = model.decode_vocab_ids(mu)
#     # print(f"det pred: {model.construct_sentence(det_sen)} with confidences; {det_confs.squeeze(0).tolist()}")
#     sam_sen, sam_confs, sam_conf = model.decode_vocab_ids(sampled)
#     # print(f"sam pred: {model.construct_sentence(sam_sen)} with confidences; {sam_confs.squeeze(0).tolist()}")


# final_state = model.get_final_state()
# c_signals, c_state = model.think(final_state, c_state, prev_output=prev)
# mu, log_sig, p_state = model.propagate_action(c_signals, p_state)
# f_sampled, log_prob = model.sample_action(mu, log_sig)
# # print(f"log_prob: {log_prob.squeeze(0).mean():.2f}, min: {log_prob.squeeze(0).min():.2f}, max: {log_prob.squeeze(0).max():.2f}")
# det_sen, det_confs, det_conf = model.decode_vocab_ids(mu)
# # print(f"final det pred: {model.construct_sentence(det_sen)} with confidences; {det_confs.squeeze(0).tolist()}")
# sam_sen, sam_confs, sam_conf = model.decode_vocab_ids(sampled)
# # print(f"final sam pred: {model.construct_sentence(sam_sen)} with confidences; {sam_confs.squeeze(0).tolist()}")

# end = time.perf_counter()
# print(f"Time taken: {end - start:.2f} seconds")

#run all diagnostics
outputs, results, grad_norms = run_phase1_diagnostics(
    model, train_tensor, sentence_id_tensor, sentence_mapping,
    vocab_list, sentence_encoder, device
)


results = run_phase3_diagnostics(
    model, train_tensor, sentence_id_tensor, sentence_mapping, 
    vocab_list, segment_length=500
)