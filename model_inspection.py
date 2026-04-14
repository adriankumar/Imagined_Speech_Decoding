from utilities import (
    load_model_checkpoint, load_vocab_embedding, load_all_data, organise_dataset, get_train_list,
    run_phase1_diagnostics, run_phase3_diagnostics
)
import torch 
from model_architecture import NWM
from sentence_transformers import SentenceTransformer

#model paths
model_p = r"demo_weights_metrics\vocab_and_model\policy_best.pt"
vocab_p = r"demo_weights_metrics\vocab_and_model\vocab_embedding.pt"

#globals
dataset = organise_dataset(load_all_data())
num_sentences = 3 #895
sentence_encoder = SentenceTransformer('all-mpnet-base-v2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device {device}")

#get full vocab list
# _, _, _, vocab_list, vocab_dict = get_train_list(
#     train_size=895, dataset=dataset
# )

train_tensor, sentence_id_tensor, sentence_mapping, vocab_list, _ = get_train_list(
    train_size=num_sentences, dataset=dataset
)
vocab_embedding, _ = load_vocab_embedding(vocab_p, device, normalise=False)

model = NWM(
    vocab_list=vocab_list,
    vocab_embedding=vocab_embedding,
    embedding_size=vocab_embedding.shape[-1]
)

model = load_model_checkpoint(model_p, model, device)

#run all diagnostics
outputs, results, grad_norms = run_phase1_diagnostics(
    model, train_tensor, sentence_id_tensor, sentence_mapping,
    vocab_list, sentence_encoder, device
)


results = run_phase3_diagnostics(
    model, train_tensor, sentence_id_tensor, sentence_mapping, 
    vocab_list, segment_length=500
)