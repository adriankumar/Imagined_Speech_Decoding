from model_architecture import NWMv1
from utilities import (
    load_all_data, organise_dataset, get_train_list, 
    load_vocab_embedding, segment_eeg_tensor
    )
from sentence_transformers import SentenceTransformer
import torch, time

SEED = 24573471 
DR = 0.2 #global drop out rate for architecture
#feature extractors
RAW_CONVS = [(16, 7, 3), (24, 5, 2), (8, 3, 1)] #used on raw eeg window
RAW_SPATIAL_FPF = 4
SPECTRO_CONVS = RAW_CONVS #used on spectrogram window
SPECTRO_FPF = 4
#others
PRINCIPAL_COMPONENTS = 16
ATTENTION_SYNC_VECTOR_SIZE = 50
SINC_OUT_DIM = 24
BP_OUT_DIM = 16


ENV_STATE_DIM = (RAW_CONVS[-1][0] * RAW_SPATIAL_FPF) + (SPECTRO_CONVS[-1][0] * SPECTRO_FPF) + (SINC_OUT_DIM + BP_OUT_DIM)
WINDOW_SIZE=500


CTM_ATTENTION_DEFAULT = {
    'attention_amount': 1,
    'dropout': DR,
    'final_dim': 0, #use attention dim arg in ctm
    'use_dense': False, #use raw attended output for ctm 

    'attention_configs':[
        {'name': 'ctm_attention', 
         'embed_dim': ATTENTION_SYNC_VECTOR_SIZE, #make same as action_sync_vector size
         'num_heads': 5, 
         'pattern': 'cross-attention'
         }],
}

ATT_CONFIG = {
    'attention_amount': 3,
    'dropout': 0.2,
    'final_dim': ATTENTION_SYNC_VECTOR_SIZE, #each attention output is projected into a final dim so shape is b x arbitrary_seq x final dim, should be same as ctm action sync
    'use_dense': True,

    'attention_configs': [
        {
            'name': 'env_state',
            'embed_dim': ENV_STATE_DIM, #the last dim of expected query and key value input to mha, should be same as the env state dim
            'num_heads': 4, #although handled in class, ensure its divisble by embed dim; basically splits embedding into even smaller sizes to use with mutliple attention heads, like a 'partial' attention
            'pattern': 'cross-attention' #cross attention with current average state
        },

        {
            'name': 'imagined_state',
            'embed_dim': ENV_STATE_DIM, #same as env state dim 
            'num_heads': 4,
            'pattern': 'cross-attention' #cross attention with env state (which can be the predicted next state, basically making this self attention in imagined scenarios)
        },

        {
            'name': 'semantic_acc', #semantic accumulation attention 
            'embed_dim': 2 * PRINCIPAL_COMPONENTS, #2 because of mu and sigma
            'num_heads': 4,
            'pattern': 'cross-attention' #on own semantic history
        },

        #skipping sentence reconstruction attention module
    ]

}

#config 
V1_CONFIG_DF = {
    #general/global args
    'seed': SEED,
    'max_windows': 4, #this is kept fixed for training, but when training on better, real-time accomodated data, this will have to be experimented with
    'eeg_channels': 122,
    'segment_length': WINDOW_SIZE,
    'sfreq': 500,
    'dropout': DR,

    #feature extractor args
    'sinc_output_dim': SINC_OUT_DIM,
    'bp_output_dim': BP_OUT_DIM,
    'sinc_filters': 64,
    'raw_spatial_fpf': RAW_SPATIAL_FPF,
    'spectro_fpf': SPECTRO_FPF,
    'raw_spatial_convs': RAW_CONVS,
    'spectro_convs': SPECTRO_CONVS,

    #f-exct activation and dropout
    'sinc_activation': 'silu',
    'raw_spatial_activation': 'leaky-relu',
    'spectro_activation': 'silu',

    #core CTM args
    'num_neurons': 50, #number of neurons the ctm uses
    'memory_length': 24, #the history/length of pre-activations from each of the n neurons to use for the NLM component in calculating the post activations for each n neuron; must be less than num_neurons
    #note the sync vector sizes also determines the number of neurons pairs that can be formed; we use random sparse pairing so not all n neurons will be used to compute synchronisation due to computational constraints
    'pred_sync_vector_size': 50, #size of latent synchronisation vector used for prediction; projects neural synchronisation into this size as latent representation; this sync vector will be decoded into the output dim (actual predictions)
    'action_sync_vector_size': ATTENTION_SYNC_VECTOR_SIZE, #size of latent synchronisation vector used for action; projects neural synchronisation into this size as latent representation; this sync vector will be used as input into the attention module 
    'self_pairing_count': 36, #number of neurons that will be paired with themselves when computing the sync vectors; ensure self_pairing_count < min(pred_sync_vector_size, action_sync_vector_size)
    'thinking_steps': 16, #number of thinking steps to perform for 1 input; each step consists of attention -> synapse model -> NLM -> Synchronisation
    'ctm_pred_dim': 50, #final output dim that the pred sync vector is decoded into i.e nn.Linear(pred_sync_vector_size, output_dim) projecting sync to prediction outputs

    'ctm_attention': CTM_ATTENTION_DEFAULT,

    #synapse model arguments
    'unet_depth': 6, #depth of the unet used for the synapse model
    'min_unet_width': 16, #minimum width of the unet used for the synapse model
    'synapse_bias': True, #whether to use bias in synapse model

    #arguments for NLM
    'use_deep_nlm': True, #whether to use a deep NLM (2 layer MLP) or a shallow NLM (1 layer linear)
    'use_layer_norm': True, #whether to normalise input (pre-activation history) before passing into MLPs inside NLM
    'temperature': 1.0, #starting temperature for computing post activations in NLM; is a learnable parameter so it will change


   #global propagator args
    'input_mapping': 'affine', #affine or linear are the valid options
    'output_mapping': 'affine',
    'ode_unfolds': 6, #internal ltc approximation steps for differential
    'epsilon': 1e-6, 

    #motor action args
    'a_R1': 10,
    'a_R2': 5,
    'a_R3': 5,
    #number of neurons connected to neighbouring region
    'a_input_fanout': 4,
    'a_R1_fanout': 3,
    'a_R2_fanout': 3,
    'a_recurrent_connections': 8,

    'num_mixtures': 2, #from https://arxiv.org/pdf/1803.10122 - used MDN for next state pred using Mixture of Guassians

    #for neural wire circuit used in LTC/motor decoder/world propagator
    'w_R1': 10,
    'w_R2': 5,
    'w_R3': 5, 
    #number of neurons connected to neighbouring region
    'w_input_fanout': 4,
    'w_R1_fanout': 3,
    'w_R2_fanout': 3,
    'w_recurrent_connections': 8,

    'semantic_components': PRINCIPAL_COMPONENTS, #number of principle components to use for semantic space
    'attention_heads': ATT_CONFIG,
    'latent_world_size': 64
}

#-------------------------------------------------------------------------

dataset = organise_dataset(load_all_data())
num_sentences = 5 #895
sentences_for_vocab_list = 500
sentence_encoder = SentenceTransformer('all-mpnet-base-v2')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f"using device {device}")


vocab_p = r"vocab_embedding.pt"

train_tensor, sentence_id_tensor, sentence_mapping, _, _ = get_train_list(
    train_size=num_sentences, dataset=dataset
)

#get correct voab list separately from training data
_, _, _, vocab_list, _ = get_train_list(
    train_size=sentences_for_vocab_list, dataset=dataset
)

vocab_embedding, _ = load_vocab_embedding(vocab_p, device, normalise=False)

world_model = NWMv1(config=V1_CONFIG_DF, vocab_list=vocab_list, vocab_embedding=vocab_embedding)

sample = train_tensor[0, : ,:] #shape channels x timepoints
windows = segment_eeg_tensor(sample, window_size=WINDOW_SIZE)
window = windows[0].unsqueeze(0) #add batch dim b x chans x segment_length

#recurrent variables for model
env_state = None #kinda redundant to put here but just including it anyways, because this is the only variable on an initial forward pass that 'exists'
cognitive_state = None 
action_state = None 
world_state = None 
semantic_state = None 

env_state = world_model.extract_features(window=window)
print(f"env_state shape: {env_state.shape}")


#prestate is a list with elements [env_input, img_input, semantic_input]
pre_state = world_model.prepare_sensory_inputs(
    env_state=env_state, semantic_state=semantic_state, is_dreaming=False 
    )

print(f"pre state shapes")
print(f"env state: {pre_state[0].shape} | img_state {pre_state[1].shape} | semantic state: {pre_state[2].shape}")

state_t = world_model.sensory_attention(state_inputs=pre_state)
print(f"attended state t shape: {state_t.shape}")

#pass state_t as list so its compatible with the mmha mod
thinking_signals, cognitive_state = world_model.think(kv=[state_t], cognitive_states=cognitive_state)
print(f"thinking signals shape: {thinking_signals.shape}")

#first output path - main action
semantic_state, action_state = world_model.propagate_action(
    thinking_signals=thinking_signals, motor_state=action_state, return_prop=False
    )

print(f"mu shape: {semantic_state[0].shape} | var shape: {semantic_state[1].shape}")

#deterministic decoding 
det_action = semantic_state[0] #mu 
det_ids, det_confs, det_avg_conf = world_model.decode_vocab_ids(coefficients=det_action, return_confidences=True)
det_sentence = world_model.construct_sentence(det_ids)
print(f"det sentence: {det_sentence} | confidence {det_avg_conf.squeeze(0).item():.3f}")
#samped
sam_actions, log_prob = world_model.sample_action(semantic_state=semantic_state)
sam_ids, sam_confs, sam_avg_conf = world_model.decode_vocab_ids(coefficients=sam_actions, return_confidences=True)
sam_sentence = world_model.construct_sentence(sam_ids)
print(f"sam sentence: {sam_sentence} | confidence {sam_avg_conf.squeeze(0).item():.3f}")

#next state pred - dream
next_state, mdn_mu, mdn_ls, mdn_pi, world_state = world_model.predict_next_state(thinking_signals=thinking_signals, world_state=world_state, temp=1.0)
print(f"img state pred shape: {next_state.shape} | mdn_mu: {mdn_mu.shape} | mdn_log_sig: {mdn_ls.shape} | mdn_pi: {mdn_pi.shape}")

#print parameter count
_ = world_model.print_parameter_count()



