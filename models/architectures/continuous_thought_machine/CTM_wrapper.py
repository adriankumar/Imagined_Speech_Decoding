import torch
import torch.nn as nn 
import numpy as np
from global_lvl import SEED
import os

from .NeuronActivations import NeuronActivations
from .NeuronlvlModel import NeuronLevelModel
from .SynapseUNet import SynapseUNet
from .SyncGate import SyncGate
from ..components import MAH

#config alone is enough to rebuild an identical CTM, seed is stored raw and rnd_seed
#is re-derived
def save_ctm(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict()}, path)

#rebuilds 
def load_ctm(path, device=None):
    checkpoint = torch.load(path, map_location=device)

    model = CTM(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])

    if device is not None:
        model.to(device)

    return model

class CTM(nn.Module):
    def __init__(self, input_dim=60, num_neurons=42, memory_length=20, pred_sync_vector_size=40,
                 action_sync_vector_size=40, self_pairing_count=26, thinking_steps=12,
                 pred_dim=64, seed=SEED, #main ctm args

                 attn_head_dim=20, #dim per attention head, num_heads is derived from this against action_sync_vector_size

                 unet_depth=6, min_unet_width=16, dr_synapse=0.2, synapse_bias=True, #synapse model arg
                 use_deep_nlm=True, use_layer_norm=True, dr_nlm=0.2, temperature=1.0): #nlm arg

        super(CTM, self).__init__() #inherent from nn.Module

        self._initialise_variables(input_dim, num_neurons, memory_length, pred_sync_vector_size,
                                   action_sync_vector_size, self_pairing_count, thinking_steps,
                                   pred_dim, seed, attn_head_dim, unet_depth, min_unet_width, dr_synapse,
                                   synapse_bias, use_deep_nlm, use_layer_norm, dr_nlm, temperature)

        self._record_config() #store into self.config for save loading
        self._build_model() #build model components; attention, synch gate, synapse model, NLM 

    #private methods (used internally only)
    def _initialise_variables(self, input_dim, num_neurons, memory_length, pred_sync_vector_size,
                              action_sync_vector_size, self_pairing_count, thinking_steps,
                              pred_dim, seed, attn_head_dim, unet_depth, min_unet_width, dr_synapse,
                              synapse_bias, use_deep_nlm, use_layer_norm, dr_nlm, temperature):
        #core CTM args
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.memory_length = memory_length
        self.pred_sync_vector_size = pred_sync_vector_size
        self.action_sync_vector_size = action_sync_vector_size

        self.self_pairing_count = self_pairing_count
        self.thinking_steps = thinking_steps
        self.output_dim = pred_dim
        self.seed = seed 
        self.rnd_seed = np.random.RandomState(seed=seed) #create numpy random seed for reproducibility

        self.attn_head_dim = attn_head_dim

        #synapse model arguments
        self.unet_depth = unet_depth
        self.min_unet_width = min_unet_width
        self.dropout_synapse = dr_synapse
        self.synapse_bias = synapse_bias

        #arguments for NLM
        self.use_deep_nlm = use_deep_nlm
        self.use_layer_norm = use_layer_norm
        self.dropout_NLM = dr_nlm
        self.temperature = temperature

    #builds self.config 
    def _record_config(self):
        self.config = {
            'input_dim': self.input_dim,
            'num_neurons': self.num_neurons,
            'memory_length': self.memory_length,
            'pred_sync_vector_size': self.pred_sync_vector_size,
            'action_sync_vector_size': self.action_sync_vector_size,
            'self_pairing_count': self.self_pairing_count,
            'thinking_steps': self.thinking_steps,
            'pred_dim': self.output_dim,
            'seed': self.seed,

            'attn_head_dim': self.attn_head_dim,

            'unet_depth': self.unet_depth,
            'min_unet_width': self.min_unet_width,
            'dr_synapse': self.dropout_synapse,
            'synapse_bias': self.synapse_bias,

            'use_deep_nlm': self.use_deep_nlm,
            'use_layer_norm': self.use_layer_norm,
            'dr_nlm': self.dropout_NLM,
            'temperature': self.temperature,
        }

    #derives the attention spec MAH needs from ctm's own dims, num_heads comes from attn_head_dim
    def _build_attn_config(self):
        assert self.action_sync_vector_size % self.attn_head_dim == 0, f"action_sync_vector_size={self.action_sync_vector_size} not divisible by attn_head_dim={self.attn_head_dim}"

        num_heads = self.action_sync_vector_size // self.attn_head_dim

        return {
            'input_attention': {
                'q': self.action_sync_vector_size, #query is the action sync vector
                'k': self.input_dim, #key/value are the raw input features
                'v': self.input_dim,
                'num_heads': num_heads,
            }
        }

    #builds model components
    def _build_model(self):

        #synchronisation gate to compute latent synchronisation vectors from neuron pairings and their post activations
        self.sync_gate = SyncGate(
            num_neurons=self.num_neurons,
            pred_sync_vector_size=self.pred_sync_vector_size,
            action_sync_vector_size=self.action_sync_vector_size,
            self_pairing_count=self.self_pairing_count,
            seed=self.rnd_seed
        )

        attn_conf = self._build_attn_config()

        #attention module to compute attended features from input features and action sync vector
        self.attention_head = MAH(head_specs=attn_conf,
                                  use_dense=False,
                                  dropout=0.2)

        #synapse model to compute new pre-activations from attended features
        self.synapse_model = SynapseUNet(  
            input_dim=self.num_neurons,
            layers=self.unet_depth,
            min_feature_dim=self.min_unet_width,
            dropout=self.dropout_synapse,
            bias=self.synapse_bias
        )

        self.nlm = NeuronLevelModel( #neuron level model to compute post activations from pre-activation history for each neuron
            num_neurons=self.num_neurons,
            memory_length=self.memory_length,
            is_deep=self.use_deep_nlm,
            use_layernorm=self.use_layer_norm,
            dropout=self.dropout_NLM,
            temperature=self.temperature
        )

        self.neural_activations = NeuronActivations(num_neurons=self.num_neurons, memory_length=self.memory_length) #handles neuron activations, pre-activation history and post activations

        self.output_projection = nn.Linear(self.pred_sync_vector_size, self.output_dim) #final projection from pred sync vector to output dim (predictions)

    #initialises neural states if none are provided; neural states consist of pre-activation history, post-activations, and synchronisation states
    def _init_neural_states(self, batch_size, device):
        pre_activation_history = self.neural_activations.initialise_pre_activation_history(batch_size, device) #initialise pre-activation history; #shape: batch x num_neurons x memory_length
        synch_states = self.sync_gate.init_sync_states() #initialise synchronisation states; #dict containing pred and action sync states
        initial_post_activations = self.neural_activations.initalise_post_activations(batch_size, device) #initial post activations; #shape: batch x num_neurons

        return pre_activation_history, initial_post_activations, synch_states #return as a tuple

    #public methods (can be used externally)
    #executes single forward pass with thinking steps, acts as a recurrent layer (can pass neural states from previous forward pass)
    def forward(self, input_features, neural_states=None, suppress_warning=False):

        batch_size = input_features.shape[0]  
        device = input_features.device

        #1.
        #initialise or use previous neural states; neural states consist of pre-activation history, post-activations, and synchronisation states 
        if neural_states is None:
            pre_history, current_post_activations, synch_state = self._init_neural_states(batch_size, device)
        else:
            pre_history = neural_states['pre_activation_history']
            current_post_activations = neural_states['post_activations']
            synch_state = neural_states['synch_states']
            
        #2.
        #storage for predictions and certainties (we iterate over thinking steps so create a temporary storage for outputs for single forward pass)
        #stores either internal reasoning or sentence reconstruction depending on thinking_loop
        predictions = torch.zeros(batch_size, self.output_dim, self.thinking_steps, device=device)

        #3.
        #start thinking steps loop
        for thought_step in range(self.thinking_steps):
            #a - compute action sync vector from previous post activations 
            action_sync_vector = self.sync_gate.compute_sync_vector(synch_state, current_post_activations, 'action', batch_size, device)

            #assume input is a vector of B x input_dim (lazy linear init)
            attention_input = {
                'input_attention': {'q': action_sync_vector.unsqueeze(1), #add sequence dim of 1
                                    'k': input_features, #expect shape b x arbitrary_seq x dim
                                    'v': input_features} #same as key
            }

            #should be shape b x action sync vector dim
            attended_features = self.attention_head(inputs=attention_input,
                                                    aggregate='list',
                                                    suppress_seq_warning=suppress_warning)[0] #only one attention head for ctm

            #assume attended features has seq dim of 1 because its based off the query seq length (which is always 1 from the action sync)
            attended_features = attended_features.squeeze(1) #b x dim;

            #c - pass attended features, concat with previous post activations into synapse model to compute new pre-activations for each neuron  
            prev_activ_new_info = torch.cat([current_post_activations, attended_features], dim=-1) #concat across feature dim
            new_pre_activations = self.synapse_model(prev_activ_new_info)

            #d - append current new pre-activations to pre-activation history for each neuron
            pre_history = self.neural_activations.update_pre_activation_history(pre_history, new_pre_activations)

            #e - pass updated pre-activation history into NLM to compute new post-activations for each neuron
            current_post_activations = self.nlm(pre_history)

            #f - compute prediction sync vector from new post-activations 
            pred_sync_vector = self.sync_gate.compute_sync_vector(synch_state, current_post_activations, 'pred', batch_size, device)

            #g - pass prediction sync vector into output projection to compute predictions
            current_predictions = self.output_projection(pred_sync_vector)
            predictions[:, :, thought_step] = current_predictions #store predictions for this thinking step, size is batch x embedding size x thinking steps

        #4.
        #update neural states with final states from this forward pass to pass in next forward pass
        final_neural_states = {'pre_activation_history': pre_history, 'post_activations': current_post_activations, 'synch_states': synch_state}

        #5.
        #return predictions and updated neural states; word decoding and sentence construction handled outside model 
        return predictions, final_neural_states

    #returns the config dictionary used to initialise the CTM
    def return_config(self):
        return self.config

    #property getters for state visualisation access
    @property
    def pred_dim(self):
        return self.output_dim

    @property 
    def think_steps(self):
        return self.thinking_steps
    
    def get_parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'non_trainable': total - trainable}

    def print_param_count(self):
        for p_type, count in self.get_parameter_counts().items():
            print(f"{p_type.lower()} parameters: {count}")