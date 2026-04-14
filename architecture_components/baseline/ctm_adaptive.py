import torch 
import torch.nn as nn 
import numpy as np
from .components import mma, dora_a, lora_a

#DEFAULT CONFIG
DR = 0.3

ctm_attention = {
    'use_dense': False,
    'dr': DR,

    'attention_configs':{
        'ctm_attention':{
            'embed_dim': 40, #make embedding dim same as action sync vector size in ctm, because thats what the query is
            'num_heads': 2
        }
    }
}

CTM_DEFAULT_CONFIG = {
    'num_neurons': 42,
    'memory_length': 20,
    'pred_sync_vector_size': 40,
    'action_sync_vector_size': 40,
    'self_pairing_count': 26,
    'thinking_steps': 16,
    'output_dim': 64,
    'seed': 24573471,

    #synapse model arguments
    'unet_depth': 6, #depth of the unet used for the synapse model
    'min_unet_width': 16, #minimum width of the unet used for the synapse model
    'dropout_synapse': 0.2, #dropout rate for synapse model
    'synapse_bias': True, #whether to use bias in synapse model

    #arguments for NLM
    'use_deep_nlm': True, #whether to use a deep NLM (2 layer MLP) or a shallow NLM (1 layer linear)
    'use_layer_norm': True, #whether to normalise input (pre-activation history) before passing into MLPs inside NLM
    'dropout_NLM': 0.2, #dropout rate for NLM
    'temperature': 1.0, #starting temperature for computing post activations in NLM; is a learnable parameter so it will change
}

class CTM_adaptive(nn.Module):

    def __init__(self, config=None, attention_config=ctm_attention, **kwargs): #**kwargs unpacks a dictionary into key-value pairs so if i dont input a config and do num_neurons=128, keyword args will unpack it into a dictionary
        super(CTM_adaptive, self).__init__() #inherent from nn.Module

        #assuming inputs are always valid; handle error later or at higher level
        self._build_config(config, **kwargs) #build config dictionary from default and any user provided args; is self.config
        self._initialise_variables(self.config) #initialise variables from config dictionary; i.e self.num_neurons etc etc
        self._build_model(attention_config) #build model components; attention, synch gate, synapse model, NLM 

    #private methods (used internally only)

   #creates self.config dictionary
    def _build_config(self, config=None, **kwargs):
        self.config = CTM_DEFAULT_CONFIG.copy() #copy default config

        if config is not None: #if a config dictionary is provided, update the default config with it
            self.config.update(config)
        
        self.config.update(kwargs) #update config with any additional keyword args if provided

    #initialise variables
    def _initialise_variables(self, config):
        #core CTM args
        self.num_neurons = config['num_neurons']
        self.memory_length = config['memory_length']
        self.pred_sync_vector_size = config['pred_sync_vector_size']
        self.action_sync_vector_size = config['action_sync_vector_size']

        self.self_pairing_count = config['self_pairing_count']
        self.thinking_steps = config['thinking_steps']
        self.output_dim = config['output_dim']
        self.rnd_seed = np.random.RandomState(seed=config['seed']) #create numpy random seed for reproducibility

        #synapse model arguments
        self.unet_depth = config['unet_depth']
        self.min_unet_width = config['min_unet_width']
        self.dropout_synapse = config['dropout_synapse']
        self.synapse_bias = config['synapse_bias']

        #arguments for NLM
        self.use_deep_nlm = config['use_deep_nlm']
        self.use_layer_norm = config['use_layer_norm']
        self.dropout_NLM = config['dropout_NLM']
        self.temperature = config['temperature']

        #adapters - no active inference params at this stage
        #modulator state flags
        self._modulators_initialised = False #are modulators built; set true by build_modulators()
        self._use_modulators = False #whether to apply modulators in forward pass; toggled by enable/disable_modulators()
       
        #input features dim; captured on first forward pass since input dim is determined by upstream sensory layer
        self._input_features_dim = None 
        self._input_dim_captured = False    

    #builds model components
    def _build_model(self, attention_config):

        assert attention_config['attention_configs']['ctm_attention']['embed_dim'] == self.action_sync_vector_size, "attention embedding dim and action sync vector size must match"

        #synchronisation gate to compute latent synchronisation vectors from neuron pairings and their post activations
        self.sync_gate = SyncGate(
            num_neurons=self.num_neurons,
            pred_sync_vector_size=self.pred_sync_vector_size,
            action_sync_vector_size=self.action_sync_vector_size,
            self_pairing_count=self.self_pairing_count,
            seed=self.rnd_seed
        )

        #attention module to compute attended features from input features and action sync vector
        self.attention_head = mma(config=attention_config)

        #synapse model to compute new pre-activations from attended features
        self.synapse_model = SynapseModelUNet(  
            neurons=self.num_neurons,
            depth=self.unet_depth,
            min_width=self.min_unet_width,
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

        #modulator container; populated by build_modulators() after pretraining
        self.modulators = nn.ModuleDict()
 

    #initialises neural states if none are provided; neural states consist of pre-activation history, post-activations, and synchronisation states
    def _init_neural_states(self, batch_size, device):
        pre_activation_history = self.neural_activations.initialise_pre_activation_history(batch_size, device) #initialise pre-activation history; #shape: batch x num_neurons x memory_length
        synch_states = self.sync_gate.init_sync_states() #initialise synchronisation states; #dict containing pred and action sync states
        initial_post_activations = self.neural_activations.initalise_post_activations(batch_size, device) #initial post activations; #shape: batch x num_neurons

        return pre_activation_history, initial_post_activations, synch_states #return as a tuple

#public methods (can be used externally)

    #executes single forward pass with thinking steps, acts as a recurrent layer (can pass neural states from previous forward pass)
    def forward(self, input_features, neural_states=None, final_reasoning_length=None):
 
        batch_size = input_features.shape[0]  
        device = input_features.device
 
        #capture input features dim on first forward pass for modulator initialisation later
        if not self._input_dim_captured:
            self._input_features_dim = input_features.shape[-1]
            self._input_dim_captured = True
 
        thinking_loop = None #dynamically change what the internal thinking loop represents or does for a single forward pass
 
        if final_reasoning_length is not None:
            thinking_loop = final_reasoning_length #final full sentence reconstruction reasoning/one-shot
        else:
            thinking_loop = self.thinking_steps #internal reasoning steps
 
        #0.
        #apply input feature modulation if modulators are active; additive correction on input before it enters attention
        if self._use_modulators and 'input_features' in self.modulators:
            input_features = input_features + self.modulators['input_features'](input_features)
 
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
        predictions = torch.zeros(batch_size, self.output_dim, thinking_loop, device=device)
 
        #3.
        #start thinking steps loop
        for thought_step in range(thinking_loop):
            #a - compute action sync vector from previous post activations 
            action_sync_vector = self.sync_gate.compute_sync_vector(synch_state, current_post_activations, 'action', batch_size, device)
 
            #b - pass input features and action sync vector into attention module to compute attended features
            attention_input = [
                {
                    'label': 'ctm_attention', #registered name of attention mod
                    'q': action_sync_vector.unsqueeze(1), #add sequence dim of 1 for compatability: b x 1 x dim
                    'k': input_features, #expect shape b x arbitrary_seq x dim
                    'v': input_features, #same as key
                    'attn_mask': None
                }
            ]
 
            #should be shape b x action sync vector dim
            attended_features = self.attention_head(inputs=attention_input)
 
            #assume attended features has seq dim of 1 because its based off the query seq length, which we made 1
            attended_features = attended_features.squeeze(1) #b x dim
 
            #c - pass attended features, concat with previous post activations into synapse model to compute new pre-activations for each neuron  
            prev_activ_new_info = torch.cat([current_post_activations, attended_features], dim=-1) #concat across feature dim
            new_pre_activations = self.synapse_model(prev_activ_new_info)
 
            #c.1 - apply lora modulation to synapse input if modulators are active
            if self._use_modulators:
                new_pre_activations = new_pre_activations + self.modulators['synapse_input'](prev_activ_new_info)
 
            #d - append current new pre-activations to pre-activation history for each neuron
            pre_history = self.neural_activations.update_pre_activation_history(pre_history, new_pre_activations)
 
            #e - pass updated pre-activation history into NLM to compute new post-activations for each neuron
            current_post_activations = self.nlm(pre_history)
 
            #f - compute prediction sync vector from new post-activations 
            pred_sync_vector = self.sync_gate.compute_sync_vector(synch_state, current_post_activations, 'pred', batch_size, device)
 
            #g - pass prediction sync vector into output projection to compute predictions
            #when modulators are active, dora replaces the output projection with an adapted version
            if self._use_modulators and 'output_projection' in self.modulators:
                current_predictions = self.modulators['output_projection'](pred_sync_vector)
            else:
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
 
#----------------------------
# modulator management 
#----------------------------
 
    #builds lora modulators for fine-tuning; call after pretraining is complete
    #requires at least one forward pass to have been run so input features dim is captured
    #synapse input dim is known from config: prev_activ_new_info = concat(post_activations, attended_features) = num_neurons + action_sync_vector_size
    #synapse output dim matches synapse model output which is num_neurons (one pre-activation per neuron)
    def build_modulators(self, rank=1, alpha=1.0, dropout=0.2, nonlinear=True, activation='silu'):
        
        if self._modulators_initialised:
            raise RuntimeError("modulators already initialised; rebuild not permitted to prevent accidental overwrite of trained modulators")
 
        if not self._input_dim_captured:
            raise RuntimeError("input features dim has not been captured yet; run at least one forward pass before building modulators")
 
        #input features modulator: additive correction on input features before they enter attention
        #input and output dim are the same since lora(input) + input requires matching shapes
        self.modulators['input_features'] = lora_a(
            input_dim=self._input_features_dim,
            output_dim=self._input_features_dim,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            nonlinear=nonlinear,
            activation=activation
        )
 
        #synapse input modulator: additive correction on synapse output, operating on the same input (prev post activations + attended features)
        synapse_input_dim = self.num_neurons + self.action_sync_vector_size #dim of prev_activ_new_info
        synapse_output_dim = self.num_neurons #synapse model outputs one value per neuron
 
        self.modulators['synapse_input'] = lora_a(
            input_dim=synapse_input_dim,
            output_dim=synapse_output_dim,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            nonlinear=nonlinear,
            activation=activation
        )
 
        #output projection modulator: dora replaces the output projection with an adapted version
        #wraps the pretrained output projection weights so direction-magnitude decomposition preserves the learned mapping
        #the original output projection is frozen since dora has captured its weights as a buffer
        output_weight = self.output_projection.weight.data #shape: output_dim x pred_sync_vector_size (nn.Linear convention)
        output_bias = self.output_projection.bias.data if self.output_projection.bias is not None else None
 
        self.modulators['output_projection'] = dora_a(
            frozen_weight=output_weight,
            bias=output_bias,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
 
        #freeze original output projection; dora now owns this transformation
        for p in self.output_projection.parameters():
            p.requires_grad = False
 
        self._modulators_initialised = True
 
        #freeze by default; explicitly unfreeze when starting fine-tuning
        self.freeze_modulators()
 
    #activates modulator usage in forward pass
    def enable_modulators(self):
        if not self._modulators_initialised:
            raise RuntimeError("cannot enable modulators: they have not been initialised yet; call build_modulators() first")
        
        self._use_modulators = True 
 
    #deactivates modulator usage in forward pass; modulators still exist but are skipped
    def disable_modulators(self):
        self._use_modulators = False 
 
    #freezes all modulator parameters so gradients are not computed
    def freeze_modulators(self):
        if not self._modulators_initialised:
            raise RuntimeError("cannot freeze modulators: they have not been initialised yet; call build_modulators() first")
        
        for modulator in self.modulators.values():
            modulator.freeze_adapter()
 
    #unfreezes all modulator parameters so gradients are computed during fine-tuning
    def unfreeze_modulators(self):
        if not self._modulators_initialised:
            raise RuntimeError("cannot unfreeze modulators: they have not been initialised yet; call build_modulators() first")
        
        for modulator in self.modulators.values():
            modulator.unfreeze_adapter()
 
    #returns whether modulators are currently active in the forward pass
    @property
    def modulators_active(self):
        return self._use_modulators and self._modulators_initialised
 
#----------------------------
# properties and utilities
#----------------------------
 
    #property getters for state visualisation access
    @property 
    def thought_steps(self):
        return self.thinking_steps
 
    
    @property
    def action_sync_vector(self):
        return self._current_action_sync
    
    @property
    def prediction_sync_vector(self):
        return self._current_pred_sync
    
    @property
    def current_pre_activations(self):
        return self._current_pre_activations
    
    @property
    def current_post_activations(self):
        return self._current_post_activations
    
    @property
    def pre_activations_history(self):
        return self._pre_activation_history
 
    #returns count in [trainable, non-trainable, total]
    def get_parameter_counts(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
 
        return [trainable_params, non_trainable_params, total_params]
 
    #returns modulator parameter counts separately for monitoring; same format as get_parameter_counts
    def get_modulator_parameter_counts(self):
        if not self._modulators_initialised:
            return [0, 0, 0]
 
        total = sum(p.numel() for p in self.modulators.parameters())
        trainable = sum(p.numel() for p in self.modulators.parameters() if p.requires_grad)
        non_trainable = total - trainable
 
        return [trainable, non_trainable, total]


#neuron activations handling (pre_activation history and post activations)
class NeuronActivations(nn.Module):
    def __init__(self, num_neurons, memory_length):
        super(NeuronActivations, self).__init__()

        #register once at construction of model 
        #pre-activation history initialised using Xavier/Glorot initialisation (+- 1/sqrt(fan_in + fan_out))
        #learnable initialisation template, but eventually gets replaced by actual pre-activations during forward pass
        self.register_parameter('initial_pre_history', 
                                nn.Parameter(
                                    torch.zeros(num_neurons, memory_length).uniform_(
                                        -np.sqrt(1/(num_neurons + memory_length)), 
                                        np.sqrt(1/(num_neurons + memory_length))
                                        )
                                    )
                                ) #shape of num_neurons x memory_length
        
        #initial post-activation state for neurons - shape: num_neurons; also learnable starting point
        self.register_parameter('initial_post_activations', 
                              nn.Parameter(torch.zeros(num_neurons).uniform_(
                                  -np.sqrt(1/num_neurons), np.sqrt(1/num_neurons)
                              )))
        
    #just expand existing params for batch
    def initialise_pre_activation_history(self, batch_size, device):
        pre_history = self.initial_pre_history.unsqueeze(0).expand(batch_size, -1, -1).to(device).clone() #expand for batch processing, shape: batch x num_neurons x memory_length
        return pre_history
    
    def initalise_post_activations(self, batch_size, device):

        inital_post_activations = self.initial_post_activations.unsqueeze(0).expand(batch_size, -1).to(device).clone() #expand for batch processing, shape: batch x num_neurons
        return inital_post_activations

    def update_pre_activation_history(self, pre_activation_history, new_pre_activations):
        return torch.cat((pre_activation_history[:, :, 1:], new_pre_activations.unsqueeze(-1)), dim=-1) #remove oldest pre-activation and append new pre-activation at the end; shape: batch x num_neurons x memory_length

#computes a latent synchronisation vector from random sprase neuron pairings and their post activations
#need to figure out what variables to use for visualisation
class SyncGate(nn.Module):
    def __init__(self, num_neurons, pred_sync_vector_size, action_sync_vector_size, self_pairing_count, seed):
        super(SyncGate, self).__init__()

        #which also builds neuron pairings using seed for reproducibility
        self._initialise_variables(num_neurons, pred_sync_vector_size, action_sync_vector_size, self_pairing_count, seed)
        
#----------------------------
# private methods
#----------------------------
    def _initialise_variables(self, num_neurons, pred_sync_vector_size, action_sync_vector_size, self_pairing_count, seed):
        self.num_neurons = num_neurons
        self.pred_sync_vector_size = pred_sync_vector_size
        self.action_sync_vector_size = action_sync_vector_size
        self.self_pairing_count = self_pairing_count
        self.rnd_seed = seed

        self.register_parameter('w_pred_sync', nn.Parameter(torch.zeros(self.pred_sync_vector_size))) #weights (that will be exponentiated; exponential decay) for computing the latent sync vector for prediction output path
        self.register_parameter('w_action_sync', nn.Parameter(torch.zeros(self.action_sync_vector_size)))

        self._builld_neuron_pairings() #build random sparse neuron pairings for computing synchronisation
    
    #register the indices for neuron pairs for both prediction and action sync vectors, these neurons (when indexed using these registers) will contribute to their respective sync vectors
    def _builld_neuron_pairings(self):
        #neuron pairings for prediction sync vector 
        first_indices_for_pred, second_indices_for_pred = self._generate_random_pairing_indices(self.pred_sync_vector_size) 
        self.register_buffer('pred_first_indices', first_indices_for_pred) 
        self.register_buffer('pred_second_indices', second_indices_for_pred) #register as buffer so they are saved in state dict but not trained 

        #neuron pairising for action sync vector
        first_indices_for_action, second_indices_for_action = self._generate_random_pairing_indices(self.action_sync_vector_size)
        self.register_buffer('action_first_indices', first_indices_for_action)
        self.register_buffer('action_second_indices', second_indices_for_action)
    
    #generates random sparse neuron pairing for synchronisation
    def _generate_random_pairing_indices(self, sync_vector_size):
        #error handling
        if self.self_pairing_count >= sync_vector_size:
            raise ValueError(f"self_pairing_count: {self.self_pairing_count} must be less than {sync_vector_size}")

        first_indices = torch.from_numpy(self.rnd_seed.choice(self.num_neurons, size=sync_vector_size, replace=False)) #imagine a neuron pair as x,y, then x here is the first indices and y is x's pair; note not all num_nuerons will be used, it is random sparse pairing

        #get the second indicies to have self_pairing_count neurons paired with themselves from the first indices
        second_indices = torch.zeros_like(first_indices)
        second_indices[:self.self_pairing_count] = first_indices[:self.self_pairing_count] #first self_pairing_count neurons paired with themselves

        #fill the rest of second indices with random choices
        if self.self_pairing_count < sync_vector_size:
            remaining_indices = torch.from_numpy(
                self.rnd_seed.choice(self.num_neurons, size=(sync_vector_size - self.self_pairing_count), replace=False)
            )
            second_indices[self.self_pairing_count:] = remaining_indices
        
        return first_indices, second_indices
    
    #computes post activation products of neuron pairs
    def _compute_neuron_sync(self, post_activations, sync_type):
        if sync_type == 'pred':
            first_indices = self.pred_first_indices.to(post_activations.device) #ensure indices are on same device as post activations
            second_indices = self.pred_second_indices.to(post_activations.device)
        elif sync_type == 'action':
            first_indices = self.action_first_indices.to(post_activations.device) 
            second_indices = self.action_second_indices.to(post_activations.device)
        else:
            raise ValueError(f"sync_type must be either 'pred' or 'action', got {sync_type} instead")
        
        #select neurons from post activations using the indices 
        left_neurons = post_activations[:, first_indices] #shape (batch_size, sync_vector_size)
        right_neurons = post_activations[:, second_indices]

        neuron_syncs = left_neurons * right_neurons #element-wise product, shape (batch_size, sync_vector_size)
        return neuron_syncs
    
    #exponentiates the weights for either pred or action sync vector to get exponential decay weights, then expands for batch processing
    def _exponentiate_weights(self, batch_size, sync_type, device):
        if sync_type == 'pred':
            weights = self.w_pred_sync
        elif sync_type == 'action':
            weights = self.w_action_sync
        else:
            raise ValueError(f"sync_type must be either 'pred' or 'action', got {sync_type} instead")
        
        clamped_weights = torch.clamp(weights, 0, 15) #prevent extreme values
        exp_weights = torch.exp(-clamped_weights) #exponential decay weights, shape (sync_vector_size,)

        #expand for batch processing: sync_size -> batch x sync_size
        return exp_weights.unsqueeze(0).expand(batch_size, -1).to(device)
    
#----------------------------
# public methods
#----------------------------
    #initialises the states of raw neuron synchronisation (pairwise product of post activations from first and second indices) and variable accumulators used in the exponential decay computation to reduce parameter overload in keeping post activation history
    def init_sync_states(self):
        sync_states = {
            'pred': {'neuron_sync_accumulator': None, 'beta': None},
            'action': {'neuron_sync_accumulator': None, 'beta': None}
        }

        return sync_states 
    
    #computes actual latent synchronisation vector for either pred or action
    def compute_sync_vector(self, sync_state, post_activations, sync_type, batch_size, device):
        pairwise_products = self._compute_neuron_sync(post_activations, sync_type) #get pairwise products of neuron pairs
        exp_weights = self._exponentiate_weights(batch_size, sync_type, device) #get weights
        
        state = sync_state[sync_type]
        
        #first iteration initialises accumulators
        if state['neuron_sync_accumulator'] is None:
            state['neuron_sync_accumulator'] = pairwise_products.clone() #starts as raw pairwise products
            state['beta'] = torch.ones_like(pairwise_products)
        else:
            #recurrent update with exponential temporal decay
            state['neuron_sync_accumulator'] = exp_weights * state['neuron_sync_accumulator'] + pairwise_products
            state['beta'] = exp_weights * state['beta'] + 1
        
        #compute normalised synchronisation representation
        synchronisation_vector = state['neuron_sync_accumulator'] / torch.sqrt(state['beta'])

        #shape: batch x sync_size
        return synchronisation_vector
    
#neuron level model component to compute post activations from pre-activation history for each neuron
class NeuronLevelModel(nn.Module):
    def __init__(self, num_neurons, memory_length, is_deep=False, use_layernorm=False, dropout=0.0, temperature=1.0):
        super().__init__()

        self._build(num_neurons, memory_length, is_deep, use_layernorm, dropout, temperature)

#----------------------------
# Architecture stuff
#----------------------------
    def _build(self, num_neurons, memory_length, is_deep, use_layernorm, dropout, temperature):
        if memory_length >= num_neurons:
            raise ValueError(f"memory_length: {memory_length} must be less than num_neurons: {num_neurons}")

        self.memory_length = memory_length #pre-activation history length
        self.num_neurons = num_neurons
        self.is_deep = is_deep

        #dropout and layernorm
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        #elementwise_affine introduces learnable weight and bias to the normalised output and performs a standard perceptron based calculation (weight * normalised + bias)
        self.layernorm = nn.LayerNorm(self.memory_length, elementwise_affine=True) if use_layernorm else nn.Identity()

        #learnable temperature scaling parameter
        self.register_parameter('temperature', nn.Parameter(torch.tensor(temperature)))

        if self.is_deep:
            self._build_deep_nlm() #2 layer nlm
        else:
            self._build_nlm() 

    def _build_deep_nlm(self):
        # first layer: memory_length -> 2*hidden_dim (for glu)
        self.register_parameter('w1', nn.Parameter(torch.empty(self.memory_length, 2 * 2, self.num_neurons).uniform_(-1/np.sqrt(self.memory_length + 2 * 2), 1/np.sqrt(self.memory_length + 2 * 2))))
        self.register_parameter('b1', nn.Parameter(torch.zeros(1, self.num_neurons, 2 * 2)))

        #second layer: hidden_dim -> 2 (for glu then squeeze to 1)
        self.register_parameter('w2', nn.Parameter(torch.empty(2, 2, self.num_neurons).uniform_(-1/np.sqrt(2 + 2), 1/np.sqrt(2 + 2))))
        self.register_parameter('b2', nn.Parameter(torch.zeros(1, self.num_neurons, 2)))
    
    def _build_nlm(self):
        #w1 has shape: memory_length x 2 x num_neurons, b1 has shape: 1 x num_neurons x 2
        self.register_parameter('w1', nn.Parameter(torch.empty(self.memory_length, 2, self.num_neurons).uniform_(-1/np.sqrt(self.memory_length + 2), 1/np.sqrt(self.memory_length + 2))))
        self.register_parameter('b1', nn.Parameter(torch.zeros(1, self.num_neurons, 2)))

    def _forward_deep(self, x):
        #first layer with glu activation
        out = torch.einsum('bnm,mhn->bnh', x, self.w1) + self.b1
        out = nn.functional.glu(out, dim=-1)  #splits last dim and applies gating
        
        #second layer with glu activation and squeeze
        out = torch.einsum('bnh,hrn->bnr', out, self.w2) + self.b2
        out = nn.functional.glu(out, dim=-1)  #results in single output per neuron

        post_activations = out.squeeze(-1) / torch.clamp(self.temperature, min=1e-8) #small epislon to prevent division by zero even tho its initialised as 1 
        
        return post_activations

    def _forward_shallow(self, x):
        #single layer with glu activation and squeeze
        out = torch.einsum('bnm,mrn->bnr', x, self.w1) + self.b1
        out = nn.functional.glu(out, dim=-1)  #single output per neuron
        
        post_activations = out.squeeze(-1) / self.temperature
        
        return post_activations
#----------------------------
# Forward Processing
#----------------------------
    def forward(self, pre_activation_history):
        x = self.dropout(pre_activation_history) #input shape: batch, num_neurons, memory_length
        x = self.layernorm(x) #normalise

        #output shape should be batch x num neurons -> each neuron has one post activation 
        if self.is_deep:
            return self._forward_deep(x)
        else:
            return self._forward_shallow(x)

#synapse model to compute new pre-activations from attended features
class SynapseModelUNet(nn.Module):
    def __init__(self, neurons, depth, min_width=16, dropout=0.0, bias=False):
        super().__init__()

        self._build(neurons, depth, min_width, dropout, bias)

#----------------------------
# Forward Processing
#----------------------------
    def forward(self, input):
        input_mapping = self.input_projection(input) #map raw input to num of neurons

        skip_activations = self.traverse_down(input_mapping) #down the u-net

        pre_activations = self.traverse_up(skip_activations)

        return pre_activations

    def traverse_up(self, skip_activations):
        current_activation = skip_activations[-1] #start from end/bottleneck layer
        layers = len(self.up_path)

        for layer_id in range(layers):
            reversed_layer_id = layers - 1 - layer_id #layer index backwards

            current_activation = self.up_path[reversed_layer_id](current_activation) #project in upward layer

            #add skip connection and normalise
            current_activation = self.skip_norm[reversed_layer_id](current_activation + skip_activations[reversed_layer_id])

        return current_activation #return the final outputs
    
    def traverse_down(self, input_mapping):
        #initial pass
        current_activation = input_mapping
        skip_activations = [current_activation] #keep a list of all layer activations for skip connection

        for layer in self.down_path:
            current_activation = layer(current_activation) #downsized until it reaches bottleneck
            skip_activations.append(current_activation) #store layer-wise activations for skip connection
        
        return skip_activations


#----------------------------
# Architecture stuff
#----------------------------
    def _build(self, num_neurons, depth, min_width, dropout, bias):
        self.num_neurons = num_neurons #same number as neurons to be used in CTM
        self.depth = depth
        self.min_width = min_width #smallest bottleneck
        self.dr = dropout
        self.bias = bias

        self.layer_widths = self._interpolate_width(self.num_neurons, self.depth, self.min_width) #list of neurons in each layer from top->bottom
        self.input_projection = self._input_projection_layer()

        #down, up, and skip connections
        self.down_path, self.up_path, self.skip_norm = self._build_network(self.dr)

    #returns a list of linearly interpolated number of neurons in each layer
    def _interpolate_width(self, num_neurons, depth, min_neurons):
        widths = np.linspace(num_neurons, min_neurons, depth) #start:num_neurons, end: min_neurons, number of elements:depth
        return [int(w) for w in widths]

    #maps input -> neurons to feed the network
    def _input_projection_layer(self):
        return nn.Sequential(
            nn.LazyLinear(self.layer_widths[0], bias=self.bias), #lazy infers the input dim, which will come from out concatennated input with post neuron activation
            nn.LayerNorm(self.layer_widths[0], bias=self.bias), #normalise
            nn.SiLU() #activation function - basically copying the offical ctm code
        )
    
    #reusable layer for each block
    def _create_projection(self, input_size, output_size, dr):
        return nn.Sequential(
            nn.Dropout(dr),
            nn.Linear(input_size, output_size), #single linear projection
            nn.LayerNorm(output_size), #normalise b4 silu activation
            nn.SiLU()
        )    
    
    #builds the actual u-net structures
    def _build_network(self, dropout_rate):
        down_path = nn.ModuleList() #downward layers
        up_path = nn.ModuleList() #upward layers
        skip_norm = nn.ModuleList() #normaliser for each skip connection in the upward layer

        #loop through layers
        for layer in range(len(self.layer_widths) - 1):
            #down
            down_block = self._create_projection(self.layer_widths[layer], self.layer_widths[layer + 1], dropout_rate) #using current layer as input size, and next layer as output size
            down_path.append(down_block) #append to module list

            #up
            up_block = self._create_projection(self.layer_widths[layer + 1], self.layer_widths[layer], dropout_rate) #same as down block but swap input size to be layer + 1 and output to just layer
            up_path.append(up_block)

            #skip connection normalisers
            skip_norm.append(nn.LayerNorm(self.layer_widths[layer])) #normaliser size will be same as output size of the up block 

        return down_path, up_path, skip_norm