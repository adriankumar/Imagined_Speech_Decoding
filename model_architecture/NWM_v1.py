import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np 
from model_architecture import CTM, FeatureExtractorv1, build_propagator


seed = 24573471
dropout_Rate = 0.2
raw_spatial_convs = [(16, 7, 3), (24, 5, 2), (8, 3, 1)]
spectro_convs = [(32, 8, 5), (24, 5, 3), (16, 4, 2), (8, 3, 1)]

V1_CONFIG_DF = {
    #general/global args
    'seed': seed,
    'max_windows': 4, #this is kept fixed for training, but when training on better, real-time accomodated data, this will have to be experimented with
    'eeg_channels': 122,
    'segment_length': 500,
    'sfreq': 500,

    #feature extractor args -- output dim doesn't have to be the same since they get concat but for now keep consistent
    'sinc_output_dim': 32,
    'bp_output_dim': 32,
    'sinc_filters': 64,
    'raw_spatial_fpf': 4,
    'spectro_fpf': 4,
    'raw_spatial_convs': raw_spatial_convs,
    'spectro_convs': spectro_convs,

    #f-exct activation and dropout
    'sinc_activation': 'silu',
    'raw_spatial_activation': 'silu',
    'spectro_activation': 'silu',
    'dropout': dropout_Rate,

    #ltc
    'input_mapping': 'affine', #affine or linear are the valid options
    'output_mapping': 'affine',
    'ode_unfolds': 6, #internal ltc approximation steps for differential
    'epsilon': 1e-6, 

    #for cognitive layer/ctm
    'num_neurons': 37, #number of neurons; note must be higher than action and pred sync vector sizes
    'memory_length': 24, #pre-activation history length for each individual neuron
    'pred_sync_vector': 28, #size of latent representation used to project predictions
    'action_sync_vector': 28, #also size of latent representation used during input in the attention module
    'self_pairing_count': 16, #number of neurons that will pair with themselves when computing the neural 'synchronisation' between two neuron pairs
    'thinking_steps': 16, #internal thinking/reasoning steps per forward pass
    'output_dim': 64, #output dimension 
    'attention_size': 64, #attention size will be set automatically
    'attention_heads': 8,
    'dropout_attention': 0.2, #dropout rate for attention module
    'unet_depth': 6, #depth of unet used for synapse model
    'min_unet_width': 16, #final compression size in unet before upwards projection
    'dropout_synapse': 0.2, #dropout rate for synapse model
    'synapse_bias': True, #use bias in synapse model
    'use_deep_nlm': True, #whether to use a deep NLM (2 layer MLP) or a shallow NLM (1 layer linear)
    'use_layer_norm': True, #whether to normalise input (pre-activation history) before passing into MLPs inside NLM
    'dropout_NLM': 0.2, #dropout rate for NLM
    'temperature': 1.0, #starting temperature for computing post activations in NLM; is a learnable parameter so it will change
    

#uses same mapping, ode unfolds, just diff neurons
    #for neural wire circuit used in LTC/motor policy propagator
    'm_R1': 16,
    'm_R2': 8,
    'm_R3': 3,
    #number of neurons connected to neighbouring region
    'm_input_fanout': 12,
    'm_R1_fanout': 8,
    'm_R2_fanout': 4,
    'm_recurrent_connections': 8,

    #add policy params
    'embedding_size': 64, #word embedding size

}

#no hjorth, no sensory prop, different buffer accumulation
class NeuralWorldModelv1(nn.Module):
    def __init__(self, config=None, vocab_list=None, vocab_embedding=None, **kwargs):
        super(NeuralWorldModelv1, self).__init__()

        if vocab_list is None or vocab_embedding is None:
            raise ValueError(f"Must provide a vocab list/embedding to initialise | vocab_list: {vocab_list if vocab_list is None else 'provided'} | vocab_embedding: {vocab_embedding if vocab_embedding is None else 'provided'}")

        self.vocab_list = vocab_list
        self.vocab_embedding = nn.Parameter(vocab_embedding, requires_grad=False)

        self._initialise_config(config, **kwargs)
        self._initialise_network()

        self.state_buffer = None #accumulates window information in fixed size, and used to create an 'avg' as final state t window for final reasoning
        self.window_count = 0.0

    def _initialise_config(self, config=None, **kwargs):
        self.config = V1_CONFIG_DF.copy() #provide default case
        if config is not None:
            self.config.update(config) #update default config custom config if any provided
        self.config.update(kwargs) #then update config with any keyword args provided
    

    def _initialise_network(self):
        self.seed = self.config['seed']
        self.max_windows = self.config['max_windows']
        self.eeg_chans = self.config['eeg_channels']
        self.seg_len = self.config['segment_length']
        self.sfreq = self.config['sfreq']

        #init feature exteactors - prepared state_t for world model
        self.feature_extractor = FeatureExtractorv1(
            eeg_channels=self.eeg_chans, segment_length=self.seg_len, sfreq=self.sfreq,
            sinc_output_dim=self.config['sinc_output_dim'],
            bp_output_dim=self.config['bp_output_dim'], sinc_filters=self.config['sinc_filters'],
            raw_spatial_fpf=self.config['raw_spatial_fpf'], spectro_fpf=self.config['spectro_fpf'],
            raw_spatial_convs=self.config['raw_spatial_convs'], spectro_convs=self.config['spectro_convs'],
            sinc_activation=self.config['sinc_activation'], raw_spatial_activation=self.config['raw_spatial_activation'],
            spectro_activation=self.config['spectro_activation'], dropout=self.config['dropout']
        )

        #helper func for getting state_t dim
        self.state_t_dim = self._get_state_t_dim()

        #cognitive layer
        self.cognitive_layer = CTM(num_neurons=self.config['num_neurons'],
                                   memory_length=self.config['memory_length'],
                                   pred_sync_vector_size=self.config['pred_sync_vector'],
                                   action_sync_vector_size=self.config['action_sync_vector'],
                                   self_pairing_count=self.config['self_pairing_count'],
                                   thinking_steps=self.config['thinking_steps'],
                                   output_dim=self.config['output_dim'],
                                   seed=self.seed,
                                   attention_size=self.config['attention_size'],
                                   attention_heads=self._get_divisble_attention_head(self.config['attention_size'], 
                                                                                     self.config['attention_heads']),
                                   dropout_attention=self.config['dropout_attention'],
                                   unet_depth=self.config['unet_depth'],
                                   min_unet_width=self.config['min_unet_width'],
                                   dropout_synapse=self.config['dropout_synapse'],
                                   synapse_bias=self.config['synapse_bias'],
                                   use_deep_nlm=self.config['use_deep_nlm'],
                                   use_layer_norm=self.config['use_layer_norm'],
                                   dropout_NLM=self.config['dropout_NLM'],
                                   temperature=self.config['temperature'])

        
        self._build_motor_policy() #build motor policy propagator -- action

    def _get_state_t_dim(self):
        raw_spatial_output_dim = self.config['raw_spatial_convs'][-1][0] * self.config['raw_spatial_fpf']
        spectro_output_dim = self.config['spectro_convs'][-1][0] * self.config['spectro_fpf']
        dim = self.config['sinc_output_dim'] + self.config['bp_output_dim']
        total = dim + raw_spatial_output_dim + spectro_output_dim
        return total

    
    def _get_divisble_attention_head(self, attention_size, attention_heads):
        for h in reversed(range(1, attention_heads + 1)):
            if attention_size % h == 0:
                # print(f"cognitive attention head num is: {h}")
                return h 
        
        return 1 #otherwise just use 1

    def _build_motor_policy(self):
        #build motor policy propagator
        self.policy_prop = build_propagator(
            r1=self.config['m_R1'], r2=self.config['m_R2'], r3=self.config['m_R3'], 
            in_fanout=self.config['m_input_fanout'], r1_fanout=self.config['m_R1_fanout'], 
            r2_fanout=self.config['m_R2_fanout'], recurrent=self.config['m_recurrent_connections'], seed=self.seed,
            input_dim=self.config['output_dim'], input_mapping=self.config['input_mapping'], 
            output_mapping=self.config['output_mapping'], ode_unfolds=self.config['ode_unfolds'], 
            epsilon=self.config['epsilon'], project_output=False
        )

        #layernorm at the end of these projections? (also layernorm before)
        self.mean_projector = nn.Linear(in_features=self.policy_prop.wire.internal_neurons + self.config['output_dim'],
                                        out_features=self.config['embedding_size'])
        
        # nn.init.uniform_(self.mean_projector.weight, -0.01, 0.01)
        # nn.init.zeros_(self.mean_projector.bias)
        self.init_policy_mu()

        self.variance_projector = nn.Linear(in_features=self.policy_prop.wire.internal_neurons + self.config['output_dim'],
                                            out_features=self.config['embedding_size'])
        
        #however since we pre-train, we need to reinitialise this so we'll just use ahelper function
        nn.init.xavier_uniform_(self.variance_projector.weight, gain=0.01)
        nn.init.constant_(self.variance_projector.bias, 0.0)

        # self.log_prob_weight = nn.Parameter(torch.tensor(0.0)) #learnable weight to project log prob into something that helps with objective function for SAC

    def init_policy_mu(self):
        with torch.no_grad():
            nn.init.uniform_(self.mean_projector.weight, -0.01, 0.01)
            
            # Sample single random vocab embedding for bias
            pad_idx = self.vocab_list.index('<PAD>')
            valid_indices = [i for i in range(len(self.vocab_list)) if i != pad_idx]
            sampled_idx = valid_indices[torch.randint(0, len(valid_indices), (1,)).item()]
            
            bias_init = self.vocab_embedding[sampled_idx]  # (embedding_dim,)
            self.mean_projector.bias.data.copy_(bias_init)

#----------------------------------------------------------------
# Isolated forward pass functions for each module of the network
#----------------------------------------------------------------
    def extract_features(self, x): 
        #assumes x is in shape batch x chans x seg_length
        state_t = self.feature_extractor(x)
        self._update_buffer(state_t) #accumulate
        self.window_count += 1

        return state_t #batch x state_t_dim

    #happens internally
    def _update_buffer(self, state_t):
        if self.state_buffer is None :
            #initialise 
            batch_size = state_t.shape[0]
            self.state_buffer = torch.zeros(batch_size, state_t.shape[-1], device=state_t.device)
            self.state_buffer += state_t #additive
        else: 
            self.state_buffer += state_t

    #computes mean of state input across windows and resets buffer variable; call this when ready to final reason
    def get_final_state(self):
        final_state = self.state_buffer / self.window_count 
        self.state_buffer = None
        self.window_count = 0.0
        return final_state #batch x state_t_dim

    #buffer cannot be none here
    def think(self, state_t, cognitive_states=None, final_reasoning_length=None, prev_output=None):
        #concate previous prediction output
        if prev_output is None:
            prev_output = torch.zeros(state_t.shape[0], self.config['embedding_size'], device=state_t.device) #shape b x 1 x embedding dim
        else: 
            prev_output = prev_output.mean(dim=-1)#in shape batch x embed x thouhgt_steps, mean across thought steps to get shape batch x embed
            
        state_t = torch.cat([state_t, prev_output], dim=-1).unsqueeze(1) #concat across feature dim shape is batch x 1 x feature_dim for attetion

        #create attetion mask; no attention mask but keep compatible with ctm architecture
        attention_mask = None

        cognitive_signals, cognitive_states = self.cognitive_layer(
            input_features=state_t, neural_states=cognitive_states, attention_mask=attention_mask, final_reasoning_length=final_reasoning_length)

        #signals has shape batch x output_dim x thought_steps
        #cognitive states is a dictionary
        #normalise cognitive signals across feature dim only to treat each thought step as 
        #an independent 'cognitive' evolution
        cognitive_signals = cognitive_signals.permute(0, 2, 1) #permute to b x t x output_dim
        cognitive_signals = F.silu(cognitive_signals) #apply activation
        cognitive_signals = F.layer_norm(cognitive_signals, [cognitive_signals.shape[-1]]) #across feature dim
        cognitive_signals = cognitive_signals.permute(0, 2, 1) #permute back to b x output_dim x t
        return cognitive_signals, cognitive_states 

    def propagate_action(self, x, motor_policy_state=None, return_prop=False):
        #cognitive signals already normalised
        length = x.shape[-1] #shape is batch x cog_signals x thought_steps

        motor_history = []
        for t in range(length):
            x_t = x[:, :, t] #batch x signals, maybe add skip connection here for propagator
            _, motor_policy_state = self.policy_prop(x_t, motor_policy_state)
            motor_history.append(motor_policy_state) #append as raw
        
        motor_prop = torch.stack(motor_history, dim=1) #b x thought_steps x neurons
        motor_prop = F.silu(motor_prop) #apply activation before normalisation?
        motor_prop = F.layer_norm(motor_prop, [motor_prop.shape[-1]]) #across feature/neuron dim

        mus = [] #means
        log_sigma = [] #log std

        for t in range(length): #over same thought-steps length
            motor_signal = motor_prop[:, t, :] #batch x neurons #or maybe add skip connection here
            motor_signal = torch.cat([motor_signal, x[:, :, t]], dim=-1) #concate, motor and cognitive signals across feature dim 

            mu = self.mean_projector(motor_signal) #batch x embedding dim
            log_std = self.variance_projector(motor_signal) #batch x embedding dim
            # log_std = torch.clamp(log_std, min=-4.0, max=1.0) #clamp

            mus.append(mu)
            log_sigma.append(log_std)
        
        #reshape back into batch x embedding dim x thought _steps
        mu = torch.stack(mus, dim=2)
        log_sigma = torch.stack(log_sigma, dim=2)

        if return_prop:
            return mu, log_sigma, motor_policy_state, motor_prop

        return mu, log_sigma, motor_policy_state

    #treat each thought step as an action
    def sample_action(self, mu, log_sigma):
        #input shapes for both batch x embedding dim x thought_steps
        sigma = torch.exp(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)

        #reparameterised sample
        action = dist.rsample()
        log_probs = dist.log_prob(action).sum(dim=1) #b x thought_steps

        #mean over thought steps to get single scalar per batch
        # log_prob = log_probs.mean(dim=-1, keepdim=True) #b x 1
        log_prob = log_probs.sum(dim=-1, keepdim=True) #b x 1; use cumulative sum of log probs instead 
        #to properly represent action dimenstionality to scale

        #learnable log prob scaling 
        # log_prob = log_prob * F.softplus(self.log_prob_weight)

        return action, log_prob #should be negative

    def decode_vocab_ids(self, mu, return_confidences=True):
        #noramlise across embedding dim; note mu can also be sampled action
        mu = mu.transpose(dim0=1, dim1=2) #b x t x e

        mu_norm = F.normalize(mu, p=2, dim=-1) #norm across embedding dim
        vocab_norm = F.normalize(self.vocab_embedding, p=2, dim=-1)
        
        #b x thought_step x similarity
        similarity = torch.matmul(
            mu_norm, #b x thoughtsteps x embedding_dim
            vocab_norm.T #embedding_dim x vocab size
        )

        #get word indices; batch x thought_steps
        word_ids = similarity.argmax(dim=-1) #get highest similarity per thought_step

        if return_confidences:
            confidences = similarity.max(dim=-1)[0] #batch x thought_steps
            avg_confidence = confidences.mean(dim=-1, keepdim=True) / self.config['thinking_steps']
            return word_ids, confidences, avg_confidence #batch x thought_steps 
        
        return word_ids #b x thought_steps

    #only used during experience replay and inference where batch dim = 1
    def construct_sentence(self, word_ids):
        word_ids = word_ids.squeeze(0) #thought_steps
        sentence = []

        for id in word_ids:
            word = self.vocab_list[id.item()]
            if word != '<PAD>': #exlcude any padding outputs from sentence generation
                sentence.append(word)
        
        sentence = ' '.join(sentence)
        return sentence


    def print_parameter_count(self):
        total = 0
        #first print feature extractor params
        feature_extractor_total = self.feature_extractor.print_parameter_count()
        print('-----------------------------------------------------------')
        cog_params = sum(p.numel() for p in self.cognitive_layer.parameters())
        total += cog_params
        policy_prop_params = sum(p.numel() for p in self.policy_prop.parameters())
        total += policy_prop_params
        policy_mu_params = sum(p.numel() for p in self.mean_projector.parameters())
        total += policy_mu_params
        policy_var_params = sum(p.numel() for p in self.variance_projector.parameters())
        total += policy_var_params
        vocab_params = self.vocab_embedding.numel()
        total += vocab_params
        

        print(f"Total Parameter count for Model: {total}")
        print(f"cognitive parameters           : {cog_params}")
        print(f"policy propagtor parameters    : {policy_prop_params}")
        print(f"policy mu projector parameters : {policy_mu_params}")
        print(f"policy var projector parameters: {policy_var_params}")
        print(f"vocab embedding parameters     : {vocab_params}")
        print('-----------------------------------------------------------')
        print(f"Entire model params            : {total + feature_extractor_total}")

        return total + feature_extractor_total

#----------------------------------------------------------------
# class properties
#----------------------------------------------------------------
    #amount of words in model's vocab
    @property
    def vocab_size(self):
        return len(self.vocab_list)
    
    @property
    def state_dim(self):
        return self.state_t_dim
    
    @property 
    def policy_neurons(self):
        return self.policy_prop.wire.internal_neurons
    
    @property
    def cognitive_dim(self):
        return self.config['output_dim']
    
    @property
    def embedding_dim(self):
        return self.config['embedding_size']

    @property
    def current_attention(self):
        return self.window_count
    
    @property
    def thought_steps(self):
        return self.cognitive_layer.thought_steps
    
    @property
    def windows_counted(self):
        return self.window_count