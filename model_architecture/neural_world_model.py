import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np 
from model_architecture import CTM, FeatureExtractorv0, build_propagator


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
    'hjorth_output_dim': 32,
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

    #sensory propagator
    'num_prop': 4, 
    'R1_count': 12,
    'R2_count': 6,
    'R3_count': 3,
    #number of neurons connected to neighbouring region
    'input_fanout': 7,
    'R1_fanout': 5,
    'R2_fanout': 4,
    'recurrent_connections': 8,

    #for sensory propagator/ltc
    'input_mapping': 'affine', #affine or linear are the valid options
    'output_mapping': 'affine',
    'ode_unfolds': 6, #internal ltc approximation steps for differential
    'epsilon': 1e-6, 

    #for cognitive layer/ctm
    'num_neurons': 32, #number of neurons; note must be higher than action and pred sync vector sizes
    'memory_length': 16, #pre-activation history length for each individual neuron
    'pred_sync_vector': 24, #size of latent representation used to project predictions
    'action_sync_vector': 24, #also size of latent representation used during input in the attention module
    'self_pairing_count': 16, #number of neurons that will pair with themselves when computing the neural 'synchronisation' between two neuron pairs
    'thinking_steps': 16, #internal thinking/reasoning steps per forward pass
    'output_dim': 20, #output dimension 
    # 'attention_size': 250, #attention size will be set automatically
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
    'm_R1': 6,
    'm_R2': 8,
    'm_R3': 3,
    #number of neurons connected to neighbouring region
    'm_input_fanout': 4,
    'm_R1_fanout': 5,
    'm_R2_fanout': 4,
    'm_recurrent_connections': 8,

    #add policy params
    'embedding_size': 64, #word embedding size
    'num_mixtures': 2, #from https://arxiv.org/pdf/1803.10122 - used MDN for next state pred using Mixture of Guassians

    #for neural wire circuit used in LTC/motor decoder/world propagator
    'd_R1': 3,
    'd_R2': 2,
    'd_R3': 1, 
    #number of neurons connected to neighbouring region
    'd_input_fanout': 3,
    'd_R1_fanout': 2,
    'd_R2_fanout': 2,
    'd_recurrent_connections': 3,
    # 'd_project_output': True #use projection to predict next_state_t with output_dim = b x 72

}

class NeuralWorldModelv0(nn.Module):
    def __init__(self, config=None, vocab_list=None, vocab_embedding=None, **kwargs):
        super(NeuralWorldModelv0, self).__init__()

        if vocab_list is None or vocab_embedding is None:
            raise ValueError(f"Must provide a vocab list/embedding to initialise | vocab_list: {vocab_list if vocab_list is None else 'provided'} | vocab_embedding: {vocab_embedding if vocab_embedding is None else 'provided'}")

        self.vocab_list = vocab_list
        self.vocab_embedding = nn.Parameter(vocab_embedding, requires_grad=False)

        self._initialise_config(config, **kwargs)
        self._initialise_network()
        self.buffer_count = 0 #counts number of individual sensory states in buffer 

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
        self.feature_extractor = FeatureExtractorv0(
            eeg_channels=self.eeg_chans, segment_length=self.seg_len, sfreq=self.sfreq,
            hjorth_output_dim=self.config['hjorth_output_dim'], sinc_output_dim=self.config['sinc_output_dim'],
            bp_output_dim=self.config['bp_output_dim'], sinc_filters=self.config['sinc_filters'],
            raw_spatial_fpf=self.config['raw_spatial_fpf'], spectro_fpf=self.config['spectro_fpf'],
            raw_spatial_convs=self.config['raw_spatial_convs'], spectro_convs=self.config['spectro_convs'],
            sinc_activation=self.config['sinc_activation'], raw_spatial_activation=self.config['raw_spatial_activation'],
            spectro_activation=self.config['spectro_activation'], dropout=self.config['dropout']
        )

        #helper func for getting state_t dim
        self.state_t_dim = self._get_state_t_dim()

        #build sensory propagator
        self.sensory_prop_amount = self.config['num_prop'] 
        self.sensory_propagator = build_propagator(
            r1=self.config['R1_count'], r2=self.config['R2_count'], r3=self.config['R3_count'], 
            in_fanout=self.config['input_fanout'], r1_fanout=self.config['R1_fanout'], 
            r2_fanout=self.config['R2_fanout'], recurrent=self.config['recurrent_connections'], seed=self.seed,
            input_dim=self.state_t_dim, input_mapping=self.config['input_mapping'], 
            output_mapping=self.config['output_mapping'], ode_unfolds=self.config['ode_unfolds'], 
            epsilon=self.config['epsilon'], project_output=False
        )
        
        self.max_attention_capacity = self.sensory_prop_amount * self.max_windows

        #cognitive layer
        self.cognitive_layer = CTM(num_neurons=self.config['num_neurons'],
                                   memory_length=self.config['memory_length'],
                                   pred_sync_vector_size=self.config['pred_sync_vector'],
                                   action_sync_vector_size=self.config['action_sync_vector'],
                                   self_pairing_count=self.config['self_pairing_count'],
                                   thinking_steps=self.config['thinking_steps'],
                                   output_dim=self.config['output_dim'],
                                   seed=self.seed,
                                   attention_size=self.max_attention_capacity,
                                   attention_heads=self._get_divisble_attention_head(self.max_attention_capacity, 
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
        self._build_motor_decoder() #build motor decoder -- world model

    def _get_state_t_dim(self):
        raw_spatial_output_dim = self.config['raw_spatial_convs'][-1][0] * self.config['raw_spatial_fpf']
        spectro_output_dim = self.config['spectro_convs'][-1][0] * self.config['spectro_fpf']
        dim = self.config['hjorth_output_dim'] + self.config['sinc_output_dim'] + self.config['bp_output_dim']
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
        self.mean_projector = nn.Linear(in_features=self.policy_prop.wire.internal_neurons,
                                        out_features=self.config['embedding_size'])
        
        # nn.init.uniform_(self.mean_projector.weight, -0.01, 0.01)
        # nn.init.zeros_(self.mean_projector.bias)
        self.init_policy_mu()

        self.variance_projector = nn.Linear(in_features=self.policy_prop.wire.internal_neurons,
                                            out_features=self.config['embedding_size'])
        
        nn.init.uniform_(self.variance_projector.weight, -0.01, 0.01)
        nn.init.constant_(self.variance_projector.bias, -1.5)

    def init_policy_mu(self):
        with torch.no_grad():
            nn.init.uniform_(self.mean_projector.weight, -0.01, 0.01)
            
            # Sample single random vocab embedding for bias
            pad_idx = self.vocab_list.index('<PAD>')
            valid_indices = [i for i in range(len(self.vocab_list)) if i != pad_idx]
            sampled_idx = valid_indices[torch.randint(0, len(valid_indices), (1,)).item()]
            
            bias_init = self.vocab_embedding[sampled_idx]  # (embedding_dim,)
            self.mean_projector.bias.data.copy_(bias_init)

    def _build_motor_decoder(self):
        #build motor decoder propagator
        self.decoder_prop = build_propagator(
            r1=self.config['d_R1'], r2=self.config['d_R2'], r3=self.config['d_R3'], 
            in_fanout=self.config['d_input_fanout'], r1_fanout=self.config['d_R1_fanout'], 
            r2_fanout=self.config['d_R2_fanout'], recurrent=self.config['d_recurrent_connections'], seed=self.seed,
            input_dim=self.config['output_dim'], input_mapping=self.config['input_mapping'], 
            output_mapping=self.config['output_mapping'], ode_unfolds=self.config['ode_unfolds'], 
            epsilon=self.config['epsilon'], project_output=False
        )

        self.num_mixtures = self.config['num_mixtures']
        input_dim = self.decoder_prop.wire.internal_neurons * self.config['thinking_steps']
        output_dim = self.num_mixtures * self.state_t_dim

        self.mdn_mu = nn.Linear(
            in_features=input_dim,
            out_features=output_dim 
            )
        
        nn.init.xavier_uniform_(self.mdn_mu.weight, gain=0.5)
        nn.init.uniform_(self.mdn_mu.bias, -0.1, 0.1)

        self.mdn_log_sig = nn.Linear(
            in_features=input_dim, 
            out_features=output_dim
            )
        
        nn.init.uniform_(self.mdn_log_sig.weight, -0.01, 0.01)
        nn.init.constant_(self.mdn_log_sig.bias, -1.0)

        self.mdn_pi = nn.Linear(
            in_features=input_dim, 
            out_features=self.num_mixtures
            )
        
        nn.init.zeros_(self.mdn_pi.weight)
        nn.init.zeros_(self.mdn_pi.bias)

#----------------------------------------------------------------
# Isolated forward pass functions for each module of the network
#----------------------------------------------------------------
    def extract_features(self, x): 
        #assumes x is in shape batch x chans x seg_length
        state_t = self.feature_extractor(x)

        return state_t #batch x state_t_dim

    def propagate_sensory(self, x, sensory_state=None): 
        #assume x is state_t with shape batch x state_t_dim
        propagation = [] #store each timestep evolution of the sensory's hidden state

        #internal time dim/axis
        for _ in range(self.sensory_prop_amount):
            #ignoring output because it's none, use full internal neurons as features
            _, sensory_state = self.sensory_propagator(x, sensory_state) 
            propagation.append(sensory_state) #batch x sesnory_neurons; dont apply activation, keep state raw for recurrent processing
        
        #cat/stack along internal time dim
        sensory_propagation = torch.stack(propagation, dim=1) #batch x prop_amount (t) x sensory_neurons

        #dont normalise or apply activation function to propagation, do that in buffer prep
        return sensory_propagation, sensory_state

    #creates or updates buffer
    def update_buffer(self, sensory_prop=None, buffer=None):
        #if buffer is none, then assume its first forward pass, which requires sensory prop to make the buffer shape
        
        #if both are none, then thats an error
        if sensory_prop is None and buffer is None:
            raise ValueError(f"buffer cannot be none if sensory prop is also none")

        #prepare buffer if not provided (assuming first forward pass or state reset)
        if sensory_prop is not None and buffer is None:
            batch_size = sensory_prop.shape[0]
            buffer = torch.zeros(batch_size, self.max_buffer_size, self.sensory_neurons, device=sensory_prop.device)

        #we either update the buffer or we dont (final reasoning)
        if sensory_prop is not None:
            #check capacity and update or raise error
            length = sensory_prop.shape[1]
            if not self.can_accept_segment(length):
                raise ValueError("Cannot accept segment, reset buffer and buffer count via reset_buffer()")
            #update the buffer
            buffer = torch.cat([buffer, sensory_prop], dim=1)
            buffer = buffer[:, -self.max_buffer_size:, :] #slice?
            buffer = F.silu(buffer) #apply activation
            buffer = F.layer_norm(buffer, buffer.shape[1:]) # b x attention_capacity x sensory_neurons; across time and neuron dim
            self.buffer_count += length #update count
        else:
            #leave buffer as is for final reasoning where sensory prop is not provided
            pass #dont have to normalise buffer again for final reasoning, this function
        #doesnt even need to be called during final reasoning but just incase it is just leave as is dont normalise it
        #again since its already normalised and untouched from before 
        
        return buffer # b x attention_capacity x sensory_neurons; no need to apply activation again?

    def can_accept_segment(self, incoming_length):
        remaining = self.max_attention_capacity - self.buffer_count
        return remaining >= incoming_length

    #buffer cannot be none here
    def think(self, buffer, cognitive_states=None, final_reasoning_length=None):

        #create attetion mask; true where buffer is padded (zeros)
        #buffer shape: batch x max_attention_capacity x sensory neurons
        attention_mask = (buffer.abs().sum(dim=-1) == 0) #batch x max_attention_capacity

        cognitive_signals, cognitive_states = self.cognitive_layer(
            input_features=buffer, neural_states=cognitive_states, attention_mask=attention_mask, final_reasoning_length=final_reasoning_length)
    
        #signals has shape batch x output_dim x thought_steps
        #cognitive states is a dictionary
        #normalise cognitive signals across feature dim only to treat each thought step as 
        #an independent 'cognitive' evolution
        cognitive_signals = cognitive_signals.permute(0, 2, 1) #permute to b x t x output_dim
        cognitive_signals = F.silu(cognitive_signals) #apply activation
        cognitive_signals = F.layer_norm(cognitive_signals, [cognitive_signals.shape[-1]]) #across feature dim
        cognitive_signals = cognitive_signals.permute(0, 2, 1) #permute back to b x output_dim x t
        return cognitive_signals, cognitive_states, buffer #might not even need to return buffer but just doing it just incase

    def reset_buffer(self, buffer):
        buffer = None
        self.buffer_count = 0
        return buffer #in future should make buffer a self variable but for now just manage recurrently


    def propagate_action(self, x, motor_policy_state=None):
        #cognitive signals already normalised
        length = x.shape[-1] #shape is batch x cog_signals x thought_steps

        motor_history = []
        for t in range(length):
            x_t = x[:, :, t] #batch x signals
            _, motor_policy_state = self.policy_prop(x_t, motor_policy_state)
            motor_history.append(motor_policy_state) #append as raw
        
        motor_prop = torch.stack(motor_history, dim=1) #b x thought_steps x neurons
        motor_prop = F.silu(motor_prop) #apply activation before normalisation?
        motor_prop = F.layer_norm(motor_prop, [motor_prop.shape[-1]]) #across feature/neuron dim

        mus = [] #means
        log_sigma = [] #log std

        for t in range(length): #over same thought-steps length
            motor_signal = motor_prop[:, t, :] #batch x neurons
            mu = self.mean_projector(motor_signal) #batch x embedding dim
            log_std = self.variance_projector(motor_signal) #batch x embedding dim
            # log_std = torch.clamp(log_std, min=-4.0, max=1.0) #clamp

            mus.append(mu)
            log_sigma.append(log_std)
        
        #reshape back into batch x embedding dim x thought _steps
        mu = torch.stack(mus, dim=2)
        log_sigma = torch.stack(log_sigma, dim=2)

        #should i layer norm mu and log sigma?

        #during training, mu and sigma both get used for sampling, but
        #during inference, mu becomes the embedding action itself and sigma isn't used

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
        log_prob = log_probs.mean(dim=-1, keepdim=True) #b x 1
        return action, log_prob

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

    def predict_next_state(self, x, motor_decoder_state=None, temp=1.0):
        #cognitive signals already normalised (b x cog_signals x thoughtsteps)
        length = x.shape[-1]

        prop = []
        for t in range(length):
            x_t = x[:, :, t] #batch x cog_signals
            _, motor_decoder_state = self.decoder_prop(x_t, motor_decoder_state)
            prop.append(motor_decoder_state) #append raw

        prop = torch.stack(prop, dim=1) #b x thought_steps x neurons
        prop_vector = prop.view(prop.shape[0], -1) #reshape to b x thought_steps * neurons
        next_state_logits = F.silu(prop_vector) #apply activation
        next_state_logits = F.layer_norm(next_state_logits, [next_state_logits.shape[-1]])

        #project mixture parameters
        mu = self.mdn_mu(next_state_logits) #batch x num_mixtures*state_t_dim
        log_sigma = self.mdn_log_sig(next_state_logits) #batch x num_mixtures*state_t_dim
        pi_logits = self.mdn_pi(next_state_logits) #batch x num_mixtures

        #reshape mu and log_sigma
        batch_size = mu.shape[0]
        mu = mu.view(batch_size, self.num_mixtures, self.state_t_dim)  # [batch, K, 72]
        log_sigma = log_sigma.view(batch_size, self.num_mixtures, self.state_t_dim)
        
        #mu: no activation (can be any value)
        #log_sigma: clamp to prevent numerical issues
        # log_sigma = torch.clamp(log_sigma, min=-10, max=2)
        sigma = torch.exp(log_sigma)  #batch, K, 72 - must be positive
        
        # pi: softmax to sum to 1
        pi = F.softmax(pi_logits / temp, dim=-1)  # [batch, K]
        
        #Sample from mixture
        next_state = self.sample_mdn(mu, sigma, pi)  # [batch, 72]

        #next state pred is shape batch x 72 (or feature_dim * 3)
        return next_state, mu, log_sigma, pi, motor_decoder_state

    def sample_mdn(self, mu, sigma, pi):
        batch_size = mu.shape[0]
        
        #sample which mixture component to use for each batch element
        mixture_idx = torch.multinomial(pi, num_samples=1).squeeze(-1)  # [batch]
        
        #gather parameters for selected mixture
        batch_indices = torch.arange(batch_size, device=mu.device)
        selected_mu = mu[batch_indices, mixture_idx]  # [batch, state_dim]
        selected_sigma = sigma[batch_indices, mixture_idx]
        
        #sample from Gaussian
        epsilon = torch.randn_like(selected_mu)
        samples = selected_mu + selected_sigma * epsilon
        
        return samples #b x state_t_dim


    def print_parameter_count(self):
        total = 0
        #first print feature extractor params
        feature_extractor_total = self.feature_extractor.print_parameter_count()
        print('-----------------------------------------------------------')
        sensory_params = sum(p.numel() for p in self.sensory_propagator.parameters())
        total += sensory_params
        cog_params = sum(p.numel() for p in self.cognitive_layer.parameters())
        total += cog_params
        policy_prop_params = sum(p.numel() for p in self.policy_prop.parameters())
        total += policy_prop_params
        policy_mu_params = sum(p.numel() for p in self.mean_projector.parameters())
        total += policy_mu_params
        policy_var_params = sum(p.numel() for p in self.variance_projector.parameters())
        total += policy_var_params
        decoder_prop_params = sum(p.numel() for p in self.decoder_prop.parameters())
        total += decoder_prop_params
        decoder_mu_params = sum(p.numel() for p in self.mdn_mu.parameters())
        total += decoder_mu_params
        decoder_ls_params = sum(p.numel() for p in self.mdn_log_sig.parameters())
        total += decoder_ls_params
        decoder_pi_params = sum(p.numel() for p in self.mdn_pi.parameters())
        total += decoder_pi_params

        print(f"Total Parameter count for World Model: {total}")
        print(f"sensory parameters                   : {sensory_params}")
        print(f"cognitive parameters                 : {cog_params}")
        print(f"policy propagtor parameters          : {policy_prop_params}")
        print(f"policy mu projector parameters       : {policy_mu_params}")
        print(f"policy var projector parameters      : {policy_var_params}")
        print(f"decoder prop parameters              : {decoder_prop_params}")
        print(f"decoder mu projector parameters      : {decoder_mu_params}")
        print(f"decoder var projector parameters     : {decoder_ls_params}")
        print(f"decoder pi projector parameters      : {decoder_pi_params}")
        print('-----------------------------------------------------------')
        print(f"Entire model params                  : {total + feature_extractor_total}")

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
    
    #max attention capacity for buffer
    @property
    def max_buffer_size(self):
        return self.max_attention_capacity
    
    @property 
    def sensory_neurons(self):
        return self.sensory_propagator.wire.internal_neurons

    @property 
    def policy_neurons(self):
        return self.policy_prop.wire.internal_neurons

    @property
    def decoder_neurons(self):
        return self.decoder_prop.wire.internal_neurons
    
    @property
    def cognitive_dim(self):
        return self.config['output_dim']
    
    @property
    def embedding_dim(self):
        return self.config['embedding_size']

    @property
    def current_attention(self):
        return self.buffer_count