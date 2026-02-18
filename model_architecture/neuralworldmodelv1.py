import torch.nn as nn 
import torch.nn.functional as F
import torch 
from model_architecture import CTMv2, FEv1, MMHA, build_propagator

SEED = 24573471 
DR = 0.2 #global drop out rate for architecture
#feature extractors
RAW_CONVS = [(16, 7, 3), (24, 5, 2), (8, 3, 1)] #used on raw eeg window
SPECTRO_CONVS = [(24, 8, 5), (16, 5, 3), (12, 4, 2), (8, 3, 1)] #used on spectrogram window
PRINCIPAL_COMPONENTS = 16


CTM_ATTENTION_DEFAULT = {
    'attention_amount': 1,
    'dropout': DR,
    'final_dim': 0, #use attention dim arg in ctm
    'use_dense': False, #use raw attended output for ctm 

    'attention_configs':[
        {'name': 'ctm_attention', 
         'embed_dim': 80, #make same as action_sync_vector size
         'num_heads': 4, 
         'pattern': 'cross-attention'
         }],
}

ATT_CONFIG = {
    'attention_amount': 3,
    'dropout': 0.2,
    'final_dim': 76, #each attention output is projected into a final dim so shape is b x arbitrary_seq x final dim
    'use_dense': True,

    'attention_configs': [
        {
            'name': 'env_state',
            'embed_dim': 128, #the last dim of expected query and key value input to mha, should be same as the env state dim
            'num_heads': 4, #although handled in class, ensure its divisble by embed dim; basically splits embedding into even smaller sizes to use with mutliple attention heads, like a 'partial' attention
            'pattern': 'cross-attention' #cross attention with current average state
        },

        {
            'name': 'imagined_state',
            'embed_dim': 128, #same as env state dim 
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
    'segment_length': 500,
    'sfreq': 500,
    'dropout': DR,

    #feature extractor args
    'sinc_output_dim': 32,
    'bp_output_dim': 32,
    'sinc_filters': 64,
    'raw_spatial_fpf': 4,
    'spectro_fpf': 4,
    'raw_spatial_convs': RAW_CONVS,
    'spectro_convs': SPECTRO_CONVS,

    #f-exct activation and dropout
    'sinc_activation': 'silu',
    'raw_spatial_activation': 'leaky-relu',
    'spectro_activation': 'silu',

    #core CTM args
    'num_neurons': 80, #number of neurons the ctm uses
    'memory_length': 24, #the history/length of pre-activations from each of the n neurons to use for the NLM component in calculating the post activations for each n neuron; must be less than num_neurons
    #note the sync vector sizes also determines the number of neurons pairs that can be formed; we use random sparse pairing so not all n neurons will be used to compute synchronisation due to computational constraints
    'pred_sync_vector_size': 80, #size of latent synchronisation vector used for prediction; projects neural synchronisation into this size as latent representation; this sync vector will be decoded into the output dim (actual predictions)
    'action_sync_vector_size': 80, #size of latent synchronisation vector used for action; projects neural synchronisation into this size as latent representation; this sync vector will be used as input into the attention module 
    'self_pairing_count': 46, #number of neurons that will be paired with themselves when computing the sync vectors; ensure self_pairing_count < min(pred_sync_vector_size, action_sync_vector_size)
    'thinking_steps': 12, #number of thinking steps to perform for 1 input; each step consists of attention -> synapse model -> NLM -> Synchronisation
    'ctm_pred_dim': 80, #final output dim that the pred sync vector is decoded into i.e nn.Linear(pred_sync_vector_size, output_dim) projecting sync to prediction outputs

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


class NWMv1(nn.Module):
    def __init__(self, config=None, vocab_list=None, vocab_embedding=None, **kwargs):
        super(NWMv1, self).__init__()

        if vocab_list is None or vocab_embedding is None:
            raise ValueError(f"Must provide a vocab list/embedding to initialise | vocab_list: {vocab_list if vocab_list is None else 'provided'} | vocab_embedding: {vocab_embedding if vocab_embedding is None else 'provided'}")

        self.vocab_list = vocab_list
        self.vocab_embedding = nn.Parameter(vocab_embedding, requires_grad=False)
        
        #sensory state stuff
        self.s_types = ['env', 'img', 'semantic'] #skipping sentence 
        self.env_state = None
        self.img_state = None
        self.semantic_state = None
        self.window_count = 0
        self.dream_count = 0
        self.state_t = None

        self._initialise_config(config, **kwargs)
        self._initialise_network()

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

        #init feature extractor
        self.feature_extractor = FEv1(
            eeg_channels=self.eeg_chans, segment_length=self.seg_len, sfreq=self.sfreq,
            sinc_output_dim=self.config['sinc_output_dim'],
            bp_output_dim=self.config['bp_output_dim'], sinc_filters=self.config['sinc_filters'],
            raw_spatial_fpf=self.config['raw_spatial_fpf'], spectro_fpf=self.config['spectro_fpf'],
            raw_spatial_convs=self.config['raw_spatial_convs'], spectro_convs=self.config['spectro_convs'],
            sinc_activation=self.config['sinc_activation'], raw_spatial_activation=self.config['raw_spatial_activation'],
            spectro_activation=self.config['spectro_activation'], dropout=self.config['dropout']

        )

        self.env_state_dim = self._get_env_state_dim()
        self.img_state_dim = self.env_state_dim

        self.register_parameter('pre_env_states', nn.Parameter(torch.zeros(1, self.env_state_dim)))
        self.register_parameter('pre_semantic_state', nn.Parameter(torch.zeros(self.config['thinking_steps'], (self.config['semantic_components'] * 2))))

        #to do: add state t management and attention heads here
        self.attention_heads = MMHA(
            config=self.config['attention_heads']
        )

        self.congnitive_layer = CTMv2(
            num_neurons=self.config['num_neurons'],
            memory_length=self.config['memory_length'],
            pred_sync_vector_size=self.config['pred_sync_vector_size'],
            action_sync_vector_size=self.config['action_sync_vector_size'],
            self_pairing_count=self.config['self_pairing_count'],
            thinking_steps=self.config['thinking_steps'],
            output_dim=self.config['ctm_pred_dim'],
            seed=self.seed,
            attention=self.config['ctm_attention'],
            unet_depth=self.config['unet_depth'],
            min_unet_width=self.config['min_unet_width'],
            dropout_synapse=self.config['dropout'],
            synapse_bias=self.config['synapse_bias'],
            use_deep_nlm=self.config['use_deep_nlm'],
            use_layer_norm=self.config['use_layer_norm'],
            dropout_NLM=self.config['dropout'],
            temperature=self.config['temperature']
        )

        #build output layers
        self._build_motor_action()
        self._build_motor_world()
        self._compute_semantic_subspace()

    def _compute_semantic_subspace(self):
        #compute svd decomposition of vocabulary embeddings to find principal semantic axes
        #centre embeddings before svd
        vocab_mean = self.vocab_embedding.mean(dim=0, keepdim=True) #1 x embedding dim
        centered_vocab = self.vocab_embedding - vocab_mean #vocab size x embedding dim

        #perform svd centred_vocab = U @ diag(S) @ Vt
        U, S, Vt = torch.linalg.svd(centered_vocab, full_matrices=False)

        #keep top k components 
        k = self.config['semantic_components']
        Vt_k = Vt[:k, :]  # k x embedding_dim (right singular vectors)
        S_k = S[:k] #k singular values

        #register as buffers so they move with model but aren't trained
        self.register_buffer('vocab_mean', vocab_mean) #1 x embedding_dim
        self.register_buffer('Vt_k', Vt_k) #k x embedding_dim
        self.register_buffer('S_k', S_k) #k
        self.register_buffer('vocab_centered', centered_vocab) #vocab_size x embedding_dim

    def _init_pre_env_states(self, batch_size, device):
        pre_env_state = self.pre_env_states.unsqueeze(0).expand(batch_size, -1, -1).to(device).clone() #expand to shape b x 1 x state_dim
        return pre_env_state

    def _init_pre_semantic_state(self, batch_size, device):
        pre_semantic_state = self.pre_semantic_state.unsqueeze(0).expand(batch_size, -1, -1).to(device).clone()
        return pre_semantic_state

    def _get_env_state_dim(self):
        #for convolution heads, its the last number of filters in the conv list (filter, kernel, stride) * fpf (features per filter)
        raw_spatial_output_dim = self.config['raw_spatial_convs'][-1][0] * self.config['raw_spatial_fpf']
        spectro_output_dim = self.config['spectro_convs'][-1][0] * self.config['spectro_fpf']

        #sinc and band power output dims
        dim = self.config['sinc_output_dim'] + self.config['bp_output_dim']
        total = dim + raw_spatial_output_dim + spectro_output_dim
        return total #vector dim for env state, not state_t
    
    def _build_motor_action(self):
        self.action_prop = build_propagator(
            r1=self.config['a_R1'], r2=self.config['a_R2'], r3=self.config['a_R3'], 
            in_fanout=self.config['a_input_fanout'], r1_fanout=self.config['a_R1_fanout'], 
            r2_fanout=self.config['a_R2_fanout'], recurrent=self.config['a_recurrent_connections'], seed=self.seed,
            input_dim=self.config['ctm_pred_dim'], input_mapping=self.config['input_mapping'], 
            output_mapping=self.config['output_mapping'], ode_unfolds=self.config['ode_unfolds'], 
            epsilon=self.config['epsilon'], project_output=False  
        )

        #policy outputs coefficients in principal semantic space
        k = self.config['semantic_components']
        self.coef_projector = nn.LazyLinear(out_features=k, bias=True) #mean coeffs
        self.variance_projector = nn.LazyLinear(out_features=k, bias=True) #variance over coeffs


    def _build_motor_world(self):
        self.world_prop = build_propagator(
            r1=self.config['w_R1'], r2=self.config['w_R2'], r3=self.config['w_R3'], 
            in_fanout=self.config['w_input_fanout'], r1_fanout=self.config['w_R1_fanout'], 
            r2_fanout=self.config['w_R2_fanout'], recurrent=self.config['w_recurrent_connections'], seed=self.seed,
            input_dim=self.config['ctm_pred_dim'], input_mapping=self.config['input_mapping'], 
            output_mapping=self.config['output_mapping'], ode_unfolds=self.config['ode_unfolds'], 
            epsilon=self.config['epsilon'], project_output=False
        )

        self.num_mixtures = self.config['num_mixtures']
        input_dim = (self.world_prop.wire.internal_neurons + self.config['ctm_pred_dim']) #we will average both thought steps to project into img state vector 
        output_dim = self.num_mixtures * self.config['latent_world_size']

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

        self.next_state_projector = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=self.config['latent_world_size'], out_features=self.env_state_dim, bias=True
                ),
            nn.LeakyReLU(negative_slope=0.2)
            )

    def init_pre_states(self, batch_size, device):
        #initialise all state buffers if they don't exist
        #called once at start of forward, then all states guaranteed to exist
        if self.env_state is None:
            self.env_state = self._init_pre_env_states(batch_size, device)
        if self.img_state is None:
            self.img_state = self._init_pre_env_states(batch_size, device)
        if self.semantic_state is None:
            self.semantic_state = self._init_pre_semantic_state(batch_size, device)

    def extract_features(self, window):
        #assume window is shape b x chans x seg_length
        env_state = self.feature_extractor(window)
        env_state = env_state.unsqueeze(1) #shape b x 1 x state_dim for attention module

        return env_state

    #assume state and count are model.env_state or model.img_state etc, they are self variables
    def get_final_state(self, state, count):
        final_state = state / count 
        state, count = None, 0.0 #reset count and accumulated state, even if its dreamed
        return final_state


    def update_input_activity(self, state, s_type, count_dream=False):
        if s_type not in self.s_types:
            raise ValueError(f"s_type {s_type} not in {' '.join(self.s_types)}")

        #if env or img state
        if s_type == self.s_types[0] or s_type == self.s_types[1]:

            #if env state 
            if s_type == self.s_types[0]:
                self.env_state += state #accumulate
                
                if count_dream:
                    self.dream_count += 1
                else:
                    self.window_count += 1

            #else img state
            else:
                self.img_state = state #dont accumulate
        
        #if semantic state
        elif s_type == self.s_types[-1]:
            self.semantic_state += state #accumulate

    def prepare_sensory_inputs(self, env_state, semantic_state=None, is_dreaming=False):
        b_size = env_state.shape[0] #b x 1 x state dim
        self.init_pre_states(batch_size=b_size, device=env_state.device) #if states are none it will init, otherwise it will do nothing 
        
        #env state always valid (current input)
        env_input = env_state #b x 1 x state_dim
        self.update_input_activity(state=env_input, s_type='env', count_dream=is_dreaming)
        
        #img state from buffer - assume we use update input activity after next state pred 
        img_input = self.img_state #b x 1 x state_dim
        
        #semantic state - combine mu and variance; assume automatically passed in list as [mu, var]
        if semantic_state is None:
            semantic_input = self.semantic_state #pre initialised
        else:
            mu_t = semantic_state[0].transpose(1, 2) #b x thought_steps x k
            var_t = semantic_state[-1].transpose(1, 2) #b x thought_steps x k
            semantic_input = torch.cat([mu_t, var_t], dim=-1) #b x thought_steps x 2k
            self.update_input_activity(state=semantic_input, s_type='semantic') #accumulate
        
        return [env_input, img_input, semantic_input]

    #assume state_inputs is a list; assume shapes handled externally
    def sensory_attention(self, state_inputs):
        #assumes accumulated states exist via _ensure_states_initialised
        env_state = state_inputs[0] #b x 1 x state_dim
        img_state = state_inputs[1] #b x 1 x state_dim
        semantic_state = state_inputs[2] #b x thought_steps x 2k
        
        #use accumulated states directly (guaranteed to exist)
        queries = [env_state, img_state, semantic_state] #current states
        #accumulated states (except for env state) for keyvalue in attention
        keys = [self.env_state, env_state, self.semantic_state]
        values = [self.env_state, self.env_state, self.semantic_state]
        
        attended_output = self.attention_heads(
            queries=queries,
            keys=keys,
            values=values
        )

        self.state_t = attended_output
        
        return attended_output #pass as input features into ctm attention, which treats this as KV for ctm attention
    

    def think(self, kv, cognitive_states=None, final_reasoning_length=None):

        thinking_signals, cognitive_states = self.congnitive_layer(
            input_features=kv, #attended output passed as keyvalue for input into attention
            neural_states=cognitive_states,
            final_reasoning_length=final_reasoning_length
        )

        #signals has shape batch x outputdim x thought_steps,
        #states is a dict of pre activation history, post activation and sync parameters
        #return raw thinking signals, since ctm should be self-governing, no need for activation or layer norm
        return thinking_signals, cognitive_states
    
    def propagate_action(self, thinking_signals, motor_state=None, return_prop=False):
        #thinking signals in shape batch x dim x thought_steps
        thought_steps = thinking_signals.shape[-1]

        motor_prop = []

        #get latent motor outputs
        for t in range(thought_steps):
            x_t = thinking_signals[:, :, t] #batch x dim
            _, motor_state = self.action_prop(x_t, motor_state)
            motor_prop.append(motor_state) #append hidden state as raw
        
        motor_prop = torch.stack(motor_prop, dim=-1) #batch x neurons x thought_steps
        latent_action = torch.cat([thinking_signals, motor_prop], dim=1) #concat across feature dim batch x (thinking_signal + neurons) x thought_steps
        latent_action = F.layer_norm(latent_action, [latent_action.shape[-1]])

        #guassian output parameters for each thought step
        mus = []
        log_stds = []

        for t in range(thought_steps):
            latent_t = latent_action[:, :, t] #batch x feature dim
            mu = self.coef_projector(latent_t) #b x k
            log_std = self.variance_projector(latent_t) #b x k

            mus.append(mu)
            log_stds.append(log_std)
        
        latent_mus = torch.stack(mus, dim=-1) #batch x k x thought_steps
        log_sig = torch.stack(log_stds, dim=-1) #b x k x thought_steps

        if return_prop:
            return [latent_mus, log_sig], motor_state, motor_prop

        return [latent_mus, log_sig], motor_state

    def sample_action(self, semantic_state):
        #input shapes for both batch x k x thought_steps (coeffs in semantic space)
        mu = semantic_state[0]
        log_sigma = semantic_state[1]

        sigma = torch.exp(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)

        #reparameteristation trick for SAC 
        coeffs = dist.rsample() #batch x k x thought_steps
        log_probs = dist.log_prob(coeffs).sum(dim=1) #batch x thought_steps
        log_prob = log_probs.sum(dim=-1, keepdim=True) #batch x 1

        return coeffs, log_prob
    
    def decode_vocab_ids(self, coefficients, return_confidences=True):
        #coefficients shape: batch x k x thought_steps
        #reconstruct embeddings from principal components
        batch_size = coefficients.shape[0]
        thought_steps = coefficients.shape[-1]
        
        #transpose to batch x thought_steps x k for matmul
        coeffs = coefficients.transpose(1, 2) #batch x thought_steps x k
        
        #weight by singular values and project through Vt_k
        weighted_coeffs = coeffs * self.S_k.unsqueeze(0).unsqueeze(0) #batch x thought_steps x k
        reconstructed = torch.matmul(weighted_coeffs, self.Vt_k) #batch x thought_steps x embedding_dim
        reconstructed = reconstructed + self.vocab_mean #add back mean
        
        #find nearest neighbor in vocabulary via cosine similarity
        recon_norm = F.normalize(reconstructed, p=2, dim=-1) #batch x thought_steps x embedding_dim
        vocab_norm = F.normalize(self.vocab_embedding, p=2, dim=-1) #vocab_size x embedding_dim
        
        #compute similarities
        similarities = torch.matmul(recon_norm, vocab_norm.T) #batch x thought_steps x vocab_size
        
        #get word indices
        word_ids = similarities.argmax(dim=-1) #batch x thought_steps
        
        if return_confidences:
            confidences = similarities.max(dim=-1)[0] #batch x thought_steps
            avg_confidence = confidences.mean(dim=-1, keepdim=True) / thought_steps
            return word_ids, confidences, avg_confidence
        
        return word_ids
    
    def construct_sentence(self, word_ids):
        #only used during experience replay and inference where batch dim = 1
        word_ids = word_ids.squeeze(0) #thought_steps
        sentence = []

        for id in word_ids:
            word = self.vocab_list[id.item()]
            sentence.append(word)
        
        sentence = ' '.join(sentence)

        return sentence
    
    def predict_next_state(self, thinking_signals, world_state=None, temp=1.0):
        #cognitive signals already normalised (b x cog_signals x thoughtsteps)
        thought_steps = thinking_signals.shape[-1]

        prop = []
        for t in range(thought_steps):
            x_t = thinking_signals[:, :, t] #batch x dim
            _, world_state = self.world_prop(x_t, world_state)
            prop.append(world_state) #append raw

        prop = torch.stack(prop, dim=1) #b x thought_steps x neurons

        #compute averages
        thinking_avg = thinking_signals.mean(dim=-1) #b x signals
        prop_avg = prop.mean(dim=1) #b x neurons

        #shape batch x (signals + neurons)
        latent_world = torch.cat([thinking_avg, prop_avg], dim=-1) #across feature dim
        latent_world = F.layer_norm(latent_world, [latent_world.shape[-1]]) #norm across feature


        #project mixture parameters
        latent_mu = self.mdn_mu(latent_world) #batch x num_mixtures*latent_world_size
        latent_log_sigma = self.mdn_log_sig(latent_world) #batch x num_mixtures*latent_world_size
        latent_pi_logits = self.mdn_pi(latent_world) #batch x num_mixtures

        #reshape mu and log_sigma
        batch_size = latent_mu.shape[0]
        mu = latent_mu.view(batch_size, self.num_mixtures, self.config['latent_world_size'])  # [batch, K, latent world size]
        log_sigma = latent_log_sigma.view(batch_size, self.num_mixtures, self.config['latent_world_size'])
        
        #mu: no activation (can be any value)
        #log_sigma: clamp to prevent numerical issues
        # log_sigma = torch.clamp(log_sigma, min=-10, max=2)
        sigma = torch.exp(log_sigma)  #batch, K, 72 - must be positive
        
        # pi: softmax to sum to 1
        pi = F.softmax(latent_pi_logits / temp, dim=-1)  # [batch, K]
        
        #Sample from mixture
        next_state_latent = self.sample_mdn(mu, sigma, pi)  # [batch, latent world]

        #project to actual state dim
        next_state = self.next_state_projector(next_state_latent) #shape b x env state dim
        next_state = next_state.unsqueeze(1) # b x 1 x state dim

        self.update_input_activity(state=next_state, s_type='img') #count dream is only for when we use this state as the env input

        #next state pred is shape batch x 1 x statedim (or feature_dim * 3)
        return next_state, mu, log_sigma, pi, world_state

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
    
    @property
    def vocab_size(self):
        return len(self.vocab_list)

    @property
    def current_env_state(self):
        return self.env_state #accumulated state
    
    @property 
    def current_img_state(self):
        return self.img_state #accumulated state

    @property 
    def current_semantic_state(self):
        return self.semantic_state #accumulated state

    @property
    def action_dim(self):
        return self.config['semantic_components']
    
    @property 
    def current_window_count(self):
        return self.window_count
    
    @property 
    def current_dream_count(self):
        return self.dream_count
    
    @property 
    def thinking_signal_dim(self):
        return self.config['ctm_pred_dim']
    
    def print_parameter_count(self):
        print('-----------------------------------------------------------')
        
        #call component print functions that have them
        feature_params = self.feature_extractor.print_parameter_count()
        attention_params = self.attention_heads.print_parameter_count()
        
        ctm_params = sum(p.numel() for p in self.congnitive_layer.parameters())
        #manually count remaining components
        action_prop_params = sum(p.numel() for p in self.action_prop.parameters())
        coef_proj_params = sum(p.numel() for p in self.coef_projector.parameters())
        var_proj_params = sum(p.numel() for p in self.variance_projector.parameters())
        
        world_prop_params = sum(p.numel() for p in self.world_prop.parameters())
        mdn_mu_params = sum(p.numel() for p in self.mdn_mu.parameters())
        mdn_sig_params = sum(p.numel() for p in self.mdn_log_sig.parameters())
        mdn_pi_params = sum(p.numel() for p in self.mdn_pi.parameters())
        next_state_projector_params = sum(p.numel() for p in self.next_state_projector.parameters())
        
        vocab_params = self.vocab_embedding.numel()
        pre_env_params = self.pre_env_states.numel()
        pre_semantic_params = self.pre_semantic_state.numel()
        
        #svd buffers (not trainable but count for completeness)
        buffer_params = self.vocab_mean.numel() + self.Vt_k.numel() + self.S_k.numel() + self.vocab_centered.numel()
        
        #total trainable params
        total_trainable = (feature_params + attention_params + ctm_params + 
                        action_prop_params + coef_proj_params + var_proj_params +
                        world_prop_params + mdn_mu_params + mdn_sig_params + mdn_pi_params + next_state_projector_params +
                        pre_env_params + pre_semantic_params)
        
        #total including frozen
        total_all = total_trainable + vocab_params + buffer_params
        
        print(f"Total trainable parameters   : {total_trainable:,}")
        print(f"Cognitive Layer parameters   : {ctm_params:,}")
        print(f"Action propagator parameters : {action_prop_params:,}")
        print(f"Coefficient projector params : {coef_proj_params:,}")
        print(f"Variance projector params    : {var_proj_params:,}")
        print(f"World propagator parameters  : {world_prop_params:,}")
        print(f"MDN mu parameters            : {mdn_mu_params:,}")
        print(f"MDN sigma parameters         : {mdn_sig_params:,}")
        print(f"MDN pi parameters            : {mdn_pi_params:,}")
        print(f"Next State projector params  : {next_state_projector_params:,}")
        print(f"Pre-env state parameters     : {pre_env_params:,}")
        print(f"Pre-semantic state params    : {pre_semantic_params:,}")
        print(f"Vocab embedding (frozen)     : {vocab_params:,}")
        print(f"SVD buffers (frozen)         : {buffer_params:,}")
        print('-----------------------------------------------------------')
        print(f"Total (all parameters)       : {total_all:,}")
        print('-----------------------------------------------------------')
        
        return total_trainable