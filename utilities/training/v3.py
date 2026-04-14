import torch, random 
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np 
import os
from utilities.chisco_preprocessing import segment_eeg_tensor
from model_architecture import QNetwork
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

DEFAULT_BEST_POLICY_PATH = r"demo_weights_metrics\best_weights"
DEFAULT_BEST_Q_PATH = r"demo_weights_metrics\best_weights"
DEFAULT_BEST_WORLD_PATH = r"demo_weights_metrics\best_weights"
FREQUENCY_PATH = r"demo_weights_metrics\frequency_weights"
TRAINING_METRICS_PATH = r"demo_weights_metrics\training_metrics"

FUNCTION_WORDS = {
    # Articles
    'a', 'an', 'the',
    # Be verbs
    'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being',
    # Prepositions
    'to', 'of', 'in', 'on', 'at', 'for', 'with', 'by', 'from',
    'about', 'into', 'through', 'during', 'before', 'after',
    # Pronouns
    'it', 'its', 'this', 'that', 'these', 'those',
    'i', "i'll", "i'm", "i've", 'you', "you're", "you'll",
    'he', "he's", 'she', "she's", 'we', "we're", "we'll",
    'they', "they're", 'my', 'your', 'his', 'her', 'our', 'their',
    'me', 'him', 'us', 'them',
    # Conjunctions
    'and', 'or', 'but', 'so', 'if', 'then', 'than', 'as',
    # Auxiliaries
    'do', "don't", 'does', "doesn't", 'did', "didn't",
    'have', 'has', 'had', "haven't", "hasn't",
    'will', "won't", 'would', "wouldn't",
    'can', "can't", 'could', "couldn't",
    'should', "shouldn't", 'must', 'may', 'might',
    # Adverbs / Modifiers
    'just', 'very', 'really', 'also', 'too', 'only', 'such',
    'not', 'no', 'yes', 'there', "there's", 'here',
    # Question words
    'what', 'when', 'where', 'which', 'who', 'how', 'why',
    # Quantifiers
    'all', 'each', 'every', 'any', 'some', 'many', 'much',
}

DEFAULT_AUGMENT_CONFIG = {
    'noise_prob': 0.7, #higher, the more relaxed
    'noise_std_range': (0.02, 0.15), #range
    'scale_prob': 0.7,
    'scale_range': (0.75, 1.25),
    'dropout_prob': 0.4,
    'dropout_count_range': (2, 8)
}

#Memory-efficient batcher that fetches windows on-demand from replay buffer
class LazyWindowBatcher:
    def __init__(self, replay_buffer, batch_size, device):
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.device = device
        self.sampled_indices = None
        self.n_windows = None
        self._window_shape = None
    
    #sample episode_indices, - call once per update
    def sample_episodes(self):
        n_eps = len(self.replay_buffer)
        if self.batch_size > n_eps:
            raise ValueError(f"batch size {self.batch_size} cannot exceed number of episodes {n_eps}")
        
        self.sampled_indices = torch.randperm(n_eps)[:self.batch_size].tolist()
        
        # Compute max windows across sampled episodes
        self.n_windows = max(len(self.replay_buffer[i]['transitions']) for i in self.sampled_indices)
        
        # Cache window shape from first valid window
        for idx in self.sampled_indices:
            first_trans = self.replay_buffer[idx]['transitions'][0]
            if first_trans['window_t'] is not None:
                self._window_shape = first_trans['window_t'].shape
                break
        
        return self
    
    #fetch a batch for the current window 
    def get_window(self, window_idx):
        if self.sampled_indices is None:
            raise RuntimeError("Must call sample_episodes() before get_window()")
        
        raw_windows = []
        next_raw_windows = []
        actions = []
        rewards = []
        latent_states = []
        dones = []
        
        batch, chans, seg_len = self._window_shape
        
        for ep_idx in self.sampled_indices:
            transitions = self.replay_buffer[ep_idx]['transitions']
            
            if window_idx < len(transitions):
                trans = transitions[window_idx]
                
                if trans['window_t'] is not None:
                    raw_windows.append(trans['window_t'])
                else:
                    raw_windows.append(torch.zeros(batch, chans, seg_len, dtype=torch.float32))
                
                if trans['next_window_t'] is not None:
                    next_raw_windows.append(trans['next_window_t'])
                else:
                    next_raw_windows.append(torch.zeros(batch, chans, seg_len, dtype=torch.float32))
                
                actions.append(trans['action_t'])
                rewards.append(trans['reward_t'])
                latent_states.append(trans['latent_t'])
                dones.append(1.0 if trans['is_done'] else 0.0)
        
        window_batch = {
            'window_t': torch.cat(raw_windows, dim=0).to(self.device),
            'next_window_t': torch.cat(next_raw_windows, dim=0).to(self.device),
            'action_t': torch.cat(actions, dim=0).to(self.device),
            'reward_t': torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device),
            'latent_t': torch.cat(latent_states, dim=0).to(self.device),
            'is_done': torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        }
        
        return window_batch
    
    #compute avg reward
    def get_avg_reward(self):
        if self.sampled_indices is None:
            return 0.0
        
        total_reward = 0.0
        count = 0
        for ep_idx in self.sampled_indices:
            for trans in self.replay_buffer[ep_idx]['transitions']:
                total_reward += trans['reward_t']
                count += 1
        
        return total_reward / count if count > 0 else 0.0

#-----------------------------------------------
#Metrics tracker
#-----------------------------------------------
class TrainingMetricsTracker:
    def __init__(self):
        #episode-level metrics
        self.total_rewards = []

        #update-level losses
        self.world_losses = []
        self.Q1_losses = []
        self.Q2_losses = []
        self.policy_losses = []

        #epoch-level statistics
        self.epochs = []
        self.mean_total_reward = []
        self.std_total_reward = []
        self.mean_world_loss = []
        self.mean_q1_loss = []
        self.mean_q2_loss = []
        self.mean_policy_loss = []

        #track episode count per epoch
        self.episodes_per_epoch = []

    def store_episode_metrics(self, total):
        self.total_rewards.append(total)

    def store_losses(self, world, q1, q2, policy):
        self.world_losses.append(world)
        self.Q1_losses.append(q1)
        self.Q2_losses.append(q2)
        self.policy_losses.append(policy)

    def compute_epoch_stats(self, epoch_num, n_episodes_this_epoch):
        #slice indices for this epoch
        start_idx = sum(self.episodes_per_epoch)
        end_idx = start_idx + n_episodes_this_epoch

        #slice episode data
        epoch_total = self.total_rewards[start_idx:end_idx]

        self.epochs.append(epoch_num)

        #reward statistics
        self.mean_total_reward.append(np.mean(epoch_total))
        self.std_total_reward.append(np.std(epoch_total))

        #loss statistics
        self.mean_world_loss.append(self.world_losses[-1])
        self.mean_q1_loss.append(self.Q1_losses[-1])
        self.mean_q2_loss.append(self.Q2_losses[-1])
        self.mean_policy_loss.append(self.policy_losses[-1])

        self.episodes_per_epoch.append(n_episodes_this_epoch)

        return {
            'avg_reward': self.mean_total_reward[-1],
            'avg_world_loss': self.mean_world_loss[-1],
            'avg_q1_loss': self.mean_q1_loss[-1],
            'avg_q2_loss': self.mean_q2_loss[-1],
            'avg_policy_loss': self.mean_policy_loss[-1]
        }

    def save_metrics(self, filepath, epoch):
        filepath = os.path.join(filepath, f'epoch_{epoch+1}')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath,
                 epochs=np.array(self.epochs),
                 mean_total_reward=np.array(self.mean_total_reward),
                 std_total_reward=np.array(self.std_total_reward),
                 mean_world_loss=np.array(self.mean_world_loss),
                 mean_q1_loss=np.array(self.mean_q1_loss),
                 mean_q2_loss=np.array(self.mean_q2_loss),
                 mean_policy_loss=np.array(self.mean_policy_loss))

#global instance
metrics_tracker = TrainingMetricsTracker()


#-----------------------------------------------
#Reward computation helpers
#-----------------------------------------------
def get_content_words(words):
    content = [w for w in words if w.lower() not in FUNCTION_WORDS]
    return content if content else words  # Fallback if all are function words

def compute_timestep_reward(model, action, true_words, is_final=False):
    if is_final:
        weights = [0.5, 0.25, 0.25] #alignment, coverage, length; if final, be more strict
    else:
        weights = [0.7, 0.2, 0.1] #if not final, more forgiving rewards

    return compute_rewards(model, action, true_words, weights)

def compute_rewards(model, action, targets, reward_weights, thought_steps=16):
    #filter content words, only include main semantics in sentence
    content_words = get_content_words(targets)

    #get content word indices
    true_word_indices = [model.vocab_list.index(w) for w in content_words if w in model.vocab_list]
    n_content = len(true_word_indices)

    #edge case - if no content words were found i.e not included in vocab list then ignore
    if n_content == 0:
        return 0.0
    
    #pad targets to thought_steps with <PAD>
    pad_idx = model.vocab_list.index('<PAD>')
    n_pad = thought_steps - n_content
    padded_indices = true_word_indices + [pad_idx] * n_pad
    
    true_embeddings = model.vocab_embedding[padded_indices]  # (16, embed_dim)
    
    #alignment computed on PADDED targets (16 positions)
    alignment_reward, claim_counts = compute_alignment_reward(action, true_embeddings, thought_steps)
    
    #coverage: only count CONTENT words (first n_content positions)
    content_claims = claim_counts[:, :n_content]
    coverage = (content_claims > 0).float().sum(dim=-1) / n_content  # (batch,)
    
    #length ratio which exlcudes pads, note decode to word ids still includes <pad>
    word_ids = model.decode_vocab_ids(action, return_confidences=False).squeeze(0)  # (thought_steps,)
    pred_length = (word_ids != pad_idx).sum().float()  #count non-PAD
    length_ratio = torch.clamp(n_content / (pred_length + 1e-6), max=1.0)
    
    total_reward = (reward_weights[0] * alignment_reward + 
                    reward_weights[1] * coverage + 
                    reward_weights[2] * length_ratio)
    
    return total_reward.mean().item()

def compute_alignment_reward(pred_embeds, true_embeds, n_content):
    thought_steps = pred_embeds.shape[-1]  # 16
    
    # Normalize for cosine similarity
    pred_norm = F.normalize(pred_embeds.transpose(1, 2), dim=-1)  # b x T x embed
    true_norm = F.normalize(true_embeds, dim=-1)  # 16 x embed
    
    # Similarity: batch x pred_steps x target_positions
    sim_matrix = torch.matmul(pred_norm, true_norm.T)
    
    batch_size = sim_matrix.shape[0]
    
    # Track claims for all positions (needed for coverage calculation)
    claim_counts = torch.zeros(batch_size, thought_steps, device=pred_embeds.device)
    step_rewards = []
    
    for t in range(thought_steps):
        step_sim = sim_matrix[:, t, :]  # batch x 16 targets
        
        # bild discount: only penalide repetition on content positions
        discount = torch.ones_like(step_sim)

        if n_content > 0:
            #softened discount: 1/(1 + 0.5*count) instead of 1/(1 + count)
            content_discount = 1.0 / (1.0 + 0.5 * claim_counts[:, :n_content])
            discount[:, :n_content] = content_discount
        #PAD positions (n_content onwards) keep discount = 1.0
        
        discounted_sim = step_sim * discount
        
        #best match for this prediction step
        best_sim, best_idx = discounted_sim.max(dim=-1)
        
        #update claim counts
        claim_counts.scatter_add_(
            1, best_idx.unsqueeze(1),
            torch.ones_like(best_idx.unsqueeze(1), dtype=torch.float)
        )
        
        step_rewards.append(best_sim)
    
    alignment_reward = torch.stack(step_rewards).mean(dim=0)  # batch
    
    return alignment_reward, claim_counts

#-----------------------------------------------
#forward passes
#-----------------------------------------------
def experience_forward_pass(model, window, buffer=None, 
                 sensory_state=None, cognitive_state=None, policy_state=None, decoder_state=None,
                 sample_action=False, ignore_world=False):

    #assume window is in shape b x chans x seg_len 

    #get state_t
    state_t = model.extract_features(window)

    #propagate sensory
    raw_prop, sensory_state = model.propagate_sensory(state_t, sensory_state)
    
    #update buffer
    buffer = model.update_buffer(raw_prop, buffer)

    #think
    signals, cognitive_state, buffer = model.think(buffer, cognitive_state)
    
    #propagate deterministic action
    action, log_sigma, policy_state = model.propagate_action(signals, policy_state) #action here is mu

    #if using sampling output (for training), change action to the sampled action
    if sample_action:
        action, log_prob = model.sample_action(action, log_sigma)
    
    next_state_pred = None 

    if not ignore_world:
        #predict next state t
        next_state_pred, mdn_mu, mdn_ls, mdn_pi, decoder_state = model.predict_next_state(signals, decoder_state)

    return action, next_state_pred, signals, buffer, sensory_state, cognitive_state, policy_state, decoder_state

def experience_final_reasoning(model, buffer, cognitive_state=None, policy_state=None, final_reasoning_length=None, sample_action=False):
    #for final reasoning, its a repeat of the previous buffer input so no changes

    signals, cognitive_state, buffer = model.think(buffer, cognitive_state, final_reasoning_length=None)
    action, log_sigma, policy_state = model.propagate_action(signals, policy_state)

    if sample_action:
        action, log_prob = model.sample_action(action, log_sigma)
    
    #reset buffer after final reasoning - final reasoning is just another name for buffer is full
    _ = model.reset_buffer(buffer)

    return action, signals, cognitive_state, policy_state

#-----------------------------------------------
#Noise, amplitude scaling & channel drop out augmentation
#-----------------------------------------------
#add guassian noise with std sampled uniformly from range
def apply_gaussian_noise(eeg_tensor, std_range):
    std = random.uniform(std_range[0], std_range[1])
    noise = torch.randn_like(eeg_tensor) * std
    return eeg_tensor + noise

#scale amplitude factors sampled uniformly from range
def apply_amplitude_scaling(eeg_tensor, scale_range):
    scale = random.uniform(scale_range[0], scale_range[1])
    return eeg_tensor * scale

#zero out random channels from the eeg sequence
def apply_channel_dropout(eeg_tensor, dropout_count_range):
    n_channels = eeg_tensor.shape[0]
    n_drop = random.randint(dropout_count_range[0], dropout_count_range[1])
    n_drop = min(n_drop, n_channels - 1)  # Keep at least 1 channel
    drop_indices = random.sample(range(n_channels), n_drop)
    augmented = eeg_tensor.clone()
    augmented[drop_indices, :] = 0
    return augmented


def augment_eeg(eeg_tensor, config=None):
    if config is None:
        config = DEFAULT_AUGMENT_CONFIG
    
    augmented = eeg_tensor.clone() #channels x timepoints
    
    #Gaussian noise
    if random.random() < config['noise_prob']:
        augmented = apply_gaussian_noise(augmented, config['noise_std_range'])
    
    #Amplitude scaling
    if random.random() < config['scale_prob']:
        augmented = apply_amplitude_scaling(augmented, config['scale_range'])
    
    #Channel dropout
    if random.random() < config['dropout_prob']:
        augmented = apply_channel_dropout(augmented, config['dropout_count_range'])
    
    return augmented

#assume train_tensor is shape num_samples x channels x timepoints
#don't collect state_t here because thats only for world model training, not policy
#offline episode collection, because model 'experiences' in eval mode and not train mode
#in otherwords, the model collects experiences and learns from the replay
def episodes_rollout(model, train_tensor, id_tensor, label_mapping, n_eps, seg_len, 
                     augmentation_factor=6, augment_config=None):

    num_samples = train_tensor.shape[0]

    if n_eps > num_samples:
        raise ValueError(f"{n_eps} n_eps per epoch cannot exceed {num_samples} training samples")

    model.eval()

    # Sample n random episode indices
    indices = torch.randperm(num_samples)[:n_eps]
    epoch_sampled_episodes = train_tensor[indices]  # n_eps x channels x timepoints
    episode_sentence_labels = id_tensor[indices]    # n_eps

    # Expand dataset with augmentations
    expanded_episodes = []
    expanded_labels = []
    
    for ep_idx in range(n_eps):
        episode = epoch_sampled_episodes[ep_idx]  # channels x timepoints
        label = episode_sentence_labels[ep_idx]
        
        # original (unaugmented)
        expanded_episodes.append(episode)
        expanded_labels.append(label)
        
        # Augmented copies
        for _ in range(augmentation_factor):
            augmented = augment_eeg(episode, augment_config)
            expanded_episodes.append(augmented)
            expanded_labels.append(label)
    
    # Stack into tensors
    total_episodes = n_eps * (1 + augmentation_factor)
    expanded_episodes = torch.stack(expanded_episodes, dim=0)
    expanded_labels = torch.stack(expanded_labels, dim=0)

    replay_buffer = []
    episodes_bar = tqdm(
        range(total_episodes), 
        desc=f"rollouts ({n_eps} base * {1 + augmentation_factor} = {total_episodes})"
    )

    with torch.no_grad():
        for ep in episodes_bar:
            episode_dict = {
                'transitions': None
            }

            episode = expanded_episodes[ep, :, :]  # channels x timepoints
            ep_lbl_id = expanded_labels[ep].item()
            true_words = label_mapping[ep_lbl_id]
            windows = segment_eeg_tensor(episode, seg_len)

            sensory_state, buffer, cognitive_state, policy_state = None, None, None, None
            transition_list = []
            final_reward = 0 #just for diplay before it gets updated with actual final reward
            
            # Process windows
            for i, window in enumerate(windows):
                window = window.unsqueeze(0)

                sampled_action, _, signals, buffer, sensory_state, cognitive_state, policy_state, _ = experience_forward_pass(
                    model, window, buffer, sensory_state, cognitive_state, policy_state, 
                    sample_action=True, ignore_world=True
                )

                reward_t = compute_timestep_reward(model, sampled_action, true_words, is_final=False)
    
                episodes_bar.set_postfix({'w_r': f'{reward_t:.2f}', 'f_r': f'{final_reward:.2f}'})

                if i < len(windows) - 1:
                    next_window = windows[i+1].unsqueeze(0)
                else:
                    next_window = None
                
                transition = {
                    'window_t': window.cpu(), 
                    'next_window_t': next_window.cpu() if next_window is not None else None,
                    'action_t': sampled_action.cpu(),
                    'reward_t': reward_t,
                    'latent_t': signals.cpu(),
                    'is_done': False
                }
                transition_list.append(transition)
            
            # Final reasoning
            final_act, final_signals, _, _ = experience_final_reasoning(
                model, buffer, cognitive_state, policy_state, sample_action=True
            )

            # word_ids, _, final_conf = model.decode_vocab_ids(final_act, return_confidences=True)
            # final_conf_value = final_conf.squeeze().item()
            # pred_sentence = model.construct_sentence(word_ids)
        
            final_reward = compute_timestep_reward(model, final_act, true_words, is_final=True)

            episodes_bar.set_postfix({
                'p_r': f'{reward_t:.2f}',
                'f_r': f'{final_reward:.2f}',
                # 'conf': f'{final_conf_value:.2f}',
                # 'pred': f'{pred_sentence}'
            })

            metrics_tracker.store_episode_metrics(total=final_reward)

            final_transition = {
                'window_t': None,
                'next_window_t': None,
                'action_t': final_act.cpu(),
                'reward_t': final_reward,
                'latent_t': final_signals.cpu(),
                'is_done': True
            }

            transition_list.append(final_transition)
            episode_dict['transitions'] = transition_list
            replay_buffer.append(episode_dict)
    
    return replay_buffer

def init_optim_and_schedulers(q1, q2, lr_q, model, lr_policy, lr_world, lr_fe, adaptive_lr,
                              q_mode, q_factor, q_patience, q_min_lr, 
                              fe_mode, fe_fac, fe_patience, fe_min_lr,
                              policy_mode, policy_factor, policy_patience, policy_min_lr,
                              world_mode, world_factor, world_patience, world_min_lr):

    #init optimisers
    q1_optim = torch.optim.Adam(q1.parameters(), lr=lr_q)
    q2_optim = torch.optim.Adam(q2.parameters(), lr=lr_q)

    #feature extractor - probably needs different learning rate then policy model
    feature_params = (
        list(model.feature_extractor.parameters())
    )

    policy_params = (
        # list(model.feature_extractor.parameters()) +
        list(model.sensory_propagator.parameters()) +
        list(model.cognitive_layer.parameters()) +
        list(model.policy_prop.parameters()) +
        list(model.mean_projector.parameters()) +
        list(model.variance_projector.parameters())
    )

    world_model_params = (
        # list(model.feature_extractor.parameters()) +
        # list(model.sensory_propagator.parameters()) +
        # list(model.cognitive_layer.parameters()) +
        list(model.decoder_prop.parameters()) +
        list(model.mdn_mu.parameters()) + 
        list(model.mdn_log_sig.parameters()) +
        list(model.mdn_pi.parameters())
    )

    feature_optim = torch.optim.Adam(feature_params, lr=lr_fe)
    policy_optim = torch.optim.Adam(policy_params, lr=lr_policy)
    world_optim = torch.optim.Adam(world_model_params, lr=lr_world)

    #init as none first
    scheduler_q1 = None 
    scheduler_q2 = None 
    scheduler_fe = None
    scheduler_policy = None 
    scheduler_world = None

    #if using adaptive lr, then initialse them properly
    if adaptive_lr:
        scheduler_q1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=q1_optim, mode=q_mode, factor=q_factor, patience=q_patience, min_lr=q_min_lr
            )

        scheduler_q2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=q2_optim, mode=q_mode, factor=q_factor, patience=q_patience, min_lr=q_min_lr
            )
        
        scheduler_fe = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=feature_optim, mode=fe_mode, factor=fe_fac, patience=fe_patience, min_lr=fe_min_lr
        )

        scheduler_policy = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=policy_optim, mode=policy_mode, factor=policy_factor, patience=policy_patience, min_lr=policy_min_lr
            )
        
        scheduler_world = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=world_optim, mode=world_mode, factor=world_factor, patience=world_patience, min_lr=world_min_lr
            )
    
    #add schedulers to list
    schedulers = [scheduler_q1, scheduler_q2, scheduler_fe, scheduler_policy, scheduler_world]
    
    return feature_params, policy_params, world_model_params, q1_optim, q2_optim, feature_optim, policy_optim, world_optim, schedulers


#note for training on this dataset, batched traj has an extra window
#that is not used for world modelling, only for policy 
def train_world_model(model, batcher, optim, grad_clip, params, temp=1.0):

    #initialise world model states 
    sensory_state, buffer, cognitive_state, motor_state = None, None, None, None 

    total_loss = []

    optim.zero_grad()

    for i in range(batcher.n_windows - 1):
        window = batcher.get_window(i)
        
        current_window = window['window_t']
        next_window = window['next_window_t']
        
        with torch.no_grad():
            current_state_t = model.extract_features(current_window)
            sensory_prop, sensory_state = model.propagate_sensory(current_state_t, sensory_state)
            buffer = model.update_buffer(sensory_prop, buffer)
            signals, cognitive_state, buffer = model.think(buffer, cognitive_state)
        
        #predict next state
        _, mu, ls, pi, motor_state = model.predict_next_state(signals, motor_state, temp)
        
        #get true next state
        with torch.no_grad():
            true_next_state = model.extract_features(next_window)
        
        loss = compute_mdn_loss(mu, ls, pi, true_next_state)
        total_loss.append(loss)
    
    avg_loss = torch.stack(total_loss, dim=0).mean()
    avg_loss.backward() #compute gradients
    torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip) #clip gradients
    optim.step() #update parameters

    #reset buffer
    _ = model.reset_buffer(buffer)

    return avg_loss.item()

def compute_mdn_loss(mu, log_sigma, pi, target):
    target = target.unsqueeze(1)
    
    sigma = torch.exp(log_sigma).clamp(min=1e-6)
    diff = (target - mu) / sigma
    log_prob = -0.5 * (diff**2 + 2*log_sigma + np.log(2*np.pi))
    log_prob = log_prob.sum(dim=-1)
    
    log_pi = torch.log(pi.clamp(min=1e-8))
    max_log = (log_prob + log_pi).max(dim=-1, keepdim=True)[0]
    log_mixture = max_log + torch.log(
        torch.exp(log_prob + log_pi - max_log).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    )
    
    nll_loss = -log_mixture.mean()
    
    # Diversity regularization
    uniform_pi = torch.ones_like(pi) / pi.shape[-1]
    diversity_loss = F.kl_div(torch.log(pi + 1e-8), uniform_pi, reduction='batchmean')
    
    total_loss = nll_loss + 0.1 * diversity_loss
    
    # Floor to prevent numerical collapse
    total_loss = torch.clamp(total_loss, min=1e-4) #world model keeps going into negative loss, need to figure out whats going on
    
    return total_loss


def compute_bellman_targets(model, batcher, TQ1, TQ2, gamma, alpha):
    #store target computations over sequence (auxillary + final reasoning rewards/q_values)

    targets = []
    TQ1_l_state, TQ1_a_state = None, None 
    TQ2_l_state, TQ2_a_state = None, None

    motor_state = None 

    for i in range(batcher.n_windows):
        window = batcher.get_window(i) #get batch for current window

        reward_t = window['reward_t']
        done = window['is_done']
        current_latent = window['latent_t']

        #on first iter, advance policy forward to get updated motor state
        if i == 0:
            with torch.no_grad():
                _, _, motor_state = model.propagate_action(current_latent, motor_state)
        
        #if on final reasoning, there is no 'next latent' to predict on so use current
        if i == batcher.n_windows - 1:
            next_latent = current_latent
        else:
            next_latent = batcher.get_window(i+1)['latent_t']
        

        with torch.no_grad():

            #generate next action from current policy
            next_mu, next_ls, motor_state = model.propagate_action(next_latent, motor_state)
            next_action, next_log_prob = model.sample_action(next_mu, next_ls)

            #compute target q values
            tq1_pred, TQ1_l_state, TQ1_a_state = TQ1(next_latent, next_action, TQ1_l_state, TQ1_a_state)
            tq2_pred, TQ2_l_state, TQ2_a_state = TQ2(next_latent, next_action, TQ2_l_state, TQ2_a_state)

            #take minimum to reduce overestimation bias: double clipped q trick or whatever its called
            min_q = torch.min(tq1_pred, tq2_pred)

            #apply bellman equation with entropy regularisation for bootstrap target value
            target = reward_t + gamma * (1.0 - done) * (min_q - alpha * next_log_prob) #batch x 1
            # target = torch.clamp(target, 0.0, 1.0) #clamp; also may need to experiment here but not yet
            targets.append(target)
    
    return targets

def update_critics(batcher, Q1, Q2, targets, q1_optim, q2_optim, grad_clip):
    #put into train mode
    Q1.train()
    Q2.train()

    Q1_loss = update_q_network(batcher, Q1, targets, q1_optim, grad_clip)
    Q2_loss = update_q_network(batcher, Q2, targets, q2_optim, grad_clip)

    return Q1_loss, Q2_loss

def update_q_network(batcher, q, targets, optim, grad_clip):
    optim.zero_grad()
    Q_loss = get_critic_losses(batcher, q, targets)

    #once per entire sequence
    Q_loss.backward() #compute gradients
    torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=grad_clip) #clip gradients
    optim.step() #update params
    return Q_loss.item()

def get_critic_losses(batcher, q, targets):
    Q_losses = [] #accumulate losses over time

    #initialise hiddens states
    q_l_state, q_a_state = None, None 

    for i in range(batcher.n_windows):
        window = batcher.get_window(i)
        current_latent = window['latent_t']
        current_action = window['action_t']

        #predict q values 
        q_pred, q_l_state, q_a_state = q(current_latent, current_action, q_l_state, q_a_state)

        #calculate loss 
        timestep_loss = F.mse_loss(q_pred, targets[i]) #returns scalar
        Q_losses.append(timestep_loss)
    
    return torch.stack(Q_losses, dim=0).mean() #turn list into tensor and get mean

def update_actor(model, batcher, q1, q2, alpha, optim, grad_clip, params):
    q1.eval()
    q2.eval()

    q1_l, q1_a = None, None 
    q2_l, q2_a = None, None
    sensory_state, buffer, cognitive_state, motor_state = None, None, None, None 

    policy_losses = []

    optim[0].zero_grad() #feature extractor optim
    optim[1].zero_grad() #policy optim

    for i in range(batcher.n_windows):
        window = batcher.get_window(i) 

        #only process feature extraction for non-final windows
        if i < batcher.n_windows - 1:
            eeg_input = window['window_t']
            state_t = model.extract_features(eeg_input)
            sensory_prop, sensory_state = model.propagate_sensory(state_t, sensory_state)
            buffer = model.update_buffer(sensory_prop, buffer)
        #final reasoning uses existing buffer, no new window to process
        
        #policy update happens for ALL timesteps including final reasoning
        current_latent, cognitive_state, buffer = model.think(buffer, cognitive_state)
        mu, ls, motor_state = model.propagate_action(current_latent, motor_state)
        action_t, log_prob_t = model.sample_action(mu, ls)

        with torch.no_grad():
            q1_pred, q1_l, q1_a = q1(current_latent, action_t, q1_l, q1_a)
            q2_pred, q2_l, q2_a = q2(current_latent, action_t, q2_l, q2_a)

        policy_loss = compute_policy_loss(q1_pred, q2_pred, log_prob_t, alpha)
        policy_losses.append(policy_loss)


    loss_avg = torch.stack(policy_losses, dim=0).mean()
    loss_avg.backward()
    
    torch.nn.utils.clip_grad_norm_(params[0], max_norm=grad_clip * 2)  # Higher for FE
    torch.nn.utils.clip_grad_norm_(params[1], max_norm=grad_clip) #clip policy params
    
    optim[0].step() #update feature extractor params
    optim[1].step() #update policy params

    _ = model.reset_buffer(buffer)

    return loss_avg.item()

def compute_policy_loss(q1v, q2v, log_prob, entropy_coefficient):
    min_q = torch.min(q1v, q2v) #reduce overestimation bias 
    policy_loss = -(min_q - entropy_coefficient * log_prob).mean()
    return policy_loss

def soft_update_target(Q, TQ, tau):
    #soft update the target q network via polyak averaging
    for source_param, target_param in zip(Q.parameters(), TQ.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

def dynamic_update_steps(epoch, avg_reward=None, min_steps=8, max_steps=40, warmup_epochs=50):
    #scale updates based on reward progress
    #early epochs or no reward data: use minimum
    #as reward improves: scale up updates
    
    if epoch < warmup_epochs or avg_reward is None:
        return min_steps
    
    #reward typically 0-1 range, scale proportionally
    reward_factor = max(0.0, min(1.0, avg_reward))
    update_steps = int(min_steps + reward_factor * (max_steps - min_steps))
    
    return update_steps

def training_V3(model, n_epochs, n_eps, segment_length,
                train_tensor, label_id_tensor, label_mapping,
                batch_size, device,
                lr_q, lr_fe, lr_policy, lr_world, 
                qmode, qfac, qpat, qmin_lr, 
                fmode, ffac, fpat, fmin_lr,
                pmode, pfac, ppat, pmin_lr, 
                wmode, wfac, wpat, wmin_lr,
                adaptive_lr=True, gamma=0.95, alpha=3e-2, tau=5e-3, 
                critic_grad_clip=10.0, actor_grad_clip=10.0, world_grad_clip=10.0,
                world_temp=1.0, save_freq=500, world_model_threshold=0.70, 
                aug_fac=6, aug_config=DEFAULT_AUGMENT_CONFIG, save_path=FREQUENCY_PATH):
    
    #move shit to correct device
    model.to(device)
    train_tensor = train_tensor.to(device)
    label_id_tensor = label_id_tensor.to(device)

    #initalise qnetworks; move to device - init with model class properties 
    #target networks that get soft updated
    TQ1 = QNetwork(cognitive_dim=model.cognitive_dim, embedding_dim=model.embedding_dim).to(device)
    TQ2 = QNetwork(cognitive_dim=model.cognitive_dim, embedding_dim=model.embedding_dim).to(device)
    #trained q networks
    Q1 = QNetwork(cognitive_dim=model.cognitive_dim, embedding_dim=model.embedding_dim).to(device)
    Q2 = QNetwork(cognitive_dim=model.cognitive_dim, embedding_dim=model.embedding_dim).to(device)

    #copy weights from q-networks to target qnetworks
    TQ1.load_state_dict(Q1.state_dict())
    TQ2.load_state_dict(Q2.state_dict())

    #initialise optimisers and schedulers
    #q networks get same learning rates
    fe_params, p_params, w_params, q1_optim, q2_optim, fe_optim, policy_optim, world_optim, schedulers = init_optim_and_schedulers(
        Q1, Q2, lr_q, model, lr_policy, lr_world, lr_fe, adaptive_lr,
        qmode, qfac, qpat, qmin_lr,
        fmode, ffac, fpat, fmin_lr, 
        pmode, pfac, ppat, pmin_lr, 
        wmode, wfac, wpat, wmin_lr)

    #best reward trackers for policy and q-networks; probably need to add world model?
    best_reward_tracker = {'policy': float('-inf'), 'q_networks': float('-inf'), 'world_model': float('-inf')}

    #target networks always on eval mode, they're soft updated not trained
    TQ1.eval()
    TQ2.eval()
    model.train() #main model always in train mode, even if gradients not used just used torch no grad

    #track avg reward for dynamic update steps
    avg_reward_tracker = None

    enable_world_training = False #boolean to trigger world model training only when policy improves

    for epoch in range(n_epochs):
        update_steps = dynamic_update_steps(epoch, avg_reward_tracker)

        #offline episode experience; TO DO, what to collect? and add augmentation?
        #NOTE: episode replay collects experience on individual samples, and is only used for policy updates
        #but the input windows are also collected here so that the world model training
        #can batch the segments
        experience_replay = episodes_rollout(
            model, train_tensor, label_id_tensor, label_mapping, n_eps, segment_length,
            augmentation_factor=aug_fac, augment_config=aug_config
        )

        #create batcher once
        batcher =  LazyWindowBatcher(experience_replay, batch_size, device)

        #update loop
        update_bar = tqdm(range(update_steps), desc=f"Epoch {epoch+1}/{n_epochs} | steps: {update_steps}")

        for _ in update_bar:
            
            #randomly batch samples from experiences
            batcher.sample_episodes()
            update_bar.set_postfix({'status': 'sampled episodes'})
            avg_r = batcher.get_avg_reward()

            #dont train via truncated BPTT because trajectory is only 4-5 timesteps long
            #note schedulers are for every epoch not per update step, so dont pass them
            # print(f"computing world model loss")
            #train world model; no
            if enable_world_training:
                world_model_loss = train_world_model(
                    model, batcher, world_optim, world_grad_clip, w_params, world_temp
                    )
                
                update_bar.set_postfix({'status': 'updated world model'})

            else:
                world_model_loss = 0.0

            # print("computing bell man targets")
            #compute target q values 
            q_targets = compute_bellman_targets(model, batcher, TQ1, TQ2, gamma, alpha)
            update_bar.set_postfix({'status': 'computed targets'})

            # print("updating critics")
            #update q networks 
            Q1_loss, Q2_loss = update_critics(batcher, Q1, Q2, q_targets, q1_optim, q2_optim, critic_grad_clip)
            update_bar.set_postfix({'status': 'updated critics'})

            # print("updating actor")
            #update policy 
            policy_loss = update_actor(
                model, batcher, Q1, Q2, alpha, 
                [fe_optim, policy_optim], actor_grad_clip, [fe_params, p_params]
                )
            update_bar.set_postfix({'status': 'updated actor'})

            # print(f"applying soft update")
            #soft update targets
            soft_update_target(Q1, TQ1, tau)
            soft_update_target(Q2, TQ2, tau)
            update_bar.set_postfix({'status': 'soft updated targets'})

            #update progress bar
            # update_bar.set_postfix({
            #     'avg_r': f'{avg_r:.2f}',
            #     'policy': f'{policy_loss:.2f}',
            #     'q1': f'{Q1_loss:.2f}',
            #     'q2': f'{Q2_loss:.2f}',
            #     'w': f'{world_model_loss:.2f}'
            # })
        
        #store epoch level metrics 
        metrics_tracker.store_losses(world_model_loss, Q1_loss, Q2_loss, policy_loss)
        #save epoch metrics
        stats = metrics_tracker.compute_epoch_stats(epoch+1, n_eps)
        
        #when saving, it will save all n epochs so far, but we only call it
        #at certain frequency to avoid every epoch I/O operation
        if (epoch + 1) % save_freq == 0 or epoch == n_epochs - 1:
            metrics_tracker.save_metrics(TRAINING_METRICS_PATH, (epoch+1))

        #update avg reward tracker for next epoch's dynamic steps
        avg_reward_tracker = stats['avg_reward']

        if stats['avg_reward'] > world_model_threshold: #only train world model when reward increases
            enable_world_training = True

        print(f"avg reward: {stats['avg_reward']:.4f} | avg policy loss: {stats['avg_policy_loss']:.4f} | avg world loss: {stats['avg_world_loss']:.4f} | avg Q1 loss: {stats['avg_q1_loss']:.4f} | avg Q2 loss: {stats['avg_q2_loss']:.4f}")

        #save best models
        if enable_world_training:
            save_best_world_model(model, epoch+1, stats['avg_reward'], best_reward_tracker, save_path=save_path)

        save_best_policy(model, epoch+1, stats['avg_reward'], best_reward_tracker, save_path=save_path)
        save_best_q_networks(Q1, Q2, epoch+1, stats['avg_reward'], best_reward_tracker, save_path=save_path)

        #frequency based saving
        if (epoch + 1) % save_freq == 0:
            os.makedirs(save_path, exist_ok=True)
            path = os.path.join(save_path, f'model_epoch_{epoch+1}.pt')

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
            }, path)

        #schedulers if True
        if adaptive_lr:
            schedulers[0].step(Q1_loss) 
            schedulers[1].step(Q2_loss)
            #feature extractor and policy operate on same loss
            schedulers[2].step(policy_loss)
            schedulers[3].step(policy_loss)
            if enable_world_training:
                schedulers[4].step(world_model_loss)
        
        if epoch == 0:
            model_params = model.print_parameter_count()
            q_params = Q1.print_parameter_count()
            print(f"total params for training: {model_params + (q_params * 4)}")


#-----------------------------------------------
#Training helper functions/saving utilities
#-----------------------------------------------
def save_best_q_networks(Q1, Q2, epoch, mean_reward, best_reward_tracker, save_path=DEFAULT_BEST_Q_PATH):
    #if the new mean reward is greater than the previous best, save the q network parameters
    if mean_reward > best_reward_tracker['q_networks']:
        #get file path
        os.makedirs(save_path, exist_ok=True)
        q1_filepath = os.path.join(save_path, 'q1_best.pt')

        #save q1 parameters
        torch.save({
            'epoch': epoch,
            'model_state_dict': Q1.state_dict(),
            'best_reward': mean_reward
        }, q1_filepath)
        
        #save q2 parameters
        q2_filepath = os.path.join(save_path, 'q2_best.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': Q2.state_dict(),
            'best_reward': mean_reward
        }, q2_filepath)
        
        #update best reward tracker
        best_reward_tracker['q_networks'] = mean_reward
        print(f"  → Saved best Q-networks (reward: {mean_reward:.4f})")

        return True
    
    return False

def save_best_policy(model, epoch, mean_reward, best_reward_tracker, save_path=DEFAULT_BEST_POLICY_PATH):
    #if the new mean reward is greater than the previous best, save the policy's parameters
    if mean_reward > best_reward_tracker['policy']:
        #get file path
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, 'policy_best.pt')

        #save model weights
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_reward': mean_reward
        }, filepath)

        #update best reward tracker
        best_reward_tracker['policy'] = mean_reward
        print(f"  → Saved best policy (reward: {mean_reward:.4f})")

        return True
    
    return False

def save_best_world_model(model, epoch, mean_reward, best_reward_tracker, save_path=DEFAULT_BEST_WORLD_PATH):
    if mean_reward > best_reward_tracker['world_model']:
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, 'world_model_best.pt')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_reward': mean_reward
        }, filepath)

        best_reward_tracker['world_model'] = mean_reward
        print(f"  → Saved best world model (reward: {mean_reward:.4f})")

        return True
    
    return False