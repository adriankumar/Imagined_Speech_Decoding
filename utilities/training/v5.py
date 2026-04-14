#here we use the raw state t as input for q networks
import torch, random, torch.nn as nn 
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np 
import os
from utilities.chisco_preprocessing import segment_eeg_tensor
from model_architecture import QNetworkv2 as QNetwork
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
        state_t_states = []
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
                state_t_states.append(trans['state_t']) #should i make trans['state_t'].shape() -> 1 x dim x thought_steps -> NO DON't, manually do that for qnetwork
                dones.append(1.0 if trans['is_done'] else 0.0)
        
        window_batch = {
            'window_t': torch.cat(raw_windows, dim=0).to(self.device),
            'next_window_t': torch.cat(next_raw_windows, dim=0).to(self.device),
            'action_t': torch.cat(actions, dim=0).to(self.device),
            'reward_t': torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device),
            'state_t': torch.cat(state_t_states, dim=0).to(self.device), #shape b x state_dim
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
        self.Q1_losses = []
        self.Q2_losses = []
        self.policy_losses = []
        self.alpha_losses = []

        #epoch-level statistics
        self.epochs = []
        self.mean_total_reward = []
        self.std_total_reward = []
        self.mean_q1_loss = []
        self.mean_q2_loss = []
        self.mean_policy_loss = []
        self.mean_alpha_loss = []

        #track episode count per epoch
        self.episodes_per_epoch = []

    def store_episode_metrics(self, total):
        self.total_rewards.append(total)

    def store_losses(self, q1, q2, policy, alpha):
        self.Q1_losses.append(q1)
        self.Q2_losses.append(q2)
        self.policy_losses.append(policy)
        self.alpha_losses.append(alpha)

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
        self.mean_q1_loss.append(self.Q1_losses[-1])
        self.mean_q2_loss.append(self.Q2_losses[-1])
        self.mean_policy_loss.append(self.policy_losses[-1])
        self.mean_alpha_loss.append(self.alpha_losses[-1])

        self.episodes_per_epoch.append(n_episodes_this_epoch)

        return {
            'avg_reward': self.mean_total_reward[-1],
            'avg_q1_loss': self.mean_q1_loss[-1],
            'avg_q2_loss': self.mean_q2_loss[-1],
            'avg_policy_loss': self.mean_policy_loss[-1],
            'avg_alpha_loss': self.mean_alpha_loss[-1]
        }

    def save_metrics(self, filepath, epoch):
        filepath = os.path.join(filepath, f'epoch_{epoch+1}')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath,
                 epochs=np.array(self.epochs),
                 mean_total_reward=np.array(self.mean_total_reward),
                 std_total_reward=np.array(self.std_total_reward),
                 mean_q1_loss=np.array(self.mean_q1_loss),
                 mean_q2_loss=np.array(self.mean_q2_loss),
                 mean_policy_loss=np.array(self.mean_policy_loss),
                 mean_alpha_loss=np.array(self.mean_alpha_loss))

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
        weights = [0.8, 0.2] #if final, be more strict
    else:
        weights = [0.7, 0.3] #if not final, more forgiving rewards

    return compute_rewards(model, action, true_words, weights)

def compute_rewards(model, action, true_words, weights, threshold=0.7):
    #filter content words
    content_words = get_content_words(true_words)

    #get target word indices to extract vocab embedding
    true_indices = [model.vocab_list.index(w) for w in content_words if w in model.vocab_list]
    n_content = len(true_indices) #how many semantic words as labels

    #dge case
    if n_content == 0:
        return 0.0 
    
    true_embeddings = model.vocab_embedding[true_indices] #n_content x embedding

    semantic_coverage, efficiency = compute_semantic_coverage(action, true_embeddings, threshold=threshold)

    reward = weights[0] * semantic_coverage + (weights[1] * (efficiency * semantic_coverage))
    return reward

def compute_semantic_coverage(pred_embeds, true_embeds, threshold):
    #pred shape is 1 x embed x thought_steps
    #true shape is n_content x embed
    thought_steps = pred_embeds.shape[-1]

    pred = F.normalize(pred_embeds.squeeze(0).T, dim=-1)  # thought_steps x embed
    true = F.normalize(true_embeds, dim=-1)  # n_content x embed

    # n_content x thought_steps
    sim = torch.mm(true, pred.T)

    # For each content word: was it matched by ANY thought step?
    max_sim_per_word = sim.max(dim=1)[0]  # n_content
    words_found = (max_sim_per_word >= threshold).float()

    # Coverage: fraction of content words successfully found
    coverage = words_found.sum() / true_embeds.shape[0] 

    #bonus
    steps_used = (sim >= threshold).any(dim=0).sum() #how many thought steps contributed
    efficiency = 1.0 - (steps_used / thought_steps) #higher if fewer steps needed

    return coverage.item(), efficiency.item()

#-----------------------------------------------
#forward passes
#-----------------------------------------------
def experience_forward_pass(model, window, cognitive_state=None, policy_state=None, sample_action=False, prev_output=None):

    #assume window is in shape b x chans x seg_len 

    #get state_t
    state_t = model.extract_features(window) #buffer internally updates; handles count

    #think
    signals, cognitive_state = model.think(state_t, cognitive_state, prev_output=prev_output)
    
    #propagate deterministic action
    action, log_sigma, policy_state = model.propagate_action(signals, policy_state) #action here is mu

    #if using sampling output (for training), change action to the sampled action
    if sample_action:
        action, log_prob = model.sample_action(action, log_sigma)
    
    
    return action, state_t, cognitive_state, policy_state

def experience_final_reasoning(model, cognitive_state=None, policy_state=None, final_reasoning_length=None, sample_action=False, prev_output=None):
    final_state_t = model.get_final_state() #averages accumulated state info by number of windows passed; handles reset internally

    signals, cognitive_state = model.think(final_state_t, cognitive_state, final_reasoning_length=None, prev_output=prev_output)
    action, log_sigma, policy_state = model.propagate_action(signals, policy_state)

    if sample_action:
        action, log_prob = model.sample_action(action, log_sigma)
    
    return action, final_state_t, cognitive_state, policy_state

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

    # model.eval()

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

            cognitive_state, policy_state, prev_output = None, None, None 
            transition_list = []
            final_reward = 0 #just for diplay before it gets updated with actual final reward
            
            # Process windows
            for i, window in enumerate(windows):
                window = window.unsqueeze(0)

                sampled_action, state_t, cognitive_state, policy_state = experience_forward_pass(
                    model, window, cognitive_state, policy_state, sample_action=True, prev_output=prev_output
                )

                #update previous output with sampled action
                prev_output = sampled_action #shape 1 x embedding x thought_steps, already handled inside think function

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
                    'state_t': state_t.cpu(), #1 x state_t dim
                    'is_done': False
                }
                transition_list.append(transition)
            
            # Final reasoning
            final_act, final_state_t, _, _ = experience_final_reasoning(
                model, cognitive_state, policy_state, sample_action=True, prev_output=prev_output
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
                'state_t': final_state_t.cpu(),
                'is_done': True
            }

            transition_list.append(final_transition)
            episode_dict['transitions'] = transition_list
            replay_buffer.append(episode_dict)
    
    return replay_buffer

def init_optim_and_schedulers(q1, q2, lr_q, model, lr_policy, lr_fe, adaptive_lr,
                              q_mode, q_factor, q_patience, q_min_lr, 
                              fe_mode, fe_fac, fe_patience, fe_min_lr,
                              policy_mode, policy_factor, policy_patience, policy_min_lr
                              ):

    #init optimisers
    q1_optim = torch.optim.Adam(q1.parameters(), lr=lr_q)
    q2_optim = torch.optim.Adam(q2.parameters(), lr=lr_q)

    #feature extractor - probably needs different learning rate then policy model
    feature_params = (
        list(model.feature_extractor.parameters())
    )

    policy_params = (
        # list(model.feature_extractor.parameters()) +
        # list(model.sensory_propagator.parameters()) +
        list(model.cognitive_layer.parameters()) +
        list(model.policy_prop.parameters()) +
        list(model.mean_projector.parameters()) +
        list(model.variance_projector.parameters())
    )

    feature_optim = torch.optim.Adam(feature_params, lr=lr_fe)
    policy_optim = torch.optim.Adam(policy_params, lr=lr_policy)

    #init as none first
    scheduler_q1 = None 
    scheduler_q2 = None 
    scheduler_fe = None
    scheduler_policy = None 

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
        
    
    #add schedulers to list
    schedulers = [scheduler_q1, scheduler_q2, scheduler_fe, scheduler_policy]
    
    return feature_params, policy_params, q1_optim, q2_optim, feature_optim, policy_optim, schedulers


def compute_bellman_targets(model, batcher, TQ1, TQ2, gamma, alpha, q_clamps=[-1, 6]):
    #store target computations over sequence (auxillary + final reasoning rewards/q_values)

    targets = []
    TQ1_l_state, TQ1_a_state = None, None 
    TQ2_l_state, TQ2_a_state = None, None

    prev_output, cognitive_state, motor_state = None, None, None 

    for i in range(batcher.n_windows):
        window = batcher.get_window(i) #get batch for current window

        reward_t = window['reward_t']
        done = window['is_done']
        current_state = window['state_t'] #shape b x state_dim

        # index out of bounds handle
        if i < batcher.n_windows - 1:
            next_state = batcher.get_window(i+1)['state_t']
        else:
            next_state = current_state  # Doesn't matter, done=1.0 zeros it out

        #on first iter, advance policy forward to get updated motor state
        if i == 0:
            with torch.no_grad():
                current_latent, cognitive_state = model.think(current_state, cognitive_state, prev_output=prev_output)
                _, _, motor_state = model.propagate_action(current_latent, motor_state)
        
        prev_output = window['action_t'] #previous output is the current action

        with torch.no_grad():

            next_latent, cognitive_state = model.think(next_state, cognitive_state, prev_output=prev_output)

            #generate next action from current policy
            next_mu, next_ls, motor_state = model.propagate_action(next_latent, motor_state)
            next_action, next_log_prob = model.sample_action(next_mu, next_ls)

            #make next_state shape expand to thought_step amount or just once
            next_state = next_state.unsqueeze(-1) #shape b x state_dim x 1

            #compute target q values
            tq1_pred, TQ1_l_state, TQ1_a_state = TQ1(next_state, next_action, TQ1_l_state, TQ1_a_state)
            tq2_pred, TQ2_l_state, TQ2_a_state = TQ2(next_state, next_action, TQ2_l_state, TQ2_a_state)

            tq1_pred = torch.clamp(tq1_pred, min=q_clamps[0], max=q_clamps[-1])
            tq2_pred = torch.clamp(tq2_pred, min=q_clamps[0], max=q_clamps[-1])

            #take minimum to reduce overestimation bias: double clipped q trick or whatever its called
            min_q = torch.min(tq1_pred, tq2_pred)

            #apply bellman equation with entropy regularisation for bootstrap target value
            target = reward_t + gamma * (1.0 - done) * (min_q - alpha * next_log_prob) #batch x 1
            # target = torch.clamp(target, 0.0, 1.0) #clamp; also may need to experiment here but not yet
            targets.append(target)
    
    return targets

def update_critics(batcher, Q1, Q2, targets, q1_optim, q2_optim, grad_clip, q_clamps=[-1, 6]):
    #put into train mode
    Q1.train()
    Q2.train()

    Q1_loss = update_q_network(batcher, Q1, targets, q1_optim, grad_clip, q_clamps)
    Q2_loss = update_q_network(batcher, Q2, targets, q2_optim, grad_clip, q_clamps)

    return Q1_loss, Q2_loss

def update_q_network(batcher, q, targets, optim, grad_clip, q_clamps):
    optim.zero_grad()
    Q_loss = get_critic_losses(batcher, q, targets, q_clamps)

    #once per entire sequence
    Q_loss.backward() #compute gradients
    torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=grad_clip) #clip gradients
    optim.step() #update params
    return Q_loss.item()

def get_critic_losses(batcher, q, targets, q_clamps):
    Q_losses = [] #accumulate losses over time

    #initialise hiddens states
    q_l_state, q_a_state = None, None 

    for i in range(batcher.n_windows):
        window = batcher.get_window(i)
        current_state = window['state_t']
        current_action = window['action_t']

        #expand current state shape to have thought_step amount 
        current_state = current_state.unsqueeze(-1) #b x state dim x 1

        #predict q values 
        q_pred, q_l_state, q_a_state = q(current_state, current_action, q_l_state, q_a_state)

        #clamp q pred 
        q_pred = torch.clamp(q_pred, min=q_clamps[0], max=q_clamps[-1])

        #calculate loss 
        timestep_loss = F.mse_loss(q_pred, targets[i]) #returns scalar
        Q_losses.append(timestep_loss)
    
    return torch.stack(Q_losses, dim=0).mean() #turn list into tensor and get mean

def update_actor(model, batcher, q1, q2, alpha, optim, grad_clip, params, q_clamps=[-1, 6]):
    q1.eval()
    q2.eval()

    q1_l, q1_a = None, None 
    q2_l, q2_a = None, None
    cognitive_state, motor_state, prev_output = None, None, None 

    policy_losses = []
    # log_probs = []

    optim[0].zero_grad() #feature extractor optim
    optim[1].zero_grad() #policy optim

    for i in range(batcher.n_windows):
        window = batcher.get_window(i) 

        #only process feature extraction for non-final windows
        if i < batcher.n_windows - 1:
            eeg_input = window['window_t']
            state_t = model.extract_features(eeg_input)
        else:
            state_t = model.get_final_state()

        #final reasoning 
        
        #policy update happens for ALL timesteps including final reasoning
        current_latent, cognitive_state = model.think(state_t, cognitive_state, prev_output=prev_output)
        mu, ls, motor_state = model.propagate_action(current_latent, motor_state)
        action_t, log_prob_t = model.sample_action(mu, ls)
        prev_output = action_t
        # log_probs.append(log_prob_t) #append for alpha loss

        with torch.no_grad():
            #expand state t shape to thought step amount

            q1_pred, q1_l, q1_a = q1(state_t.unsqueeze(-1), action_t, q1_l, q1_a)
            q2_pred, q2_l, q2_a = q2(state_t.unsqueeze(-1), action_t, q2_l, q2_a)

            q1_pred = torch.clamp(q1_pred, min=q_clamps[0], max=q_clamps[-1])
            q2_pred = torch.clamp(q2_pred, min=q_clamps[0], max=q_clamps[-1])

        policy_loss = compute_policy_loss(q1_pred, q2_pred, log_prob_t, alpha)
        policy_losses.append(policy_loss)

    #compute policy update first
    loss_avg = torch.stack(policy_losses, dim=0).mean()
    loss_avg.backward()
    
    torch.nn.utils.clip_grad_norm_(params[0], max_norm=grad_clip * 2)  # Higher for FE
    torch.nn.utils.clip_grad_norm_(params[1], max_norm=grad_clip) #clip policy params
    
    optim[0].step() #update feature extractor params
    optim[1].step() #update policy params

    #update alpha
    # optim[2].zero_grad()
    # avg_log_probs = torch.stack(log_probs, dim=0).mean().detach()
    # alpha_loss = (log_alpha.exp() * (-avg_log_probs - target_entropy)).mean()
    # alpha_loss.backward()
    # optim[2].step()
    #clamp alpha
    # with torch.no_grad():
    #     log_alpha.data.clamp_(min=1e-4, max=1.0) #clamp negative

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

def training_V5(model, n_epochs, n_eps, segment_length,
                train_tensor, label_id_tensor, label_mapping,
                batch_size, device,
                lr_q, lr_fe, lr_policy, 
                qmode, qfac, qpat, qmin_lr, 
                fmode, ffac, fpat, fmin_lr,
                pmode, pfac, ppat, pmin_lr, 
                adaptive_lr=True, gamma=0.95, tau=5e-3, 
                critic_grad_clip=7.0, actor_grad_clip=7.0, base_alpha=-2.0, alpha_lr=1e-3,
                save_freq=500, q_clamps=[-1, 6],
                aug_fac=6, aug_config=DEFAULT_AUGMENT_CONFIG, save_path=FREQUENCY_PATH):
    
    #move shit to correct device
    model.to(device)
    train_tensor = train_tensor.to(device)
    label_id_tensor = label_id_tensor.to(device)

    #initalise qnetworks; move to device - init with model class properties 
    #target networks that get soft updated
    TQ1 = QNetwork(state_dim=model.state_dim, embedding_dim=model.embedding_dim).to(device)
    TQ2 = QNetwork(state_dim=model.state_dim, embedding_dim=model.embedding_dim).to(device)
    #trained q networks
    Q1 = QNetwork(state_dim=model.state_dim, embedding_dim=model.embedding_dim).to(device)
    Q2 = QNetwork(state_dim=model.state_dim, embedding_dim=model.embedding_dim).to(device)

    #copy weights from q-networks to target qnetworks
    TQ1.load_state_dict(Q1.state_dict())
    TQ2.load_state_dict(Q2.state_dict())

    #initialise optimisers and schedulers
    #q networks get same learning rates
    fe_params, p_params, q1_optim, q2_optim, fe_optim, policy_optim, schedulers = init_optim_and_schedulers(
        Q1, Q2, lr_q, model, lr_policy, lr_fe, adaptive_lr,
        qmode, qfac, qpat, qmin_lr,
        fmode, ffac, fpat, fmin_lr, 
        pmode, pfac, ppat, pmin_lr, 
    )
    
    #learnable alpha -- if ever using leanrable alpha ensure you always .exp() when using
    # log_alpha = nn.Parameter(torch.tensor(base_alpha, device=device))
    # log_alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
    # target_entropy = -model.embedding_dim
    alpha = base_alpha / model.embedding_dim

    #best reward trackers for policy and q-networks; probably need to add world model?
    best_reward_tracker = {'policy': float('-inf'), 'q_networks': float('-inf')}

    #target networks always on eval mode, they're soft updated not trained
    TQ1.eval()
    TQ2.eval()
    model.train() #main model always in train mode, even if gradients not used just used torch no grad

    #track avg reward for dynamic update steps
    avg_reward_tracker = None
    alpha_loss = 0.0
    # fe_train = False

    #freeze feature extractors for warm up epochs to get policy learning before end to end learning
    # for param in model.feature_extractor.parameters():
    #     param.requires_grad = False 

    # print("Feature extractors frozen")

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


            # print("computing bell man targets")
            #compute target q values 
            q_targets = compute_bellman_targets(model, batcher, TQ1, TQ2, gamma, alpha, q_clamps)
            update_bar.set_postfix({'status': 'computed targets'})

            # print("updating critics")
            #update q networks 
            Q1_loss, Q2_loss = update_critics(batcher, Q1, Q2, q_targets, q1_optim, q2_optim, critic_grad_clip, q_clamps)
            update_bar.set_postfix({'status': 'updated critics'})

            # print("updating actor")
            #update policy 
            policy_loss = update_actor(
                model, batcher, Q1, Q2, alpha,
                [fe_optim, policy_optim], actor_grad_clip, [fe_params, p_params], q_clamps
                )
            update_bar.set_postfix({'status': 'updated actor'})

            # print(f"applying soft update")
            #soft update targets
            soft_update_target(Q1, TQ1, tau)
            soft_update_target(Q2, TQ2, tau)
            update_bar.set_postfix({'status': 'soft updated targets'})

        
        #store epoch level metrics 
        metrics_tracker.store_losses(Q1_loss, Q2_loss, policy_loss, alpha_loss)
        #save epoch metrics
        stats = metrics_tracker.compute_epoch_stats(epoch+1, n_eps)
        
        #when saving, it will save all n epochs so far, but we only call it
        #at certain frequency to avoid every epoch I/O operation
        if (epoch + 1) % save_freq == 0 or epoch == n_epochs - 1:
            metrics_tracker.save_metrics(TRAINING_METRICS_PATH, (epoch+1))

        #update avg reward tracker for next epoch's dynamic steps
        avg_reward_tracker = stats['avg_reward']

        # if epoch == fe_threshold and not fe_train:
        #     for params in model.feature_extractor.parameters():
        #         params.requires_grad = True
        #     fe_train = True
        #     print("Unfroze feature extractor parameters")



        print(f"avg reward: {stats['avg_reward']:.4f} | avg policy loss: {stats['avg_policy_loss']:.4f} | avg Q1 loss: {stats['avg_q1_loss']:.4f} | avg Q2 loss: {stats['avg_q2_loss']:.4f} | avg alpha loss: {stats['avg_alpha_loss']:.4f}")


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
