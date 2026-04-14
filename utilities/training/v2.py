import torch 
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

def compute_timestep_reward(model, action, true_words, is_final=False):
    if is_final:
        weights = [0.5, 0.25, 0.25] #alignment, coverage, length; if final, be more strict
    else:
        weights = [0.7, 0.2, 0.1] #if not final, more forgiving rewards

    return compute_rewards(model, action, true_words, weights)

def compute_rewards(model, action, targets, reward_weights, thought_steps=16):
    #get content word indices
    true_word_indices = [model.vocab_list.index(w) for w in targets if w in model.vocab_list]
    n_content = len(true_word_indices)
    
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

def compute_alignment_reward(pred_embeds, true_embeds, true_length):
    #normalise predicted and true embeddings for cosine scores
    pred_norm = F.normalize(pred_embeds.transpose(dim0=1, dim1=2), dim=-1) #transpose to shape b x t x embedding, and normalise across vectors
    true_norm = F.normalize(true_embeds, dim=-1) #vocab x embedding dim

    sim_matrix = torch.matmul(pred_norm, true_norm.T)

    batch_size, t, _ = sim_matrix.shape #note batch size will always be 1 for episode roll out but just for compatibility

    claim_counts = torch.zeros(batch_size, true_length, device=pred_embeds.device)
    step_rewards = [] #per thought_step on model level not per timestep on the window level

    for t_s in range(t):
        step_similarity = sim_matrix[:, t_s, :] #batch x true_sentence_length 

        #penalise matching already-claimed counts; i.e first match in thought step is 100% reward, but next thought step if second match exists decreases say 50% and so on for every thought step for repetition
        discount = 1.0 / (1.0 + claim_counts)
        discounted_similarity = step_similarity * discount 

        #best match for this step
        best_sim, best_idx = discounted_similarity.max(dim=-1) #batch,

        claim_counts.scatter_add_(1, best_idx.unsqueeze(1), torch.ones_like(best_idx.unsqueeze(1), dtype=torch.float))
        step_rewards.append(best_sim)
    
    #aggregate per thought step rewards into mean reward
    alignment_reward = torch.stack(step_rewards).mean(dim=0)  # (batch,)

    return alignment_reward, claim_counts

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

#assume train_tensor is shape num_samples x channels x timepoints
#don't collect state_t here because thats only for world model training, not policy
#offline episode collection, because model 'experiences' in eval mode and not train mode
#in otherwords, the model collects experiences and learns from the replay
def episodes_rollout(model, train_tensor, id_tensor, label_mapping, n_eps, seg_len):
    num_samples = train_tensor.shape[0]

    if n_eps > num_samples:
        raise ValueError(f"{n_eps} n_eps per epoch cannot exceed {num_samples} training samples")

    model.eval() #set model to eval mode for offline experience

    #sample n random eps, so not every training sample will be used at each epoch for efficiency
    #so just have high number of epochs (trade off between processing speed and time)
    indices = torch.randperm(num_samples)[:n_eps] #permute sample indicies and only use n_eps amount to 'experience' for training per epoch
    epoch_sampled_episodes = train_tensor[indices] #n_eps x channels x timepoints
    episode_sentence_labels = id_tensor[indices] #n_eps  

    replay_buffer = [] #stores n_eps of elements, where each element is a dictionary of that episode experience
    episodes_bar = tqdm(range(n_eps), desc="episode rollouts")

    with torch.no_grad():

        for ep in episodes_bar:
            #make episode specific dictionary
            episode_dict = {
                'transitions': None #store transitions/episode experience in sequential order (as list of dicts)
            }

            episode = epoch_sampled_episodes[ep, :, :] #shape channels x timepoints
            ep_lbl_id = episode_sentence_labels[ep].item() #index
            true_words = label_mapping[ep_lbl_id]
            windows = segment_eeg_tensor(episode, seg_len) #list of channels x seg_len

            #model states exlucde world model state here
            sensory_state, buffer, cognitive_state, policy_state = None, None, None, None

            transition_list = []

            #loop through windows (auxillary rewards), before final reasoning (main objective reward)
            for i, window in enumerate(windows):
                window = window.unsqueeze(0)

                #forward pass
                sampled_action, _, signals, buffer, sensory_state, cognitive_state, policy_state, _ = experience_forward_pass(
                    model, window, buffer, sensory_state, cognitive_state, policy_state, sample_action=True, ignore_world=True
                )

                #per-timestep reward
                reward_t = compute_timestep_reward(model, sampled_action, true_words, is_final=False)

                episodes_bar.set_postfix({
                    'w_r': f'{reward_t:.2f}'
                })

                #next window handling
                if i < len(windows) - 1:
                    next_window = windows[i+1].unsqueeze(0)
                else:
                    next_window = None
                
                transition = {
                    'window_t': window.cpu(), 
                    'next_window_t': next_window.cpu() if next_window is not None else None,
                    'action_t': sampled_action.cpu(),
                    'reward_t': reward_t, #scalar
                    'latent_t': signals.cpu(), #critics evaluate cognitive signals as input for policy
                    'is_done': False
                }
                transition_list.append(transition)
            
            #final reasoning; buffer reset handled internally 
            final_act, final_signals, _, _ = experience_final_reasoning(
                model, buffer, cognitive_state, policy_state, sample_action=True
            )

            #for debugging
            word_ids, _, final_conf = model.decode_vocab_ids(final_act, return_confidences=True)
            final_conf_value = final_conf.squeeze().item() #in episode replay batch size = 1
            pred_sentence = model.construct_sentence(word_ids)
        
            #final reward computation 
            final_reward = compute_timestep_reward(model, final_act, true_words, is_final=True)

            episodes_bar.set_postfix({
                    'p_r': f'{reward_t:.2f}', #previous reward
                    'f_r': f'{final_reward:.2f}',
                    'conf': f'{final_conf_value:.2f}',
                    'pred': f'{pred_sentence}'
                })

            #store episode metrics
            metrics_tracker.store_episode_metrics(
                total=final_reward
            )

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

#batch experience replay for training
def batch_samples(replay_buffer, batch_size=32, device='cpu'):

    n_eps = len(replay_buffer)
    if batch_size > n_eps:
        raise ValueError(f"batch size {batch_size} cannot exceed number of episodes {n_eps}")
    
    #sample random episode indices
    ep_indices = torch.randperm(n_eps)[:batch_size]
    sampled_episodes = [replay_buffer[i] for i in ep_indices]
    
    #determine max number of windows across sampled episodes; should all be the same
    max_windows = max(len(ep['transitions']) for ep in sampled_episodes)
    
    #transpose: from episodes[ep][window] to batches[window][ep]
    batched_windows = []

    batch, chans, seg_len = None, None, None #global for state shape, to build zero tensors of states; assume all equal size
    
    for window_idx in range(max_windows):
        raw_windows = []
        next_raw_windows = []
        actions = []
        rewards = []
        latent_states = [] #the state t that world model decoder and policy share; i.e cognitive signals
        dones = []

        #collect all episodes' data at this window index
        for ep in sampled_episodes:
            transitions = ep['transitions']

            #check if this episode has this window index (hanlding variable length by getting rid of them but it should all be the same)
            if window_idx < len(transitions):
                trans = transitions[window_idx]

                #on first window set variable shapes
                if window_idx == 0 and batch is None and chans is None and seg_len is None:
                    batch, chans, seg_len = trans['window_t'].shape
            
                #if state is none (final reasoning), then make window zeros
                if trans['window_t'] is not None:
                    raw_windows.append(trans['window_t']) #b x c x seg
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
    
        #create batch for this window timestep
        window_batch = {
            'window_t': torch.cat(raw_windows, dim=0).to(device), #batch x channels x seg
            'next_window_t': torch.cat(next_raw_windows, dim=0).to(device), #batch x channels x seg
            'action_t': torch.cat(actions, dim=0).to(device), #batch x embedding_dim x thought_steps
            'reward_t': torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device), #batch x 1
            'latent_t': torch.cat(latent_states, dim=0).to(device), # batch x cog_sginals x thought_steps
            'is_done': torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device) #batch x 1
        }

        batched_windows.append(window_batch)
    
    return batched_windows #list of batches


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
def train_world_model(model, batched_traj, optim, grad_clip, params, temp=1.0):

    #initialise world model states 
    sensory_state, buffer, cognitive_state, motor_state = None, None, None, None 

    total_loss = []

    optim.zero_grad()

    for i, window in enumerate(batched_traj):
        if i == len(batched_traj) - 1:
            break #break out before 'final reasoning' window which is only used for policy

        current_window = window['window_t'] #b x chans x seg_len
        next_window = window['next_window_t']

        #forward pass current window to get next state pred
        with torch.no_grad():
            current_state_t = model.extract_features(current_window)
            sensory_prop, sensory_state = model.propagate_sensory(current_state_t, sensory_state)
            buffer = model.update_buffer(sensory_prop, buffer)
            signals, cognitive_state, buffer = model.think(buffer, cognitive_state)

        #ignoring policy component
        _, mu, ls, pi, motor_state = model.predict_next_state(signals, motor_state, temp) 

        #get actual next state:
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


def compute_bellman_targets(model, batched_traj, TQ1, TQ2, gamma, alpha):
    #store target computations over sequence (auxillary + final reasoning rewards/q_values)

    targets = []
    TQ1_l_state, TQ1_a_state = None, None 
    TQ2_l_state, TQ2_a_state = None, None

    motor_state = None 

    for i, window in enumerate(batched_traj):
        reward_t = window['reward_t']
        done = window['is_done']
        current_latent = window['latent_t']

        #on first iter, advance policy forward to get updated motor state
        if i == 0:
            with torch.no_grad():
                _, _, motor_state = model.propagate_action(current_latent, motor_state)
        
        #if on final reasoning, there is no 'next latent' to predict on so use current
        if i == len(batched_traj) - 1:
            next_latent = current_latent
        else:
            next_latent = batched_traj[i+1]['latent_t']
        

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

def update_critics(batched_traj, Q1, Q2, targets, q1_optim, q2_optim, grad_clip):
    #put into train mode
    Q1.train()
    Q2.train()

    Q1_loss = update_q_network(batched_traj, Q1, targets, q1_optim, grad_clip)
    Q2_loss = update_q_network(batched_traj, Q2, targets, q2_optim, grad_clip)

    return Q1_loss, Q2_loss

def update_q_network(batched_traj, q, targets, optim, grad_clip):
    optim.zero_grad()
    Q_loss = get_critic_losses(batched_traj, q, targets)

    #once per entire sequence
    Q_loss.backward() #compute gradients
    torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=grad_clip) #clip gradients
    optim.step() #update params
    return Q_loss.item()

def get_critic_losses(batched_traj, q, targets):
    Q_losses = [] #accumulate losses over time

    #initialise hiddens states
    q_l_state, q_a_state = None, None 

    for i, window in enumerate(batched_traj):

        current_latent = window['latent_t']
        current_action = window['action_t']

        #predict q values 
        q_pred, q_l_state, q_a_state = q(current_latent, current_action, q_l_state, q_a_state)

        #calculate loss 
        timestep_loss = F.mse_loss(q_pred, targets[i]) #returns scalar
        Q_losses.append(timestep_loss)
    
    return torch.stack(Q_losses, dim=0).mean() #turn list into tensor and get mean

def update_actor(model, batched_traj, q1, q2, alpha, optim, grad_clip, params):
    q1.eval()
    q2.eval()

    q1_l, q1_a = None, None 
    q2_l, q2_a = None, None
    sensory_state, buffer, cognitive_state, motor_state = None, None, None, None 

    policy_losses = []

    optim[0].zero_grad() #feature extractor optim
    optim[1].zero_grad() #policy optim

    for i, window in enumerate(batched_traj):
        #only process feature extraction for non-final windows
        if i < len(batched_traj) - 1:
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

def training_V2(model, n_epochs, n_eps, segment_length,
                train_tensor, label_id_tensor, label_mapping,
                batch_size, device,
                lr_q, lr_fe, lr_policy, lr_world, 
                qmode, qfac, qpat, qmin_lr, 
                fmode, ffac, fpat, fmin_lr,
                pmode, pfac, ppat, pmin_lr, 
                wmode, wfac, wpat, wmin_lr,
                adaptive_lr=True, gamma=0.95, alpha=3e-2, tau=5e-3, 
                critic_grad_clip=10.0, actor_grad_clip=10.0, world_grad_clip=10.0,
                world_temp=1.0, save_freq=500, world_model_threshold=0.70, save_path=FREQUENCY_PATH):
    
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
            model, train_tensor, label_id_tensor, label_mapping, n_eps, segment_length
        )

        #update loop
        update_bar = tqdm(range(update_steps), desc=f"Epoch {epoch+1}/{n_epochs} | steps: {update_steps}")

        for _ in update_bar:
            
            # print(f"batching samples")
            #randomly batch samples from experiences
            batched_list = batch_samples(experience_replay, batch_size, device) 
            avg_r = torch.cat([wb['reward_t'] for wb in batched_list], dim=0).mean().item()

            #dont train via truncated BPTT because trajectory is only 4-5 timesteps long
            #note schedulers are for every epoch not per update step, so dont pass them
            # print(f"computing world model loss")
            #train world model; no
            if enable_world_training:
                world_model_loss = train_world_model(
                    model, batched_list, world_optim, world_grad_clip, w_params, world_temp
                    )
            else:
                world_model_loss = 0.0

            # print("computing bell man targets")
            #compute target q values 
            q_targets = compute_bellman_targets(model, batched_list, TQ1, TQ2, gamma, alpha)

            # print("updating critics")
            #update q networks 
            Q1_loss, Q2_loss = update_critics(batched_list, Q1, Q2, q_targets, q1_optim, q2_optim, critic_grad_clip)

            # print("updating actor")
            #update policy 
            policy_loss = update_actor(
                model, batched_list, Q1, Q2, alpha, 
                [fe_optim, policy_optim], actor_grad_clip, [fe_params, p_params]
                )

            # print(f"applying soft update")
            #soft update targets
            soft_update_target(Q1, TQ1, tau)
            soft_update_target(Q2, TQ2, tau)

            #update progress bar
            update_bar.set_postfix({
                'avg_r': f'{avg_r:.2f}',
                'policy': f'{policy_loss:.2f}',
                'q1': f'{Q1_loss:.2f}',
                'q2': f'{Q2_loss:.2f}',
                'w': f'{world_model_loss:.2f}'
            })
        
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
            save_best_world_model(model, epoch+1, stats['avg_reward'], best_reward_tracker)

        save_best_policy(model, epoch+1, stats['avg_reward'], best_reward_tracker)
        save_best_q_networks(Q1, Q2, epoch+1, stats['avg_reward'], best_reward_tracker)

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