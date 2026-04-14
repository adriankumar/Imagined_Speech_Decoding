import torch 
import torch.nn.functional as F
from tqdm import tqdm
import os
from utilities import segment_eeg_tensor
import random

#-----------------------------------------------
#Constants
#-----------------------------------------------
DEFAULT_SAVE_PATH = r"demo_weights_metrics\pretrained"

DEFAULT_AUGMENT_CONFIG = {
    'noise_prob': 0.7,
    'noise_std_range': (0.02, 0.15),
    'scale_prob': 0.7,
    'scale_range': (0.75, 1.25),
    'dropout_prob': 0.4,
    'dropout_count_range': (2, 8)
}

FUNCTION_WORDS = {
    # # Articles
    'a', 'an', 'the',
    # # Be verbs
    # 'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being',
    'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being',

    # # Prepositions
    # 'to', 'of', 'in', 'on', 'at', 'for', 'with', 'by', 'from',
    # 'about', 'into', 'through', 'during', 'before', 'after',
    'to', 'of', 'in', 'on', 'at', 'for', 'with', 'by', 'from',
    'about', 'into', 'through', 'during', 'before', 'after',

    # # Pronouns
    'it', 'its', 'this', 'that', 'these', 'those',
    # 'i', "i'll", "i'm", "i've", 'you', "you're", "you'll",
    "i'll", "i'm", "i've", 'you', "you're", "you'll",
    # 'he', "he's", 'she', "she's", 'we', "we're", "we'll",
    # 'they', "they're", 'my', 'your', 'his', 'her', 'our', 'their',
    # 'me', 'him', 'us', 'them',
    # # Conjunctions
    # 'and', 'or', 'but', 'so', 'if', 'then', 'than', 'as',
    # # Auxiliaries
    # 'do', "don't", 'does', "doesn't", 'did', "didn't",
    # 'have', 'has', 'had', "haven't", "hasn't",
    # 'will', "won't", 'would', "wouldn't",
    # 'can', "can't", 'could', "couldn't",
    # 'should', "shouldn't", 'must', 'may', 'might',
    # # Adverbs / Modifiers
    # 'just', 'very', 'really', 'also', 'too', 'only', 'such',
    # 'not', 'no', 'yes', 'there', "there's", 'here',
    # # Question words
    # 'what', 'when', 'where', 'which', 'who', 'how', 'why',
    # # Quantifiers
    # 'all', 'each', 'every', 'any', 'some', 'many', 'much',
}

#-----------------------------------------------
#Helper functions
#-----------------------------------------------
def get_content_embeddings(model, true_words):
    content_indices = []
    
    for word in true_words:
        #skip function words and padding
        if word.lower() not in FUNCTION_WORDS:
            try:
                idx = model.vocab_list.index(word)
                content_indices.append(idx)
            except ValueError:
                continue  #word not in vocab, skip
    
    if len(content_indices) == 0:
        return torch.empty(0, model.embedding_dim, device=model.vocab_embedding.device)
    
    return model.vocab_embedding[content_indices]  #n_content x embed_dim

#Add gaussian noise. Works on batch x channels x timepoints
def apply_gaussian_noise(eeg_tensor, std_range):
    std = random.uniform(std_range[0], std_range[1])
    noise = torch.randn_like(eeg_tensor) * std
    return eeg_tensor + noise

#Scale amplitude. Works on batch x channels x timepoints
def apply_amplitude_scaling(eeg_tensor, scale_range):
    scale = random.uniform(scale_range[0], scale_range[1])
    return eeg_tensor * scale

#Zero out random channels. Works on batch x channels x timepoints
def apply_channel_dropout(eeg_tensor, dropout_count_range):
    n_channels = eeg_tensor.shape[1]  # batch x channels x timepoints
    n_drop = random.randint(dropout_count_range[0], dropout_count_range[1])
    n_drop = min(n_drop, n_channels - 1)
    drop_indices = random.sample(range(n_channels), n_drop)
    augmented = eeg_tensor.clone()
    augmented[:, drop_indices, :] = 0  # zero across entire batch
    return augmented

def augment_batch(batch_eeg, config=None):
    if config is None:
        config = DEFAULT_AUGMENT_CONFIG
    
    augmented = batch_eeg.clone()
    
    if random.random() < config['noise_prob']:
        augmented = apply_gaussian_noise(augmented, config['noise_std_range'])
    
    if random.random() < config['scale_prob']:
        augmented = apply_amplitude_scaling(augmented, config['scale_range'])
    
    if random.random() < config['dropout_prob']:
        augmented = apply_channel_dropout(augmented, config['dropout_count_range'])
    
    return augmented #batch x channels x timepoints

# Returns:
#     target_embeds: batch x thought_steps x embed_dim (zero-padded)
#     target_lengths: batch tensor of actual content word counts
#     target_mask: batch x thought_steps (1.0 for real content, 0.0 for pad)
def prepare_batch_targets(model, batch_label_ids, label_mapping, thought_steps, device):
    batch_size = len(batch_label_ids)
    embed_dim = model.embedding_dim
    
    #initialise output tensors
    target_embeds = torch.zeros(batch_size, thought_steps, embed_dim, device=device)
    target_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    target_mask = torch.zeros(batch_size, thought_steps, device=device)
    
    for b, label_id in enumerate(batch_label_ids):
        true_words = label_mapping[label_id.item()]
        content_embeds = get_content_embeddings(model, true_words)  #n_content x embed_dim
        
        n_content = content_embeds.shape[0]
        if n_content > 0:
            #clamp to thought_steps if more content words than thought steps
            n_to_copy = min(n_content, thought_steps)
            target_embeds[b, :n_to_copy, :] = content_embeds[:n_to_copy]
            target_lengths[b] = n_to_copy
            target_mask[b, :n_to_copy] = 1.0
    
    return target_embeds, target_lengths, target_mask


def compute_similarity_loss(mu, target_embeds, target_mask, diversity_weight=0.2):

    batch_size = mu.shape[0]
    thought_steps = mu.shape[-1]
    
    #reshape mu to batch x thought_steps x embed_dim
    mu_t = mu.permute(0, 2, 1)
    
    #normalise for cosine similarity
    mu_norm = F.normalize(mu_t, dim=-1)  #batch x thought_steps x embed_dim
    target_norm = F.normalize(target_embeds, dim=-1)  #batch x thought_steps x embed_dim
    
    #compute similarity matrix: batch x pred_steps x target_steps
    sim = torch.bmm(mu_norm, target_norm.transpose(1, 2))
    
    #for each content word (target), find best matching prediction
    #max over prediction dimension for each target position
    max_sim_per_target = sim.max(dim=1)[0]  #batch x thought_steps
    
    #mask out padded target positions
    masked_sim = max_sim_per_target * target_mask  #batch x thought_steps
    
    #average over valid positions per sample (avoid div by zero)
    valid_counts = target_mask.sum(dim=1).clamp(min=1)  #batch
    sample_scores = masked_sim.sum(dim=1) / valid_counts  #batch
    
    #coverage loss: 1 - average best similarity (lower is better)
    coverage_loss = 1.0 - sample_scores.mean()
    
    #-----------------------------------------------
    #diversity penalty: discourage thought steps from being too similar
    #-----------------------------------------------
    #compute self-similarity of predictions
    self_sim = torch.bmm(mu_norm, mu_norm.transpose(1, 2))  #batch x t x t
    
    #mask out diagonal (self-similarity of 1.0)
    diag_mask = ~torch.eye(thought_steps, dtype=torch.bool, device=mu.device)
    diag_mask = diag_mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    #average off-diagonal similarity (penalise high similarity between different steps)
    diversity_loss = self_sim[diag_mask].view(batch_size, -1).mean()
    
    #total loss combines coverage and diversity
    total_loss = coverage_loss + diversity_weight * diversity_loss
    
    return total_loss

#-----------------------------------------------
#Pretraining step
#-----------------------------------------------
def pretrain_step(model, batch_eeg, target_embeds, target_mask, seg_len,
                  diversity_weight=0.2, augment=True, augment_config=None, augment_factor=6):
 
    device = batch_eeg.device
    
    #apply augmentation if enabled
    if augment:
        augmented_samples = [batch_eeg]  #original first
        expanded_embeds = [target_embeds]
        expanded_masks = [target_mask]
        
        for _ in range(augment_factor):
            augmented = augment_batch(batch_eeg, augment_config)
            augmented_samples.append(augmented)
            expanded_embeds.append(target_embeds)
            expanded_masks.append(target_mask)

        #concatenate along batch dimension (dim=0), not stack
        batch_eeg = torch.cat(augmented_samples, dim=0)  #(batch * (1+aug_factor)) x channels x timepoints
        target_embeds = torch.cat(expanded_embeds, dim=0)  #(batch * (1+aug_factor)) x thought_steps x embed_dim
        target_mask = torch.cat(expanded_masks, dim=0)  #(batch * (1+aug_factor)) x thought_steps
    
    #get batch size AFTER augmentation expansion
    batch_size = batch_eeg.shape[0]
    
    #initialise recurrent states
    cognitive_state, motor_state, prev_output = None, None, None
    
    #segment each sample into windows
    #note: all samples have same total length so same number of windows
    sample_windows = []
    for b in range(batch_size):
        windows = segment_eeg_tensor(batch_eeg[b], seg_len)  #list of segment tensors
        sample_windows.append(windows)
    
    n_windows = len(sample_windows[0])
    
    losses = []

    #process windows sequentially, batched across samples
    for w in range(n_windows):
        #stack window w from all samples: batch x channels x seg_len
        window_batch = torch.stack([sample_windows[b][w] for b in range(batch_size)], dim=0)
        
        #forward through feature extractor (internally accumulates state_t)
        state_t = model.extract_features(window_batch)
        
        #forward through cognitive layer
        signals, cognitive_state = model.think(state_t, cognitive_state, prev_output=prev_output)
        
        #forward through policy (only using mu, ignoring variance for pretraining)
        mu, _, motor_state = model.propagate_action(signals, motor_state)
        loss = compute_similarity_loss(mu, target_embeds, target_mask, diversity_weight)
        losses.append(loss)
        #update previous output for next window
        prev_output = mu
    
    #final reasoning using averaged accumulated state
    final_state = model.get_final_state()  #averages buffer and resets
    signals, _ = model.think(final_state, cognitive_state, prev_output=prev_output)
    mu, _, _ = model.propagate_action(signals, motor_state)
    
    #compute similarity loss (differentiable)
    loss = compute_similarity_loss(mu, target_embeds, target_mask, diversity_weight)
    losses.append(loss * 2.0) #weight final loss higher
    total_loss = torch.stack(losses, dim=0).mean() #avg loss
    return total_loss, mu, batch_size


def dynamic_update_steps(epoch, avg_loss=None, min_steps=8, max_steps=40, warmup_epochs=50):
    #scale updates based on reward progress
    #early epochs or no reward data: use minimum
    #as reward improves: scale up updates
    
    if epoch < warmup_epochs or avg_loss is None:
        return min_steps
    
    #loss typically 0-1 range (coverage loss)
    #higher loss = more steps needed, lower loss = fewer steps
    loss_factor = max(0.0, min(1.0, avg_loss))
    update_steps = int(min_steps + loss_factor * (max_steps - min_steps))
    
    return update_steps

#-----------------------------------------------
#Main pretraining function
#-----------------------------------------------
def pretrain_feature_extractor(model, train_tensor, label_id_tensor, label_mapping,
                               n_epochs, batch_size, seg_len, lr, thought_steps=None,
                               diversity_weight=0.1, grad_clip=5.0, save_freq=50,
                               augment=True, augment_config=None, augment_factor=6,
                               device='cuda', save_path=DEFAULT_SAVE_PATH):
    
    #get thought_steps from model if not provided
    if thought_steps is None:
        thought_steps = model.thought_steps
    
    #move to device
    model.to(device)
    train_tensor = train_tensor.to(device)
    label_id_tensor = label_id_tensor.to(device)
    
    #-----------------------------------------------
    #isolate parameters to train (exclude variance projector - that's for RL)
    #-----------------------------------------------
    pretrain_params = (
        list(model.feature_extractor.parameters()) +
        list(model.cognitive_layer.parameters()) +
        list(model.policy_prop.parameters()) +
        list(model.mean_projector.parameters())
    )
    
    #initialise optimiser and scheduler
    optim = torch.optim.Adam(pretrain_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optim, mode='min', factor=0.5, patience=100, min_lr=1e-6
    )
    
    best_loss_tracker = float('inf')  #track lowest loss (lower is better)
    avg_loss_tracker = None
    num_samples = train_tensor.shape[0]
    
    model.train()  #put model in train mode for dropout etc
    
    print(f"Starting pretraining: {n_epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"Training samples: {num_samples}, thought_steps: {thought_steps}")
    print(f"Augmentation: {augment}")
    print("-" * 60)
    
    for epoch in range(n_epochs):
        
        #shuffle indices each epoch
        # perm = torch.randperm(num_samples, device=device)
        epoch_loss = 0.0
        n_batches = 0
        
        #create progress bar for batches
        n_steps = dynamic_update_steps(epoch, avg_loss_tracker)
        update_bar = tqdm(range(n_steps), desc=f"Epoch {epoch+1}/{n_epochs} | steps: {n_steps}")

        for _ in update_bar:
            #randomly sample batch indices each step
            batch_indices = torch.randint(0, num_samples, (batch_size,), device=device)
            
            #get batch data
            batch_eeg = train_tensor[batch_indices]  #batch x channels x timepoints
            batch_label_ids = label_id_tensor[batch_indices]  #batch
            
            #prepare padded target embeddings
            target_embeds, target_lengths, target_mask = prepare_batch_targets(
                model, batch_label_ids, label_mapping, thought_steps, device
            )
            
            #skip batch if no valid targets (all padding)
            if target_mask.sum() == 0:
                continue
            
            #zero gradients
            optim.zero_grad()
            
            #forward pass and loss computation (with augmentation)
            loss, _, actual_batch_size = pretrain_step(
                model, batch_eeg, target_embeds, target_mask, seg_len, 
                diversity_weight, augment, augment_config, augment_factor
            )
            
            #backward pass
            loss.backward()
            
            #gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(pretrain_params, max_norm=grad_clip)
            
            #update weights
            optim.step()
            
            #accumulate loss
            epoch_loss += loss.item()
            n_batches += 1
            
            #update progress bar
            update_bar.set_postfix({'batch_size': f'{actual_batch_size}', 
                                    'loss': f'{loss.item():.4f}'})
        
        #compute average loss for epoch
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_loss_tracker = avg_loss
        
        #step scheduler based on epoch loss
        scheduler.step(avg_loss)
        
        #get current learning rate
        current_lr = optim.param_groups[0]['lr']
        
        #print epoch stats
        print(f"Epoch {epoch+1}/{n_epochs} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
        #save best model (updates tracker internally)
        best_loss_tracker = save_best_model(
            model, epoch+1, avg_loss, best_loss_tracker, save_path=save_path
        )
        
        #periodic checkpoint saving
        if (epoch + 1) % save_freq == 0:
            save_checkpoint(model, epoch+1, avg_loss, save_path=save_path)
    
    print("-" * 60)
    print(f"Pretraining complete. Best loss: {best_loss_tracker:.4f}")
    
    return best_loss_tracker


#-----------------------------------------------
#Saving utilities
#-----------------------------------------------
def save_best_model(model, epoch, mean_loss, best_loss_tracker, save_path=DEFAULT_SAVE_PATH):

    if mean_loss < best_loss_tracker:
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, 'pretmodel_best.pt')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_loss': mean_loss
        }, filepath)
        
        print(f"  → Saved best model (loss: {mean_loss:.4f})")
        
        return mean_loss  #return new best
    
    return best_loss_tracker  #return unchanged


def save_checkpoint(model, epoch, loss, save_path=DEFAULT_SAVE_PATH):

    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, f'pretmodel_epoch_{epoch}.pt')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss
    }, filepath)
    
    print(f"  → Saved checkpoint at epoch {epoch}")


def load_pretrained_model(model, checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', None),
        'loss': checkpoint.get('best_loss', checkpoint.get('loss', None))
    }
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint_info['epoch']}, Loss: {checkpoint_info['loss']:.4f}")
    
    return model, checkpoint_info