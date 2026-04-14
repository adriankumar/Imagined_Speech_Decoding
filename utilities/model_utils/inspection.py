import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter

def load_inference_data(num_sentences, dataset, device='cpu'):
    #wrapper around existing data loading for inspection
    #import here to avoid circular imports
    from utilities.chisco_preprocessing import get_train_list
    
    train_tensor, label_id_tensor, label_mapping, vocab_list, vocab_dict = get_train_list(
        train_size=num_sentences, dataset=dataset
    )
    
    train_tensor = train_tensor.to(device)
    label_id_tensor = label_id_tensor.to(device)
    
    print(f"loaded {train_tensor.shape[0]} samples for {num_sentences} sentences")
    print(f"vocab size: {len(vocab_list)}")
    
    return train_tensor, label_id_tensor, label_mapping, vocab_list, vocab_dict


#-----------------------------------------------
#Forward pass inspection
#-----------------------------------------------
def run_full_inference(model, eeg_sample, segment_length=500):
    #runs complete forward pass through all windows + final reasoning
    #returns dict with all intermediate tensors for inspection
    #assumes eeg_sample is shape: channels x timepoints (no batch dim)
    
    from utilities.chisco_preprocessing import segment_eeg_tensor
    
    model.eval()
    device = next(model.parameters()).device
    
    #add batch dim and move to device
    if eeg_sample.dim() == 2:
        eeg_sample = eeg_sample.unsqueeze(0)
    eeg_sample = eeg_sample.to(device)
    
    #segment into windows
    windows = segment_eeg_tensor(eeg_sample.squeeze(0), segment_length)
    
    #storage for intermediates
    outputs = {
        'windows': [],
        'state_t': [],
        'sensory_prop': [],
        'buffer_states': [],
        'cognitive_signals': [],
        'mu': [],
        'log_sigma': [],
        'sampled_action': None,
        'final_mu': None,
        'final_log_sigma': None,
        'final_action': None,
        'word_ids': None,
        'confidences': None,
        'predicted_sentence': None
    }
    
    #model states
    sensory_state, buffer, cognitive_state, policy_state = None, None, None, None
    
    with torch.no_grad():
        #process each window
        for i, window in enumerate(windows):
            window = window.unsqueeze(0).to(device)
            outputs['windows'].append(window.cpu())
            
            #feature extraction
            state_t = model.extract_features(window)
            outputs['state_t'].append(state_t.cpu())
            
            #sensory propagation
            sensory_prop, sensory_state = model.propagate_sensory(state_t, sensory_state)
            outputs['sensory_prop'].append(sensory_prop.cpu())
            
            #buffer update
            buffer = model.update_buffer(sensory_prop, buffer)
            outputs['buffer_states'].append(buffer.cpu().clone())
            
            #cognitive processing
            signals, cognitive_state, buffer = model.think(buffer, cognitive_state)
            outputs['cognitive_signals'].append(signals.cpu())
            
            #policy output
            mu, log_sigma, policy_state = model.propagate_action(signals, policy_state)
            outputs['mu'].append(mu.cpu())
            'log_sigma'
            outputs['log_sigma'].append(log_sigma.cpu())
            
            #sample action for this window
            action, log_prob = model.sample_action(mu, log_sigma)
            
        #final reasoning
        final_signals, cognitive_state, buffer = model.think(buffer, cognitive_state)
        final_mu, final_log_sigma, policy_state = model.propagate_action(final_signals, policy_state)
        final_action, final_log_prob = model.sample_action(final_mu, final_log_sigma)
        
        outputs['final_mu'] = final_mu.cpu()
        outputs['final_log_sigma'] = final_log_sigma.cpu()
        outputs['final_action'] = final_action.cpu()
        outputs['final_log_prob'] = final_log_prob.cpu()
        
        #decode to words
        word_ids, confidences, avg_conf = model.decode_vocab_ids(final_mu, return_confidences=True)
        outputs['word_ids'] = word_ids.cpu()
        outputs['confidences'] = confidences.cpu()
        outputs['avg_confidence'] = avg_conf.cpu()
        
        #construct sentence
        predicted_sentence = model.construct_sentence(word_ids)
        outputs['predicted_sentence'] = predicted_sentence
        
        #reset buffer
        model.reset_buffer(buffer)
    
    return outputs


def print_tensor_stats(tensor, name, indent=2):
    #prints shape and statistics for a tensor
    prefix = " " * indent
    
    if tensor is None:
        print(f"{prefix}{name}: None")
        return
    
    if isinstance(tensor, str):
        print(f"{prefix}{name}: '{tensor}'")
        return
    
    if isinstance(tensor, list):
        if len(tensor) == 0:
            print(f"{prefix}{name}: empty list")
            return
        print(f"{prefix}{name}: list of {len(tensor)} tensors")
        #print stats for first and last
        if torch.is_tensor(tensor[0]):
            print_tensor_stats(tensor[0], f"{name}[0]", indent + 2)
            if len(tensor) > 1:
                print_tensor_stats(tensor[-1], f"{name}[-1]", indent + 2)
        return
    
    if not torch.is_tensor(tensor):
        print(f"{prefix}{name}: {type(tensor).__name__} = {tensor}")
        return
    
    t = tensor.float()
    num_zeros = (tensor == 0).sum().item()
    total = tensor.numel()
    pct_zeros = 100 * num_zeros / total if total > 0 else 0
    
    if total > 1:
        std_val = t.std().item()
    else:
        std_val = 0.0 #std undefined for single element

    print(f"{prefix}{name}:")
    print(f"{prefix}  shape: {list(tensor.shape)}")
    print(f"{prefix}  mean: {t.mean().item():.6f}, std: {std_val:.6f}")
    print(f"{prefix}  min: {t.min().item():.6f}, max: {t.max().item():.6f}")
    print(f"{prefix}  zeros: {pct_zeros:.1f}%")


def print_inference_summary(outputs):
    #prints summary statistics for all intermediate outputs
    print("=" * 60)
    print("INFERENCE SUMMARY")
    print("=" * 60)
    
    print("\n[WINDOW PROCESSING]")
    print(f"  num windows: {len(outputs['windows'])}")
    
    print("\n[FEATURE EXTRACTION - state_t]")
    print_tensor_stats(outputs['state_t'], 'state_t')
    
    print("\n[SENSORY PROPAGATION]")
    print_tensor_stats(outputs['sensory_prop'], 'sensory_prop')
    
    print("\n[BUFFER STATES]")
    print_tensor_stats(outputs['buffer_states'], 'buffer_states')
    
    print("\n[COGNITIVE SIGNALS]")
    print_tensor_stats(outputs['cognitive_signals'], 'cognitive_signals')
    
    print("\n[POLICY OUTPUT - intermediate]")
    print_tensor_stats(outputs['mu'], 'mu')
    print_tensor_stats(outputs['log_sigma'], 'log_sigma')
    
    print("\n[FINAL REASONING]")
    print_tensor_stats(outputs['final_mu'], 'final_mu')
    print_tensor_stats(outputs['final_log_sigma'], 'final_log_sigma')
    print_tensor_stats(outputs['final_action'], 'final_action')
    print_tensor_stats(outputs['final_log_prob'], 'final_log_prob')
    
    print("\n[DECODING]")
    print_tensor_stats(outputs['word_ids'], 'word_ids')
    print_tensor_stats(outputs['confidences'], 'confidences')
    print_tensor_stats(outputs['avg_confidence'], 'avg_confidence')
    print(f"  predicted: '{outputs['predicted_sentence']}'")
    
    print("=" * 60)


#-----------------------------------------------
#Single prediction inspection
#-----------------------------------------------
def inspect_single_prediction(model, eeg_sample, true_words, vocab_list, segment_length=500):
    #runs inference and prints detailed comparison with ground truth
    
    outputs = run_full_inference(model, eeg_sample, segment_length)
    
    print("=" * 60)
    print("SINGLE PREDICTION INSPECTION")
    print("=" * 60)
    
    #ground truth
    true_sentence = ' '.join(true_words)
    print(f"\n[GROUND TRUTH]")
    print(f"  sentence: '{true_sentence}'")
    print(f"  words: {true_words}")
    print(f"  length: {len(true_words)}")
    
    #prediction
    pred_sentence = outputs['predicted_sentence']
    pred_words = pred_sentence.split() if pred_sentence else []
    print(f"\n[PREDICTION]")
    print(f"  sentence: '{pred_sentence}'")
    print(f"  words: {pred_words}")
    print(f"  length: {len(pred_words)}")
    
    #per-word confidence breakdown
    word_ids = outputs['word_ids'].squeeze(0)  #thought_steps
    confidences = outputs['confidences'].squeeze(0)  #thought_steps
    
    print(f"\n[PER-STEP BREAKDOWN] (16 thinking steps)")
    print(f"  {'step':<6} {'word_id':<10} {'word':<20} {'confidence':<12}")
    print(f"  {'-'*48}")
    
    for step in range(len(word_ids)):
        wid = word_ids[step].item()
        word = vocab_list[wid]
        conf = confidences[step].item()
        marker = "" if word == "<PAD>" else "←"
        print(f"  {step:<6} {wid:<10} {word:<20} {conf:<12.4f} {marker}")
    
    #confidence statistics
    avg_conf = outputs['avg_confidence'].item()
    non_pad_mask = word_ids != 0  #assuming PAD is index 0
    if non_pad_mask.any():
        non_pad_conf = confidences[non_pad_mask].mean().item()
    else:
        non_pad_conf = 0.0
    
    print(f"\n[CONFIDENCE STATS]")
    print(f"  avg (all steps): {avg_conf:.4f}")
    print(f"  avg (non-PAD): {non_pad_conf:.4f}")
    print(f"  min: {confidences.min().item():.4f}")
    print(f"  max: {confidences.max().item():.4f}")
    
    #word overlap analysis
    true_set = set(true_words)
    pred_set = set(pred_words)
    overlap = true_set & pred_set
    
    print(f"\n[WORD OVERLAP]")
    print(f"  true words: {true_set}")
    print(f"  pred words: {pred_set}")
    print(f"  overlap: {overlap}")
    print(f"  jaccard: {len(overlap) / len(true_set | pred_set) if pred_set else 0:.4f}")
    
    #policy output statistics
    print(f"\n[POLICY OUTPUT STATS]")
    final_mu = outputs['final_mu']
    final_ls = outputs['final_log_sigma']
    print(f"  mu - mean: {final_mu.mean().item():.4f}, std: {final_mu.std().item():.4f}")
    print(f"  mu - range: [{final_mu.min().item():.4f}, {final_mu.max().item():.4f}]")
    print(f"  log_sigma - mean: {final_ls.mean().item():.4f}, std: {final_ls.std().item():.4f}")
    print(f"  log_sigma - range: [{final_ls.min().item():.4f}, {final_ls.max().item():.4f}]")
    
    print("=" * 60)
    
    return outputs


def inspect_batch_predictions(model, train_tensor, label_id_tensor, label_mapping, 
                              vocab_list, n_samples=None, segment_length=500):
    #runs inference on multiple samples and collects statistics
    
    if n_samples is None:
        n_samples = train_tensor.shape[0]
    n_samples = min(n_samples, train_tensor.shape[0])
    
    results = {
        'predicted_sentences': [],
        'true_sentences': [],
        'confidences': [],
        'word_ids': [],
        'semantic_scores': [],
        'jaccard_scores': []
    }
    
    print(f"running inference on {n_samples} samples...")
    
    for i in range(n_samples):
        eeg_sample = train_tensor[i]
        label_id = label_id_tensor[i].item()
        true_words = label_mapping[label_id]
        
        outputs = run_full_inference(model, eeg_sample, segment_length)
        
        pred_sentence = outputs['predicted_sentence']
        pred_words = pred_sentence.split() if pred_sentence else []
        
        #collect results
        results['predicted_sentences'].append(pred_sentence)
        results['true_sentences'].append(' '.join(true_words))
        results['confidences'].append(outputs['avg_confidence'].item())
        results['word_ids'].append(outputs['word_ids'].squeeze(0).tolist())
        
        #jaccard
        true_set = set(true_words)
        pred_set = set(pred_words)
        jaccard = len(true_set & pred_set) / len(true_set | pred_set) if pred_set else 0
        results['jaccard_scores'].append(jaccard)
    
    #print summary
    print("\n" + "=" * 60)
    print("BATCH PREDICTION SUMMARY")
    print("=" * 60)
    
    print(f"\n[PREDICTIONS]")
    for i in range(min(n_samples, 10)):  #show first 10
        print(f"  [{i}] true: '{results['true_sentences'][i]}'")
        print(f"       pred: '{results['predicted_sentences'][i]}'")
        print(f"       conf: {results['confidences'][i]:.4f}, jaccard: {results['jaccard_scores'][i]:.4f}")
        print()
    
    print(f"[OVERALL STATS]")
    print(f"  avg confidence: {np.mean(results['confidences']):.4f}")
    print(f"  avg jaccard: {np.mean(results['jaccard_scores']):.4f}")
    
    #check for word collapse
    all_word_ids = [wid for wids in results['word_ids'] for wid in wids]
    unique_words = len(set(all_word_ids))
    print(f"  unique words predicted: {unique_words} / {len(vocab_list)} vocab")
    
    print("=" * 60)
    
    return results


#-----------------------------------------------
#Gradient inspection
#-----------------------------------------------
def compute_gradient_norms(model, eeg_sample, true_words, sentence_encoder, 
                           segment_length=500, device='cpu'):
    #computes gradient L2 norms for each module after one forward-backward pass
    #uses semantic similarity as the loss signal
    
    from utilities.chisco_preprocessing import segment_eeg_tensor
    
    model.train()
    model.zero_grad()
    
    #prepare sample
    if eeg_sample.dim() == 2:
        eeg_sample = eeg_sample.unsqueeze(0)
    eeg_sample = eeg_sample.to(device)
    
    #segment
    windows = segment_eeg_tensor(eeg_sample.squeeze(0), segment_length)
    
    #model states
    sensory_state, buffer, cognitive_state, policy_state = None, None, None, None
    
    #forward pass through all windows
    for window in windows:
        window = window.unsqueeze(0).to(device)
        
        state_t = model.extract_features(window)
        sensory_prop, sensory_state = model.propagate_sensory(state_t, sensory_state)
        buffer = model.update_buffer(sensory_prop, buffer)
        signals, cognitive_state, buffer = model.think(buffer, cognitive_state)
        mu, log_sigma, policy_state = model.propagate_action(signals, policy_state)
    
    #final reasoning
    final_signals, cognitive_state, buffer = model.think(buffer, cognitive_state)
    final_mu, final_log_sigma, policy_state = model.propagate_action(final_signals, policy_state)
    
    #decode and compute semantic loss
    #layer norm for cosine similarity
    mu_norm = F.layer_norm(final_mu.transpose(1, 2), [final_mu.shape[1]])  #b x t x e
    vocab_norm = F.layer_norm(model.vocab_embedding, [model.vocab_embedding.shape[-1]])
    
    #get word predictions
    similarity = torch.matmul(mu_norm, vocab_norm.T)
    word_ids = similarity.argmax(dim=-1).squeeze(0)
    
    #construct predicted sentence
    pred_words = []
    for wid in word_ids:
        word = model.vocab_list[wid.item()]
        if word != '<PAD>':
            pred_words.append(word)
    pred_sentence = ' '.join(pred_words) if pred_words else ''
    true_sentence = ' '.join(true_words)
    
    #compute semantic similarity as target
    with torch.no_grad():
        pred_emb = sentence_encoder.encode([pred_sentence])[0]
        true_emb = sentence_encoder.encode([true_sentence])[0]
        target_similarity = sklearn_cosine([pred_emb], [true_emb])[0][0]
    
    #use negative similarity as loss (we want to maximize similarity)
    #create a differentiable proxy: maximize mean similarity to vocab embeddings of correct class
    #simplified: just use mean of action as proxy loss for gradient flow analysis
    loss = -final_mu.mean()  #simple proxy to check gradient flow
    
    #backward
    loss.backward()
    
    #collect gradient norms per module
    grad_norms = {}
    
    #feature extractor components
    grad_norms['feature_extractor'] = get_module_grad_norm(model.feature_extractor)
    
    #try to get sub-components if they exist
    if hasattr(model.feature_extractor, 'hjorth'):
        grad_norms['fe_hjorth'] = get_module_grad_norm(model.feature_extractor.hjorth)
    if hasattr(model.feature_extractor, 'sinc_net'):
        grad_norms['fe_sinc'] = get_module_grad_norm(model.feature_extractor.sinc_net)
    if hasattr(model.feature_extractor, 'bp_projector'):
        grad_norms['fe_bandpower'] = get_module_grad_norm(model.feature_extractor.bp_projector)
    if hasattr(model.feature_extractor, 'raw_spatial'):
        grad_norms['fe_raw_spatial'] = get_module_grad_norm(model.feature_extractor.raw_spatial)
    if hasattr(model.feature_extractor, 'spectro_conv'):
        grad_norms['fe_spectro'] = get_module_grad_norm(model.feature_extractor.spectro_conv)
    
    #sensory propagator
    grad_norms['sensory_propagator'] = get_module_grad_norm(model.sensory_propagator)
    
    #cognitive layer
    grad_norms['cognitive_layer'] = get_module_grad_norm(model.cognitive_layer)
    
    #policy components
    grad_norms['policy_prop'] = get_module_grad_norm(model.policy_prop)
    grad_norms['mean_projector'] = get_module_grad_norm(model.mean_projector)
    grad_norms['variance_projector'] = get_module_grad_norm(model.variance_projector)
    
    #reset
    model.zero_grad()
    model.reset_buffer(buffer)
    
    return grad_norms


def get_module_grad_norm(module):
    #computes L2 norm of gradients for all parameters in a module
    total_norm = 0.0
    num_params = 0
    
    for param in module.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
            num_params += 1
    
    if num_params == 0:
        return 0.0
    
    return np.sqrt(total_norm)


def print_gradient_summary(grad_norms):
    #prints formatted table of gradient norms
    print("=" * 60)
    print("GRADIENT NORMS BY MODULE")
    print("=" * 60)
    
    #sort by gradient magnitude
    sorted_norms = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)
    
    max_norm = max(grad_norms.values()) if grad_norms.values() else 1.0
    
    print(f"\n  {'module':<25} {'grad norm':<15} {'relative':<10}")
    print(f"  {'-'*50}")
    
    for name, norm in sorted_norms:
        relative = norm / max_norm if max_norm > 0 else 0
        bar = "█" * int(relative * 20)
        print(f"  {name:<25} {norm:<15.6f} {bar}")
    
    print(f"\n[ANALYSIS]")
    
    #check for gradient issues
    fe_norm = grad_norms.get('feature_extractor', 0)
    policy_norm = grad_norms.get('mean_projector', 0) + grad_norms.get('variance_projector', 0)
    
    if fe_norm < 1e-6:
        print("  ⚠️  WARNING: Feature extractor gradients near zero - gradient starvation!")
    elif fe_norm < policy_norm * 0.01:
        print("  ⚠️  WARNING: Feature extractor gradients 100x smaller than policy - potential starvation")
    else:
        print("  ✓ Gradients flowing to feature extractor")
    
    cog_norm = grad_norms.get('cognitive_layer', 0)
    if cog_norm < 1e-6:
        print("  ⚠️  WARNING: Cognitive layer gradients near zero")
    
    print("=" * 60)


def plot_gradient_norms(grad_norms, figsize=(12, 6)):
    #bar plot of gradient norms across modules
    
    #separate into feature extractor sub-components and main modules
    fe_keys = [k for k in grad_norms.keys() if k.startswith('fe_')]
    main_keys = [k for k in grad_norms.keys() if not k.startswith('fe_')]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    #main modules
    if main_keys:
        names = main_keys
        values = [grad_norms[k] for k in names]
        colors = ['red' if v < 1e-6 else 'steelblue' for v in values]
        
        axes[0].barh(names, values, color=colors)
        axes[0].set_xlabel('gradient L2 norm')
        axes[0].set_title('main modules')
        axes[0].grid(True, alpha=0.3)
    
    #feature extractor sub-components
    if fe_keys:
        names = [k.replace('fe_', '') for k in fe_keys]
        values = [grad_norms[k] for k in fe_keys]
        colors = ['red' if v < 1e-6 else 'green' for v in values]
        
        axes[1].barh(names, values, color=colors)
        axes[1].set_xlabel('gradient L2 norm')
        axes[1].set_title('feature extractor components')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


#-----------------------------------------------
#Quick diagnostic runner
#-----------------------------------------------
def run_phase1_diagnostics(model, train_tensor, label_id_tensor, label_mapping, 
                           vocab_list, sentence_encoder, device='cpu', segment_length=500):
    #runs all phase 1 diagnostics in sequence
    
    print("\n" + "="*70)
    print("PHASE 1 DIAGNOSTICS")
    print("="*70)
    
    #pick first sample for detailed inspection
    sample_idx = 0
    eeg_sample = train_tensor[sample_idx]
    label_id = label_id_tensor[sample_idx].item()
    true_words = label_mapping[label_id]
    
    print(f"\nusing sample {sample_idx} with label: '{' '.join(true_words)}'")
    
    #1. full inference inspection
    print("\n[1/4] FULL INFERENCE INSPECTION")
    outputs = run_full_inference(model, eeg_sample, segment_length)
    print_inference_summary(outputs)
    
    #2. single prediction inspection
    print("\n[2/4] SINGLE PREDICTION INSPECTION")
    inspect_single_prediction(model, eeg_sample, true_words, vocab_list, segment_length)
    
    #3. batch predictions
    print("\n[3/4] BATCH PREDICTIONS")
    results = inspect_batch_predictions(model, train_tensor, label_id_tensor, 
                                        label_mapping, vocab_list, n_samples=None, 
                                        segment_length=segment_length)
    
    #4. gradient analysis
    print("\n[4/4] GRADIENT ANALYSIS")
    grad_norms = compute_gradient_norms(model, eeg_sample, true_words, 
                                        sentence_encoder, segment_length, device)
    print_gradient_summary(grad_norms)
    plot_gradient_norms(grad_norms)
    
    return outputs, results, grad_norms

#-----------------------------------------------
#Prediction analysis
#-----------------------------------------------
def plot_confidence_histogram(model, train_tensor, label_id_tensor, segment_length=500, 
                               n_samples=None, figsize=(12, 5)):
    #plots distribution of confidence values across samples
    #checks if model is uniformly confident or has meaningful variance
    
    from utilities.chisco_preprocessing import segment_eeg_tensor
    
    model.eval()
    device = next(model.parameters()).device
    
    if n_samples is None:
        n_samples = train_tensor.shape[0]
    n_samples = min(n_samples, train_tensor.shape[0])
    
    all_confidences = []  #per-word confidences
    avg_confidences = []  #per-sample average
    
    with torch.no_grad():
        for i in range(n_samples):
            eeg_sample = train_tensor[i].unsqueeze(0).to(device)
            windows = segment_eeg_tensor(eeg_sample.squeeze(0), segment_length)
            
            #forward pass through windows
            sensory_state, buffer, cognitive_state, policy_state = None, None, None, None
            
            for window in windows:
                window = window.unsqueeze(0).to(device)
                state_t = model.extract_features(window)
                sensory_prop, sensory_state = model.propagate_sensory(state_t, sensory_state)
                buffer = model.update_buffer(sensory_prop, buffer)
                signals, cognitive_state, buffer = model.think(buffer, cognitive_state)
                mu, log_sigma, policy_state = model.propagate_action(signals, policy_state)
            
            #final reasoning
            final_signals, cognitive_state, buffer = model.think(buffer, cognitive_state)
            final_mu, _, _ = model.propagate_action(final_signals, policy_state)
            
            #decode
            _, confidences, avg_conf = model.decode_vocab_ids(final_mu, return_confidences=True)
            
            all_confidences.extend(confidences.squeeze(0).cpu().numpy().tolist())
            avg_confidences.append(avg_conf.item())
            
            model.reset_buffer(buffer)
    
    #plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    #per-word confidence distribution
    n_bins = min(30, max(5, len(all_confidences) // 10))
    axes[0].hist(all_confidences, bins=n_bins, edgecolor='black', alpha=0.7, color='orange')
    axes[0].axvline(np.mean(all_confidences), color='red', linestyle='--', 
                    label=f'mean={np.mean(all_confidences):.3f}')
    axes[0].set_xlabel('confidence')
    axes[0].set_ylabel('count')
    axes[0].set_title('per-word confidence distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    #per-sample average confidence
    n_bins_avg = min(20, max(5, len(avg_confidences) // 2))
    axes[1].hist(avg_confidences, bins=n_bins_avg, edgecolor='black', alpha=0.7, color='teal')
    axes[1].axvline(np.mean(avg_confidences), color='red', linestyle='--',
                    label=f'mean={np.mean(avg_confidences):.3f}')
    axes[1].set_xlabel('average confidence')
    axes[1].set_ylabel('count')
    axes[1].set_title('per-sample average confidence')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    #print stats
    print("=" * 50)
    print("CONFIDENCE STATISTICS")
    print("=" * 50)
    print(f"  per-word mean: {np.mean(all_confidences):.4f}")
    print(f"  per-word std: {np.std(all_confidences):.4f}")
    print(f"  per-word range: [{np.min(all_confidences):.4f}, {np.max(all_confidences):.4f}]")
    print(f"  per-sample mean: {np.mean(avg_confidences):.4f}")
    print(f"  per-sample std: {np.std(avg_confidences):.4f}")
    print("=" * 50)
    
    return all_confidences, avg_confidences


def plot_word_frequency(model, train_tensor, vocab_list, segment_length=500, 
                        n_samples=None, figsize=(12, 6)):
    #plots frequency of predicted words - checks for word collapse
    
    from utilities.chisco_preprocessing import segment_eeg_tensor
    
    model.eval()
    device = next(model.parameters()).device
    
    if n_samples is None:
        n_samples = train_tensor.shape[0]
    n_samples = min(n_samples, train_tensor.shape[0])
    
    all_word_ids = []
    
    with torch.no_grad():
        for i in range(n_samples):
            eeg_sample = train_tensor[i].unsqueeze(0).to(device)
            windows = segment_eeg_tensor(eeg_sample.squeeze(0), segment_length)
            
            #forward pass
            sensory_state, buffer, cognitive_state, policy_state = None, None, None, None
            
            for window in windows:
                window = window.unsqueeze(0).to(device)
                state_t = model.extract_features(window)
                sensory_prop, sensory_state = model.propagate_sensory(state_t, sensory_state)
                buffer = model.update_buffer(sensory_prop, buffer)
                signals, cognitive_state, buffer = model.think(buffer, cognitive_state)
                mu, log_sigma, policy_state = model.propagate_action(signals, policy_state)
            
            #final reasoning
            final_signals, cognitive_state, buffer = model.think(buffer, cognitive_state)
            final_mu, _, _ = model.propagate_action(final_signals, policy_state)
            
            #decode
            word_ids, _, _ = model.decode_vocab_ids(final_mu, return_confidences=True)
            all_word_ids.extend(word_ids.squeeze(0).cpu().numpy().tolist())
            
            model.reset_buffer(buffer)
    
    #count frequencies
    word_counts = Counter(all_word_ids)
    
    #plot
    plt.figure(figsize=figsize)
    
    words = [vocab_list[i] for i in range(len(vocab_list))]
    counts = [word_counts.get(i, 0) for i in range(len(vocab_list))]
    
    colors = ['red' if c == 0 else 'steelblue' for c in counts]
    
    plt.bar(range(len(vocab_list)), counts, color=colors, alpha=0.7)
    plt.xticks(range(len(vocab_list)), words, rotation=90, fontsize=8)
    plt.xlabel('word')
    plt.ylabel('frequency')
    plt.title(f'word prediction frequency ({n_samples} samples × 16 steps = {n_samples * 16} predictions)')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    #print stats
    print("=" * 50)
    print("WORD FREQUENCY STATISTICS")
    print("=" * 50)
    
    total_predictions = len(all_word_ids)
    unique_predicted = len([c for c in counts if c > 0])
    
    print(f"  total predictions: {total_predictions}")
    print(f"  unique words predicted: {unique_predicted} / {len(vocab_list)}")
    print(f"  vocab utilization: {unique_predicted / len(vocab_list):.2%}")
    
    #top predicted words
    sorted_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  top 5 predicted:")
    for wid, count in sorted_counts[:5]:
        pct = count / total_predictions * 100
        print(f"    '{vocab_list[wid]}': {count} ({pct:.1f}%)")
    
    #never predicted words
    never_predicted = [vocab_list[i] for i in range(len(vocab_list)) if word_counts.get(i, 0) == 0]
    if never_predicted:
        print(f"\n  never predicted ({len(never_predicted)}): {never_predicted}")
    
    #collapse check
    if sorted_counts:
        top_word_pct = sorted_counts[0][1] / total_predictions
        if top_word_pct > 0.5:
            print(f"\n  ⚠️  WARNING: top word accounts for {top_word_pct:.1%} - severe collapse!")
        elif top_word_pct > 0.3:
            print(f"\n  ⚠️  WARNING: top word accounts for {top_word_pct:.1%} - moderate collapse")
    
    print("=" * 50)
    
    return word_counts, all_word_ids


def compute_prediction_entropy(word_counts, vocab_size):
    #computes entropy of prediction distribution - higher = more diverse
    
    total = sum(word_counts.values())
    if total == 0:
        return 0.0
    
    probs = np.array([word_counts.get(i, 0) / total for i in range(vocab_size)])
    probs = probs[probs > 0]  #filter zeros for log
    
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(vocab_size)  #uniform distribution
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    print("=" * 50)
    print("PREDICTION ENTROPY")
    print("=" * 50)
    print(f"  entropy: {entropy:.4f} bits")
    print(f"  max entropy (uniform): {max_entropy:.4f} bits")
    print(f"  normalized entropy: {normalized_entropy:.4f} (1.0 = uniform)")
    
    if normalized_entropy < 0.5:
        print(f"  ⚠️  LOW entropy - predictions heavily biased toward few words")
    elif normalized_entropy > 0.8:
        print(f"  ✓ HIGH entropy - predictions well distributed")
    
    print("=" * 50)
    
    return entropy, normalized_entropy


#-----------------------------------------------
#Action space analysis
#-----------------------------------------------
def plot_action_vs_vocab(mu, vocab_embeddings, vocab_list, method='pca', figsize=(12, 10)):
    #visualizes where actions land relative to vocab embeddings
    #checks if actions are actually near valid vocab points
    
    #handle tensors
    if torch.is_tensor(mu):
        mu = mu.detach().cpu().numpy()
    if torch.is_tensor(vocab_embeddings):
        vocab_emb = vocab_embeddings.detach().cpu().numpy()
    else:
        vocab_emb = vocab_embeddings
    
    #reshape mu if needed: b x e x t -> (b*t) x e
    if mu.ndim == 3:
        mu = mu.transpose(0, 2, 1).reshape(-1, mu.shape[1])  #flatten to n_actions x embedding_dim
    
    #combine for joint reduction
    combined = np.vstack([vocab_emb, mu])
    n_vocab = len(vocab_list)
    
    #reduce
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(combined)
        title = 'action vs vocab embeddings (PCA)'
    else:
        perplexity = min(30, len(combined) - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        reduced = reducer.fit_transform(combined)
        title = 'action vs vocab embeddings (t-SNE)'
    
    vocab_reduced = reduced[:n_vocab]
    action_reduced = reduced[n_vocab:]
    
    #plot
    plt.figure(figsize=figsize)
    
    #vocab points
    plt.scatter(vocab_reduced[:, 0], vocab_reduced[:, 1], 
                c='blue', s=100, marker='o', label='vocab', alpha=0.8)
    
    #action points
    plt.scatter(action_reduced[:, 0], action_reduced[:, 1],
                c='red', s=30, marker='x', label='actions', alpha=0.5)
    
    #annotate vocab
    for i, word in enumerate(vocab_list):
        plt.annotate(word, (vocab_reduced[i, 0], vocab_reduced[i, 1]),
                    fontsize=9, alpha=0.8)
    
    plt.xlabel('dimension 1')
    plt.ylabel('dimension 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compute_nearest_vocab_distances(mu, vocab_embeddings, vocab_list):
    #computes distance from each action to nearest vocab embedding
    #checks if actions are landing near valid vocab points
    
    #handle tensors
    if torch.is_tensor(mu):
        mu_t = mu.detach().cpu()
    else:
        mu_t = torch.tensor(mu)
    
    if torch.is_tensor(vocab_embeddings):
        vocab_t = vocab_embeddings.detach().cpu()
    else:
        vocab_t = torch.tensor(vocab_embeddings)
    
    #reshape mu if needed: b x e x t -> b x t x e
    if mu_t.ndim == 3:
        mu_t = mu_t.permute(0, 2, 1)  #b x t x e
        batch_size, n_steps, emb_dim = mu_t.shape
        mu_flat = mu_t.reshape(-1, emb_dim)  #(b*t) x e
    else:
        mu_flat = mu_t
        n_steps = 1
    
    #normalize for cosine similarity
    mu_norm = F.normalize(mu_flat, dim=-1)
    vocab_norm = F.normalize(vocab_t, dim=-1)
    
    #compute cosine similarities
    similarities = torch.matmul(mu_norm, vocab_norm.T)  #(b*t) x vocab
    
    #get max similarity (nearest vocab) per action
    max_sims, nearest_ids = similarities.max(dim=-1)
    
    #reshape back to b x t if needed
    if mu_t.ndim == 3:
        max_sims = max_sims.reshape(batch_size, n_steps)
        nearest_ids = nearest_ids.reshape(batch_size, n_steps)
    
    #compute stats
    max_sims_np = max_sims.numpy().flatten()
    
    print("=" * 60)
    print("ACTION-TO-VOCAB DISTANCE ANALYSIS")
    print("=" * 60)
    print(f"\n[COSINE SIMILARITY TO NEAREST VOCAB]")
    print(f"  mean: {max_sims_np.mean():.4f}")
    print(f"  std: {max_sims_np.std():.4f}")
    print(f"  min: {max_sims_np.min():.4f}")
    print(f"  max: {max_sims_np.max():.4f}")
    
    #per-step analysis (if batch=1)
    if mu_t.ndim == 3 and batch_size == 1:
        print(f"\n[PER-STEP BREAKDOWN]")
        print(f"  {'step':<6} {'nearest word':<20} {'similarity':<12}")
        print(f"  {'-'*38}")
        for t in range(n_steps):
            word = vocab_list[nearest_ids[0, t].item()]
            sim = max_sims[0, t].item()
            print(f"  {t:<6} {word:<20} {sim:<12.4f}")
    
    #diagnostic
    print(f"\n[DIAGNOSTIC]")
    low_sim_count = np.sum(max_sims_np < 0.5)
    if low_sim_count > 0:
        print(f"  ⚠️  {low_sim_count} actions have similarity < 0.5 to nearest vocab")
    if max_sims_np.mean() < 0.3:
        print(f"  ⚠️  LOW mean similarity - actions not landing near vocab points")
    elif max_sims_np.mean() > 0.7:
        print(f"  ✓ HIGH mean similarity - actions close to vocab points")
    
    print("=" * 60)
    
    return max_sims, nearest_ids


def plot_sigma_distribution(log_sigma, figsize=(12, 5)):
    #plots distribution of log_sigma values
    #checks if policy variance is collapsing or exploding
    
    #handle tensor
    if torch.is_tensor(log_sigma):
        ls = log_sigma.detach().cpu().numpy()
    else:
        ls = log_sigma
    
    #flatten
    ls_flat = ls.flatten()
    sigma_flat = np.exp(ls_flat)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    #log sigma distribution
    n_bins = min(30, max(5, len(ls_flat) // 10))
    axes[0].hist(ls_flat, bins=n_bins, edgecolor='black', alpha=0.7, color='purple')
    axes[0].axvline(ls_flat.mean(), color='red', linestyle='--', 
                    label=f'mean={ls_flat.mean():.3f}')
    axes[0].set_xlabel('log(σ)')
    axes[0].set_ylabel('count')
    axes[0].set_title('log_sigma distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    #sigma distribution
    axes[1].hist(sigma_flat, bins=n_bins, edgecolor='black', alpha=0.7, color='green')
    axes[1].axvline(sigma_flat.mean(), color='red', linestyle='--',
                    label=f'mean={sigma_flat.mean():.3f}')
    axes[1].set_xlabel('σ')
    axes[1].set_ylabel('count')
    axes[1].set_title('sigma distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    #print stats
    print("=" * 50)
    print("POLICY VARIANCE STATISTICS")
    print("=" * 50)
    print(f"  log_sigma mean: {ls_flat.mean():.4f}")
    print(f"  log_sigma std: {ls_flat.std():.4f}")
    print(f"  log_sigma range: [{ls_flat.min():.4f}, {ls_flat.max():.4f}]")
    print(f"  sigma mean: {sigma_flat.mean():.4f}")
    print(f"  sigma range: [{sigma_flat.min():.4f}, {sigma_flat.max():.4f}]")
    
    #diagnostic
    if ls_flat.mean() < -3:
        print(f"\n  ⚠️  LOW variance (mean log_σ < -3) - policy may be overconfident/collapsed")
    elif ls_flat.mean() > 1:
        print(f"\n  ⚠️  HIGH variance (mean log_σ > 1) - policy too uncertain")
    else:
        print(f"\n  ✓ Variance in reasonable range")
    
    print("=" * 50)


#-----------------------------------------------
#Intermediate state visualization
#-----------------------------------------------
def plot_state_t(state_t, figsize=(12, 4)):
    #plots heatmap of feature extractor output
    
    #handle tensor
    if torch.is_tensor(state_t):
        st = state_t.detach().cpu().numpy()
    else:
        st = state_t
    
    #squeeze batch if needed
    if st.ndim == 2:
        st = st.squeeze(0)  #remove batch dim
    
    plt.figure(figsize=figsize)
    
    if st.ndim == 1:
        #single vector - bar plot
        plt.bar(range(len(st)), st, alpha=0.7)
        plt.xlabel('feature dimension')
        plt.ylabel('value')
        plt.title(f'state_t features ({len(st)} dims)')
    else:
        #multiple vectors - heatmap
        plt.imshow(st, aspect='auto', cmap='RdBu_r')
        plt.colorbar(label='value')
        plt.xlabel('feature dimension')
        plt.ylabel('sample')
        plt.title(f'state_t features (shape: {st.shape})')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_cognitive_signals(signals, figsize=(10, 6)):
    #plots heatmap of CTM output (output_dim x thinking_steps)
    
    #handle tensor
    if torch.is_tensor(signals):
        sig = signals.detach().cpu().numpy()
    else:
        sig = signals
    
    #squeeze batch if needed
    if sig.ndim == 3:
        sig = sig.squeeze(0)  #b x output_dim x thinking_steps -> output_dim x thinking_steps
    
    plt.figure(figsize=figsize)
    plt.imshow(sig, aspect='auto', cmap='RdBu_r')
    plt.colorbar(label='activation')
    plt.xlabel('thinking step')
    plt.ylabel('output dimension')
    plt.title(f'cognitive signals (shape: {sig.shape})')
    plt.tight_layout()
    plt.show()
    
    #print stats per thinking step
    print("=" * 50)
    print("COGNITIVE SIGNALS STATS")
    print("=" * 50)
    print(f"  overall mean: {sig.mean():.4f}, std: {sig.std():.4f}")
    print(f"  per-step mean range: [{sig.mean(axis=0).min():.4f}, {sig.mean(axis=0).max():.4f}]")
    print(f"  per-step std range: [{sig.std(axis=0).min():.4f}, {sig.std(axis=0).max():.4f}]")
    print("=" * 50)


def plot_action_heatmap(mu, figsize=(10, 6)):
    #plots heatmap of policy output (embedding_dim x thinking_steps)
    
    #handle tensor
    if torch.is_tensor(mu):
        m = mu.detach().cpu().numpy()
    else:
        m = mu
    
    #squeeze batch if needed
    if m.ndim == 3:
        m = m.squeeze(0)  #b x embedding_dim x thinking_steps -> embedding_dim x thinking_steps
    
    plt.figure(figsize=figsize)
    plt.imshow(m, aspect='auto', cmap='RdBu_r')
    plt.colorbar(label='value')
    plt.xlabel('thinking step')
    plt.ylabel('embedding dimension')
    plt.title(f'policy output μ (shape: {m.shape})')
    plt.tight_layout()
    plt.show()
    
    #print stats
    print("=" * 50)
    print("POLICY OUTPUT (μ) STATS")
    print("=" * 50)
    print(f"  overall mean: {m.mean():.4f}, std: {m.std():.4f}")
    print(f"  range: [{m.min():.4f}, {m.max():.4f}]")
    print(f"  per-step mean range: [{m.mean(axis=0).min():.4f}, {m.mean(axis=0).max():.4f}]")
    print("=" * 50)


#-----------------------------------------------
#Phase 3 runner
#-----------------------------------------------
def run_phase3_diagnostics(model, train_tensor, label_id_tensor, label_mapping, 
                           vocab_list, segment_length=500):
    #runs all phase 3 deeper analysis diagnostics
    
    print("\n" + "="*70)
    print("PHASE 3 DIAGNOSTICS - DEEPER ANALYSIS")
    print("="*70)
    
    #1. confidence distribution
    print("\n[1/6] CONFIDENCE DISTRIBUTION")
    all_conf, avg_conf = plot_confidence_histogram(model, train_tensor, label_id_tensor, 
                                                    segment_length)
    
    #2. word frequency
    print("\n[2/6] WORD FREQUENCY")
    word_counts, all_word_ids = plot_word_frequency(model, train_tensor, vocab_list, 
                                                     segment_length)
    
    #3. prediction entropy
    print("\n[3/6] PREDICTION ENTROPY")
    entropy, norm_entropy = compute_prediction_entropy(word_counts, len(vocab_list))
    
    #4. get a sample output for visualization
    print("\n[4/6] INTERMEDIATE STATE VISUALIZATION")

    sample_outputs = run_full_inference(model, train_tensor[0], segment_length)
    
    print("\n  [4a] State_t visualization")
    plot_state_t(sample_outputs['state_t'][-1])  #last window state_t
    
    print("\n  [4b] Cognitive signals visualization")
    plot_cognitive_signals(sample_outputs['cognitive_signals'][-1])
    
    print("\n  [4c] Action heatmap")
    plot_action_heatmap(sample_outputs['final_mu'])
    
    #5. sigma distribution
    print("\n[5/6] POLICY VARIANCE ANALYSIS")
    plot_sigma_distribution(sample_outputs['final_log_sigma'])
    
    #6. action vs vocab analysis
    print("\n[6/6] ACTION VS VOCAB ANALYSIS")
    plot_action_vs_vocab(sample_outputs['final_mu'], model.vocab_embedding, vocab_list)
    compute_nearest_vocab_distances(sample_outputs['final_mu'], model.vocab_embedding, vocab_list)
    
    return {
        'all_confidences': all_conf,
        'avg_confidences': avg_conf,
        'word_counts': word_counts,
        'entropy': entropy,
        'normalized_entropy': norm_entropy,
        'sample_outputs': sample_outputs
    }