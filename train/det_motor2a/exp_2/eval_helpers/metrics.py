import numpy as np

#the five task heads, column order matches the label encoding rows
CLASS_NAMES = ["hand", "feet", "tongue", "left", "right"]

#a window is task-active when any task head is positive, all-zeros is the implicit rest
def active_windows(targets):
    return targets.max(axis=-1) > 0 #N,

#accuracy of always predicting the more common outcome
def majority_rate(binary_targets):
    positive = binary_targets.mean()
    return float(max(positive, 1.0 - positive))

#task-active vs idle over every window
def task_vs_idle(gate_probs, targets, threshold):
    active = active_windows(targets) #N,
    predicted = gate_probs.squeeze(-1) > threshold #N,

    return float((predicted == active).mean()), majority_rate(active)

#per-head accuracy over ground-truth active windows only, so 1D and 2D are comparable
#gated scores a wrongly-closed window as all-negative, the denominator stays the active set
def per_head(probs, gate_probs, targets, threshold, gated):
    active = active_windows(targets) #N,

    head_probs = probs[active] #n_active x classes
    head_targets = targets[active] #n_active x classes
    predicted = head_probs > threshold

    if gated:
        opened = gate_probs.squeeze(-1)[active] > threshold #n_active,
        predicted = predicted & opened[:, None] #closed windows predict all-negative

    accuracies = (predicted == (head_targets > 0)).mean(axis=0) #classes,
    baselines = [majority_rate(head_targets[:, i] > 0) for i in range(head_targets.shape[-1])]

    return {name: float(acc) for name, acc in zip(CLASS_NAMES, accuracies)}, \
           {name: base for name, base in zip(CLASS_NAMES, baselines)}

#one model on one window size; gated only means something for the two decoder
def alignment(outputs, threshold=0.5, gated=False):
    gate_acc, gate_base = task_vs_idle(gate_probs=outputs['gate_probs'], targets=outputs['targets'],
                                       threshold=threshold)

    head_acc, head_base = per_head(probs=outputs['probs'], gate_probs=outputs['gate_probs'],
                                   targets=outputs['targets'], threshold=threshold, gated=gated)

    return {'task_vs_idle': gate_acc, 'baseline_task_vs_idle': gate_base,
            'per_head': head_acc, 'baseline_per_head': head_base,
            'n_windows': int(len(outputs['targets'])),
            'n_active': int(active_windows(outputs['targets']).sum())}

#every window size for one model; pooled is the window-count weighted average
def alignment_per_window(per_window_outputs, threshold=0.5, gated=False):
    scores = {wz: alignment(outputs=outputs, threshold=threshold, gated=gated)
              for wz, outputs in per_window_outputs.items()}

    return scores, pool_alignment(scores)

#weighted by window count so the pooled number reflects the whole distribution
def pool_alignment(scores):
    weights = np.array([s['n_windows'] for s in scores.values()], dtype=np.float64)
    weights = weights / weights.sum()

    values = list(scores.values())

    pooled = {'task_vs_idle': float(np.average([s['task_vs_idle'] for s in values], weights=weights)),
              'baseline_task_vs_idle': float(np.average([s['baseline_task_vs_idle'] for s in values], weights=weights)),
              'per_head': {}, 'baseline_per_head': {}}

    #per-head pooling weights by active count, since that is the per-head denominator
    active = np.array([s['n_active'] for s in values], dtype=np.float64)
    active = active / active.sum()

    for name in CLASS_NAMES:
        pooled['per_head'][name] = float(np.average([s['per_head'][name] for s in values], weights=active))
        pooled['baseline_per_head'][name] = float(np.average([s['baseline_per_head'][name] for s in values], weights=active))

    return pooled