import torch

#need to update with config
def load_model_checkpoint(path, model, device='cpu'):
    #loads model state dict from checkpoint file
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    #print checkpoint info if available
    print(f"loaded checkpoint from: {path}")
    if 'epoch' in checkpoint:
        print(f"  epoch: {checkpoint['epoch']}")
    if 'best_reward' in checkpoint:
        print(f"  best reward: {checkpoint['best_reward']:.4f}")
    
    return model

#-----------------------------------------------
#forward passes
#-----------------------------------------------
def forward_pass(model, window, buffer=None, 
                 sensory_state=None, cognitive_state=None, policy_state=None, decoder_state=None,
                 sample_action=False):

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
    
    #predict next state t
    next_state_pred, mdn_mu, mdn_ls, mdn_pi, decoder_state = model.predict_next_state(signals, decoder_state)

    return action, next_state_pred, buffer, sensory_state, cognitive_state, policy_state, decoder_state

#for this demo, final_reasoning_length is not ever used because we dont have the data
#to allow for dynamic length reasoning since all samples are fixed and we have no module to help predict
#a value for final reasoning, and the architecture is not compatible with a different final reasoning
#length other than the thinking_steps (however we could make the final reasoning be any length but only slice out the last
#thinking_steps amount of the output to keep it compatible); this is only for future reference to include when we can
#make the architecture more advanced and dynamic; note as well since our dataset is fixed there is no need
#to predict next state t, but in future we will if we want it to work in real-time 
def final_reasoning(model, buffer, cognitive_state=None, policy_state=None, final_reasoning_length=None, sample_action=False):
    #for final reasoning, its a repeat of the previous buffer input so no changes

    signals, cognitive_state = model.think(buffer, cognitive_state, final_reasoning_length=None)
    action, log_sigma, policy_state = model.propagate_action(signals, policy_state)
    if sample_action:
        action, log_prob = model.sample_action(action, log_sigma)
    
    #reset buffer after final reasoning - final reasoning is just another name for buffer is full
    model.reset_buffer(buffer)

    return action, cognitive_state, policy_state

