from architecture_components import CTM_adaptive 
import torch 

batch = 4
input_dim = 40 #should be same as the embedding dim used in the attention and the action sync vector, unless you explicitly set vdim kdim 

model = CTM_adaptive() #using default configs

input_features = torch.rand(batch, 3, input_dim)
hidden_states = None #will be a dictionary

#--- forward pass test ---
predictions, hidden_states = model(input_features=input_features, neural_states=hidden_states)

print(f"prediction output size: {predictions.shape}")
pc = model.get_parameter_counts()
print(f"ctm parameter count: trainable: {pc[0]} | non-trainable: {pc[1]} | total: {pc[2]}")

#--- recurrency test ---
#pass hidden states back into the model for a second forward pass to simulate temporal processing
predictions_2, hidden_states_2 = model(input_features=input_features, neural_states=hidden_states)

print(f"\nrecurrent prediction output size: {predictions_2.shape}")

#verify hidden states are actually different after second pass (recurrence is doing something)
pre_hist_changed = not torch.equal(hidden_states['pre_activation_history'], hidden_states_2['pre_activation_history'])
post_activ_changed = not torch.equal(hidden_states['post_activations'], hidden_states_2['post_activations'])
print(f"pre-activation history changed after recurrence: {pre_hist_changed}")
print(f"post activations changed after recurrence: {post_activ_changed}")

#--- modulator tests ---
print("\n--- modulator tests ---")

#confirm input dim was captured during forward pass
print(f"input dim captured: {model._input_dim_captured} | dim: {model._input_features_dim}")

#confirm modulators cant be enabled before being built
try:
    model.enable_modulators()
    print("ERROR: should have raised RuntimeError")
except RuntimeError as e:
    print(f"correct guard on enable before build: {e}")

#build modulators
model.build_modulators(rank=2, alpha=1.0, dropout=0.2)
print(f"\nmodulators built successfully")
print(f"modulators initialised: {model._modulators_initialised}")
print(f"registered modulators: {list(model.modulators.keys())}")

#confirm rebuild is blocked
try:
    model.build_modulators(rank=2)
    print("ERROR: should have raised RuntimeError")
except RuntimeError as e:
    print(f"correct guard on rebuild: {e}")

#check modulator parameter counts (should be frozen by default after build)
mpc = model.get_modulator_parameter_counts()
print(f"\nmodulator params: trainable: {mpc[0]} | non-trainable: {mpc[1]} | total: {mpc[2]}")

#unfreeze and verify counts change
model.unfreeze_modulators()
mpc_unfrozen = model.get_modulator_parameter_counts()
print(f"after unfreeze: trainable: {mpc_unfrozen[0]} | non-trainable: {mpc_unfrozen[1]} | total: {mpc_unfrozen[2]}")

#enable and run forward pass with modulators active
model.enable_modulators()
print(f"\nmodulators active: {model.modulators_active}")

predictions_mod, hidden_states_mod = model(input_features=input_features, neural_states=None)
print(f"modulated prediction output size: {predictions_mod.shape}")

#verify modulated output differs from non-modulated (lora/dora should change the output even at init)
outputs_differ = not torch.equal(predictions, predictions_mod)
print(f"modulated output differs from base: {outputs_differ}")

#disable and verify fallback to original behaviour
model.disable_modulators()
print(f"\nafter disable - modulators active: {model.modulators_active}")

predictions_disabled, _ = model(input_features=input_features, neural_states=None)
print(f"disabled prediction output size: {predictions_disabled.shape}")

#freeze and verify
model.freeze_modulators()
mpc_frozen = model.get_modulator_parameter_counts()
print(f"after refreeze: trainable: {mpc_frozen[0]} | non-trainable: {mpc_frozen[1]} | total: {mpc_frozen[2]}")

#--- overall parameter summary ---
print("\n--- parameter summary ---")
pc_final = model.get_parameter_counts()
print(f"total model params: trainable: {pc_final[0]} | non-trainable: {pc_final[1]} | total: {pc_final[2]}")
print(f"of which modulators: {mpc_frozen[2]}")
print(f"original output projection frozen: {not any(p.requires_grad for p in model.output_projection.parameters())}")