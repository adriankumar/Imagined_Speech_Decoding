from architecture_components import build_ltc_adaptive_prop
import torch

#args
r1 = 3 #region 1 neurons
r2 = 5 #region 2 neurons
r3 = 5 #region 3 neurons
input_fanout = 1 
r1_fanout = 3
r2_fanout = 3
self_connections = 4
input_dim = 10

#build model 
propagator = build_ltc_adaptive_prop(
    r1=r1, r2=r2, r3=r3, 
    input_fanout=input_fanout, r1_fanout=r1_fanout, r2_fanout=r2_fanout,
    self_connections=self_connections, input_dim=input_dim
)

#test input
batch = 4
timesteps = 7
random_input = torch.rand(batch, input_dim, timesteps)

#--- forward pass and recurrency test ---
hidden_state = None #init hidden state as none on first iter
prop_history = [] #store all hidden states

for t in range(timesteps):
    x_t = random_input[:, :, t] #b x input_dim
    _, hidden_state = propagator(x_t, hidden_state) #recurrent hidden states
    prop_history.append(hidden_state) #b x internal neurons

prop_tensor = torch.stack(prop_history, dim=-1) #batch x internal neurons x timesteps
print(f"shape of propagation hidden states: {prop_tensor.shape}")
print(f"number of neurons in model: {propagator.internal_neurons} | total neurons (including input neurons): {propagator.total_neuron_count}")

#verify hidden states change across timesteps (recurrence is doing something)
states_differ = not torch.equal(prop_history[0], prop_history[1])
states_evolve = not torch.equal(prop_history[-2], prop_history[-1])
print(f"hidden state changed after first recurrence: {states_differ}")
print(f"hidden state still evolving at final timesteps: {states_evolve}")

pc = propagator.get_parameter_counts()
print(f"\nltc parameter count: trainable: {pc[0]} | non-trainable: {pc[1]} | total: {pc[2]}")

#--- modulator tests ---
print("\n--- modulator tests ---")

#confirm modulators cant be enabled before being built
try:
    propagator.enable_modulators()
    print("ERROR: should have raised RuntimeError")
except RuntimeError as e:
    print(f"correct guard on enable before build: {e}")

#build modulators; no forward pass needed since dims come from wiring
propagator.build_modulators(rank=2, alpha=1.0, dropout=0.2, adapt_membrane=True)
print(f"\nmodulators built successfully")
print(f"modulators initialised: {propagator._modulators_initialised}")
print(f"registered modulators: {list(propagator.modulators.keys())}")

#confirm rebuild is blocked
try:
    propagator.build_modulators(rank=2)
    print("ERROR: should have raised RuntimeError")
except RuntimeError as e:
    print(f"correct guard on rebuild: {e}")

#verify base params are frozen after build
base_trainable = sum(1 for key in propagator.params if isinstance(propagator.params[key], torch.nn.Parameter) and propagator.params[key].requires_grad)
print(f"\nbase params still trainable after build: {base_trainable} (should be 0)")

#check modulator parameter counts (should be frozen by default after build)
mpc = propagator.get_modulator_parameter_counts()
print(f"modulator params: trainable: {mpc[0]} | non-trainable: {mpc[1]} | total: {mpc[2]}")

#unfreeze and verify counts change
propagator.unfreeze_modulators()
mpc_unfrozen = propagator.get_modulator_parameter_counts()
print(f"after unfreeze: trainable: {mpc_unfrozen[0]} | non-trainable: {mpc_unfrozen[1]} | total: {mpc_unfrozen[2]}")

#enable and run forward pass with modulators active
propagator.enable_modulators()
print(f"\nmodulators active: {propagator.modulators_active}")

#run full recurrent sequence with modulators
hidden_state_mod = None
prop_history_mod = []

for t in range(timesteps):
    x_t = random_input[:, :, t]
    _, hidden_state_mod = propagator(x_t, hidden_state_mod)
    prop_history_mod.append(hidden_state_mod)

prop_tensor_mod = torch.stack(prop_history_mod, dim=-1)
print(f"modulated propagation shape: {prop_tensor_mod.shape}")

#verify modulated output differs from non-modulated
outputs_differ = not torch.equal(prop_tensor, prop_tensor_mod)
print(f"modulated output differs from base: {outputs_differ}")

#verify recurrence still works under modulation
mod_states_evolve = not torch.equal(prop_history_mod[-2], prop_history_mod[-1])
print(f"hidden state still evolving under modulation: {mod_states_evolve}")

#disable and verify fallback
propagator.disable_modulators()
print(f"\nafter disable - modulators active: {propagator.modulators_active}")

hidden_state_disabled = None
for t in range(timesteps):
    x_t = random_input[:, :, t]
    _, hidden_state_disabled = propagator(x_t, hidden_state_disabled)

print(f"disabled forward pass completed successfully")

#freeze and verify
propagator.freeze_modulators()
mpc_frozen = propagator.get_modulator_parameter_counts()
print(f"after refreeze: trainable: {mpc_frozen[0]} | non-trainable: {mpc_frozen[1]} | total: {mpc_frozen[2]}")

#--- overall parameter summary ---
print("\n--- parameter summary ---")
pc_final = propagator.get_parameter_counts()
print(f"total model params: trainable: {pc_final[0]} | non-trainable: {pc_final[1]} | total: {pc_final[2]}")
print(f"of which modulators: {mpc_frozen[2]}")