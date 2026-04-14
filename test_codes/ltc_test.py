from architecture_components import build_ltc_prop
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
propagator = build_ltc_prop(
    r1=r1, r2=r2, r3=r3, 
    input_fanout=input_fanout, r1_fanout=r1_fanout, r2_fanout=r2_fanout,
    self_connections=self_connections, input_dim=input_dim
)

#test input
batch= 4
timesteps = 7
random_input = torch.rand(batch, input_dim, timesteps)

hidden_state = None #init hidden state as none on first iter
prop_history = [] #store all hidden states

for t in range(timesteps):
    x_t = random_input[:, :, t] #b x input_dim
    _, hidden_state = propagator(x_t, hidden_state) #recurrent hidden states
    prop_history.append(hidden_state) #b x internal neurons

prop_tensor = torch.stack(prop_history, dim=-1) #batch x internal neurons x timesteps
print(f"shape of propagation hidden states: {prop_tensor.shape}")
print(f"number of neurons in model: {propagator.internal_neurons} | total neurons (including input neurons): {propagator.total_neuron_count}")