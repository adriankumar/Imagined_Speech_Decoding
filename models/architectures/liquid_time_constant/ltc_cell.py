import torch 
import torch.nn as nn 
import numpy as np 
from global_lvl import EPSILON


#euler -substep approximation of ode, not closed-form (cfc)
#implementation taken from official ltc github: https://github.com/mlech26l/ncps/blob/master/ncps/torch/ltc_cell.py
#modified variable names and separation of functions

#official ltc library: https://ncps.readthedocs.io/en/latest/quickstart.html

class LTCCell(nn.Module):
    def __init__(self, neural_graph, output_dim=None,
                 input_mapping="affine", output_mapping="affine", 
                 ode_unfolds=6, epsilon=EPSILON):
        
        super(LTCCell, self).__init__() #inherit from nn.mod

        self._store_config(wire=neural_graph,
                           io_mapping=[input_mapping, output_mapping],
                           ode_unfolds=ode_unfolds,
                           epsilon=epsilon)
    
        #smooth approximation to the ReLU function and 
        #used to constrain the output to always be positive
        self.positive_constraint = nn.Softplus()

        #initialisation ranges for neural parameters - taken from original ltc source code
        self.init_ranges = {
            "leakage_conductance": (0.001, 1.0),
            "reverse_potential": (-0.2, 0.2),
            "membrane_capacitance": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3.0, 8.0),
            "mu": (0.3, 0.8),

            #input projection stuff
            "input_w": (0.001, 1.0),
            "input_sigma": (3.0, 8.0),
            "input_mu": (0.3, 0.8),
        }

        self.out_dim = output_dim

        self._initialise_parameters()

    def _store_config(self, wire, io_mapping, ode_unfolds, epsilon):
        self.graphs = wire
        self.input_mapping, self.output_mapping = io_mapping[:]
        self.ode_unfolds = ode_unfolds
        self.epsilon = epsilon

    #initialising parameter types with weight values    
    def _add_weight(self, name, init_value, requires_grad=True):
        param = nn.Parameter(init_value, requires_grad=requires_grad)
        self.register_parameter(name, param) #store the parameter
        return param
    
    #helper function used to initialise parameters with min max values from init_ranges dictionary
    def _get_init_value(self, shape, param_name):
        minval, maxval = self.init_ranges[param_name] 

        if minval == maxval:
            return torch.ones(shape) * minval 
        else:
            return torch.rand(*shape) * (maxval - minval) + minval #*shape passes shape iterable and unpacks every element
        
    def _initialise_parameters(self):
        self._params = {} #param dictionary

        #per (internal) neuron per layer parameters
        keys = ["leakage_conductance", "reverse_potential", "membrane_capacitance"]
        for name in keys:
            self._params[name] = self._add_weight(name=name, init_value=self._get_init_value((self.graphs.internal_neurons,), name))

        #3 separate weight matrices of shape internal neurons x internal neurons
        keys = ["w", "sigma", "mu"]
        for name in keys:
            self._params[name] = self._add_weight(name=name, init_value=self._get_init_value((self.graphs.internal_neurons, self.graphs.internal_neurons), name))

        #another weight matrix parameter for internal neuron graph
        self._params['synapse_reverse_potential'] = self._add_weight(name="synapse_reverse_potential", init_value=torch.Tensor(self.graphs.get_graph_copy(self.graphs.internal_graph)))

        #same as above but for input to layer 1
        keys = ["input_w", "input_sigma", "input_mu"]
        for name in keys:
            self._params[name] = self._add_weight(name=name, init_value=self._get_init_value((self.graphs.input_size, self.graphs.internal_neurons), name))
        
        self._params["input_reverse_potential"] = self._add_weight(name="input_reverse_potential", init_value=torch.Tensor(self.graphs.get_graph_copy(self.graphs.input_graph)))
        
        #sparsity masks, they are non-trainable; because above
        #weight matrices will have values in all row, column entries,
        #sparsity masks keep and update the existing connectivity
        keys = ["sparsity_mask", "input_sparsity_mask"]
        self._params[keys[0]] = self._add_weight(name=keys[0], init_value=torch.Tensor(np.abs(self.graphs.internal_graph)), requires_grad=False)
        self._params[keys[1]] = self._add_weight(name=keys[1], init_value=torch.Tensor(np.abs(self.graphs.input_graph)), requires_grad=False)

        #input mapping
        if self.input_mapping in ["affine", "linear"]:
            self._params["input_weights"] = self._add_weight(name="input_weights", init_value=torch.ones((self.graphs.input_size,)))

        if self.input_mapping == "affine":
            self._params["input_bias"] = self._add_weight(name="input_bias", init_value=torch.zeros((self.graphs.input_size,)))
        

        if self.output_mapping in ["affine", "linear"]:
            self._params["output_weights"] = self._add_weight(name="output_weights", init_value=torch.ones((self.graphs.output_size,)))

        if self.output_mapping == "affine":
            self._params["output_bias"] = self._add_weight(name="output_bias", init_value=torch.zeros((self.graphs.output_size,)))

        if self.out_dim is not None:
            self.readout = nn.Linear(self.graphs.output_size, self.out_dim)

    #maps raw input features through simple affine transformation if enabled
    def map_input(self, x):
        if self.input_mapping in ["affine", "linear"]:
            x = x * self._params["input_weights"] #applies for both affine and linear regardless

        if self.input_mapping == "affine":
            x = x + self._params["input_bias"]
        return x

    #map layer 3 neurons to outputs where output size is the number of neurons in layer 3
    def map_output(self, state):
        output = state[:, 0:self.graphs.output_size] #layer 3 to output - > projected if projected

        if self.output_mapping in ["affine", 'linear']:
            output = output * self._params["output_weights"]
        if self.output_mapping == "affine":
            output = output + self._params['output_bias']

        if self.out_dim is not None: #to arbitrary output dim if specified
            output = self.readout(output)
        
        return output

    #sigmoid synapse gate, shapes a presynaptic potential into a bounded activation
    def sigmoid_gate(self, presynaptic, mu, sigma):
        presynaptic = torch.unsqueeze(presynaptic, -1) #(batch, src) -> (batch, src, 1), broadcasts across dst
        return torch.sigmoid(sigma * (presynaptic - mu))

    #accumulates synaptic conductance into each postsynaptic neuron
    #returns the two terms the ode update needs, split as numerator/denominator contributions
    def synapse_terms(self, presynaptic, weight, mu, sigma, sparsity_mask, reverse_potential):
        #conductance g_ij = w_ij * gate(v_j), masked so only wired synapses contribute
        conductance = (weight * self.sigmoid_gate(presynaptic, mu, sigma)) * sparsity_mask

        #weighted reversal pulls the neuron toward its synaptic targets, conductance is the total drive
        weighted_reversal = torch.sum(conductance * reverse_potential, dim=1) #sum over src -> (batch, dst)
        total_conductance = torch.sum(conductance, dim=1)
        return weighted_reversal, total_conductance

    #advances the hidden state across one external timestep via the paper's fused solver
    #each unfold is one implicit-euler sub-step refining this single step's solution
    #the loop converges toward the step, it does not accumulate information across passes like a ctm loop
    def ode_solve(self, projected_input, state, elapsed_time):
        #sensory synapses are held constant over the step, so compute them once outside the loop
        input_reversal, input_conductance = self.synapse_terms(
            presynaptic=projected_input,
            weight=self.positive_constraint(self._params['input_w']),
            mu=self._params['input_mu'],
            sigma=self._params['input_sigma'],
            sparsity_mask=self._params['input_sparsity_mask'],
            reverse_potential=self._params['input_reverse_potential'])

        #cm divided by sub-step size, controls how far each unfold moves the state
        step_size = elapsed_time / self.ode_unfolds
        scaled_capacitance = self.positive_constraint(self._params['membrane_capacitance']) / step_size

        leak_conductance = self.positive_constraint(self._params['leakage_conductance'])
        leak_reversal = self._params['reverse_potential']

        for _ in range(self.ode_unfolds):
            #internal synapses re-read the evolving state each unfold, presynaptic gates are explicit
            synapse_reversal, synapse_conductance = self.synapse_terms(
                presynaptic=state,
                weight=self.positive_constraint(self._params['w']),
                mu=self._params['mu'],
                sigma=self._params['sigma'],
                sparsity_mask=self._params['sparsity_mask'],
                reverse_potential=self._params['synapse_reverse_potential'])

            #numerator gathers every drive term acting on the postsynaptic potential
            numerator = (scaled_capacitance * state
                         + leak_conductance * leak_reversal
                         + input_reversal + synapse_reversal)

            #denominator is the total conductance, the inverse of the effective time constant
            denominator = (scaled_capacitance
                           + leak_conductance
                           + input_conductance + synapse_conductance)

            #implicit-euler solve for the postsynaptic potential at this sub-step
            state = numerator / (denominator + self.epsilon)

        return state

    def init_state(self, batch_dim, device, dtype):
        return torch.zeros(batch_dim, self.graphs.internal_neurons, device=device, dtype=dtype)

    #forward pass for a single external timestep, the sequence loop lives outside the cell
    def forward(self, x, state=None, elapsed_sub_time=1.0):

        #if state is none, assume its a new sequence
        if state is None:
            state = self.init_state(batch_dim=x.shape[0], device=x.device, dtype=x.dtype) #shape batch x internal neurons is hidden state shape (one dimensional)
        
        projected_input = self.map_input(x) #batch x input size (vector)
        
        #note Cfc implements closed form solution, this is an euler sub-delta approximation towards a converged timestep point outside the cell, not an accumulation of information like the CTM
        new_state = self.ode_solve(projected_input, state, elapsed_time=elapsed_sub_time) #solve for new state; shape batch x internal neurons

        output = self.map_output(new_state) #shape is B x output dim or B x layer 3 neurons
        
        return output, new_state 
    
    def get_parameter_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total_params, 'trainable': trainable_params, 'non_trainable': total_params - trainable_params}

    @property 
    def internal_neurons(self):
        return self.graphs.internal_neurons
    
    @property 
    def total_neurons(self):
        return self.graphs.internal_neurons + self.graphs.input_size
    
    @property
    def input_size(self):
        return self.graphs.input_size
    
    @property 
    def output_size(self):
        return self.graphs.output_size
    
    @property 
    def connectivity_graphs(self):
        return {'input': self.graphs.input_graph, 'internal': self.graphs.internal_graph}

    @property 
    def params(self): #return params dictionary and plot for current values
        return self._params