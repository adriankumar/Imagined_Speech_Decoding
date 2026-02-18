import torch 
import torch.nn as nn 
import numpy as np 

#version 0 of LTC model, use standard architecture, later versions may change the dynamics
class LTCCell(nn.Module):
    def __init__(self, neural_wiring, input_mapping="affine", output_mapping="affine", ode_unfolds=6, epsilon=1e-8, project_output=False):
        super(LTCCell, self).__init__() #inheret from nn.mod

        self.project_output = project_output

        #removed output mapping because we are'nt projecting, we are using the hidden state's evolution as output
        self._store_config(neural_wiring, input_mapping, output_mapping, ode_unfolds, epsilon)

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

        self._initialise_parameters()
    
    def _store_config(self, neural_wiring, input_mapping, output_mapping, ode_unfolds, epsilon):
        #store config
        self.wire = neural_wiring
        self.input_mapping = input_mapping #"affine" or linear, in otherwords, affine -> includes bias term
        self.output_mapping = output_mapping
        self.ode_unfolds = ode_unfolds
        self.epsilon = epsilon

#initialising parameter types with weight values    
    def add_weight(self, name, init_value, requires_grad=True):
        param = nn.Parameter(init_value, requires_grad=requires_grad)
        self.register_parameter(name, param) #store the parameter
        return param
    
    #helper function used to initialise parameters with min max values from init_ranges dictionary
    def get_init_value(self, shape, param_name):
        minval, maxval = self.init_ranges[param_name] 

        if minval == maxval:
            return torch.ones(shape) * minval 
        else:
            return torch.rand(*shape) * (maxval - minval) + minval #*shape passes shape iterable and unpacks every element
        
    def _initialise_parameters(self):
        self.params = {}

        #internal neuron parameters, separate from input projections
        keys = ["leakage_conductance", "reverse_potential", "membrane_capacitance"]
        for name in keys:
            self.params[name] = self.add_weight(name=name, init_value=self.get_init_value((self.wire.internal_neurons,), name))

        #neuron to neuron connection parameters
        keys = ["w", "sigma", "mu"]
        for name in keys:
            self.params[name] = self.add_weight(name=name, init_value=self.get_init_value((self.wire.internal_neurons, self.wire.internal_neurons), name))

        #note that neuron and sensory reverse potentials are already initialised with values
        #neuron to neuron reverse potential/used for internal neurons
        self.params['synapse_reverse_potential'] = self.add_weight(name="synapse_reverse_potential", init_value=torch.Tensor(self.wire.create_synaptic_parameters(self.wire.NAM)))
        
        #input projection parameters
        keys = ["input_w", "input_sigma", "input_mu"]
        for name in keys:
            self.params[name] = self.add_weight(name=name, init_value=self.get_init_value((self.wire.input_size, self.wire.internal_neurons), name))
        
        #input projection reverse potential
        self.params["input_reverse_potential"] = self.add_weight(name="input_reverse_potential", init_value=torch.Tensor(self.wire.create_synaptic_parameters(self.wire.INA)))
        
        #sparsity masks, they are non-trainable
        keys = ["sparsity_mask", "input_sparsity_mask"]
        self.params[keys[0]] = self.add_weight(name=keys[0], init_value=torch.Tensor(np.abs(self.wire.NAM)), requires_grad=False)
        self.params[keys[1]] = self.add_weight(name=keys[1], init_value=torch.Tensor(np.abs(self.wire.INA)), requires_grad=False)

        #input mapping
        if self.input_mapping in ["affine", "linear"]:
            self.params["input_weights"] = self.add_weight(name="input_weights", init_value=torch.ones((self.wire.input_size,)))

        if self.input_mapping == "affine":
            self.params["input_bias"] = self.add_weight(name="input_bias", init_value=torch.zeros((self.wire.input_size,)))
        
        #output mapping for singular if enabled:
        if self.project_output:
            if self.output_mapping in ["affine", "linear"]:
                self.params["output_weights"] = self.add_weight(name="output_weights", init_value=torch.ones((self.wire.output_dim,)))

            if self.output_mapping == "affine":
                self.params["output_bias"] = self.add_weight(name="output_bias", init_value=torch.zeros((self.wire.output_dim,)))

    #maps raw input features through simple affine transformation if enabled
    def map_input(self, x):
        if self.input_mapping in ["affine", "linear"]:
            x = x * self.params["input_weights"] #applies for both affine and linear regardless

        if self.input_mapping == "affine":
            x = x + self.params["input_bias"]
        return x
    
    #map region 3 neurons to outputs where output dim size is the number of neurons in  region 3
    def map_output(self, state):
        output = state 
        if self.wire.output_dim < self.wire.internal_neurons:
            output = output[:, 0:self.wire.output_dim] #output is sliced to self.wire.output_dim amount of neurons, in ncp we let sr3 indices start from 0
        
        if self.output_mapping in ["affine", 'linear']:
            output = output * self.params["output_weights"]
        if self.output_mapping == "affine":
            output = output + self.params['output_bias']
        
        return output
    
    #sigmoid-based gate shaping synaptic activation from pre to post neuron
    def sigmoid_gate(self, x, mu, sigma):
        x = torch.unsqueeze(x, -1)
        mu_term = x - mu
        gated = sigma * mu_term
        return torch.sigmoid(gated)

    #computes synaptic numerator and denominator for ode update
    def compute_synapse(self, input_state, weight, mu, sigma, sparsity_mask, reverse_potential):
        synaptic_activation = (weight * self.sigmoid_gate(input_state, mu, sigma)) * sparsity_mask

        synaptic_reverse_potential = synaptic_activation * reverse_potential
        numerator = torch.sum(synaptic_reverse_potential, dim=1)
        denominator = torch.sum(synaptic_activation, dim=1)
        return numerator, denominator

    #ode solver that evolves internal neuron state over one time step
    #this solver is a conductance based ode
    def conductance_closed_solver(self, projected_x, current_state, time_constant):
        copy_state = current_state 

        #compute synaptic activations and reverse potential for input projection
        input_numerator, input_denom = self.compute_synapse(input_state=projected_x, 
                                                            weight=self.positive_constraint(self.params['input_w']),
                                                            mu=self.params['input_mu'],
                                                            sigma=self.params['input_sigma'],
                                                            sparsity_mask=self.params['input_sparsity_mask'],
                                                            reverse_potential=self.params['input_reverse_potential'])

        scaled_membrane_capacitance = self.positive_constraint(self.params["membrane_capacitance"]) / (time_constant / self.ode_unfolds)

        #decoupled recurrent loop/ode unfolding for approximating neural dynamics
        for _ in range(self.ode_unfolds):

            #compute synaptic activations and reverse potential for internal neurons
            numerator, denom = self.compute_synapse(input_state=copy_state, 
                                                    weight=self.positive_constraint(self.params['w']),
                                                    mu=self.params['mu'],
                                                    sigma=self.params['sigma'],
                                                    sparsity_mask=self.params['sparsity_mask'],
                                                    reverse_potential=self.params['synapse_reverse_potential'])

            total_numerator = numerator + input_numerator
            total_denom = denom + input_denom 

            dh = scaled_membrane_capacitance * copy_state + self.positive_constraint(self.params['leakage_conductance']) * self.params['reverse_potential'] + total_numerator
            dt = scaled_membrane_capacitance + self.positive_constraint(self.params['leakage_conductance']) + total_denom

            copy_state = dh / (dt + self.epsilon) #unfold and approximate dynamics of new state
        
        return copy_state #return modified copy and pass as args next iteration

    #forward pass, assuming input is in shape batch x timepoints x features,
    #where timepoints = 1, so loop is outside of cell
    def forward(self, x, state=None, time_constant=1.0): #forward pass for SINGLE time step

        #if no hidden state is given, create one with the same size as the number of internal neurons
        if state is None:
            state = torch.zeros(x.shape[0], self.wire.internal_neurons, device=x.device, dtype=x.dtype)

        projected_x = self.map_input(x)
        new_hidden_state = self.conductance_closed_solver(projected_x, state, time_constant)
       
        output = None #placeholder if not using outputs
        if self.project_output:
            output = self.map_output(new_hidden_state) #batch x SR3_neuron_count=2, represents binary classes 0 (non-active classification) 1 (active classification)

        return output, new_hidden_state #use torch.cat(hidden_states, dim=1) to arrange in batch x timepoints x hidden_states, where hidden_states.append(new_hidden_state)
 


