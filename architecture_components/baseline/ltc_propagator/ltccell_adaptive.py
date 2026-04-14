import torch 
import torch.nn as nn 
import numpy as np
from ..components import lora_a
from .ltc_adapter import LTCAdapter


#adaptive version of ltccell with modulator support for fine-tuning
#all base ltc functionality is preserved; modulators are optional and built after pretraining
class LTCCell_adaptive(nn.Module):
    def __init__(self, neural_wiring, input_mapping="affine", output_mapping="affine", ode_unfolds=6, epsilon=1e-8, project_output=False):
        super(LTCCell_adaptive, self).__init__() #inheret from nn.mod

        self.project_output = project_output

        self._store_config(neural_wiring, input_mapping, output_mapping, ode_unfolds, epsilon)
        self._initialise_modulator_state()

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

    #private methods (used internally only)

    def _store_config(self, neural_wiring, input_mapping, output_mapping, ode_unfolds, epsilon):
        #store config
        self.wire = neural_wiring
        self.input_mapping = input_mapping #"affine" or linear, in otherwords, affine -> includes bias term
        self.output_mapping = output_mapping
        self.ode_unfolds = ode_unfolds
        self.epsilon = epsilon

    def _initialise_modulator_state(self):
        #modulator state flags; same pattern as ctm_adaptive
        self._modulators_initialised = False #whether modulators have been built; set true by build_modulators()
        self._use_modulators = False #whether to apply modulators in forward pass; toggled by enable/disable_modulators()

        #modulator container; populated by build_modulators() after pretraining
        self.modulators = nn.ModuleDict()

#----------------------------
# parameter initialisation
#----------------------------

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

#----------------------------
# signal mapping
#----------------------------

    #maps raw input features through simple affine transformation if enabled
    def map_input(self, x):
        if self.input_mapping in ["affine", "linear"]:
            x = x * self.params["input_weights"] #applies for both affine and linear regardless

        if self.input_mapping == "affine":
            x = x + self.params["input_bias"]
        return x
    
    #map region 3 neurons to outputs where output dim size is the number of neurons in region 3
    def map_output(self, state):
        output = state 
        if self.wire.output_dim < self.wire.internal_neurons:
            output = output[:, 0:self.wire.output_dim] #output is sliced to self.wire.output_dim amount of neurons, in ncp we let sr3 indices start from 0
        
        if self.output_mapping in ["affine", 'linear']:
            output = output * self.params["output_weights"]
        if self.output_mapping == "affine":
            output = output + self.params['output_bias']
        
        return output

#----------------------------
# ode dynamics
#----------------------------
    
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

    #creates a working copy of self.params with overlay corrections added
    #overlays are added before any positivity constraints so softplus still applies to adapted values
    #keys not present in overlays are passed through unchanged
    def _apply_overlays(self, overlays):
        adapted = {}
        for key, value in self.params.items():
            if key in overlays:
                adapted[key] = value + overlays[key]
            else:
                adapted[key] = value
        return adapted

    #ode solver that evolves internal neuron state over one time step
    #this solver is a conductance based ode
    #overlays is an optional dictionary of additive corrections from the ltc adapter
    def conductance_closed_solver(self, projected_x, current_state, time_constant, overlays=None):
        
        #get working params; original or adapted with overlay corrections
        p = self._apply_overlays(overlays) if overlays is not None else self.params

        copy_state = current_state 

        #compute synaptic activations and reverse potential for input projection
        input_numerator, input_denom = self.compute_synapse(input_state=projected_x, 
                                                            weight=self.positive_constraint(p['input_w']),
                                                            mu=p['input_mu'],
                                                            sigma=p['input_sigma'],
                                                            sparsity_mask=p['input_sparsity_mask'],
                                                            reverse_potential=p['input_reverse_potential'])

        scaled_membrane_capacitance = self.positive_constraint(p["membrane_capacitance"]) / (time_constant / self.ode_unfolds)

        #decoupled recurrent loop/ode unfolding for approximating neural dynamics
        for _ in range(self.ode_unfolds):

            #compute synaptic activations and reverse potential for internal neurons
            numerator, denom = self.compute_synapse(input_state=copy_state, 
                                                    weight=self.positive_constraint(p['w']),
                                                    mu=p['mu'],
                                                    sigma=p['sigma'],
                                                    sparsity_mask=p['sparsity_mask'],
                                                    reverse_potential=p['synapse_reverse_potential'])

            total_numerator = numerator + input_numerator
            total_denom = denom + input_denom 

            dh = scaled_membrane_capacitance * copy_state + self.positive_constraint(p['leakage_conductance']) * p['reverse_potential'] + total_numerator
            dt = scaled_membrane_capacitance + self.positive_constraint(p['leakage_conductance']) + total_denom

            copy_state = dh / (dt + self.epsilon) #unfold and approximate dynamics of new state
        
        return copy_state #return modified copy and pass as args next iteration

#----------------------------
# forward pass
#----------------------------

    #forward pass for single time step; assuming input is in shape batch x features
    def forward(self, x, state=None, time_constant=1.0):

        #if no hidden state is given, create one with the same size as the number of internal neurons
        if state is None:
            state = torch.zeros(x.shape[0], self.wire.internal_neurons, device=x.device, dtype=x.dtype)

        #0 - apply input modulation if modulators are active; additive correction before input mapping
        if self._use_modulators and 'input' in self.modulators:
            x = x + self.modulators['input'](x)

        projected_x = self.map_input(x)

        #1 - get ode coefficient overlays if modulators are active
        overlays = None
        if self._use_modulators and 'ode_coefficients' in self.modulators:
            overlays = self.modulators['ode_coefficients']()

        new_hidden_state = self.conductance_closed_solver(projected_x, state, time_constant, overlays=overlays)
       
        output = None #placeholder if not using outputs
        if self.project_output:
            output = self.map_output(new_hidden_state)

        return output, new_hidden_state

#----------------------------
# modulator management 
#----------------------------

    #builds modulators for fine-tuning; call after pretraining is complete
    #creates an input lora for modulating input features and an ltc adapter for ode coefficient overlays
    #all dimensions are known from the wiring so no forward pass is needed before building
    def build_modulators(self, rank=1, alpha=1.0, dropout=0.2, 
                         nonlinear=True, activation='silu',
                         adapt_membrane=True,
                         stochastic={'is_stoch': True, 'quantile_dims': 1, 'quantiles_per_dim': 5}):
        
        if self._modulators_initialised:
            raise RuntimeError("modulators already initialised; rebuild not permitted to prevent accidental overwrite of trained modulators")

        #input modulator: additive correction on input features before input mapping
        #input and output dim are the same since lora(input) + input requires matching shapes
        self.modulators['input'] = lora_a(
            input_dim=self.wire.input_size,
            output_dim=self.wire.input_size,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            nonlinear=nonlinear,
            activation=activation,
            stochastic=stochastic
        )

        #ode coefficient adapter: additive overlays on synaptic parameters masked by wiring topology
        self.modulators['ode_coefficients'] = LTCAdapter(
            internal_neurons=self.wire.internal_neurons,
            input_dim=self.wire.input_size,
            sparsity_mask=self.params['sparsity_mask'].data,
            input_sparsity_mask=self.params['input_sparsity_mask'].data,
            adapt_membrane=adapt_membrane,
            stochastic=stochastic
        )

        #freeze all base ltc parameters; only modulator parameters should train during fine-tuning
        for key in self.params:
            if isinstance(self.params[key], nn.Parameter):
                self.params[key].requires_grad = False

        self._modulators_initialised = True

        #freeze modulators by default; explicitly unfreeze when starting fine-tuning
        self.freeze_modulators()

    #activates modulator usage in forward pass
    def enable_modulators(self):
        if not self._modulators_initialised:
            raise RuntimeError("cannot enable modulators: they have not been initialised yet; call build_modulators() first")
        
        self._use_modulators = True 

    #deactivates modulator usage in forward pass; modulators still exist but are skipped
    def disable_modulators(self):
        self._use_modulators = False 

    #freezes all modulator parameters so gradients are not computed
    def freeze_modulators(self):
        if not self._modulators_initialised:
            raise RuntimeError("cannot freeze modulators: they have not been initialised yet; call build_modulators() first")
        
        for modulator in self.modulators.values():
            modulator.freeze_adapter()

    #unfreezes all modulator parameters so gradients are computed during fine-tuning
    def unfreeze_modulators(self):
        if not self._modulators_initialised:
            raise RuntimeError("cannot unfreeze modulators: they have not been initialised yet; call build_modulators() first")
        
        for modulator in self.modulators.values():
            modulator.unfreeze_adapter()

    #returns whether modulators are currently active in the forward pass
    @property
    def modulators_active(self):
        return self._use_modulators and self._modulators_initialised

#----------------------------
# properties and utilities
#----------------------------

    @property 
    def internal_neurons(self):
        return self.wire.internal_neurons
    
    @property 
    def total_neuron_count(self):
        return self.wire.total_neuron_count
    
    #returns count in [trainable, non-trainable, total]
    def get_parameter_counts(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        return [trainable_params, non_trainable_params, total_params]

    #returns modulator parameter counts separately for monitoring; same format as get_parameter_counts
    def get_modulator_parameter_counts(self):
        if not self._modulators_initialised:
            return [0, 0, 0]

        total = sum(p.numel() for p in self.modulators.parameters())
        trainable = sum(p.numel() for p in self.modulators.parameters() if p.requires_grad)
        non_trainable = total - trainable

        return [trainable, non_trainable, total]