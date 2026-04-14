import torch 
import torch.nn as nn 
import numpy as np
from ..components import StochasticGate
 
 
#adapter for ode-governed networks where parameters are coefficients rather than weight matrices
#produces additive overlays on synaptic coefficients (w, mu, sigma) masked by the wiring topology
#initialised to zero so adapted model starts identical to pretrained model
#designed for ltccell 
class LTCAdapter(nn.Module):
    def __init__(self, internal_neurons, input_dim,
                 sparsity_mask, input_sparsity_mask,
                 adapt_membrane=False,
                 stochastic={'is_stoch': True, 'quantile_dims': 1, 'quantiles_per_dim': 5}):
        
        super(LTCAdapter, self).__init__()
 
        self.internal_neurons = internal_neurons
        self.input_dim = input_dim
        self.adapt_membrane = adapt_membrane
 
        #register sparsity masks as non-trainable buffers
        #overlays are multiplied by these so only existing connections receive gradients
        self.register_buffer('sparsity_mask', sparsity_mask.clone().detach().float())
        self.register_buffer('input_sparsity_mask', input_sparsity_mask.clone().detach().float())
 
        self._build_overlays()
        self._build_gate(stochastic)
 
#----------------------------
# construction
#----------------------------
 
    def _build_overlays(self):
        #internal synapse overlays; shape matches ltc's (internal_neurons, internal_neurons) coefficient tensors
        #zero init means no correction at start; sparsity mask zeroes out non-existent connections
        self.overlay_w = nn.Parameter(torch.zeros(self.internal_neurons, self.internal_neurons))
        self.overlay_mu = nn.Parameter(torch.zeros(self.internal_neurons, self.internal_neurons))
        self.overlay_sigma = nn.Parameter(torch.zeros(self.internal_neurons, self.internal_neurons))
 
        #input synapse overlays; shape matches ltc's (input_dim, internal_neurons) coefficient tensors
        self.overlay_input_w = nn.Parameter(torch.zeros(self.input_dim, self.internal_neurons))
        self.overlay_input_mu = nn.Parameter(torch.zeros(self.input_dim, self.internal_neurons))
        self.overlay_input_sigma = nn.Parameter(torch.zeros(self.input_dim, self.internal_neurons))
 
        #optional membrane parameter overlays; per-neuron vectors rather than connection matrices
        #controls temporal integration dynamics (decay rate and responsiveness)
        if self.adapt_membrane:
            self.overlay_leakage = nn.Parameter(torch.zeros(self.internal_neurons))
            self.overlay_capacitance = nn.Parameter(torch.zeros(self.internal_neurons))
 
    def _build_gate(self, stochastic):
        self.is_stochastic = stochastic['is_stoch']
        self._last_quantiles = None
 
        if self.is_stochastic:
            #gate summary is built from overlay norms; 6 values (or 8 with membrane)
            #this is global not per-sample because ode coefficient corrections are model-level not input-level
            summary_dim = 8 if self.adapt_membrane else 6
 
            self.gate = StochasticGate(
                input_dim=summary_dim,
                output_dim=stochastic['quantile_dims'],
                num_quantiles=stochastic['quantiles_per_dim']
            )
 
#----------------------------
# forward
#----------------------------
 
    #computes masked overlays and optionally scales them through the stochastic gate
    #returns a dictionary of corrections to be added to the ltc's original parameters during integration
    def forward(self):
 
        #apply sparsity masks so overlays only affect existing connections
        masked_overlays = {
            'w': self.overlay_w * self.sparsity_mask,
            'mu': self.overlay_mu * self.sparsity_mask,
            'sigma': self.overlay_sigma * self.sparsity_mask,
            'input_w': self.overlay_input_w * self.input_sparsity_mask,
            'input_mu': self.overlay_input_mu * self.input_sparsity_mask,
            'input_sigma': self.overlay_input_sigma * self.input_sparsity_mask,
        }
 
        if self.adapt_membrane:
            masked_overlays['leakage_conductance'] = self.overlay_leakage
            masked_overlays['membrane_capacitance'] = self.overlay_capacitance
 
        #stochastic gate operates on overlay norms as a global summary
        #the gate asks: how much should i shift these dynamics overall?
        if self.is_stochastic:
            summary = torch.stack([v.norm() for v in masked_overlays.values()]).unsqueeze(0) #1 x num_overlays
            scale, self._last_quantiles = self.gate(summary) #1 x num_scales
 
            #scale all overlays uniformly; squeeze back to scalar for clean broadcasting
            scale = scale.squeeze(0) #num_scales (should be 1 for scalar gate)
 
            for key in masked_overlays:
                masked_overlays[key] = masked_overlays[key] * scale
 
        return masked_overlays
 
#----------------------------
# adapter interface
#----------------------------
 
    def freeze_adapter(self):
        for p in self.parameters():
            p.requires_grad = False 
    
    def unfreeze_adapter(self):
        for p in self.parameters():
            p.requires_grad = True
 
    def return_last_quantiles(self):
        return self._last_quantiles if self._last_quantiles is not None else None
 