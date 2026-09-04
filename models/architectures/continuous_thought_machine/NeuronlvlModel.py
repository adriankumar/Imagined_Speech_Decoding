import torch.nn as nn
import torch 
import numpy as np

#neuron level model component to compute post activations from pre-activation history for each neuron
class NeuronLevelModel(nn.Module):
    def __init__(self, num_neurons, memory_length, is_deep=False, use_layernorm=False, dropout=0.0, temperature=1.0):
        super().__init__()

        self._build(num_neurons, memory_length, is_deep, use_layernorm, dropout, temperature)

#----------------------------
# Architecture stuff
#----------------------------
    def _build(self, num_neurons, memory_length, is_deep, use_layernorm, dropout, temperature):
        if memory_length >= num_neurons:
            raise ValueError(f"memory_length: {memory_length} must be less than num_neurons: {num_neurons}")

        self.memory_length = memory_length #pre-activation history length
        self.num_neurons = num_neurons
        self.is_deep = is_deep

        #dropout and layernorm
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        #elementwise_affine introduces learnable weight and bias to the normalised output and performs a standard perceptron based calculation (weight * normalised + bias)
        self.layernorm = nn.LayerNorm(self.memory_length, elementwise_affine=True) if use_layernorm else nn.Identity()

        #learnable temperature scaling parameter
        self.register_parameter('temperature', nn.Parameter(torch.tensor(temperature)))

        if self.is_deep:
            self._build_deep_nlm() #2 layer nlm
        else:
            self._build_nlm() 

    def _build_deep_nlm(self):
        # first layer: memory_length -> 2*hidden_dim (for glu)
        self.register_parameter('w1', nn.Parameter(torch.empty(self.memory_length, 2 * 2, self.num_neurons).uniform_(-1/np.sqrt(self.memory_length + 2 * 2), 1/np.sqrt(self.memory_length + 2 * 2))))
        self.register_parameter('b1', nn.Parameter(torch.zeros(1, self.num_neurons, 2 * 2)))

        #second layer: hidden_dim -> 2 (for glu then squeeze to 1)
        self.register_parameter('w2', nn.Parameter(torch.empty(2, 2, self.num_neurons).uniform_(-1/np.sqrt(2 + 2), 1/np.sqrt(2 + 2))))
        self.register_parameter('b2', nn.Parameter(torch.zeros(1, self.num_neurons, 2)))
    
    def _build_nlm(self):
        #w1 has shape: memory_length x 2 x num_neurons, b1 has shape: 1 x num_neurons x 2
        self.register_parameter('w1', nn.Parameter(torch.empty(self.memory_length, 2, self.num_neurons).uniform_(-1/np.sqrt(self.memory_length + 2), 1/np.sqrt(self.memory_length + 2))))
        self.register_parameter('b1', nn.Parameter(torch.zeros(1, self.num_neurons, 2)))

#----------------------------
# Forward Processing
#----------------------------
    def _forward_deep(self, x):
        #first layer with glu activation
        out = torch.einsum('bnm,mhn->bnh', x, self.w1) + self.b1
        out = nn.functional.glu(out, dim=-1)  #splits last dim and applies gating
        
        #second layer with glu activation and squeeze
        out = torch.einsum('bnh,hrn->bnr', out, self.w2) + self.b2
        out = nn.functional.glu(out, dim=-1)  #results in single output per neuron

        post_activations = out.squeeze(-1) / torch.clamp(self.temperature, min=1e-8) #small epislon to prevent division by zero even tho its initialised as 1 
        
        return post_activations

    def _forward_shallow(self, x):
        #single layer with glu activation and squeeze
        out = torch.einsum('bnm,mrn->bnr', x, self.w1) + self.b1
        out = nn.functional.glu(out, dim=-1)  #single output per neuron
        
        post_activations = out.squeeze(-1) / self.temperature
        
        return post_activations
    
    def forward(self, pre_activation_history):
        x = self.dropout(pre_activation_history) #input shape: batch, num_neurons, memory_length
        x = self.layernorm(x) #normalise

        #output shape should be batch x num neurons -> each neuron has one post activation 
        if self.is_deep:
            return self._forward_deep(x)
        else:
            return self._forward_shallow(x)