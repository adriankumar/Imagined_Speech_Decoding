import torch.nn as nn
import torch 
import numpy as np

#neuron activations handling (pre_activation history and post activations)
class NeuronActivations(nn.Module):
    def __init__(self, num_neurons, memory_length):
        super(NeuronActivations, self).__init__()

        #register once at construction of model 
        #pre-activation history initialised using Xavier/Glorot initialisation (+- 1/sqrt(fan_in + fan_out))
        #learnable initialisation template, but eventually gets replaced by actual pre-activations during forward pass
        self.register_parameter('initial_pre_history', 
                                nn.Parameter(
                                    torch.zeros(num_neurons, memory_length).uniform_(
                                        -np.sqrt(1/(num_neurons + memory_length)), 
                                        np.sqrt(1/(num_neurons + memory_length))
                                        )
                                    )
                                ) #shape of num_neurons x memory_length
        
        #initial post-activation state for neurons - shape: num_neurons; also learnable starting point
        self.register_parameter('initial_post_activations', 
                              nn.Parameter(torch.zeros(num_neurons).uniform_(
                                  -np.sqrt(1/num_neurons), np.sqrt(1/num_neurons)
                              )))
        
    #just expand existing params for batch
    def initialise_pre_activation_history(self, batch_size, device):
        pre_history = self.initial_pre_history.unsqueeze(0).expand(batch_size, -1, -1).to(device).clone() #expand for batch processing, shape: batch x num_neurons x memory_length
        return pre_history
    
    def initalise_post_activations(self, batch_size, device):

        inital_post_activations = self.initial_post_activations.unsqueeze(0).expand(batch_size, -1).to(device).clone() #expand for batch processing, shape: batch x num_neurons
        return inital_post_activations

    def update_pre_activation_history(self, pre_activation_history, new_pre_activations):
        return torch.cat((pre_activation_history[:, :, 1:], new_pre_activations.unsqueeze(-1)), dim=-1) #remove oldest pre-activation and append new pre-activation at the end; shape: batch x num_neurons x memory_length
