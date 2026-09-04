import torch.nn as nn 
import numpy as np

#component computes pre-activations from attended features
#1 dimensional unet vector, hence each layer is a vector compression, width refers to featurs in each layer
#automatically interpolates dims from input dim -> min_feature size across unet_depth
class SynapseUNet(nn.Module):
    def __init__(self, input_dim, layers, min_feature_dim=16, dropout=0.0, bias=False):
        super().__init__()

        self._build(input_dim, layers, min_feature_dim, dropout, bias)

#----------------------------
# Architecture stuff
#----------------------------
    def _build(self, num_neurons, depth, min_width, dr, bias):
        self.num_neurons = num_neurons #what base ctm uses 
        self.depth = depth 
        self.min_width = min_width #smallest allowed bottleneck; width being number of elements each vector layer in the unet has
        self.dr = dr 
        self.bias = bias 

        #unet layer sizes (each layer is 1dim, so sizes refers to number of elements)
        self.layer_sizes = self._interpolate_layers(self.num_neurons, self.depth, self.min_width)
        self.input_proj = self._get_input_projection()

        self.down_proj, self.up_proj, self.skip_norm = self._build_unet(self.dr)

    #interpolate number of dims each layer in the 1d-unet has from input_dim -> min_feature_dim
    def _interpolate_layers(self, num_neurons, num_layers, min_feature_dim):
        layer_sizes = np.linspace(start=num_neurons, stop=min_feature_dim, num=num_layers)
        return [int(size) for size in layer_sizes]
    
    def _get_input_projection(self):
        return nn.Sequential(
            nn.LazyLinear(self.layer_sizes[0], bias=self.bias), #lazy infers input dim, 
            nn.LayerNorm(self.layer_sizes[0], bias=self.bias), #normalise
            nn.SiLU() #act used from official ctm repo
        )
    
    #helper for UNet layers
    def _get_projection(self, input_dim, output_dim, dr):
        return nn.Sequential(
            nn.Dropout(dr),
            nn.Linear(input_dim, output_dim), #single linear projection
            nn.LayerNorm(output_dim),
            nn.SiLU()
        )
    
    def _build_unet(self, dr):
        down_path = nn.ModuleList() #downward projections
        up_path = nn.ModuleList() #upward projections
        skip_norm = nn.ModuleList() #normaliser for each skip connection in the upward layer
 
        #loop through layers
        for layer in range(len(self.layer_sizes) - 1):
            #down
            down_block = self._get_projection(self.layer_sizes[layer], self.layer_sizes[layer + 1], dr) #using current layer as input size, and next layer as output size
            down_path.append(down_block) #append to module list

            #up
            up_block = self._get_projection(self.layer_sizes[layer + 1], self.layer_sizes[layer], dr) #same as down block but swap input size to be layer + 1 and output to just layer
            up_path.append(up_block)

            #skip connection normalisers
            skip_norm.append(nn.LayerNorm(self.layer_sizes[layer])) #normaliser size will be same as output size of the up block 

        return down_path, up_path, skip_norm

#----------------------------
# Forward Processing
#----------------------------
    def forward(self, input):
        input_mapped = self.input_proj(input) #map input 
        skip_act = self.project_down(input_mapped) #project down unet and store skip connections
        pre_act = self.project_up(skip_act) #project up and add skip connections; pre_activation for neurons in ctm 

        return pre_act #b x num_neurons

    def project_up(self, skip_activations):
        current_activation = skip_activations[-1] #start from end/bottleneck layer
        layers = len(self.up_proj)

        for layer_id in range(layers):
            reversed_layer_id = layers - 1 - layer_id #layer index backwards

            current_activation = self.up_proj[reversed_layer_id](current_activation) #project in upward layer

            #add skip connection and normalise
            current_activation = self.skip_norm[reversed_layer_id](current_activation + skip_activations[reversed_layer_id])

        return current_activation #return the final outputs
    
    def project_down(self, input_mapping):
        #initial pass
        current_activation = input_mapping
        skip_activations = [current_activation] #keep a list of all layer activations for skip connection

        for layer in self.down_proj:
            current_activation = layer(current_activation) #downsized until it reaches bottleneck
            skip_activations.append(current_activation) #store layer-wise activations for skip connection
        
        return skip_activations