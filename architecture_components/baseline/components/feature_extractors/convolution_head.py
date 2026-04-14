import torch 
import torch.nn.functional as F
import torch.nn as nn

#2 types of convolutions:
#1 - raw eeg to extract raw spatial (may be useless may not be) (b x c x seg)
#2 - spectrogram 
#note all will use a 2d conv, band power will have a lazy linear projector output

#globals; conv configuration, can be (filters, kernel_size_h, kernel_size_w, stride_h, stride_w) but is
#num_filters, kernel_size, stride
raw_spatial_conv_layers = [(16, 7, 3), (24, 5, 2), (8, 3, 1)] #last filter num is model arg
spectro_conv_layers = [(72, 8, 5), (42, 5, 3), (16, 4, 2), (8, 3, 1)] #compresses from 122 chans; 
compressor_conv_layers = [(32, 8, 5), (24, 5, 3), (16, 4, 2), (8, 3, 1)]

allowed_activations = ['leaky-relu', 'relu', 'silu']

class ConvHead(nn.Module):
    def __init__(self, conv_layers=raw_spatial_conv_layers, 
                 f_per_filter=8, chans=1, height=122, width=500, activation='silu'):
        
        if activation not in allowed_activations:
            raise ValueError(f"{activation} must be in {allowed_activations}. Failed to initialise")
        
        super(ConvHead, self).__init__() #inhrent from nn.mod
        self.output_filters = conv_layers[-1][0] #conv config expected in tuple with (filters, kernel_size, stride), so last element and first tuple element is the final num filters
        self.fpf = f_per_filter #features per filter
        self.height = height 
        self.width = width 
        self.colour_chans = chans
        self.activation_type = activation

        self.conv_config = conv_layers
        self.conv_layers = nn.ModuleList()
    
        self.create_conv_layers() #creates convolutions
        self.conv_to_feature_maps() #creates convolution output layers
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if self.activation_type == 'silu':
                    nn.init.xavier_uniform_(module.weight)
                else:  # leaky_relu
                    nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                if self.activation_type == 'silu':
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def create_conv_layers(self):
        in_channels = self.colour_chans

        for filter_amount, kernel_size, stride in self.conv_config:
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filter_amount,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size//2,
                    bias=True
                )
            )
            in_channels = filter_amount #dynamically change next in_channels amount

    def conv_to_feature_maps(self):
        self.dense_layers = nn.ModuleList() #each num_filter gets processed by its own dense layer

        #calculate input size
        feature_out_h, feature_out_w = self.calculate_conv_output_dim()
        flattened = feature_out_h * feature_out_w

        #create dense layers
        for _ in range(self.output_filters):
            self.dense_layers.append(
                nn.Linear(
                    in_features=flattened,
                    out_features=self.fpf,
                    bias=True
                )
            )

    #calculate feature size dims to project conv into dense layers
    def calculate_conv_output_dim(self):
        h, w = self.height, self.width

        for _, kernel_size, stride in self.conv_config:
            padding = kernel_size // 2
            h = (h + 2 * padding - kernel_size) // stride + 1
            w = (w + 2 * padding - kernel_size) // stride + 1 
        
        return h, w

    #note any F function that ends in '_' is an in-place operator as opposed to returning something
    def activation_F(self, x):
        if self.activation_type == 'leaky-relu':
            return F.leaky_relu(x, 0.2)
        elif self.activation_type == 'relu':
            return F.relu(x)
        elif self.activation_type == 'silu':
            return F.silu(x)
        

    #assume shape is handled externally
    def forward(self, x, return_list=False):

        #normalise input
        x = (x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + 1e-5)

        for conv in self.conv_layers:
            x = self.activation_F(conv(x)) 
        
        #split filters along channel dim
        filter_outputs = torch.split(x, 1, dim=1)

        #extract features
        feature_layers = []
        for i, filter_output in enumerate(filter_outputs):
            #flatten spatial dim
            flattened = filter_output.view(filter_output.size(0), -1)
            feature = self.activation_F(self.dense_layers[i](flattened)) #forward pass; shape batch x features_per_filter
            feature_layers.append(feature)

        if return_list:
            return feature_layers #list of b x fpf; number of elements is number of filters

        #concatenate all features into a single vector
        feature_vector = torch.cat(feature_layers, dim=1) #batch x (output_dim)
        return feature_vector

    #returns count in [trainable, non-trainable, total]
    def get_parameter_counts(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        return [trainable_params, non_trainable_params, total_params]
    
    @property
    def num_filters(self):
        return self.conv_config[-1][0] #last conv layer, first element in tuple is the number of filters