import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

allowed_activations = ['leaky-relu', 'relu', 'silu']
#SincNet for audio waveform processing/feature extraction: https://arxiv.org/pdf/1808.00158 | https://arxiv.org/pdf/1811.09725
# https://github.com/mravanelli/SincNet?tab=readme-ov-file 

#sincnet will process frequency content across entire segment_length window
#extracts learnable frequency band features from temporal eeg signals
#default values should be 6157 parameters
class SincNet(nn.Module):
    def __init__(self, num_sinc_filters=24, output_dim=24,
                 eeg_channels=122, segment_length=500, freq=500, activation='silu'):

        if activation not in allowed_activations:
            raise ValueError(f"{activation} must be in {allowed_activations}. Failed to initialise")
        
        super(SincNet, self).__init__()
        
        self.num_sinc_filters = num_sinc_filters #learnable frequency bands
        self.output_dim = output_dim #final feature vector size
        self.eeg_channels = eeg_channels
        self.segment_length = segment_length
        self.sample_rate = freq
        self.activation_type = activation
        
        #sinc kernel must fit within segment_length; not sure how to validate this yet
        self.sinc_kernel_size = 151 
        
        #frequency constraints for eeg bands
        self.min_low_hz = 1.0 #theta band start (skip delta due to short window)
        self.min_band_hz = 2.0 #minimum bandwidth
        
        #list of conv layers after sinc in tuple form (filters, kernel_size, stride)
        self.conv_config = [(16, 8, 3), (24, 5, 2), (32, 2, 1)]
        
        self.create_sinc_layer() #creates sinc-based bandpass filters
        self.create_conv_layers() #creates post-sinc convolutions
        self.conv_to_feature_vector() #creates projection to output
        self.init_weights()
    
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                if self.activation_type == 'silu':
                    nn.init.xavier_uniform_(module.weight)
                elif self.activation_type == 'relu':
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:  # leaky_relu
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', a=0.2, nonlinearity='leaky_relu')
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Linear):
                if self.activation_type == 'silu':
                    nn.init.xavier_uniform_(module.weight)
                elif self.activation_type == 'relu':
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:  # leaky_relu
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', a=0.2, nonlinearity='leaky_relu')
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def create_sinc_layer(self):
        #learnable parameters: low frequency cutoff and bandwidth for each filter
        #initialize across eeg frequency spectrum (theta to high gamma)
        max_freq = self.sample_rate / 2
        self.low_hz = nn.Parameter(
            torch.linspace(self.min_low_hz, max_freq - self.min_band_hz, self.num_sinc_filters)
        )
        self.band_hz = nn.Parameter(
            torch.ones(self.num_sinc_filters) * self.min_band_hz
        )
        
        #hamming window for smoother filters
        n = torch.linspace(0, self.sinc_kernel_size - 1, self.sinc_kernel_size)
        window = 0.54 - 0.46 * torch.cos(2 * np.pi * n / (self.sinc_kernel_size - 1))
        self.register_buffer('window', window)
        
        #time axis for sinc function
        t = torch.linspace(-(self.sinc_kernel_size - 1) / 2, 
                          (self.sinc_kernel_size - 1) / 2, 
                          self.sinc_kernel_size)
        t = t / self.sample_rate #normalize by sampling rate
        self.register_buffer('t', t.view(1, 1, -1))
    
    def create_conv_layers(self):
        self.conv_layers = nn.ModuleList()
        in_channels = self.num_sinc_filters
        
        for filter_amount, kernel_size, stride in self.conv_config:
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=filter_amount,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size // 2),
                    bias=True
                )
            )
            in_channels = filter_amount
    
    def conv_to_feature_vector(self):
        #calculate output size after convolutions
        final_channels = self.conv_config[-1][0]
        
        #global average pooling handles variable time dimension
        #so we only need to project from final_channels to output_dim
        self.projection = nn.Linear(
            in_features=final_channels,
            out_features=self.output_dim,
            bias=True
        )
    
    def apply_sinc_conv(self, x):
        #constrain frequencies to valid ranges
        low_hz = self.low_hz.clamp(self.min_low_hz, self.sample_rate / 2)
        high_hz = (low_hz + self.band_hz.clamp(self.min_band_hz)).clamp(
            self.min_low_hz, self.sample_rate / 2
        )
        
        #convert to normalized frequencies
        low = 2 * low_hz / self.sample_rate
        high = 2 * high_hz / self.sample_rate
        
        #expand dimensions for broadcasting
        low = low.view(self.num_sinc_filters, 1, 1)
        high = high.view(self.num_sinc_filters, 1, 1)
        t = self.t #shape: (1, 1, kernel_size)
        
        #create bandpass filters using sinc functions
        low_pass_high = 2 * high * torch.sinc(2 * high * t)
        low_pass_low = 2 * low * torch.sinc(2 * low * t)
        band_pass = low_pass_high - low_pass_low
        
        #apply hamming window
        band_pass = band_pass * self.window.view(1, 1, -1)
        
        #normalize filters
        band_pass = band_pass / (band_pass.abs().sum(dim=2, keepdim=True) + 1e-7)
        
        #apply convolution with sinc filters to each channel
        #reshape input: (batch, channels, time) -> (batch * channels, 1, time)
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size * self.eeg_channels, 1, -1)
        
        #convolve: filters shape (num_sinc_filters, 1, kernel_size)
        filtered = F.conv1d(
            x_reshaped,
            band_pass,
            padding=self.sinc_kernel_size // 2
        ) #output: (batch * channels, num_sinc_filters, time)
        
        #reshape back and aggregate across channels
        filtered = filtered.view(batch_size, self.eeg_channels, self.num_sinc_filters, -1)
        
        #average across channels: all channels share same frequency space
        filtered = filtered.mean(dim=1) #(batch, num_sinc_filters, time)
        
        return filtered

    #note any F function that ends in '_' is an in-place operator as opposed to returning something
    def activation_F(self, x):
        if self.activation_type == 'leaky-relu':
            return F.leaky_relu(x, 0.2)
        elif self.activation_type == 'relu':
            return F.relu(x)
        elif self.activation_type == 'silu':
            return F.silu(x)
        
    def forward(self, x):
        #assume x is of shape batch x eeg_channels x segment_length
        
        #normalize input before filtering
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-5)
        
        #apply sinc-based bandpass filtering
        x = self.apply_sinc_conv(x) #(batch, num_sinc_filters, time)
        x = x.permute(0, 2, 1) #shape b x time x sinc_filters
        x = F.layer_norm(x, [x.shape[2]]) #normalise on feature dim
        x = x.permute(0, 2, 1) #shape b x f x t

        x = self.activation_F(x)
        
        #pass through standard conv layers
        for conv in self.conv_layers:
            x = self.activation_F(conv(x))
        
        #global average pooling across time dimension
        x = x.mean(dim=2) #(batch, final_channels)
        
        #project to output dimension
        feature_vector = self.activation_F(self.projection(x))
        
        return feature_vector #b x output_dim
    
    def get_frequency_bands(self):
        #utility to inspect learned frequency bands
        with torch.no_grad():
            low_hz = self.low_hz.clamp(self.min_low_hz, self.sample_rate / 2)
            high_hz = (low_hz + self.band_hz.clamp(self.min_band_hz)).clamp(
                self.min_low_hz, self.sample_rate / 2
            )
            return low_hz.cpu().numpy(), high_hz.cpu().numpy()
        

#plotting function - will move to utilities soon
def plot_sinc_filters(sincnet_model, filters=[0, 4]):
    low_hz, high_hz = sincnet_model.get_frequency_bands()
    start, end = filters
    n_filters = end - start
    
    fig, axes = plt.subplots(n_filters, 1, figsize=(10, 2*n_filters))
    if n_filters == 1:
        axes = [axes]
    
    for i, filt_idx in enumerate(range(start, end)):
        # Horizontal bar showing frequency band
        axes[i].barh(0, high_hz[filt_idx] - low_hz[filt_idx], 
                     left=low_hz[filt_idx], height=0.5, color='steelblue')
        axes[i].set_xlim(0, 250)  # 0 to Nyquist (500/2)
        axes[i].set_ylim(-1, 1)
        axes[i].set_xlabel('Frequency (Hz)')
        axes[i].set_yticks([])
        axes[i].set_title(f'Filter {filt_idx}: {low_hz[filt_idx]:.1f} - {high_hz[filt_idx]:.1f} Hz')
        axes[i].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()