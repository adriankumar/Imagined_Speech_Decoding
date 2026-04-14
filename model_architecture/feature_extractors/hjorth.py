import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

#hjorth computes statistical descriptors of eeg temporal dynamics
#extracts activity, mobility, and complexity per channel (non-learnable)
#https://en.wikipedia.org/wiki/Hjorth_parameters 
#but has leanrable output projectors to convert statistical features into feature vector for model to process
class HjorthFeatures(nn.Module):
    def __init__(self, eeg_channels=122, segment_length=300, output_dim=24):
        
        super(HjorthFeatures, self).__init__()
        
        self.eeg_channels = eeg_channels
        self.segment_length = segment_length
        self.stat_output_dim = eeg_channels * 3 #activity, mobility, complexity per channel
        self.output_dim = output_dim
        
        self.eps = 1e-8 #numerical stability for division
        self.stats_to_feature_maps()
        self.final_projection = nn.LazyLinear(out_features=output_dim, bias=True)

    #like spatial conv, create independent 'dense' layers for: activity, mobility and  complexity
    def stats_to_feature_maps(self):
        self.dense_layers = nn.ModuleList()

        for _ in range(3):
            self.dense_layers.append(
                nn.Linear(in_features=self.eeg_channels, out_features=21, bias=True)
            )
    
    def compute_derivatives(self, x):
        #x shape: (batch, channels, time)
        
        #first derivative: dx/dt
        dx = x[:, :, 1:] - x[:, :, :-1] #(batch, channels, time-1)
        
        #second derivative: d²x/dt²
        d2x = dx[:, :, 1:] - dx[:, :, :-1] #(batch, channels, time-2)
        
        return dx, d2x
    
    def compute_variances(self, x, dx, d2x):
        #compute variance along time dimension for each channel
        
        #x shape: (batch, channels, time) -> var -> (batch, channels)
        var_x = torch.var(x, dim=-1, unbiased=False) #(batch, channels)
        
        #dx shape: (batch, channels, time-1) -> var -> (batch, channels)
        var_dx = torch.var(dx, dim=-1, unbiased=False) #(batch, channels)
        
        #d2x shape: (batch, channels, time-2) -> var -> (batch, channels)
        var_d2x = torch.var(d2x, dim=-1, unbiased=False) #(batch, channels)
        
        return var_x, var_dx, var_d2x
    
    def compute_hjorth_params(self, var_x, var_dx, var_d2x):
        #all inputs shape: (batch, channels)
        
        #activity: signal power
        activity = var_x #(batch, channels)
        
        #mobility: mean frequency (ratio of derivative std to signal std)
        mobility = torch.sqrt(var_dx / (var_x + self.eps)) #(batch, channels)
        
        #complexity: frequency bandwidth (mobility of derivative / mobility of signal)
        complexity = torch.sqrt(
            (var_d2x * var_x) / (var_dx * var_dx + self.eps)
        ) #(batch, channels)
        
        return activity, mobility, complexity
    
    def forward(self, x, return_raw_hjorth=False):
        #assume x is of shape batch x eeg_channels x segment_length
        #input shape: (batch, 122, 500)
        
        #normalize input to prevent scale issues
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-5)
        #shape: (batch, 122, 500)
        
        #compute first and second derivatives
        dx, d2x = self.compute_derivatives(x)
        #dx shape: (batch, 122, 499)
        #d2x shape: (batch, 122, 498)
        
        #compute variances for signal and derivatives
        #all should have shape: (batch, 122)
        var_x, var_dx, var_d2x = self.compute_variances(x, dx, d2x)

        
        #compute hjorth parameters per channel
        activity, mobility, complexity = self.compute_hjorth_params(var_x, var_dx, var_d2x)
        #all should have shape: (batch, 122)
        # print(f"Activity shape: {activity.shape} | mobility shape: {mobility.shape} | complexity shape: {complexity.shape}")
        # stat_features = torch.cat([activity, mobility, complexity], dim=1)
        #shape: (batch, 366) = (batch, 122*3)
        
        features_list = [activity, mobility, complexity]

        #pass to dense layers in order: activity, mobility and complexity
        feature_layers = [] #concats all projections
        for i in range(3):
            feature = features_list[i] #batch x channels
            # feature_vec = F.leaky_relu(self.dense_layers[i](feature), 0.2) #individal feature vec
            feature_vec = F.silu(self.dense_layers[i](feature)) #changed to silu just to keep consistent with other features
            feature_layers.append(feature_vec) #list of b x 21

        #concatenate all features into single vector
        feature_vector = torch.cat(feature_layers, dim=1) #batch x (21 * 3)
        feature_vector = F.layer_norm(feature_vector, [feature_vector.shape[-1]])

        #final output projection
        # feature_vector = F.leaky_relu(self.final_projection(feature_vector), 0.2) #shape batch x output_dim
        feature_vector = F.silu(self.final_projection(feature_vector))
        
        if return_raw_hjorth:
            return feature_vector, activity, mobility, complexity

        #remember this is unormalised, so use layernorm on this as ur state_t before passing any further
        return feature_vector



#plot function - will move to unifed utilites later
#assumes batch dim already removed
def plot_hjorth_params(activity, mobility, complexity):
    activity_np = activity.cpu().numpy()
    mobility_np = mobility.cpu().numpy()
    complexity_np = complexity.cpu().numpy()
    
    channels = np.arange(len(activity_np))
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    
    # Activity
    axes[0].plot(channels, activity_np, linewidth=1, color='blue')
    axes[0].set_xlabel('Channel')
    axes[0].set_ylabel('Activity (Variance)')
    axes[0].set_title('Hjorth Activity')
    axes[0].grid(True, alpha=0.3)
    
    # Mobility
    axes[1].plot(channels, mobility_np, linewidth=1, color='green')
    axes[1].set_xlabel('Channel')
    axes[1].set_ylabel('Mobility')
    axes[1].set_title('Hjorth Mobility')
    axes[1].grid(True, alpha=0.3)
    
    # Complexity
    axes[2].plot(channels, complexity_np, linewidth=1, color='orange')
    axes[2].set_xlabel('Channel')
    axes[2].set_ylabel('Complexity')
    axes[2].set_title('Hjorth Complexity')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()