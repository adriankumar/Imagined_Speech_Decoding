from model_architecture import (
    HjorthFeatures, SincNet, ConvHeadv0,
    extract_band_power, extract_spectrogram,
)
import torch.nn as nn
import torch.nn.functional as F
import torch

#globals
raw_spatial_convs = [(16, 7, 3), (24, 5, 2), (8, 3, 1)]
# spectro_convs = [(72, 8, 5), (42, 5, 3), (16, 4, 2), (8, 3, 1)]
spectro_convs = [(32, 8, 5), (24, 5, 3), (16, 4, 2), (8, 3, 1)]

class FeatureExtractorv0(nn.Module):
    def __init__(self, eeg_channels=122, segment_length=500, sfreq=500, 
                 hjorth_output_dim=32, sinc_filters=64, sinc_output_dim=32, sinc_activation='silu', bp_output_dim=32,
                 raw_spatial_convs=raw_spatial_convs, raw_spatial_fpf=4, raw_spatial_activation='silu',
                 spectro_convs=spectro_convs, spectro_fpf=4, spectro_activation='silu',
                 dropout=0.2
                 ):

        super(FeatureExtractorv0, self).__init__()
        
        self.eeg_channels = eeg_channels
        self.segment_length = segment_length
        self.sfreq = sfreq

        self.hjorth_out_dim = hjorth_output_dim
        self.sinc_filters = sinc_filters
        self.sinc_out_dim = sinc_output_dim
        self.sinc_actv = sinc_activation
        self.band_power_out_dim = bp_output_dim

        self.raw_spatial_convs = raw_spatial_convs
        self.raw_spatial_fpf = raw_spatial_fpf
        self.raw_spatial_actv = raw_spatial_activation

        self.spectro_convs = spectro_convs
        self.spectro_fpf = spectro_fpf 
        self.spectro_actv = spectro_activation

        self.dr = nn.Dropout(dropout) #after concat

        self._build_extractors()
    
    def _build_extractors(self):
        
        #takes b x c x seg as input
        self.hjorth = HjorthFeatures(
            eeg_channels=self.eeg_channels, segment_length=self.segment_length, output_dim=32
        )

        #takes b x c x seg as input
        self.sinc_net = SincNet(
            num_sinc_filters=self.sinc_filters, output_dim=32, eeg_channels=self.eeg_channels, 
            segment_length=self.segment_length, freq=self.sfreq, activation=self.sinc_actv
        )

        #takes b x c x seg as input
        self.raw_spatial = ConvHeadv0(
            conv_layers=self.raw_spatial_convs, f_per_filter=self.raw_spatial_fpf, 
            chans=1, height=self.eeg_channels, width=self.segment_length, activation=self.raw_spatial_actv
        )

        #note spectrogram and band power use numpy to compute features
        #so theres no gradients, thats fine because these are non-parameterised feature extractors
        #so the gradients only need to come from processing the outputs of the spectrogram and band power features
        #conv for spectrogram - get output dim of spectrogram to initialise conv layer
        spectro_h, spectro_w = self._get_spectro_dims()
        self.spectro_conv = ConvHeadv0(
            conv_layers=self.spectro_convs, f_per_filter=self.spectro_fpf, 
            chans=self.eeg_channels, height=spectro_h, width=spectro_w, activation=self.spectro_actv
        )

        #projector for band power
        self.bp_projector = nn.LazyLinear(
            out_features=self.band_power_out_dim, bias=True
        )
    
    def _get_spectro_dims(self):
        rand = torch.randn(self.eeg_channels, self.segment_length)
        spec = extract_spectrogram(rand)
        return spec.shape[1], spec.shape[-1]
    
    #even if batch dim = 1 just keep for consistency
    def batch_extract_spectrogram(self, batch_tensor):
        # batch_tensor: (batch, channels, time)
        spectrograms = []
        for i in range(batch_tensor.shape[0]):
            spec = extract_spectrogram(batch_tensor[i], fs=self.sfreq)  # (channels, freq, time)
            spectrograms.append(spec)
        return torch.stack(spectrograms, dim=0).to(batch_tensor.device)  # (batch, channels, freq, time)

    def batch_extract_bp(self, batch_tensor):
        bps = []
        for i in range(batch_tensor.shape[0]):
            bp = extract_band_power(batch_tensor[i], fs=self.sfreq)
            bps.append(bp) 
        return torch.stack(bps, dim=0).to(batch_tensor.device) #batch x channels x 5

    def forward(self, x):
        #x input shape is batch x channels x segment_length

        #pass to all feature extractors and concat output
        hjorth_vec = self.hjorth(x) #unnormalised, shape b x output_dim
        sinc_vec = self.sinc_net(x) #unnormalised, shape b x output_dim

        #add channel dim=1 for raw eeg spatial
        raw_spatial_vec = self.raw_spatial(x.unsqueeze(1)) #unnormalised, shape b x output_dim (output_dim = last num filters * fpf)

        raw_spectro_features = self.batch_extract_spectrogram(x) 
        spectro_vec = self.spectro_conv(raw_spectro_features) #unnormalised, shape b x output_dim (output_dim = last num filters * fpf)

        raw_bp_features = self.batch_extract_bp(x)
        #reshape to batch x channels * 5
        bp_flat = raw_bp_features.view(x.shape[0], -1) 
        #project
        bp_vec = F.silu(self.bp_projector(bp_flat)) #batch x output_dim

        #concatenate and layernorm
        feature_vector = torch.cat([hjorth_vec, sinc_vec, raw_spatial_vec, spectro_vec, bp_vec], dim=-1) #concate across feature dim
        feature_vector = self.dr(feature_vector) #dropout only occurs during train mode
    
        state_t = F.layer_norm(feature_vector, [feature_vector.shape[-1]])
        return state_t

    def print_parameter_count(self):
        hjorth_params = sum(p.numel() for p in self.hjorth.parameters())
        sinc_params = sum(p.numel() for p in self.sinc_net.parameters())
        raw_params = sum(p.numel() for p in self.raw_spatial.parameters())
        spectro_params = sum(p.numel() for p in self.spectro_conv.parameters())
        bp_params = sum(p.numel() for p in self.bp_projector.parameters())

        total = hjorth_params + sinc_params + raw_params + spectro_params + bp_params

        print(f"Total Parameter count for feature extractors: {total}")
        print(f"hjorth parameters                           : {hjorth_params}")
        print(f"sinc parameters                             : {sinc_params}")
        print(f"raw spatial conv parameters                 : {raw_params}")
        print(f"spectro conv parameters                     : {spectro_params}")
        print(f"band power proj parameters                  : {bp_params}")

        return total

#no hjorth features
class FeatureExtractorv1(nn.Module):
    def __init__(self, eeg_channels=122, segment_length=500, sfreq=500, 
                 sinc_filters=64, sinc_output_dim=32, sinc_activation='silu', bp_output_dim=32,
                 raw_spatial_convs=raw_spatial_convs, raw_spatial_fpf=4, raw_spatial_activation='silu',
                 spectro_convs=spectro_convs, spectro_fpf=4, spectro_activation='silu',
                 dropout=0.2
                 ):

        super(FeatureExtractorv1, self).__init__()
        
        self.eeg_channels = eeg_channels
        self.segment_length = segment_length
        self.sfreq = sfreq

        self.sinc_filters = sinc_filters
        self.sinc_out_dim = sinc_output_dim
        self.sinc_actv = sinc_activation
        self.band_power_out_dim = bp_output_dim

        self.raw_spatial_convs = raw_spatial_convs
        self.raw_spatial_fpf = raw_spatial_fpf
        self.raw_spatial_actv = raw_spatial_activation

        self.spectro_convs = spectro_convs
        self.spectro_fpf = spectro_fpf 
        self.spectro_actv = spectro_activation

        self.dr = nn.Dropout(dropout) #after concat

        self._build_extractors()
    
    def _build_extractors(self):

        #takes b x c x seg as input
        self.sinc_net = SincNet(
            num_sinc_filters=self.sinc_filters, output_dim=32, eeg_channels=self.eeg_channels, 
            segment_length=self.segment_length, freq=self.sfreq, activation=self.sinc_actv
        )

        #takes b x c x seg as input
        self.raw_spatial = ConvHeadv0(
            conv_layers=self.raw_spatial_convs, f_per_filter=self.raw_spatial_fpf, 
            chans=1, height=self.eeg_channels, width=self.segment_length, activation=self.raw_spatial_actv
        )

        #note spectrogram and band power use numpy to compute features
        #so theres no gradients, thats fine because these are non-parameterised feature extractors
        #so the gradients only need to come from processing the outputs of the spectrogram and band power features
        #conv for spectrogram - get output dim of spectrogram to initialise conv layer
        spectro_h, spectro_w = self._get_spectro_dims()
        self.spectro_conv = ConvHeadv0(
            conv_layers=self.spectro_convs, f_per_filter=self.spectro_fpf, 
            chans=self.eeg_channels, height=spectro_h, width=spectro_w, activation=self.spectro_actv
        )

        #projector for band power
        self.bp_projector = nn.LazyLinear(
            out_features=self.band_power_out_dim, bias=True
        )
    
    def _get_spectro_dims(self):
        rand = torch.randn(self.eeg_channels, self.segment_length)
        spec = extract_spectrogram(rand)
        return spec.shape[1], spec.shape[-1]
    
    #even if batch dim = 1 just keep for consistency
    def batch_extract_spectrogram(self, batch_tensor):
        # batch_tensor: (batch, channels, time)
        spectrograms = []
        for i in range(batch_tensor.shape[0]):
            spec = extract_spectrogram(batch_tensor[i], fs=self.sfreq)  # (channels, freq, time)
            spectrograms.append(spec)
        return torch.stack(spectrograms, dim=0).to(batch_tensor.device)  # (batch, channels, freq, time)

    def batch_extract_bp(self, batch_tensor):
        bps = []
        for i in range(batch_tensor.shape[0]):
            bp = extract_band_power(batch_tensor[i], fs=self.sfreq)
            bps.append(bp) 
        return torch.stack(bps, dim=0).to(batch_tensor.device) #batch x channels x 5

    def forward(self, x):
        #x input shape is batch x channels x segment_length

        #pass to all feature extractors and concat output
        sinc_vec = self.sinc_net(x) #unnormalised, shape b x output_dim

        #add channel dim=1 for raw eeg spatial
        raw_spatial_vec = self.raw_spatial(x.unsqueeze(1)) #unnormalised, shape b x output_dim (output_dim = last num filters * fpf)

        raw_spectro_features = self.batch_extract_spectrogram(x) 
        spectro_vec = self.spectro_conv(raw_spectro_features) #unnormalised, shape b x output_dim (output_dim = last num filters * fpf)

        raw_bp_features = self.batch_extract_bp(x)
        #reshape to batch x channels * 5
        bp_flat = raw_bp_features.view(x.shape[0], -1) 
        #project
        bp_vec = F.silu(self.bp_projector(bp_flat)) #batch x output_dim

        #concatenate and layernorm
        feature_vector = torch.cat([sinc_vec, raw_spatial_vec, spectro_vec, bp_vec], dim=-1) #concate across feature dim
        feature_vector = self.dr(feature_vector) #dropout only occurs during train mode
    
        state_t = F.layer_norm(feature_vector, [feature_vector.shape[-1]])
        return state_t

    def print_parameter_count(self):
        sinc_params = sum(p.numel() for p in self.sinc_net.parameters())
        raw_params = sum(p.numel() for p in self.raw_spatial.parameters())
        spectro_params = sum(p.numel() for p in self.spectro_conv.parameters())
        bp_params = sum(p.numel() for p in self.bp_projector.parameters())

        total = sinc_params + raw_params + spectro_params + bp_params

        print(f"Total Parameter count for feature extractors: {total}")
        print(f"sinc parameters                             : {sinc_params}")
        print(f"raw spatial conv parameters                 : {raw_params}")
        print(f"spectro conv parameters                     : {spectro_params}")
        print(f"band power proj parameters                  : {bp_params}")

        return total