import torch.nn as nn
import torch

from ...architectures import MLPClassifier
from ..EEGAttentionMod import EEGAttention

#xD -> x decoders
class Motor2aMLP1D(nn.Module):
    def __init__(self, input_dim, num_features, query_dim, num_attn_heads,
                 hidden_dim, output_dim, n_layers=3, dr=0.2, reduce_layers=True,
                 attn_label='decoder', input_clip=None, input_mean=None):

        super().__init__()

        #build model
        self._build(input_dim=input_dim, num_features=num_features, query_dim=query_dim, 
                    input_clip=input_clip, input_mean=input_mean, num_heads=num_attn_heads, 
                    hidden=hidden_dim, out=output_dim, layers=n_layers, dr=dr, 
                    reduce=reduce_layers, attn_label=attn_label)

        #save config
        self.config = {'input_dim': input_dim, 'num_features': num_features,
                       'query_dim': query_dim, 'num_attn_heads': num_attn_heads,
                       'hidden_dim': hidden_dim, 'output_dim': output_dim,
                       'n_layers': n_layers, 'dr': dr, 'reduce_layers': reduce_layers,
                       'attn_label': attn_label}

    def _build(self, input_dim, num_features, query_dim, 
               input_clip, input_mean, num_heads, hidden, 
               out, layers, dr, reduce, attn_label):

        self._output_dim = out
        self._attn_label = attn_label 

        #single attention module
        self._attention = EEGAttention(input_dim=input_dim, num_features=num_features,
                                       query_config={attn_label: (query_dim, num_heads)}, dr=dr)

        #attention output is B x query_dim
        self._readout_dim = self._attention.latent_dim

        self._mlp_readout = MLPClassifier(input_dim=self._readout_dim, hidden_dim=hidden,
                                          output_dim=out, n_hidden=layers, dropout=dr,
                                          reduce_layers=reduce)

        #clip un-used in the model, check @property 
        mean = torch.zeros(input_dim, num_features) if input_mean is None else torch.as_tensor(input_mean, dtype=torch.float32)
        clip = torch.full((num_features,), float("inf")) if input_clip is None else torch.as_tensor(input_clip, dtype=torch.float32)

        self.register_buffer('input_mean', mean) #input_dim x F
        self.register_buffer('input_clip', clip) #F

    #x is shape B x input_dim x F
    def attend(self, x, return_weights=False):
        x = x - self.input_mean #centering to train dist mean
        latents = self._attention(x=x, return_weights=return_weights)
        latent = latents[self._attn_label] #B x query_dim

        if return_weights:
            return latent, self.curr_attn_weights #attn weights is shape B x q_seq x k_seq

        return latent, None

    #latent is shape B x readout_dim
    def readout(self, latent):
        return self._mlp_readout(latent) #B x output_dim; raw logits

    #returns raw logits, predict() returns probabilities
    def forward(self, x, return_weights=False):
        latent, attended = self.attend(x=x, return_weights=return_weights)
        logits = self.readout(latent=latent)
        return logits, attended

    def predict(self, x, return_weights=False):
        logits, attended = self.forward(x=x, return_weights=return_weights)
        probs = torch.sigmoid(logits) #B x output_dim
        return probs, attended #do probs > threshold externally for boolean answers instead of probabilities

    #for staged training, once this path becomes a fixed feature extractor for a later one
    def freeze(self):
        self.requires_grad_(False)
        self.eval()

    @property
    def latent_dim(self):
        return self._readout_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def attn_label(self):
        return self._attn_label

    #not used internally by the model, use torch.minimum(window, model.electrode_clip)
    #externally before computing coefficients or passing as raw electrodes as input
    @property
    def electrode_clip(self):
        return self.input_clip

    @property
    def curr_attn_weights(self):
        return self._attention.curr_attn_weights #dict of label -> B x q_seq x k_seq

    def get_parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'non_trainable': total - trainable}

    def print_param_count(self):
        for p_type, count in self.get_parameter_counts().items():
            print(f"{p_type.lower()} parameters: {count}")