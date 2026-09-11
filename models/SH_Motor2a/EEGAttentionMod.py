#class module for building attention input 
#for either electrode or coefficient input in shape (B x input_dim x F)
#where input_dim (electrode or coeffs) are the sequence-axis, and F is the 'embedding'

import torch
import torch.nn as nn 
import numpy as np
from ..architectures import MAH 

class EEGAttention(nn.Module):
    def __init__(self, input_dim, num_features, query_config, dr=0.2):
        super().__init__()

        #pass only the query side of the attention config dict we need to build for MAH
        assert isinstance(query_config, dict) and len(query_config) > 0, "query_config must be a non-empty dict"

        self._eeg_in = input_dim #electrodes or SH-coefficients
        self._num_feat = num_features
        self._attn_names = {label: {'query_dim': query_dim, 
                                    'num_heads': num_heads} 
                                    for label, (query_dim, num_heads) in query_config.items()}

        self._build(dr=dr)

        #store args to rebuild when loading saved weights
        self.config = {'input_dim': input_dim, 
                       'num_features': num_features, 
                       'query_config': query_config, 
                       'dr': dr}

    #initialiser for learnable query vector in _build()
    def _init_query_vec(self, query_dim):
        bound = 1.0 / np.sqrt(query_dim)
        return torch.zeros(query_dim).uniform_(-bound, bound)

    def _build(self, dr):
        #use input dim as the position (sequence) axis, and num features as the embedding
        #since expected inputs are either coeffs or electrodes we can see which 'positions'
        #were attended to relative to a prediction; 
        attn_config = {
            label: {'q': (1, spec['query_dim']), #1 x query size
                    'k': (self._eeg_in, self._num_feat), #input x F
                    'v': (self._eeg_in, self._num_feat), #input x F
                    'num_heads': spec['num_heads']
                    }
            for label, spec in self._attn_names.items()
        }

        #dont dense project; 
        self._attn_heads = MAH(head_specs=attn_config, use_dense=False, dropout=dr)

        #one learnable query per attn, at fan-in scale so pre-softmax logits start unsaturated
        #query config specifies how many singular attention heads to use (not the num_heads per attention head)
        self.query_vecs = nn.ParameterDict({
            label: nn.Parameter(
                self._init_query_vec(spec['query_dim'])) 
                for label, spec in self._attn_names.items()
                })

    #learnable init vector taken from CTM implementation
    def _expand_query(self, label, batch_size):
        return self.query_vecs[label].unsqueeze(0).unsqueeze(1).expand(batch_size, -1, -1) #B x seq=1 x query_dim

    #eeg feature input expected in shape B x input_dim x F
    def forward(self, eeg_features):
        assert eeg_features.shape[1:] == (self._eeg_in, self._num_feat), \
            f"expected B x {self._eeg_in} x {self._num_feat}, got {eeg_features.shape}"

        attn_in = {
            label: {'q': self._expand_query(label=label, batch_size=eeg_features.shape[0]), #B x q x query 
                    'k': eeg_features, #b x input_dim x F
                    'v': eeg_features, #b x input_dim x F
                    'attn_mask': None #not used
                    } 
            for label in self._attn_names
        }

        #dont suppress warning if seq length differs, since they should be fixed in this case
        attn_outs = self._attn_heads(inputs=attn_in, aggregate='dict',
                                     return_weights=True, #always true so its accessible via attn_scores property
                                     suppress_seq_warning=False)

        #label -> B x query_dim
        return {label: attended.squeeze(1) 
                for label, attended in attn_outs.items()} 

    @property
    def registered_labels(self):
        return list(self._attn_names.keys())

    #per-attn query size
    @property
    def query_dims(self):
        return {label: spec['query_dim'] for label, spec in self._attn_names.items()}

    #readout size for a readout model to expect
    @property
    def readout_dim(self):
        return sum(spec['query_dim'] for spec in self._attn_names.values())

    @property #retrieves current attention scores
    def attn_scores(self):
        return self._attn_heads.current_att_weight #dict of label -> B x q_seq x k_seq

    def get_parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'non_trainable': total - trainable}

    def print_param_count(self):
        for p_type, count in self.get_parameter_counts().items():
            print(f"{p_type.lower()} parameters: {count}")