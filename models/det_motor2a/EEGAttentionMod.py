import torch.nn as nn
import torch
import numpy as np

from ..architectures import MAH

#one attention module per decoder path; 
#every registered name reads the same B x input_dim x F input
class EEGAttention(nn.Module):
    def __init__(self, input_dim, num_features, query_config, dr=0.2):
        super().__init__()

        assert isinstance(query_config, dict) and len(query_config) > 0, "query_config must be a non-empty dict"

        self._input_dim = input_dim #is number of coeffs or number of electrodes
        self._num_feat = num_features
        self._attn_names = {label: {'query_dim': query_dim, 'num_heads': num_heads} for label, (query_dim, num_heads) in query_config.items()}

        self._build(dr=dr)

        self.config = {'input_dim': input_dim, 'num_features': num_features, 'query_config': query_config, 'dr': dr}

    def _build(self, dr):
        #use input dim as the position (sequence) axis, and num features as the embedding
        #since expected inputs are either coeffs or electrodes we can see which 'positions'
        #were attended to relative to a prediction
        attn_config = {
            label: {'q': (1, spec['query_dim']), #1 x query size
                    'k': (self._input_dim, self._num_feat), #input x F
                    'v': (self._input_dim, self._num_feat), #input x F
                    'num_heads': spec['num_heads']
                    }
            for label, spec in self._attn_names.items()
        }

        self._attn_heads = MAH(head_specs=attn_config, use_dense=False, dropout=dr)

        #one learnable query per attn, at fan-in scale so pre-softmax logits start unsaturated
        self.query_vecs = nn.ParameterDict({label: nn.Parameter(self._init_query_vec(spec['query_dim'])) for label, spec in self._attn_names.items()})

    def _init_query_vec(self, query_dim):
        bound = 1.0 / np.sqrt(query_dim)
        return torch.zeros(query_dim).uniform_(-bound, bound)

    #expand query across the batch
    def _expand_query(self, label, batch_size):
        return self.query_vecs[label].unsqueeze(0).unsqueeze(1).expand(batch_size, -1, -1) #B x seq=1 x query_dim

    #x is shape B x input_dim x F, shared by every registered stream
    def forward(self, x, return_weights=False):
        assert x.shape[1:] == (self._input_dim, self._num_feat), f"expected B x {self._input_dim} x {self._num_feat}, got {x.shape}"

        attn_input = {
            label: {'q': self._expand_query(label=label, batch_size=x.shape[0]),
                    'k': x, #B x input_dim x F
                    'v': x, #B x input_dim x F
                    'attn_mask': None
                    }
            for label in self._attn_names
        }

        #dont suppress warning if seq length differs,
        #since they should be fixed in this case
        attn_outs = self._attn_heads(inputs=attn_input, aggregate='dict',
                                     return_weights=return_weights,
                                     suppress_seq_warning=False)

        return {label: attended.squeeze(1) for label, attended in attn_outs.items()} #label -> B x query_dim

    @property
    def registered_labels(self):
        return list(self._attn_names.keys())

    #per-attn latent width for a readout 
    @property
    def latent_dims(self):
        return {label: spec['query_dim'] for label, spec in self._attn_names.items()}

    #total latent width for a readout if concatennated
    @property
    def latent_dim(self):
        return sum(spec['query_dim'] for spec in self._attn_names.values())

    @property
    def curr_attn_weights(self):
        return self._attn_heads.current_att_weight #dict of label -> B x q_seq x k_seq

    def get_parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'non_trainable': total - trainable}

    def print_param_count(self):
        for p_type, count in self.get_parameter_counts().items():
            print(f"{p_type.lower()} parameters: {count}")