import torch 
import torch.nn as nn 
import torch.nn.functional as F
from ..architectures import build_ltc, auto_layers, auto_fanouts
from .EEGAttentionMod import EEGAttention
from .helpers import DECISIONS

#LTC readout after EEGAttention on input
class Motor2aLTC(nn.Module):
    def __init__(self, input_dim, num_features, query_dim, 
                 num_heads, total_neurons, output_dim,
                 dr=0.2, attn_label="coeff_attn", decision="sigmoid", 
                 input_clip=None, input_mean=None): 

        assert decision in list(DECISIONS.keys()), f"decision must be either 'sigmoid' or 'softmax', got {decision}"

        super().__init__()

        self._build(input_dim=input_dim, num_features=num_features, query_dim=query_dim, 
                    num_heads=num_heads, neurons=total_neurons, out=output_dim, 
                    dr=dr, attn_label=attn_label, 
                    input_clip=input_clip, input_mean=input_mean)

        self._act_fn = DECISIONS[decision] #returns the torch.nn.functional object

        #store args to rebuild when loading saved weights;
        #input clip and input mean are registered buffers in the model as they are tensors
        #so not stored in config
        self.config = {'input_dim': input_dim, 'num_features': num_features,
                       'query_dim': query_dim, 'num_heads': num_heads,
                       'total_neurons': total_neurons, 'output_dim': output_dim, 'dr': dr, 
                       'attn_label': attn_label, 'decision': decision}

    def _build(self, input_dim, num_features, query_dim, num_heads, 
               neurons, out, dr, attn_label, input_clip, input_mean):

        import numpy as np #local import for learnable prev

        self._out = out 
        self._attn_label = attn_label 

        #eeg-attention module
        #input dim is seq axis for KV, features are embedding dim
        self._attention = EEGAttention(input_dim=input_dim, num_features=num_features,
                                       query_config={attn_label: (query_dim, num_heads)},
                                       dr=dr)

        #read out classifier is recurrent, allow carry
        #of previous prediction as input
        readout_dim = self._attention.readout_dim + out 

        #neurons and fanouts in each ltc layer
        l1, l2, l3 = auto_layers(total=neurons)
        ifo, l1fo, l2fo = auto_fanouts(l1=l1, l2=l2, l3=l3, input_dim=readout_dim)

        self._readout = build_ltc(layer_1=l1, layer_2=l2, layer_3=l3, 
                                  input_fanout=ifo, l1_fanout=l1fo, l2_fanout=l2fo,
                                  self_connections=(neurons//2), 
                                  input_dim=readout_dim, output_dim=out)

        #learnable init for no previous prediction, at fan-in scale to match the latent it concats with
        bound = 1.0 / np.sqrt(out)
        self.register_parameter('init_prev', param=nn.Parameter(torch.zeros(out).uniform_(-bound, bound)))

        #clip un-used in the model, check @property 
        mean = torch.zeros(input_dim, num_features) if input_mean is None else torch.as_tensor(input_mean, dtype=torch.float32)
        clip = torch.full((num_features,), float("inf")) if input_clip is None else torch.as_tensor(input_clip, dtype=torch.float32)

        self.register_buffer('input_mean', mean) #input_dim x F
        self.register_buffer('input_clip', clip) #F

    #expand the learnable prev across the batch
    def _expand_prev(self, batch_size):
        return self.init_prev.unsqueeze(0).expand(batch_size, -1) #B x output_dim

    #eeg features is shape B x input_dim x F
    def attend(self, eeg_features):
        x = eeg_features - self.input_mean #center input to training dist mean
        latents = self._attention(eeg_features)[self._attn_label] #B x query_dim
        return latents 

    #use for training w sigmoid()
    def forward(self, eeg_features, state=None, prev=None, elapse=1.0):
        latent = self.attend(eeg_features=eeg_features) #B x query_dim

        #recurrent readout 
        if prev is None:
            prev = self._expand_prev(batch_size=eeg_features.shape[0])

        #concat readout input
        readin = F.layer_norm(torch.cat([latent, prev], dim=-1), #B x readout_size
                              normalized_shape=[self.readout_size])

        assert readin.shape[-1] == self.readout_size, \
            f"expected readout size of {self.readout_size}, got {readin.shape[-1]}"

        #readout
        logits, state = self._readout(x=readin, state=state, elapsed_sub_time=elapse) #B x output_dim
        
        return logits, state 

    #use for training w softmax(),
    def predict(self, eeg_features, state=None, prev=None, elapse=1.0):
        logits = self.forward(eeg_features=eeg_features, state=state, prev=prev, elapse=elapse) #B x output_dim
        return self._act_fn(logits)

    #not used internally by the model, use torch.minimum(window, model.electrode_clip)
    #externally before computing coefficients or passing as raw electrodes as input
    @property
    def electrode_clip(self):
        return self.input_clip

    @property
    def curr_attn_scores(self):
        return self._attention.attn_scores #dict of label -> B x q_seq x k_seq

    @property 
    def readout_size(self):
        return self._attention.readout_dim + self._out 

    @property
    def output_dim(self):
        return self._out

    @property
    def attn_label(self):
        return self._attn_label

    def get_parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'non_trainable': total - trainable}

    def print_param_count(self):
        for p_type, count in self.get_parameter_counts().items():
            print(f"{p_type.lower()} parameters: {count}")