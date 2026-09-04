import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from ...architectures import build_ltc, auto_layers, auto_fanouts
from ..EEGAttentionMod import EEGAttention

#xD -> x decoders
class Motor2aLTC1D(nn.Module):
    def __init__(self, input_dim, num_features, query_dim, num_attn_heads,
                 total_neurons, output_dim, dr=0.2, attn_label='decoder'):

        super().__init__()

        #build model
        self._build(input_dim=input_dim, num_features=num_features, query_dim=query_dim,
                    num_heads=num_attn_heads, num_neurons=total_neurons, out=output_dim,
                    dr=dr, attn_label=attn_label)

        #save config
        self.config = {'input_dim': input_dim, 'num_features': num_features,
                       'query_dim': query_dim, 'num_attn_heads': num_attn_heads,
                       'total_neurons': total_neurons, 'output_dim': output_dim,
                       'dr': dr, 'attn_label': attn_label}

    def _build(self, input_dim, num_features, query_dim, num_heads, num_neurons, out, dr, attn_label):

        self._output_dim = out
        self._attn_label = attn_label 

        self._attention = EEGAttention(input_dim=input_dim, num_features=num_features,
                                       query_config={attn_label: (query_dim, num_heads)}, dr=dr)

        #readout is the LTC; concatenates previous logits into the input
        self._readout_dim = self._attention.latent_dim + out

        l1, l2, l3 = auto_layers(num_neurons) #ltc layers
        ifo, l1fo, l2fo = auto_fanouts(l1, l2, l3, input_dim=self._readout_dim) #ltc fanouts
        self._ltc = build_ltc(layer_1=l1, layer_2=l2, layer_3=l3,
                              input_fanout=ifo, l1_fanout=l1fo, l2_fanout=l2fo,
                              self_connections=(num_neurons//2), input_dim=self._readout_dim,
                              output_dim=out, print_parameters=False)

        #learnable init for no previous prediction, at fan-in scale to match the latent it concats with
        bound = 1.0 / np.sqrt(out)
        self.register_parameter('init_prev', param=nn.Parameter(torch.zeros(out).uniform_(-bound, bound)))

    #expand the learnable prev across the batch
    def _init_prev(self, batch_size):
        return self.init_prev.unsqueeze(0).expand(batch_size, -1) #B x output_dim

    #x is shape B x input_dim x F
    def attend(self, x, return_weights=False):
        latents = self._attention(x=x, return_weights=return_weights)
        latent = latents[self._attn_label] #B x query_dim

        if return_weights:
            return latent, self.curr_attn_weights #attn weights is shape B x q_seq x k_seq

        return latent, None

    #latent is B x query_dim, prev is B x output_dim
    def _readout_input(self, latent, prev):
        readout_input = torch.cat([latent, prev], dim=-1) #B x (query + output_dim)

        #layernorm to prevent previous preds from having unbounded growth
        #but trade off is we lose absolute magnitude information from their norm
        return F.layer_norm(readout_input, [self._readout_dim]) #same shape

    #x is shape B x (query_dim + output_dim)
    def readout(self, x, state=None, elapse=1.0):
        assert x.dim() == 2, f"expected input shape B x dim, got {x.shape}"
        assert x.shape[-1] == self._readout_dim, f"expected readout size of {self._readout_dim}, got {x.shape[-1]}"

        logits, state = self._ltc(x=x, state=state, elapsed_sub_time=elapse) #B x output_dim
        return logits, state

    #single window; prev is logits not probabilities
    def forward(self, x, state=None, prev=None, elapse=1.0, return_weights=False):
        latent, attended = self.attend(x=x, return_weights=return_weights) #B x query dim

        if prev is None:
            prev = self._init_prev(batch_size=x.shape[0])

        readout_input = self._readout_input(latent=latent, prev=prev)
        logits, state = self.readout(x=readout_input, state=state, elapse=elapse) #B x output_dim

        return logits, state, attended

    def predict(self, x, state=None, prev=None, elapse=1.0, return_weights=False):
        logits, state, attended = self.forward(x=x, state=state, prev=prev, elapse=elapse, return_weights=return_weights)
        probs = torch.sigmoid(logits) #B x output_dim
        return probs, state, attended #do probs > threshold externally for boolean answers instead of probabilities

    #normalise elapse to one value per window, a scalar means uniform spacing
    def _step_elapse(self, elapse, n_windows):
        if isinstance(elapse, (int, float)):
            return [float(elapse)] * n_windows

        assert len(elapse) == n_windows, f"expected {n_windows} elapse values, got {len(elapse)}"
        return elapse

    #x_list is a list of single windows, each B x input_dim x F
    #calls forward() per window with state and prev;
    def unroll(self, x_list, elapse=1.0, return_weights=False):
        assert len(x_list) > 0, "expected at least one window"

        step_elapse = self._step_elapse(elapse=elapse, n_windows=len(x_list))

        state, prev = None, None
        all_logits, all_weights = [], []

        for x, step in zip(x_list, step_elapse):
            logits, state, attended = self.forward(x=x, state=state, prev=prev,
                                                   elapse=step, return_weights=return_weights)

            #the cache is overwritten each pass, so collect per window
            all_weights.append(attended)
            all_logits.append(logits)

            prev = logits #attached, the logits path is trained alongside the state path

        return torch.stack(all_logits, dim=1), state, all_weights #B x n_windows x output_dim

    #for staged training
    def freeze(self):
        self.requires_grad_(False)
        self.eval()

    @property
    def latent_dim(self):
        return self._attention.latent_dim #attention latent alone, prev concat is internal

    @property
    def readout_dim(self):
        return self._readout_dim #latent and prev concat

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def attn_label(self):
        return self._attn_label

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