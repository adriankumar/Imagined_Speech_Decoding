import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from dataclasses import dataclass

from ...architectures import build_ltc, auto_layers, auto_fanouts
from ..EEGAttentionMod import EEGAttention
from ..one_decoder.attn_ltc import Motor2aLTC1D

#two decoders; one for task-active vs idle/rest, the other for task-specific gated by the first decoder
#decoder 1 is a whole 1D so it trains and checkpoints standalone in stage one


#the recurrent values threaded between windows, one per decoder
#None throughout means a fresh sequence
@dataclass
class Motor2aCarry:
    state_1: torch.Tensor = None
    state_2: torch.Tensor = None
    prev_1: torch.Tensor = None
    prev_2: torch.Tensor = None


class Motor2aLTC2D(nn.Module):
    def __init__(self, input_dim, num_features, query1_dim, query2_dim,
                 num_heads1, num_heads2, total_neurons1, total_neurons2,
                 out1_dim, out2_dim, dr=0.2, gate_threshold=0.7):

        super().__init__()

        assert 0 < gate_threshold <= 1, f"gate threshold must be between 0-1, got {gate_threshold}"

        #build model
        self._build(input_dim=input_dim, num_features=num_features, query1_dim=query1_dim,
                    query2_dim=query2_dim, num_heads1=num_heads1, num_heads2=num_heads2,
                    num_neurons1=total_neurons1, num_neurons2=total_neurons2,
                    out1=out1_dim, out2=out2_dim, dr=dr, gate_threshold=gate_threshold)

        #save config
        self.config = {'input_dim': input_dim, 'num_features': num_features,
                       'query1_dim': query1_dim, 'query2_dim': query2_dim,
                       'num_heads1': num_heads1, 'num_heads2': num_heads2,
                       'total_neurons1': total_neurons1, 'total_neurons2': total_neurons2,
                       'out1_dim': out1_dim, 'out2_dim': out2_dim,
                       'dr': dr, 'gate_threshold': gate_threshold}

    def _build(self, input_dim, num_features, query1_dim, query2_dim, num_heads1, num_heads2,
               num_neurons1, num_neurons2, out1, out2, dr, gate_threshold):

        self._gate_threshold = gate_threshold
        self._out2_dim = out2
        self._attn_label_2 = 'task_spec'

        self._frozen_1 = False
        self._frozen_2 = False

        #for task-active vs idle/rest
        self._decode_1 = Motor2aLTC1D(input_dim=input_dim, num_features=num_features, query_dim=query1_dim,
                                      num_attn_heads=num_heads1, total_neurons=num_neurons1, output_dim=out1,
                                      dr=dr, attn_label='active_vs_idle')

        #for specialised, attends the same input against its own learnable query
        self._attention_2 = EEGAttention(input_dim=input_dim, num_features=num_features,
                                         query_config={self._attn_label_2: (query2_dim, num_heads2)}, dr=dr)

        #concat decode 2 with decode 1's latent and decode 2's previous logits
        #decode 1 is detached at the concat
        self._readout2_dim = self._attention_2.latent_dim + self._decode_1.latent_dim + out2

        l1, l2, l3 = auto_layers(num_neurons2) #ltc layers
        ifo, l1fo, l2fo = auto_fanouts(l1, l2, l3, input_dim=self._readout2_dim) #ltc fanouts
        self._ltc_2 = build_ltc(layer_1=l1, layer_2=l2, layer_3=l3,
                                input_fanout=ifo, l1_fanout=l1fo, l2_fanout=l2fo,
                                self_connections=(num_neurons2//2), input_dim=self._readout2_dim,
                                output_dim=out2, print_parameters=False)

        #learnable init for no previous prediction, at fan-in scale to match the latents it concats with
        bound = 1.0 / np.sqrt(out2)
        self.register_parameter('init_prev_2', param=nn.Parameter(torch.zeros(out2).uniform_(-bound, bound)))

    #expand the learnable prev across the batch
    def _init_prev_2(self, batch_size):
        return self.init_prev_2.unsqueeze(0).expand(batch_size, -1) #B x out2_dim

    #===decoder 1, task-active vs idle/rest===
    #x is shape B x input_dim (coeffs or electrodes) x F; train this path first
    def attend_gate(self, x, return_weights=False):
        return self._decode_1.attend(x=x, return_weights=return_weights) #B x query1_dim

    def gate_logits(self, latent_1, state_1=None, prev_1=None, elapse=1.0):
        if prev_1 is None:
            prev_1 = self._decode_1._init_prev(batch_size=latent_1.shape[0])

        readout_input = self._decode_1._readout_input(latent=latent_1, prev=prev_1)
        return self._decode_1.readout(x=readout_input, state=state_1, elapse=elapse) #B x out1_dim; raw logits

    #===decoder 2, task-specific===
    #x is the same input shape
    def attend_task(self, x, return_weights=False):
        latents = self._attention_2(x=x, return_weights=return_weights)
        latent = latents[self._attn_label_2] #B x query2_dim

        if return_weights:
            return latent, self.curr_attn_weights_2 #attn weights is shape B x q_seq x k_seq

        return latent, None

    #latent 2 is B x query2_dim, latent 1 is B x query1_dim, prev 2 is B x out2_dim
    #latent 1 is detached here, the task attention adjusts to the gate context without shaping it
    #layer norm over the concat since the latents come from independently trained attentions
    def _readout_input(self, latent_1, latent_2, prev_2):
        readout_input = torch.cat([latent_2, latent_1.detach(), prev_2], dim=-1) #B x (query2 + query1 + out2); task first
        return F.layer_norm(readout_input, [self._readout2_dim])

    def task_logits(self, latent_1, latent_2, state_2=None, prev_2=None, elapse=1.0):
        if prev_2 is None:
            prev_2 = self._init_prev_2(batch_size=latent_2.shape[0])

        readout_input = self._readout_input(latent_1=latent_1, latent_2=latent_2, prev_2=prev_2)

        assert readout_input.shape[-1] == self._readout2_dim, f"expected readout size of {self._readout2_dim}, got {readout_input.shape[-1]}"

        logits, state_2 = self._ltc_2(x=readout_input, state=state_2, elapsed_sub_time=elapse) #B x out2_dim
        return logits, state_2

    #===full paths===
    #single window; both paths run unconditionally, gating is applied in predict
    def forward(self, x, carry=None, elapse=1.0, return_weights=False):
        if carry is None:
            carry = Motor2aCarry()

        latent_1, attended_1 = self.attend_gate(x=x, return_weights=return_weights)
        latent_2, attended_2 = self.attend_task(x=x, return_weights=return_weights)

        logits_1, state_1 = self.gate_logits(latent_1=latent_1, state_1=carry.state_1,
                                             prev_1=carry.prev_1, elapse=elapse)

        logits_2, state_2 = self.task_logits(latent_1=latent_1, latent_2=latent_2,
                                             state_2=carry.state_2, prev_2=carry.prev_2, elapse=elapse)

        carry = Motor2aCarry(state_1=state_1, state_2=state_2, prev_1=logits_1, prev_2=logits_2)

        return logits_1, logits_2, carry, attended_1, attended_2

    #returns both probability sets and the gate as a boolean, masking is left to the caller
    def predict(self, x, carry=None, elapse=1.0, return_weights=False):
        logits_1, logits_2, carry, attended_1, attended_2 = self.forward(x=x, carry=carry, elapse=elapse,
                                                                         return_weights=return_weights)

        gate_probs = torch.sigmoid(logits_1) #B x out1_dim
        task_probs = torch.sigmoid(logits_2) #B x out2_dim
        gate = gate_probs > self._gate_threshold #B x out1_dim

        return gate_probs, task_probs, gate, carry, attended_1, attended_2

    #decoder 2 only carries across confident windows, so its recurrence resets where the gate is closed
    #per-sample since the gate varies within a batch
    def reset_task_carry(self, carry, gate):
        state_2 = torch.where(gate, carry.state_2, torch.zeros_like(carry.state_2))
        prev_2 = torch.where(gate, carry.prev_2, self._init_prev_2(batch_size=carry.prev_2.shape[0]))

        return Motor2aCarry(state_1=carry.state_1, state_2=state_2, prev_1=carry.prev_1, prev_2=prev_2)

    #normalise elapse to one value per window, a scalar means uniform spacing
    def _step_elapse(self, elapse, n_windows):
        if isinstance(elapse, (int, float)):
            return [float(elapse)] * n_windows

        assert len(elapse) == n_windows, f"expected {n_windows} elapse values, got {len(elapse)}"
        return elapse

    #x_list is a list of single windows, each B x input_dim x F
    #drives forward() per window, threading the carry; nothing about the step lives here
    #prev stays attached so gradient reaches how a logit at t-1 shaped the input at t
    def unroll(self, x_list, elapse=1.0, return_weights=False):
        assert len(x_list) > 0, "expected at least one window"

        step_elapse = self._step_elapse(elapse=elapse, n_windows=len(x_list))

        carry = None
        all_logits_1, all_logits_2 = [], []
        all_weights_1, all_weights_2 = [], []

        for x, step in zip(x_list, step_elapse):
            logits_1, logits_2, carry, attended_1, attended_2 = self.forward(x=x, carry=carry, elapse=step,
                                                                             return_weights=return_weights)

            #the cache is overwritten each pass, so collect per window
            all_weights_1.append(attended_1)
            all_weights_2.append(attended_2)
            all_logits_1.append(logits_1)
            all_logits_2.append(logits_2)

        #B x n_windows x out_dim
        return torch.stack(all_logits_1, dim=1), torch.stack(all_logits_2, dim=1), carry, all_weights_1, all_weights_2

    #===staged training===
    def freeze_decode_1(self):
        self._decode_1.freeze()
        self._frozen_1 = True

    def freeze_decode_2(self):
        self._attention_2.requires_grad_(False)
        self._attention_2.eval()
        self._ltc_2.requires_grad_(False)
        self._ltc_2.eval()
        self.init_prev_2.requires_grad_(False)
        self._frozen_2 = True

    #train() propagates to every child, so re-assert eval on whatever is frozen
    def train(self, mode=True):
        super().train(mode)

        if self._frozen_1:
            self._decode_1.eval()

        if self._frozen_2:
            self._attention_2.eval()
            self._ltc_2.eval()

        return self

    @property
    def decode_1(self):
        return self._decode_1 #exposed so stage one loads a 1D checkpoint straight into it

    @property
    def gate_threshold(self):
        return self._gate_threshold

    @property
    def readout2_dim(self):
        return self._readout2_dim

    @property
    def curr_attn_weights_1(self):
        return self._decode_1.curr_attn_weights #dict of label -> B x q_seq x k_seq

    @property
    def curr_attn_weights_2(self):
        return self._attention_2.curr_attn_weights #dict of label -> B x q_seq x k_seq

    def get_parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'non_trainable': total - trainable}

    def print_param_count(self):
        for p_type, count in self.get_parameter_counts().items():
            print(f"{p_type.lower()} parameters: {count}")