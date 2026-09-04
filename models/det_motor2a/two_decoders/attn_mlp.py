import torch.nn as nn
import torch
import torch.nn.functional as F

from ...architectures import MLPClassifier
from ..EEGAttentionMod import EEGAttention
from ..one_decoder.attn_mlp import Motor2aMLP1D

#two decoders; one for task-active vs idle/rest, the other for task-specific gated by the first decoder
#decoder 1 is a whole 1D so it trains and checkpoints standalone in stage one

class Motor2aMLP2D(nn.Module):
    def __init__(self, input_dim, num_features, query1_dim, query2_dim,
                 num_heads1, num_heads2, hidden_dim, n_layers, out1_dim, out2_dim,
                 dr=0.2, reduce_layers=True, gate_threshold=0.7, input_mean=None, input_clip=None):

        super().__init__()

        assert 0 < gate_threshold <= 1, f"gate threshold must be between 0-1, got {gate_threshold}"

        #build model
        self._build(input_dim=input_dim, num_features=num_features, query1_dim=query1_dim,
                    query2_dim=query2_dim, num_heads1=num_heads1, num_heads2=num_heads2,
                    hidden=hidden_dim, layers=n_layers, out1=out1_dim, out2=out2_dim,
                    dr=dr, reduce=reduce_layers, gate_threshold=gate_threshold, input_mean=input_mean,
                    input_clip=input_clip)

        #save config
        self.config = {'input_dim': input_dim, 'num_features': num_features,
                       'query1_dim': query1_dim, 'query2_dim': query2_dim,
                       'num_heads1': num_heads1, 'num_heads2': num_heads2,
                       'hidden_dim': hidden_dim, 'n_layers': n_layers,
                       'out1_dim': out1_dim, 'out2_dim': out2_dim,
                       'dr': dr, 'reduce_layers': reduce_layers, 'gate_threshold': gate_threshold}

    def _build(self, input_dim, num_features, query1_dim, query2_dim, num_heads1, num_heads2,
               hidden, layers, out1, out2, dr, reduce, gate_threshold, input_mean, input_clip):

        self._gate_threshold = gate_threshold
        self._attn_label_2 = 'task_spec'

        self._frozen_1 = False
        self._frozen_2 = False

        #for task-active vs idle/rest
        self._decode_1 = Motor2aMLP1D(input_dim=input_dim, num_features=num_features, query_dim=query1_dim,
                                      num_attn_heads=num_heads1, hidden_dim=hidden, output_dim=out1,
                                      n_layers=layers, dr=dr, reduce_layers=reduce,
                                      attn_label='active_vs_idle', input_mean=input_mean, input_clip=input_clip)

        #for specialised, attends the same input against its own learnable query
        self._attention_2 = EEGAttention(input_dim=input_dim, num_features=num_features,
                                         query_config={self._attn_label_2: (query2_dim, num_heads2)}, dr=dr)

        #concat decode 2 with decode 1's latent, decode 1 is detached at the concat
        self._readout2_dim = self._attention_2.latent_dim + self._decode_1.latent_dim

        self._readout2 = MLPClassifier(input_dim=self._readout2_dim, hidden_dim=hidden, output_dim=out2,
                                       n_hidden=layers, dropout=dr, reduce_layers=reduce)


    #===decoder 1, task-active vs idle/rest===
    #x is shape B x input_dim (coeffs or electrodes) x F; train this path first
    def attend_gate(self, x, return_weights=False):
        return self._decode_1.attend(x=x, return_weights=return_weights) #B x query1_dim

    def gate_logits(self, latent_1):
        return self._decode_1.readout(latent=latent_1) #B x out1_dim; raw logits

    #===decoder 2, task-specific===
    #x is the same input shape
    def attend_task(self, x, return_weights=False):
        x = x - self._decode_1.input_mean
        latents = self._attention_2(x=x, return_weights=return_weights)
        latent = latents[self._attn_label_2] #B x query2_dim

        if return_weights:
            return latent, self.curr_attn_weights_2 #attn weights is shape B x q_seq x k_seq

        return latent, None

    #latent 1 is shape B x query1_dim; latent 2 is shape B x query2_dim
    #latent 1 is detached here, the task attention adjusts to the gate context without shaping it
    #layer norm over the concat since the two latents come from independently trained attentions
    def task_logits(self, latent_1, latent_2):
        readout_input = torch.cat([latent_2, latent_1.detach()], dim=-1) #B x (query2 + query1); task first
        readout_input = F.layer_norm(readout_input, [self._readout2_dim])
        return self._readout2(readout_input) #B x out2_dim; raw logits

    #===full paths===
    #x is shape B x input dim x F; both paths run unconditionally, gating is applied in predict
    def forward(self, x, return_weights=False):
        latent_1, attended_1 = self.attend_gate(x=x, return_weights=return_weights)
        latent_2, attended_2 = self.attend_task(x=x, return_weights=return_weights)

        logits_1 = self.gate_logits(latent_1=latent_1)
        logits_2 = self.task_logits(latent_1=latent_1, latent_2=latent_2)

        return logits_1, logits_2, attended_1, attended_2

    #returns both probability sets and the gate as a boolean,
    def predict(self, x, return_weights=False):
        logits_1, logits_2, attended_1, attended_2 = self.forward(x=x, return_weights=return_weights)

        gate_probs = torch.sigmoid(logits_1) #B x out1_dim
        task_probs = torch.sigmoid(logits_2) #B x out2_dim
        gate = gate_probs > self._gate_threshold #B x out1_dim

        return gate_probs, task_probs, gate, attended_1, attended_2

    #===staged training===
    def freeze_decode_1(self):
        self._decode_1.freeze()
        self._frozen_1 = True

    def freeze_decode_2(self):
        self._attention_2.requires_grad_(False)
        self._attention_2.eval()
        self._readout2.requires_grad_(False)
        self._readout2.eval()
        self._frozen_2 = True

    #train() propagates to every child, so re-assert eval on whatever is frozen
    def train(self, mode=True):
        super().train(mode)

        if self._frozen_1:
            self._decode_1.eval()

        if self._frozen_2:
            self._attention_2.eval()
            self._readout2.eval()

        return self

    @property
    def decode_1(self):
        return self._decode_1 #exposed so stage one loads a 1D checkpoint straight into it

    @property
    def gate_threshold(self):
        return self._gate_threshold

    @property
    def curr_attn_weights_1(self):
        return self._decode_1.curr_attn_weights #dict of label -> B x q_seq x k_seq

    @property
    def curr_attn_weights_2(self):
        return self._attention_2.curr_attn_weights #dict of label -> B x q_seq x k_seq

    @property
    def electrode_clip(self):
        return self._decode_1.electrode_clip

    def get_parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'non_trainable': total - trainable}

    def print_param_count(self):
        for p_type, count in self.get_parameter_counts().items():
            print(f"{p_type.lower()} parameters: {count}")