import torch 
import torch.nn as nn 
from ..architectures import MLPClassifier
from .EEGAttentionMod import EEGAttention
from .helpers import DECISIONS

#standard MLP readout after EEGAttention on input
class Motor2aMLP(nn.Module):
    def __init__(self, input_dim, num_features, query_dim, 
                 num_heads, hidden_dim, n_layers, output_dim,
                 dr=0.2, reduce_layers=True, attn_label="coeff_attn",
                 decision="sigmoid", 

                 #clip is to clip outlier magnitudes of features over training dist
                 #input mean normalises input
                 #both need to occur for the model to work as they represent
                 #its training dist expectations to hold up during inference
                 #refers to either the sh coeffs or electrodes
                 input_clip=None, input_mean=None): 

        assert decision in list(DECISIONS.keys()), f"decision must be either 'sigmoid' or 'softmax', got {decision}"

        super().__init__()

        self._build(input_dim=input_dim, num_features=num_features, query_dim=query_dim, 
                    num_heads=num_heads, hidden=hidden_dim, out=output_dim, layers=n_layers, 
                    dr=dr, reduce=reduce_layers, attn_label=attn_label, 
                    input_clip=input_clip, input_mean=input_mean)

        self._act_fn = DECISIONS[decision] #returns the torch.nn.functional object

        #store args to rebuild when loading saved weights;
        #input clip and input mean are registered buffers in the model as they are tensors
        #so not stored in config
        self.config = {'input_dim': input_dim, 'num_features': num_features,
                       'query_dim': query_dim, 'num_heads': num_heads,
                       'hidden_dim': hidden_dim, 'n_layers': n_layers,
                       'output_dim': output_dim, 'dr': dr, 
                       'reduce_layers': reduce_layers, 'attn_label': attn_label, 
                       'decision': decision}

    def _build(self, input_dim, num_features, query_dim, num_heads, hidden, 
               out, layers, dr, reduce, attn_label, input_clip, input_mean):

        self._out = out 
        self._attn_label = attn_label 

        #eeg-attention module
        #input dim is seq axis for KV, features are embedding dim
        self._attention = EEGAttention(input_dim=input_dim, num_features=num_features,
                                       query_config={attn_label: (query_dim, num_heads)},
                                       dr=dr)

        #read out classifier; only one attention head used so
        #input size is just attn.readout_dim
        self._readout = MLPClassifier(input_dim=self._attention.readout_dim, 
                                      hidden_dim=hidden, output_dim=out, 
                                      n_hidden=layers, dropout=dr, 
                                      reduce_layers=reduce)

        #clip un-used in the model, check @property 
        mean = torch.zeros(input_dim, num_features) if input_mean is None else torch.as_tensor(input_mean, dtype=torch.float32)
        clip = torch.full((num_features,), float("inf")) if input_clip is None else torch.as_tensor(input_clip, dtype=torch.float32)

        self.register_buffer('input_mean', mean) #input_dim x F
        self.register_buffer('input_clip', clip) #F

    #eeg features is shape B x input_dim x F
    def attend(self, eeg_features):
        x = eeg_features - self.input_mean #center input to training dist mean
        latents = self._attention(eeg_features)[self._attn_label] #B x query_dim
        return latents 

    #use for training w sigmoid()
    def forward(self, eeg_features):
        latent = self.attend(eeg_features=eeg_features) #B x query_dim
        logits = self._readout(latent) #B x output_dim
        return logits 

    #use for training w softmax(),
    def predict(self, eeg_features):
        logits = self.forward(eeg_features=eeg_features) #B x output_dim
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
        return self._attention.readout_dim

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