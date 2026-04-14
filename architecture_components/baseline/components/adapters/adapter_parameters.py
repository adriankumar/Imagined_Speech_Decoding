import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F


allowed_activations = ['leaky-relu', 'relu', 'silu']

#this module is used for projection layers; inspo from original LoRA paper:
# Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). 
# LoRA: Low-rank adaptation of large language models. In Proceedings of the International Conference 
# on Learning Representations (ICLR). https://openreview.net/forum?id=nZeVKeeFYf9 

#because we are abstracting our learning process to the parameter scale
#if the intent of each training stage/objective is for a model to build a 
#distribution shaped by that objective, then the adapters should also build
#internal distributions to help modulate the prior knowledge instead of 
#relying on fixed transformations; that's what this stochastic gate does, it 
#adds a quantile regression module to build quantiles per output dim
class StochasticGate(nn.Module):
    def __init__(self, input_dim, output_dim=1, num_quantiles=5): #n quantiles per output dim
        super().__init__()
        self.output_dim = output_dim
        self.num_quantiles = num_quantiles
        
        #project modulation to compact summary
        self.summary_proj = nn.Linear(input_dim, output_dim)
        
        #predict quantile values from summary
        self.quantile_head = nn.Linear(output_dim, output_dim * num_quantiles)
        
        #initialise so all quantiles start at ~1.0 (no effect initially)
        nn.init.zeros_(self.quantile_head.weight)
        nn.init.ones_(self.quantile_head.bias)
        
        #fixed tau positions
        taus = torch.linspace(0.05, 0.95, num_quantiles)
        self.register_buffer('taus', taus)
    
    def forward(self, modulation):
        #perceptron: summarise the modulation
        summary = self.summary_proj(modulation)  # batch, output_dim
        
        #probability: predict quantile values
        q = self.quantile_head(summary)  # batch, output_dim * num_quantiles
        q = F.silu(q) #activate
        q = q.view(*q.shape[:-1], self.output_dim, self.num_quantiles)
        q = torch.sort(q, dim=-1).values  #enforce monotonicity
        
        #output: sample or commit
        if self.training:
            #uniform sample, interpolate between nearest quantiles
            u = torch.rand(*q.shape[:-1], 1, device=q.device)
            #find interpolation position across quantile bins
            bin_width = 1.0 / (self.num_quantiles - 1)
            bin_idx = (u / bin_width).clamp(0, self.num_quantiles - 2)
            lower = bin_idx.long()
            frac = bin_idx - lower.float()
            
            q_low = q.gather(-1, lower).squeeze(-1)
            q_high = q.gather(-1, (lower + 1).clamp(max=self.num_quantiles - 1)).squeeze(-1)
            scale = q_low + frac.squeeze(-1) * (q_high - q_low)
        else:
            #median quantile
            # scale = q[:, :, self.num_quantiles // 2]
            scale = q[..., self.num_quantiles // 2]
        
        return scale, q  #scale for use, q for monitoring

#LoRA, with non-linear bottleneck modification; and scaling correction
#potentially use this for feature extractors and prediction output modules
#like the MoE paper of LoRA: https://arxiv.org/pdf/2309.05444 
class LoRAAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, 
                 rank=1, alpha=1.0, dropout=0.2,
                 nonlinear=True, activation=allowed_activations[-1], 
                 stochastic={'is_stoch': True, 'quantile_dims': 1, 'quantiles_per_dim': 5}
                 ):

        super(LoRAAdapter, self).__init__()

        self.input_dim = input_dim 
        self.output_dim = output_dim
        self.rank = rank 
        self.scaling = alpha / np.sqrt(self.rank) #from https://finlora-docs.readthedocs.io/en/latest/lora_methods/rslora.html
        self.is_nonlinear = nonlinear 
        self.dr = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.activation = self._init_activation(activation) if self.is_nonlinear else None

        #A^T @ B^T --> i x r @ r x o 
        self.A = nn.Parameter(torch.empty(self.rank, self.input_dim)) #r x i
        self.B = nn.Parameter(torch.zeros(self.output_dim, self.rank)) #o x r
        
        #initialise; copying official LoRA github computations: https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))

        self.is_stochastic = stochastic['is_stoch']
        self._last_quantiles = None

        if self.is_stochastic:
            self.gate = StochasticGate(input_dim=self.output_dim, 
                                       output_dim=stochastic['quantile_dims'], #num of output dim
                                       num_quantiles=stochastic['quantiles_per_dim'] #quantiles per output dim
                                       )

    def _init_activation(self, activation):
        if activation == 'silu':
            return nn.SiLU()
        elif activation == 'leaky-relu':
            return nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'relu':
            return nn.ReLU()

    def forward(self, x):
        out = self.dr(x) @ self.A.T #batch, r; project down 

        if self.is_nonlinear:
            out = self.activation(out) #nonlinear mixing inside bottle neck 
        
        out = out @ self.B.T #project back up r, output_dim

        modulation = out * self.scaling

        if self.is_stochastic:
            scale, self._last_quantiles = self.gate(modulation)
            modulation = modulation * scale 
        
        return modulation #use externally as forward_pass(x) + LoRA(x)

    def freeze_adapter(self):
        for p in self.parameters():
            p.requires_grad = False 
    
    def unfreeze_adapter(self):
        for p in self.parameters():
            p.requires_grad = True
    
    def return_last_quantiles(self):
        return self._last_quantiles if self._last_quantiles is not None else None


class DoRAAdapter(nn.Module):
    def __init__(self, frozen_weight, bias=None,
                 rank=1, alpha=1.0, dropout=0.2,
                 stochastic={'is_stoch': True, 'quantile_dims': 1, 'quantiles_per_dim': 5}
                 ):
        
        super(DoRAAdapter, self).__init__()

        self.rank = rank
        self.output_dim, self.input_dim = frozen_weight.shape
        self.scaling = alpha / np.sqrt(self.rank)
        self.dr = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        #register frozen weights as buffer
        self.register_buffer('frozen_weight', frozen_weight.detach().clone())
        
        if bias is not None:
            self.register_buffer('frozen_bias', bias.detach().clone())
        else:
            self.frozen_bias = None

        #LoRA parameters for directional updates
        self.A = nn.Parameter(torch.empty(rank, self.input_dim))
        self.B = nn.Parameter(torch.zeros(self.output_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))

        #DoRA magnitude, one learnable scalar per output dimension
        #Initialised to match W0's row norms so output starts identical to original
        self.m = nn.Parameter(torch.linalg.norm(frozen_weight, dim=1).clone())

        self.is_stochastic = stochastic['is_stoch']
        self._last_quantiles = None

        if self.is_stochastic:
            self.gate = StochasticGate(input_dim=self.output_dim, 
                                       output_dim=stochastic['quantile_dims'], #num of output dim
                                       num_quantiles=stochastic['quantiles_per_dim'] #quantiles per output dim
                                       )

    def forward(self, x):
        #Step 1: form the adapted weight (conceptually, not stored)
        # W'= W0 + B·A·scaling
        adapted_weight = self.frozen_weight + (self.B @ self.A) * self.scaling

        #Step 2: get row norms of W' for normalisation
        #detach keeps direction and mangitude optimised independently
        row_norms = torch.linalg.norm(adapted_weight, dim=1).detach()

        #Step 3: compute scale factor = m / ‖W'_row‖
        norm_scale = self.m / (row_norms + 1e-8)

        base_out = x @ self.frozen_weight.T
        lora_out = self.dr(x) @ self.A.T @ self.B.T * self.scaling

        adapted_out = norm_scale * (base_out + lora_out)

        if self.is_stochastic:
            delta = adapted_out - base_out
            scale, self._last_quantiles = self.gate(delta)
            out = base_out + scale * delta
        else:
            out = adapted_out

        if self.frozen_bias is not None:
            out = out + self.frozen_bias

        return out #use externally as output to next layer, frozen weights compute direction normalisation so its absorbed

    def freeze_adapter(self):
        for p in self.parameters():
            p.requires_grad = False 
    
    def unfreeze_adapter(self):
        for p in self.parameters():
            p.requires_grad = True

    def return_last_quantiles(self):
        return self._last_quantiles if self._last_quantiles is not None else None


#TO DO: adapters for ODE based networks-- use coefficients instead of weight matrices