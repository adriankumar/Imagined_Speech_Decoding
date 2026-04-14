import torch 
import torch.nn as nn 
import torch.nn.functional as F

#example---
MMA_DEFAULT = {
    'use_dense': True,
    'dr': 0.2, #drop out, making global for simplicity
    'out_dim': 42,

    'attention_configs': {
        'env_state': {
            'embed_dim': 84,
            'num_heads': 4
        },

        'imagined_state': {
            'embed_dim': 84,
            'num_heads': 4
        },

        'semantic_acc': {
            'embed_dim': 64,
            'num_heads': 4,
            'kdim': 56,
            'vdim': 54
        }

    }
}

#a class that creates n number of attention modules for the 'sensory attention' layer
#i.e n=3 then 3 individual attention modules for expecting 3 different QKV inputs, and returns
#either dense projection output concatennated or attened outputs concatenated; 
#concat output is layer normed
#output of att will always be batch, query_seq, embed_dim
class mma(nn.Module):
    def __init__(self, config=None, **kwargs):
        super().__init__()

        self._build_config(config, **kwargs)
        self._build_attention_heads()

    
    def _build_config(self, config=None, **kwargs):
        self.config = MMA_DEFAULT.copy() #start with default, and replace with any changes

        if config is not None:
            assert isinstance(config, dict), "config must be a dict"
            self.config = config
        
        self.config.update(kwargs)

        assert len(self.config['attention_configs']) > 0, "need at least one attention config"

        self.use_dense = self.config['use_dense']
        self.out_dim = self.config['out_dim'] if self.use_dense else None

        if self.use_dense:
            assert self.out_dim is not None, "missing key: 'out_dim if using dense"

        self.dr = self.config['dr']
        self.registered_labels = list(self.config['attention_configs'].keys())
        self.attention_amnt = len(self.registered_labels)

    def _build_attention_heads(self):
        self.attention_heads = nn.ModuleDict()

        if self.use_dense:
            self.dense_projections = nn.ModuleDict()
        
        for label, head_config in self.config['attention_configs'].items():
            embed_dim = head_config['embed_dim'] #size of query feature dim
            num_heads = head_config['num_heads'] #split input and heads into smaller ones 
            
            #ensure embed and num heads are divisible for even splitting
            if embed_dim % num_heads != 0:
                raise ValueError(f"[{label}] embed_dim: {embed_dim} must be divisible by num_heads: {num_heads}")
            
            #make attentionm od and register under label
            self.attention_heads[label] = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=head_config.get('dr', self.dr), #custom, otherwise global
                batch_first=True,
                kdim=head_config.get('kdim', None), #optional
                vdim=head_config.get('vdim', None) #optional 
            )

            #dense projection
            if self.use_dense:
                self.dense_projections[label] = nn.LazyLinear(
                    out_features=self.out_dim
                )
    
    def _validate_labels(self, inputs):
        incoming_labels = set()

        for dict in inputs:
            assert 'label' in dict, f"each input dict must have a 'label' key; got keys {list(dict.keys())} instead"
            
            label = dict['label']
            #check if label exist in what was registered
            assert label in self.registered_labels, (
                f"unknown attention label '{label}', registered labels are: {self.registered_labels}"
            )

            assert label not in incoming_labels, f"duplicate label '{label}' in inputs"
            incoming_labels.add(label)
        
        return incoming_labels

    #assume expected input is a list of dict:
    #[{label:<>, q:<>, k:<>, v:<>, attn_mask:<>}, {...}, ...]
    def forward(self, inputs, return_weights=False):
        assert isinstance(inputs, list), "inputs must be a list of dicts"

        #validate labels
        labels = self._validate_labels(inputs) #set of attention labels for dict

        #ensure all registered heads have an incoming input
        missing = set(self.registered_labels) - labels 
        assert len(missing) == 0, f"missing inputs for attention heads: {missing}"

        att_outputs = []
        att_weights = [] 

        #run attention for each label registered
        for label in self.registered_labels:
            input_stuff = next(i for i in inputs if i['label'] == label) #any order

            #shape is b x query_s x embed_dim
            attended_features, attended_weights = self.attention_heads[label](
                query=input_stuff['q'], #b x s x embed_dim
                key=input_stuff['k'], #b x s x embed or kdim
                value=input_stuff['v'], #b x s x embed or vdim
                need_weights=return_weights,
                attn_mask=input_stuff.get('attn_mask', None)
            )

            if self.use_dense:
                projected = self.dense_projections[label](attended_features)
                att_outputs.append(F.silu(projected)) #shape b x output_dim
            else:
                att_outputs.append(attended_features) #b x s_query x embed_dim
            
            if return_weights:
                att_weights.append(attended_weights) 
        
        #concate individual attention outputs across sequence dim
        attended_outputs = torch.cat(att_outputs, dim=1) #batch x total_seq x dim

        return F.layer_norm(attended_outputs, [attended_outputs.shape[-1]])

    @property 
    def attention_types(self):
        return ' '.join(self.registered_labels)

    #returns count in [trainable, non-trainable, total]
    def get_parameter_counts(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        return [trainable_params, non_trainable_params, total_params]
            

