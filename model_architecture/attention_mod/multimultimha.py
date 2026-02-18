from torch.nested import nested_tensor
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

#self-attention: use the same input as QKV
#cross attention: use different Q and KV inputs
#encoder-decoder: idk yet
ATTENTION_PATTERNS = ['self-attention', 'cross-attention', 'encoder-decoder']

MMHA_DEFAULT_CONFIG = {
    #number of multi head attention layers to stack, based on the input
    #number, add an assertion to ensure they're are n number of args to properly 
    #build everything
    'attention_amount': 3,
    'dropout': 0.2,
    'final_dim': 76, #each attention output is projected into a final dim // attention amount size dim so concat gives this final dim
    'use_dense': True,

    'attention_configs': [
        {
            'name': 'env_state',
            'embed_dim': 84, #the last dim of expected query and key value input to mha
            'num_heads': 4, #although handled in class, ensure its divisble by embed dim; basically splits embedding into even smaller sizes to use with mutliple attention heads, like a 'partial' attention
            'pattern': ATTENTION_PATTERNS[1] #cross attention with the previous state
        },

        {
            'name': 'imagined_state',
            'embed_dim': 84,
            'num_heads': 4,
            'pattern': ATTENTION_PATTERNS[1] #cross attention with env state (which can be the predicted next state, basically making this self attention in imagined scenarios)
        },

        {
            'name': 'semantic_acc', #semantic accumulation attention 
            'embed_dim':150,
            'num_heads':5,
            'pattern': ATTENTION_PATTERNS[0]
        },

        #skipping sentence reconstruction attention module
    ]

}


MMHA_DEFAULT_CTM = {
    'attention_amount': 1,
    'dropout': 0.2,
    'final_dim': 0, #use attention dim arg in ctm
    'use_dense': False, #use raw attended output for ctm 

    'attention_configs': [
        {
            'name': 'ctm_attention',
            'embed_dim': 76, #make this same size as action sync size
            'num_heads': 4,
            'pattern': ATTENTION_PATTERNS[1] #cross attention with action sync vector as query, so make action sync size same as final_dim in the multi attention dict above
        }
    ]
}

#multi-multi head attention module
#assume inputs are handled externally, as in the dimensionality and what makes query, key and value
#use torch.compile after lazylinears have been initialised 
class MMHA(nn.Module):
    def __init__(self, config=None, **kwargs): 
        super().__init__()
    
        self._build_config(config, **kwargs)
        self.attention_amount = self.config['attention_amount']
        self.out_dim = self.config['final_dim']

        self.attention_order = [config['name'] for config in self.config['attention_configs']]
        self.dr = self.config['dropout']

        self._build_attention_heads()
    
    def _build_config(self, config=None, **kwargs):
        self.config = MMHA_DEFAULT_CONFIG.copy() 

        if config is not None:
            self.config.update(config)
        
        self.config.update(kwargs)
    
    def _build_attention_heads(self):
        self.attention_heads = nn.ModuleList()
        self.dense_projections = nn.ModuleList()
        configs = self.config['attention_configs']

        #esnure number of configs match number of attention heads
        if len(configs) != self.attention_amount:
            raise ValueError(f"there are {len(configs)} attention configs but {self.attention_amount} mha heads were specified")

        for i in range(self.attention_amount):
            config = configs[i]
            #get args
            embed_dim = config['embed_dim']
            num_heads = config['num_heads']

            #esnure embed and num heads are divisble for even splitting
            if embed_dim % num_heads != 0:
                raise ValueError(f"attention_size: {embed_dim} must be divisible by num_heads: {num_heads}")

            #make attention mod and append to list
            attention_head = nn.MultiheadAttention(
                embed_dim=embed_dim, #qkv expected to have same last dim, but can have variable sequence dim
                num_heads=num_heads,
                dropout=self.dr,
                batch_first=True #batch dim expected to come first
            )

            self.attention_heads.append(attention_head)

        if self.config['use_dense']:
            for i in range(self.attention_amount):
                projector = nn.LazyLinear( #lazy linear will keep sequence dim
                    out_features=self.out_dim
                )

                self.dense_projections.append(projector)

    #assume query, key and value args are passed as a list where each element input data 
    #matches the order of the initialised attention order    
    #expected input shape batch x sequence length x feature dim, where sequence length can vary
    #but feature dim must be the same
    #if input queries do not have a sequence dim (sequence dim <= 1 ) then has_seq should be false
    def forward(self, queries, keys, values):
        att_outputs = []
        
        for i in range(self.attention_amount):
            attended_feature, _ = self.attention_heads[i](
                query=queries[i],
                key=keys[i], 
                value=values[i],
                need_weights=False  # skip weights for speed unless debugging
            )
            
            #use dense if initialised
            if self.config['use_dense']:
                projected = self.dense_projections[i](attended_feature)
                att_outputs.append(F.silu(projected))
            else:
                att_outputs.append(attended_feature)
        
        #concat across sequence dim
        attended_output = torch.cat(att_outputs, dim=1)  # (b, total_seq, dim)
        
        #auto-detect: only squeeze if final seq_dim is 1, if it is then there was only one attention head and is a vector output
        if attended_output.shape[1] == 1:
            attended_output = attended_output.squeeze(1)  # (b, dim)
        else:
            attended_output = F.layer_norm(attended_output, [attended_output.shape[-1]])
        
        return attended_output

    @property 
    def attention_types(self):
        return ' '.join(self.attention_order)

    def print_parameter_count(self):
        print('-----------------------------------------------------------')
        attention_params = sum(p.numel() for p in self.attention_heads.parameters())
        if self.config['use_dense']:
            dense_params = sum(p.numel() for p in self.dense_projections.parameters())
        else:
            dense_params = 0        

        print(f"Total Parameter count for Attention Mod: {attention_params + dense_params}")
        print(f"attention parameters                   : {attention_params}")
        print(f"dense projections parameters           : {dense_params}")
        print(f"Number of attention modules            : {self.attention_amount}")
        print('-----------------------------------------------------------')


        return attention_params + dense_params


        

