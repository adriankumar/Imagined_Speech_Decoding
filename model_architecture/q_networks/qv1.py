from model_architecture import build_propagator, MMHA
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_QNETWORK = {
    'attention_amount': 1,
    'dropout': 0.2,
    'final_dim': 0, 
    'use_dense': False, #use raw attended output 

    'attention_configs': [
        {
            'name': 'qnetwork',
            'embed_dim': 150, #make this same size as embedding for word
            'num_heads': 5,
            'pattern': 'cross-attention' #cross attention with action and state
        }
    ]
}

#have two ltc's, one processes the state, the other processed the processed state + action into single q value estimation
class QNetworkv1(nn.Module):
    def __init__(self, state_dim=100, action_dim=150, seed=24573471):
        super(QNetworkv1, self).__init__()
        self.state_dim = state_dim #need to project state into action dim for attention
        self.action_dim = action_dim
        self.seed = seed

        self._initialise_network()
    
    def _initialise_network(self):

        self.kv_projection = nn.LazyLinear(out_features=self.action_dim, bias=True)
        
        #attention for processing state action; query is
        self.attention_head = MMHA(
            config=DEFAULT_QNETWORK
        )

        #propagator after attention
        self.value_prop = build_propagator(
            r1=15, r2=7, r3=1, in_fanout=10, r1_fanout=6, r2_fanout=4, recurrent=12, seed=self.seed,
            input_dim=self.action_dim, input_mapping='affine', output_mapping='affine', ode_unfolds=6, epsilon=1e-6,
            project_output=True  
        )

    def forward(self, state_t, action_t, network_state=None, prev=None):
        action_t = action_t.permute(0, 2, 1) #permute shape to b x thought_steps x embed_dim

        #project state into equal action dim for attention computation
        state_t_proj = self.kv_projection(state_t) #assume already in shape b x s x state_t_dim,

        #make qkv 
        query = [action_t]
        kv = [state_t_proj] #b x s x action dim

        #pass to attention
        attended_output = self.attention_head(
            queries=query,
            keys=kv,
            values=kv
        ) #expected shape of b x thoughtsteps x embed dim

        #pass to ltc
        for i in range(attended_output.shape[1]):
            x_t = attended_output[:, i, :] #shape b x features
            delta_q, network_state = self.value_prop(x_t, network_state) #logts shape b x ltc_output_dim

        #use last logit, and perform residual new_q = old_q + delta
        if prev is None:
            q_pred = delta_q
        else: 
            q_pred = prev + delta_q
        
        return q_pred, network_state
    
    def print_parameter_count(self):
        print('-----------------------------------------------------------')
        attention_params = self.attention_head.print_parameter_count()
        kv_params = sum(p.numel() for p in self.kv_projection.parameters())
        value_params = sum(p.numel() for p in self.value_prop.parameters())

        print(f"Total parameter count for Q network: {attention_params + kv_params + value_params}")
        print(f"key-value projector parameters     : {kv_params}")
        print(f"value propagator parameters        : {value_params}")
        print('-----------------------------------------------------------')

        return attention_params + kv_params + value_params