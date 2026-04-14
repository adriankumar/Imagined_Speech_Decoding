from architecture_components import mma
import torch 

input_dim = 80
batch = 4
q_seq = 5
k_seq = 7 
v_seq = k_seq #kv must be the same




#uses dense and multiple modules
config_v1 = {
    'use_dense': True,
    'dr': 0.2,
    'out_dim': 64,

    'attention_configs': {
        'random_1': {
            'embed_dim': input_dim, 
            'num_heads': 1,
        },

        'random_2':{
            'embed_dim': input_dim,
            'num_heads': 1,
        },

        'random_3': {
            'embed_dim': input_dim,
            'num_heads': 1
        }
    }
}

#non-dense and multiple modules
config_v2 = {
    'use_dense': False,
    'dr': 0.2,
    
    'attention_configs': {
        'random_1': {
            'embed_dim': input_dim, 
            'num_heads': 1,
        },

        'random_2':{
            'embed_dim': input_dim,
            'num_heads': 1,
        },

        'random_3': {
            'embed_dim': input_dim,
            'num_heads': 1
        }
    }
}


#single and dense
config_v3 = {
    'use_dense': True,
    'dr': 0.2,
    'out_dim': 64,
    
    'attention_configs': {
        'random_1': {
            'embed_dim': input_dim, 
            'num_heads': 1,
        }
    }
}

#single and non-dense
config_v4 = {
    'use_dense': False,
    'dr': 0.2,
    
    'attention_configs': {
        'random_1': {
            'embed_dim': input_dim, 
            'num_heads': 1,
        }
    }
}

configs = [config_v1, config_v2, config_v3, config_v4]

#multiple mod input:
multi_inputs = [
    {
    'label': 'random_1', 
    'q': torch.rand(batch, q_seq, input_dim), 
    'k': torch.rand(batch, k_seq, input_dim),
    'v': torch.rand(batch, v_seq, input_dim),
    'attn_mask': None
    },

    {
    'label': 'random_2', 
    'q': torch.rand(batch, q_seq, input_dim), 
    'k': torch.rand(batch, k_seq, input_dim),
    'v': torch.rand(batch, v_seq, input_dim),
    'attn_mask': None
    },

    {
    'label': 'random_3', 
    'q': torch.rand(batch, q_seq, input_dim), 
    'k': torch.rand(batch, k_seq, input_dim),
    'v': torch.rand(batch, v_seq, input_dim),
    'attn_mask': None
    },
    ] 

#single model input:
single_input = [
    {
    'label': 'random_1', 
    'q': torch.rand(batch, q_seq, input_dim), 
    'k': torch.rand(batch, k_seq, input_dim),
    'v': torch.rand(batch, v_seq, input_dim),
    'attn_mask': None
    },
    ]

sensory_attention = mma(config=configs[-1])

attended_output = sensory_attention(inputs=single_input)
print(f"shape of attended outputs: {attended_output.shape}")
pc = sensory_attention.get_parameter_counts()

print(f"attention heads parameter count:",
      f"trainable: {pc[0]} | non-trainable: {pc[1]} | total: {pc[2]}")
