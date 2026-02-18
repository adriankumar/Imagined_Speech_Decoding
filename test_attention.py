from model_architecture import (
    MMHA, MMHA_DEFAULT_CONFIG, MMHA_DEFAULT_CTM,
    CTMv2, QV1
    )
import torch 

# ctm = CTMv2()
q = QV1(state_dim=80, action_dim=150)
states = None 
prev = None 

# query = [torch.rand(3, 16, 150)]
# kv = [torch.rand(3, 18, 80)] #after attention on all input states


query = torch.rand(3, 150, 16)
kv = torch.rand(3, 18, 80)

for i in range(3):
    q_pred, states = q(state_t=kv, action_t=query, network_state=states, prev=prev)
    prev = q_pred 

print(f"shape of pred: {q_pred.shape}")

_ = q.print_parameter_count()

# preds, states = ctm(kv, states)

# print(f"pred shape: {preds.shape}")

# _ = ctm.print_parameter_count()

#has 3 attention mods
# att_mod = MMHA(
#     config=MMHA_DEFAULT_CONFIG
# )

# att_mod_ctm = MMHA(
#     config=MMHA_DEFAULT_CTM
# )

# #manual set up of queries, keys and values
# env_state_dim, img_s_dim, s_acc_dim = 84, 84, 150
# print(f"attention order: {att_mod.attention_types}")

# #so order of elements in qkv lists need to have expected last dim, but can have variable sequence length
# queries = [
#     torch.rand(3, 1, env_state_dim), #env state
#     torch.rand(3, 1, img_s_dim), #img state
#     torch.rand(3, 16, s_acc_dim)
# ]

# keys = [
#     torch.rand(3, 1, env_state_dim), #env state
#     torch.rand(3, 1, img_s_dim), #img state
#     torch.rand(3, 16, s_acc_dim)
# ]

# values = [
#     torch.rand(3, 1, env_state_dim), #env state
#     torch.rand(3, 1, img_s_dim), #img state
#     torch.rand(3, 16, s_acc_dim)
# ] 

# attended_output = att_mod(
#     queries=queries,
#     keys=keys,
#     values=values
# )

# print(f"output shape: {attended_output.shape}")

# param_total = att_mod.print_parameter_count()

# #testing single one
# print(f"attention order: {att_mod_ctm.attention_types}")
# action_query = [torch.rand(3, 1, 76)] #action syn size should be same as final dim 
# kv = [attended_output]

# ctm_att = att_mod_ctm(
#     queries=action_query,
#     keys=kv,
#     values=kv
# )

# print(f"output shape: {ctm_att.shape}")

# param_total = att_mod_ctm.print_parameter_count()