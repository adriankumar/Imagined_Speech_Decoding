from model_architecture import build_propagator
import torch
import torch.nn as nn
import torch.nn.functional as F


#Q networks evaluate the cognitive output (latent) and action
#so build 2 ltc, -> concat both final ltc states and project them to a single value
class QNetwork(nn.Module):
    def __init__(self, cognitive_dim=20, embedding_dim=64, seed=24573471):
        super(QNetwork, self).__init__()
        #note; the actual shape is in dim x thought_steps, but we leverage LTC for that recurrent processing over timesteps
        #so process each thoughtstep in time, to evaluate the full state and action input, and let
        #the ltc's hidden states continue basically twice over time, like a very losely inspired
        #version of bayes theorem, where ltc's hidden state carries temporal information 
        #of the sub_signals and each individual word, then also maintains its state for each full window
        #so it carries information about the 'belief' of one window over sub-timesteps (thought-steps) to
        #subsequent windows
        self.state_dim = cognitive_dim
        self.action_dim = embedding_dim
        self.seed = seed

        self._initialise_network()
    
    def _initialise_network(self):
        
        #processes cognitive signals (state_t for policy)
        self.latent_prop = build_propagator(
            r1=12, r2=6, r3=2, in_fanout=6, r1_fanout=4, r2_fanout=2, recurrent=9, input_dim=self.state_dim,
            input_mapping='affine', output_mapping='affine', ode_unfolds=6, epsilon=1e-6, seed=self.seed, 
            project_output=True
        )

        self.action_prop = build_propagator(
            r1=12, r2=6, r3=2, in_fanout=6, r1_fanout=4, r2_fanout=2, recurrent=9, input_dim=self.action_dim,
            input_mapping='affine', output_mapping='affine', ode_unfolds=6, epsilon=1e-6, seed=self.seed, 
            project_output=True
        )

        input_dim = self.latent_prop.wire.output_dim * self.action_prop.wire.output_dim
        # print(f"input dim for value projection: {input_dim}")

        self.value_projection = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=input_dim, out_features=1, bias=True),
            nn.SiLU()
        )
    
    def propagate(self, model, x, state=None):
        #assume shape of either input is b x dim x thought_steps 
        length = x.shape[-1]

        for i in range(length):
            x_t = x[:, :, i]
            logits, state = model(x_t, state) #logts shape b x ltc_output_dim
        
        return logits, state
    
    def forward(self, latent_t, action_t, l_state=None, a_state=None):
        log_l, l_state = self.propagate(self.latent_prop, latent_t, l_state)
        log_a, a_state = self.propagate(self.action_prop, action_t, a_state)
        # print(f"latent logit shape: {log_l.shape} | action: {log_a.shape}")

        #concat across feature dim, apply activation + layernorm before projection
        logits = torch.concat([log_l, log_a], dim=-1) #b x (ltc output_dim * 2))
        # print(f"logits after concat shape: {logits.shape}")
        logits = F.silu(logits)
        logits = F.layer_norm(logits, [logits.shape[-1]]) #norm across feature dim

        q_pred = self.value_projection(logits) #b x 1
        return q_pred, l_state, a_state #return untouched states for temporal processing across windows
    
    def print_parameter_count(self):
        latent_prop_params = sum(p.numel() for p in self.latent_prop.parameters())
        action_prop_params = sum(p.numel() for p in self.action_prop.parameters())
        value_projector_params = sum(p.numel() for p in self.value_projection.parameters())
        total = latent_prop_params + action_prop_params + value_projector_params

        print(f"Total parameter count for Q network: {total}")
        print(f"latent propagator parameters       : {latent_prop_params}")
        print(f"action propagator parameters       : {action_prop_params}")
        print(f"value projection parameters        : {value_projector_params}")

        return total


class QNetworkv2(nn.Module):
    def __init__(self, state_dim=100, embedding_dim=64, seed=24573471):
        super(QNetworkv2, self).__init__()
        #note; the actual shape is in dim x thought_steps, but we leverage LTC for that recurrent processing over timesteps
        #so process each thoughtstep in time, to evaluate the full state and action input, and let
        #the ltc's hidden states continue basically twice over time, like a very losely inspired
        #version of bayes theorem, where ltc's hidden state carries temporal information 
        #of the sub_signals and each individual word, then also maintains its state for each full window
        #so it carries information about the 'belief' of one window over sub-timesteps (thought-steps) to
        #subsequent windows
        self.state_dim = state_dim
        self.action_dim = embedding_dim
        self.seed = seed

        self._initialise_network()
    
    def _initialise_network(self):
        
        #processes cognitive signals (state_t for policy)
        self.state_prop = build_propagator(
            r1=24, r2=8, r3=6, in_fanout=16, r1_fanout=7, r2_fanout=4, recurrent=12, input_dim=self.state_dim,
            input_mapping='affine', output_mapping='affine', ode_unfolds=6, epsilon=1e-6, seed=self.seed, 
            project_output=True
        )

        self.action_prop = build_propagator(
            r1=12, r2=6, r3=2, in_fanout=6, r1_fanout=4, r2_fanout=2, recurrent=9, input_dim=self.action_dim,
            input_mapping='affine', output_mapping='affine', ode_unfolds=6, epsilon=1e-6, seed=self.seed, 
            project_output=True
        )

        input_dim = self.state_prop.wire.output_dim + self.action_prop.wire.output_dim
        # print(f"input dim for value projection: {input_dim}")

        self.value_projection = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=input_dim, out_features=1, bias=True),
            nn.SiLU()
        )
    
    def propagate(self, model, x, state=None):
        #assume shape of either input is b x dim x thought_steps 
        length = x.shape[-1]

        for i in range(length):
            x_t = x[:, :, i]
            logits, state = model(x_t, state) #logts shape b x ltc_output_dim
        
        return logits, state
    
    def forward(self, state_t, action_t, l_state=None, a_state=None):
        log_l, l_state = self.propagate(self.state_prop, state_t, l_state)
        log_a, a_state = self.propagate(self.action_prop, action_t, a_state)
        # print(f"latent logit shape: {log_l.shape} | action: {log_a.shape}")

        #concat across feature dim, apply activation + layernorm before projection
        logits = torch.concat([log_l, log_a], dim=-1) #b x (ltc output_dim * 2))
        # print(f"logits after concat shape: {logits.shape}")
        logits = F.silu(logits)
        logits = F.layer_norm(logits, [logits.shape[-1]]) #norm across feature dim

        q_pred = self.value_projection(logits) #b x 1
        return q_pred, l_state, a_state #return untouched states for temporal processing across windows
    
    def print_parameter_count(self):
        state_prop_params = sum(p.numel() for p in self.state_prop.parameters())
        action_prop_params = sum(p.numel() for p in self.action_prop.parameters())
        value_projector_params = sum(p.numel() for p in self.value_projection.parameters())
        total = state_prop_params + action_prop_params + value_projector_params

        print(f"Total parameter count for Q network: {total}")
        print(f"state propagator parameters       : {state_prop_params}")
        print(f"action propagator parameters       : {action_prop_params}")
        print(f"value projection parameters        : {value_projector_params}")

        return total
    

class QNetworkv2_1(nn.Module):
    def __init__(self, state_dim=100, embedding_dim=64, seed=24573471):
        super(QNetworkv2_1, self).__init__()
        #note; the actual shape is in dim x thought_steps, but we leverage LTC for that recurrent processing over timesteps
        #so process each thoughtstep in time, to evaluate the full state and action input, and let
        #the ltc's hidden states continue basically twice over time, like a very losely inspired
        #version of bayes theorem, where ltc's hidden state carries temporal information 
        #of the sub_signals and each individual word, then also maintains its state for each full window
        #so it carries information about the 'belief' of one window over sub-timesteps (thought-steps) to
        #subsequent windows
        self.state_dim = state_dim
        self.action_dim = embedding_dim
        self.seed = seed

        self._initialise_network()
    
    def _initialise_network(self):
        
        #processes cognitive signals (state_t for policy)
        self.state_prop = build_propagator(
            r1=24, r2=8, r3=6, in_fanout=16, r1_fanout=7, r2_fanout=4, recurrent=12, input_dim=self.state_dim,
            input_mapping='affine', output_mapping='affine', ode_unfolds=6, epsilon=1e-6, seed=self.seed, 
            project_output=True
        )

        self.action_prop = build_propagator(
            r1=12, r2=6, r3=2, in_fanout=6, r1_fanout=4, r2_fanout=2, recurrent=9, input_dim=self.action_dim,
            input_mapping='affine', output_mapping='affine', ode_unfolds=6, epsilon=1e-6, seed=self.seed, 
            project_output=True
        )

        input_dim = self.state_prop.wire.output_dim + self.action_prop.wire.output_dim
        # print(f"input dim for value projection: {input_dim}")

        self.value_projection = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=input_dim, out_features=1, bias=True),
            nn.SiLU()
        )
    
    def propagate(self, model, x, state=None):
        #assume shape of either input is b x dim x thought_steps 
        length = x.shape[-1]

        for i in range(length):
            x_t = x[:, :, i]
            logits, state = model(x_t, state) #logts shape b x ltc_output_dim
        
        return logits, state
    
    def forward(self, state_t, action_t, l_state=None, a_state=None, prev=None):
        # batch_size = state_t.shape[0]

        log_l, l_state = self.propagate(self.state_prop, state_t, l_state)
        log_a, a_state = self.propagate(self.action_prop, action_t, a_state)
        # print(f"latent logit shape: {log_l.shape} | action: {log_a.shape}")

        #concat across feature dim, apply activation + layernorm before projection
        logits = torch.concat([log_l, log_a], dim=-1) #b x (ltc output_dim * 2))
        # print(f"logits after concat shape: {logits.shape}")
        logits = F.silu(logits)
        logits = F.layer_norm(logits, [logits.shape[-1]]) #norm across feature dim

        delta_q = self.value_projection(logits) #b x 1

        if prev is None:
            # prev = torch.zeros(batch_size, 1, device=state_t.device) #shape b x 1, if prev is none then init as 0
            q_pred = delta_q
        else:
            #residual: new_q = old q + delta_q
            q_pred = prev + delta_q

        return q_pred, l_state, a_state #return untouched states for temporal processing across windows
    
    def print_parameter_count(self):
        state_prop_params = sum(p.numel() for p in self.state_prop.parameters())
        action_prop_params = sum(p.numel() for p in self.action_prop.parameters())
        value_projector_params = sum(p.numel() for p in self.value_projection.parameters())
        total = state_prop_params + action_prop_params + value_projector_params

        print(f"Total parameter count for Q network: {total}")
        print(f"state propagator parameters       : {state_prop_params}")
        print(f"action propagator parameters       : {action_prop_params}")
        print(f"value projection parameters        : {value_projector_params}")

        return total