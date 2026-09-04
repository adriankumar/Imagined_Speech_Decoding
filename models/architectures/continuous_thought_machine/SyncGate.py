import torch
import torch.nn as nn

#computes a latent synchronisation vector from random sprase neuron pairings and their post activations
#need to figure out what variables to use for visualisation
class SyncGate(nn.Module):
    def __init__(self, num_neurons, pred_sync_vector_size, action_sync_vector_size, self_pairing_count, seed):
        super(SyncGate, self).__init__()

        #which also builds neuron pairings using seed for reproducibility
        self._initialise_variables(num_neurons, pred_sync_vector_size, action_sync_vector_size, self_pairing_count, seed)
        
#----------------------------
# private methods
#----------------------------
    def _initialise_variables(self, num_neurons, pred_sync_vector_size, action_sync_vector_size, self_pairing_count, seed):
        self.num_neurons = num_neurons
        self.pred_sync_vector_size = pred_sync_vector_size
        self.action_sync_vector_size = action_sync_vector_size
        self.self_pairing_count = self_pairing_count
        self.rnd_seed = seed

        self.register_parameter('w_pred_sync', nn.Parameter(torch.zeros(self.pred_sync_vector_size))) #weights (that will be exponentiated; exponential decay) for computing the latent sync vector for prediction output path
        self.register_parameter('w_action_sync', nn.Parameter(torch.zeros(self.action_sync_vector_size)))

        self._builld_neuron_pairings() #build random sparse neuron pairings for computing synchronisation
    
    #register the indices for neuron pairs for both prediction and action sync vectors, these neurons (when indexed using these registers) will contribute to their respective sync vectors
    def _builld_neuron_pairings(self):
        #neuron pairings for prediction sync vector 
        first_indices_for_pred, second_indices_for_pred = self._generate_random_pairing_indices(self.pred_sync_vector_size) 
        self.register_buffer('pred_first_indices', first_indices_for_pred) 
        self.register_buffer('pred_second_indices', second_indices_for_pred) #register as buffer so they are saved in state dict but not trained 

        #neuron pairising for action sync vector
        first_indices_for_action, second_indices_for_action = self._generate_random_pairing_indices(self.action_sync_vector_size)
        self.register_buffer('action_first_indices', first_indices_for_action)
        self.register_buffer('action_second_indices', second_indices_for_action)
    
    #generates random sparse neuron pairing for synchronisation
    def _generate_random_pairing_indices(self, sync_vector_size):
        #error handling
        if self.self_pairing_count >= sync_vector_size:
            raise ValueError(f"self_pairing_count: {self.self_pairing_count} must be less than {sync_vector_size}")

        first_indices = torch.from_numpy(self.rnd_seed.choice(self.num_neurons, size=sync_vector_size, replace=False)) #imagine a neuron pair as x,y, then x here is the first indices and y is x's pair; note not all num_nuerons will be used, it is random sparse pairing

        #get the second indicies to have self_pairing_count neurons paired with themselves from the first indices
        second_indices = torch.zeros_like(first_indices)
        second_indices[:self.self_pairing_count] = first_indices[:self.self_pairing_count] #first self_pairing_count neurons paired with themselves

        #fill the rest of second indices with random choices
        if self.self_pairing_count < sync_vector_size:
            remaining_indices = torch.from_numpy(
                self.rnd_seed.choice(self.num_neurons, size=(sync_vector_size - self.self_pairing_count), replace=False)
            )
            second_indices[self.self_pairing_count:] = remaining_indices
        
        return first_indices, second_indices
    
    #computes post activation products of neuron pairs
    def _compute_neuron_sync(self, post_activations, sync_type):
        if sync_type == 'pred':
            first_indices = self.pred_first_indices.to(post_activations.device) #ensure indices are on same device as post activations
            second_indices = self.pred_second_indices.to(post_activations.device)
        elif sync_type == 'action':
            first_indices = self.action_first_indices.to(post_activations.device) 
            second_indices = self.action_second_indices.to(post_activations.device)
        else:
            raise ValueError(f"sync_type must be either 'pred' or 'action', got {sync_type} instead")
        
        #select neurons from post activations using the indices 
        left_neurons = post_activations[:, first_indices] #shape (batch_size, sync_vector_size)
        right_neurons = post_activations[:, second_indices]

        neuron_syncs = left_neurons * right_neurons #element-wise product, shape (batch_size, sync_vector_size)
        return neuron_syncs
    
    #exponentiates the weights for either pred or action sync vector to get exponential decay weights, then expands for batch processing
    def _exponentiate_weights(self, batch_size, sync_type, device):
        if sync_type == 'pred':
            weights = self.w_pred_sync
        elif sync_type == 'action':
            weights = self.w_action_sync
        else:
            raise ValueError(f"sync_type must be either 'pred' or 'action', got {sync_type} instead")
        
        clamped_weights = torch.clamp(weights, 0, 15) #prevent extreme values
        exp_weights = torch.exp(-clamped_weights) #exponential decay weights, shape (sync_vector_size,)

        #expand for batch processing: sync_size -> batch x sync_size
        return exp_weights.unsqueeze(0).expand(batch_size, -1).to(device)
    
#----------------------------
# public methods
#----------------------------
    #initialises the states of raw neuron synchronisation (pairwise product of post activations from first and second indices) and variable accumulators used in the exponential decay computation to reduce parameter overload in keeping post activation history
    def init_sync_states(self):
        sync_states = {
            'pred': {'neuron_sync_accumulator': None, 'beta': None},
            'action': {'neuron_sync_accumulator': None, 'beta': None}
        }

        return sync_states 
    
    #computes actual latent synchronisation vector for either pred or action
    def compute_sync_vector(self, sync_state, post_activations, sync_type, batch_size, device):
        pairwise_products = self._compute_neuron_sync(post_activations, sync_type) #get pairwise products of neuron pairs
        exp_weights = self._exponentiate_weights(batch_size, sync_type, device) #get weights
        
        state = sync_state[sync_type]
        
        #first iteration initialises accumulators
        if state['neuron_sync_accumulator'] is None:
            state['neuron_sync_accumulator'] = pairwise_products.clone() #starts as raw pairwise products
            state['beta'] = torch.ones_like(pairwise_products)
        else:
            #recurrent update with exponential temporal decay
            state['neuron_sync_accumulator'] = exp_weights * state['neuron_sync_accumulator'] + pairwise_products
            state['beta'] = exp_weights * state['beta'] + 1
        
        #compute normalised synchronisation representation
        synchronisation_vector = state['neuron_sync_accumulator'] / torch.sqrt(state['beta'])

        #shape: batch x sync_size
        return synchronisation_vector