#this defines the connectivity of neurons in the ltccell file

import numpy as np

#builds adjacency matrcies for network connection
class SynapseWiring:
    def __init__(self, internal_neurons): 
        self.total_internal_neurons = internal_neurons #internal neurons are separate from number of input dim
        self.ineuron_adjacency_matrix = np.zeros([self.total_internal_neurons, self.total_internal_neurons], dtype=np.int8)

        self.input_neuron_adjacency = None #will be initialised, stores projection from input to internal neurons 
        self.input_dim, self.final_dim = None, None #final and output dim are equivalent here

   #private method for initialising input to internal neuron size and adjacency
    def _set_input_dim(self, input_dim):
        self.input_dim = input_dim
        self.input_neuron_adjacency = np.zeros([self.input_dim, self.total_internal_neurons], dtype=np.int8) #shape: input neurons x internal neurons

    def _set_final_dim(self, final_dim):
        self.final_dim = final_dim 

    def build(self, input_dim):
        if self.input_dim is None:
            self._set_input_dim(input_dim)
        else:
            raise ValueError(f"Input dimension already set to {self.input_dim}")

    #used before adding synapse connection
    def _validate_neuron_connection(self, src, dst, polarity, reference_size, type="neurons"):
        #error handling for src input
        if src < 0 or src >= reference_size: #reference size can be internal_neurons for add_synapse or input_dim for add_sensory_synapse
            print(f"Cannot add neuron connection from neuron {src} when only {reference_size} {type} exist...")
            return False 
        
        #error handling for dst input
        if dst < 0 or dst >= self.total_internal_neurons:
            print(f"Cannot add connection to destination neuron {dst} when only {self.internal_neurons} neurons exist...")
            return False 
        
        if polarity not in [-1, 1]: #must be either -1 or 1
            print(f"Cannot add connection with polarity {polarity} (expected -1 or +1)")
            return False 
        
        return True #input args are valid
    
    #updates adjacency matrix for neurons
    def create_synapse_connection(self, src, dst, polarity, n_type="internal"):
        if n_type == "internal":
            if self._validate_neuron_connection(src, dst, polarity, self.total_internal_neurons, type=n_type):
                self.ineuron_adjacency_matrix[src, dst] = polarity

        elif n_type == "input":
            if self._validate_neuron_connection(src, dst, polarity, self.input_dim, type=n_type):
                self.input_neuron_adjacency[src, dst] = polarity

    #create a weight matrix copy from adjacency matrix
    def create_synaptic_parameters(self, adjacency_matrix, dtype=np.float32):
        return np.copy(adjacency_matrix).astype(dtype)
    
    #returns both input and internal neuron count
    @property
    def total_neuron_count(self):
        return self.total_internal_neurons + self.input_dim
    
    @property 
    def internal_neurons(self):
        return self.total_internal_neurons
    
    #return adjacency matrices as property, they will act as sparsity masks
    @property
    def NAM(self):
        return self.ineuron_adjacency_matrix 
    
    @property
    def INA(self):
        return self.input_neuron_adjacency


#Neural circuit policy for ltc cell, creates the circuit connection of neurons used by the ltccell
class NeuralCircuitPolicy(SynapseWiring):
    def __init__(self, R1_count, R2_count, R3_count,
                 input_fanout, R1_fanout, R2_fanout,
                 recurrent_connections, seed=24573471):
        
        self.rndm_sd = np.random.RandomState(seed) #random state for reproducibility
        neuron_total = R1_count + R2_count + R3_count #total internal neuron count
        super().__init__(internal_neurons=neuron_total) #initialise synapse wiring class to create adjacency matrix for internal neurons
        
        #config initialisation; building is done via function call
        self._set_final_dim(R3_count) #base class function 
        self._store_internal_neurons(R1_count, R2_count, R3_count)
        self._store_connectivity(input_fanout, R1_fanout, R2_fanout, recurrent_connections)
        self._generate_neuron_indicies()

    def build(self, input_dim):
        super().build(input_dim) #call base class build to set input dim and adjacency matrix
        self.input_neurons = [i for i in range(0, self.input_dim)] #self.input_dim from base wiring class

        self._connect_input_to_R1() #input projection layer to  region 1
        self._connect_R1_to_R2() # region 1 to region 2
        self._connect_R2_to_R3() #region 2 to region 3
        self._build_recurrent_connections()

    #connect remaining neurons with previous layer/region so one neuron may have multiple neighbours
    def _connect_remaining_neurons(self, dst_neurons, prev_neurons, fanout_neurons, total_neurons, n_type):
        mean_fanin = int(len(prev_neurons) * fanout_neurons / total_neurons)
        mean_fanin = np.clip(mean_fanin, 1, total_neurons)

        for dst in dst_neurons:
            for src in self.rndm_sd.choice(prev_neurons, size=mean_fanin, replace=False):
                polarity = self.rndm_sd.choice([-1, 1])
                self.create_synapse_connection(src, dst, polarity, n_type=n_type)
    
    def _connect_input_to_R1(self):
        dst_neurons = [neuron for neuron in self.R1_neurons]

        for src in self.input_neurons:
            for dst in self.rndm_sd.choice(self.R1_neurons, size=self.input_fanout, replace=False):
                if dst in dst_neurons:
                    dst_neurons.remove(dst)
                
                polarity = self.rndm_sd.choice([-1, 1])
                self.create_synapse_connection(src, dst, polarity, n_type="input") #input arg to specify input to internal neuron adjacency

        self._connect_remaining_neurons(dst_neurons, self.input_neurons, self.input_fanout, self.R1_neuron_count, n_type="input")

    def _connect_R1_to_R2(self):
        dst_neurons = [neuron for neuron in self.R2_neurons] #carries a copy to act as marked neighbours

        for src in self.R1_neurons:
            #using replace=False to only get unique choices of R2 indices for connection
            for dst in self.rndm_sd.choice(self.R2_neurons, size=self.R1_fanout, replace=False):
                if dst in dst_neurons:
                    dst_neurons.remove(dst) #remove connection from full list

                polarity = self.rndm_sd.choice([-1, 1]) #inhib/excitatory
                self.create_synapse_connection(src, dst, polarity, n_type="internal")

        self._connect_remaining_neurons(dst_neurons, self.R1_neurons, self.R1_fanout, self.R2_neuron_count, n_type="internal")


    def _connect_R2_to_R3(self):
        src_neurons = [neuron for neuron in self.R2_neurons] #using SR2 neurons as src

        for dst in self.R3_neurons: #assuming SR3 has less than SR2 neurons
            for src in self.rndm_sd.choice(self.R2_neurons, size=self.R2_fanout, replace=False):
                if src in src_neurons:
                    src_neurons.remove(src)
                
                polarity = self.rndm_sd.choice([-1, 1])
                self.create_synapse_connection(src, dst, polarity, n_type="internal")
        
        #connect remaining, note src and dst args are reversed here
        self._connect_remaining_neurons(src_neurons, self.R3_neurons, self.R2_fanout, self.R2_neuron_count, n_type="internal")

#i think original paper only did recurrent connections for r2 (which i think was called command layer in their version)
    def _build_recurrent_connections(self):
        all_neurons = [neuron for neuron in range(0, self.internal_neurons)] #indices from all internal neurons

        for _ in range(self.recurrent_connections):
            src = self.rndm_sd.choice(all_neurons)
            dst = self.rndm_sd.choice(all_neurons)
            polarity = self.rndm_sd.choice([-1, 1])
            self.create_synapse_connection(src, dst, polarity, n_type="internal")

    #storing config functions---
    #R = neuron Region/ layer, internal propagation layers of ltc    
    def _store_internal_neurons(self, R1_count, R2_count, R3_count=2):
        self.R1_neuron_count = R1_count
        self.R2_neuron_count = R2_count
        self.R3_neuron_count = R3_count

    def _store_connectivity(self, input_to_R1_count, R1_to_R2_count, R2_to_R3_count, recurrent_count):
        self.input_fanout = input_to_R1_count
        self.R1_fanout = R1_to_R2_count
        self.R2_fanout = R2_to_R3_count
        self.recurrent_connections = recurrent_count 

    def _generate_neuron_indicies(self):
        self.R3_neurons = [i for i in range(0, self.R3_neuron_count)]
        self.R2_neurons = [i for i in range(self.R3_neuron_count, (self.R3_neuron_count + self.R2_neuron_count))]
        self.R1_neurons = [i for i in range((self.R3_neuron_count + self.R2_neuron_count), self.internal_neurons)]

    @property 
    def input_size(self):
        return self.input_dim
    
    @property
    def output_dim(self):
        return self.R3_neuron_count 
    #TO DO
    def get_config(self):
        return {}