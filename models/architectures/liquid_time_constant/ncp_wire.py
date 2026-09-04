import numpy as np
from global_lvl import SEED

#this module implemented from the official ltc github:https://github.com/mlech26l/ncps/blob/master/ncps/wirings/wirings.py
#note this is the NCP specific implementation from the github; the github offers fully connected or sparse-random
#not 100% accurate re-creation

#builds adjacendy matrices for the ltc connections/connectivity graph
#for simplicity we replace the sensory, inter, command and motor layers as 'layer_x'
class SynapseWiring:
    def __init__(self, internal_neurons): 
        self.total_internal_neurons = internal_neurons #exlcudes input neurons (aka sensory)

        #connectivity graph for internal neurons 
        self.i_neuron_graph = np.zeros([self.total_internal_neurons, self.total_internal_neurons], dtype=np.int8)

        self.input_neuron_graph = None #will be initialised; is input to layer_1 (sensory -> inter)
        self.input_dim, self.out_dim = None, None #will be initialised

    #private method for initialising input to internal neuron size and adjacency
    def _set_input_dim(self, input_dim):
        self.input_dim = input_dim
        self.input_neuron_graph = np.zeros([self.input_dim, self.total_internal_neurons], dtype=np.int8) #shape: input neurons x internal neurons

    def _set_out_dim(self, out_dim):
        self.out_dim = out_dim 

    def build(self, input_dim):
        if self.input_dim is None:
            self._set_input_dim(input_dim)
        else:
            raise ValueError(f"Input dimension already set to {self.input_dim}")

    #used before adding connection to the graphs
    def _validate_connectivity(self, src, dst, polarity, reference_size, type="neurons"):
        #error handling for src input
        if src < 0 or src >= reference_size: #reference size can be internal_neurons for add_synapse or input_dim for add_sensory_synapse
            print(f"Cannot add neuron connection from neuron {src} when only {reference_size} {type} exist...")
            return False 
        
        #error handling for dst input
        if dst < 0 or dst >= self.total_internal_neurons:
            print(f"Cannot add connection to destination neuron {dst} when only {self.total_internal_neurons} neurons exist...")
            return False 
        
        if polarity not in [-1, 1]: #must be either -1 or 1
            print(f"Cannot add connection with polarity {polarity} (expected -1 or +1)")
            return False 
        
        return True #input args are valid
    
    def add_connection(self, src, dst, polarity, n_type="internal"):
        if n_type == "internal":
            if self._validate_connectivity(src, dst, polarity, self.total_internal_neurons, type=n_type):
                self.i_neuron_graph[src, dst] = polarity #connection between graph nodes (neurons) identified with polarity of [-1, 0, 1], where 0 is no connection
        
        elif n_type == "input":
            if self._validate_connectivity(src, dst, polarity, self.input_dim, type=n_type):
                self.input_neuron_graph[src, dst] = polarity #connection for input to internal neurons (layer_1)
    
    #return a copy of the neuron connectivity graph passed as graph=
    def get_graph_copy(self, graph, dtype=np.float32):
        return np.copy(graph).astype(dtype)
    
    #public properties
    @property
    def total_neurons(self):
        return self.total_internal_neurons + self.input_dim
    
    @property
    def internal_neurons(self):
        return self.total_internal_neurons
    
    @property 
    def internal_graph(self):
        return self.i_neuron_graph
    
    @property
    def input_graph(self):
        return self.input_neuron_graph

class NCPBackbone(SynapseWiring):
    def __init__(self, l1, l2, l3, #lx where x = 1, 2, 3 are the corresponding internal neurons (inter -> command -> motor)
                 input_fanout, l1_fanout, l2_fanout, #fan-out connections (how many neighbours each src neuron has to the next layers)
                 self_connections, seed=SEED):
        
        self.rndm_sd = np.random.RandomState(seed) #for reproducibility
        total_internal_neurons = l1 + l2 + l3 
        super().__init__(internal_neurons=total_internal_neurons) #initialise base class to build graph

        self._set_out_dim(out_dim=l3) #base class function
        #number of neurons in each layer
        self.l1_count = l1 
        self.l2_count = l2 
        self.l3_count = l3 

        #connectivity args
        self.input_fanout = input_fanout
        self.l1_fanout = l1_fanout
        self.l2_fanout = l2_fanout
        self.recurrent = self_connections #random neurons have a connection with themselves (identity diagonal of graph set with a polarity -1, 1)

        self._preload_connectivity() #computes connectivity indicies for internal neurons
        self._validate_constraints()

    def _preload_connectivity(self):
        self.l3_neurons = [i for i in range(0, self.l3_count)] #starting with last layer for indexing speed retrieval if using motor layer as output 
        self.l2_neurons = [i for i in range(self.l3_count, (self.l3_count + self.l2_count))]
        self.l1_neurons = [i for i in range((self.l3_count + self.l2_count), self.internal_neurons)] #self.internal_neurons from parent/base class property
    
    def _validate_layer(self, fanout, constraint, src, dst):
        if fanout > constraint:
            raise ValueError(f"Cannot construct {fanout} outgoing connections from {src} to {dst}, when there are only {constraint} {dst}")

    #ensure fanouts for layers do not exceed number of actual neurons in that layer
    #must be <=
    def _validate_constraints(self):
        self._validate_layer(fanout=self.input_fanout, constraint=self.l1_count, src="input", dst="layer 1")
        self._validate_layer(fanout=self.l1_fanout, constraint=self.l2_count, src="layer 1", dst="layer 2")
        self._validate_layer(fanout=self.l2_fanout, constraint=self.l3_count, src="layer 2", dst="layer 3")  

    def build(self, input_dim):
        super().build(input_dim=input_dim) #set base class attribute
        #input to internal neuron graph is separate from internal neuron graph so
        #input neurons can start from 0 index
        self.input_neurons = [i for i in range(0, self.input_dim)] #input_dim from base class

        self._connect_input_to_l1() #input to l3 (input_neuron graph)
        self._connect_l1_to_l2() #l1 to l2
        self._connect_l2_to_l3() #l2 to l3 
        self._connect_recurrent()
    
    #connect remaining dst to random src so every dst has a connection
    def _connect_remaining(self, dst, src, fanout, total, n_type):
        mean_fanin = int(len(src) * fanout / total)
        mean_fanin_clipped = np.clip(mean_fanin, 1, len(src))

        for remaining in dst:
            for src_neuron in self.rndm_sd.choice(src, size=mean_fanin_clipped, replace=False):
                pol = self.rndm_sd.choice([-1, 1])
                self.add_connection(src=src_neuron, dst=remaining, polarity=pol, n_type=n_type)

    def _connect_layers(self, dst_n, src_n, fanout, n_type):
        dst_neurons = [neuron for neuron in dst_n] #dst indices coming from destination layer

        for src in src_n:
            for dst in self.rndm_sd.choice(dst_n, size=fanout, replace=False):
                if dst in dst_neurons: #if not visited
                    dst_neurons.remove(dst) #visit
                
                #set random polarity
                pol = self.rndm_sd.choice([-1, 1])
                self.add_connection(src=src, dst=dst, polarity=pol, n_type=n_type)
        
        self._connect_remaining(dst=dst_neurons, src=src_n, fanout=fanout,
                                total=len(dst_n), n_type=n_type)

    def _connect_input_to_l1(self):
        self._connect_layers(dst_n=self.l1_neurons,
                             src_n=self.input_neurons,
                             fanout=self.input_fanout,
                             n_type="input")
        
    def _connect_l1_to_l2(self):
        self._connect_layers(dst_n=self.l2_neurons,
                             src_n=self.l1_neurons,
                             fanout=self.l1_fanout,
                             n_type="internal")

    def _connect_l2_to_l3(self):
        self._connect_layers(dst_n=self.l3_neurons,
                            src_n=self.l2_neurons,
                            fanout=self.l2_fanout,
                            n_type="internal")
        
    #original paper did recurrent connections for only layer 2 (command)
    def _connect_recurrent(self):
        all_neurons = [neuron for neuron in range(0, self.internal_neurons)] #indices from all internal neurons

        for _ in range(self.recurrent):
            neuron = self.rndm_sd.choice(all_neurons) #connection with itself
            pol = self.rndm_sd.choice([-1, 1])
            self.add_connection(src=neuron, dst=neuron, polarity=pol, n_type="internal")
    
    @property 
    def input_size(self):
        return self.input_dim
    
    @property 
    def output_size(self): #layer 3 is the output layer but can use all hidden states instead
        return self.l3_count