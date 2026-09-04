from .ltc_cell import LTCCell 
from .ncp_wire import NCPBackbone as ncp 
from global_lvl import SEED, EPSILON

#function wrapper returning the ltc
def build_ltc(layer_1=2, layer_2=3, layer_3=2,
              input_fanout=2, l1_fanout=3, l2_fanout=2, self_connections=5,
              seed=SEED, #ncp wiring arguments
              
              input_dim=10, output_dim=3, input_mapping="affine", output_mapping="affine",
              ode_unfolds=6, epsilon=EPSILON, #ltc args; 

              print_parameters=True): 
    
    total_neurons = layer_1 + layer_2 + layer_3
    assert total_neurons >= 3, "Each layer must have at least one neuron"
    
    #build graph first; any connectivity errors surfaced here
    connectivity_graph = ncp(l1=layer_1, l2=layer_2, l3=layer_3,
                             input_fanout=input_fanout, l1_fanout=l1_fanout, l2_fanout=l2_fanout,
                             self_connections=self_connections, seed=seed)
    
    #initialise
    connectivity_graph.build(input_dim=input_dim)

    #build ltc 
    ltc = LTCCell(neural_graph=connectivity_graph,
                  input_mapping=input_mapping, output_mapping=output_mapping,
                  ode_unfolds=ode_unfolds, output_dim=output_dim,
                  epsilon=epsilon)
    
    if print_parameters:
        pc = ltc.get_parameter_count() #dict

        for p_type, count in pc.items():
            print(f"{p_type.lower()} parameters: {count}")
    
    return ltc

#------------
#For LTC layers
#------------
# def auto_fanouts(l1, l2, l3):
#     input_fanout = max(1, l1 // 2) #dst is l1
#     l1_fanout = max(1, l2 // 2) #dst is l2
#     l2_fanout = max(1, l3 // 2) #dst is l3
#     return input_fanout, l1_fanout, l2_fanout

#density is the fraction of the dst layer each src neuron connects to;
#a wide input against a narrow output wires sparser, a narrow one wires denser
def auto_fanouts(l1, l2, l3, input_dim):
    import numpy as np 
    ratio = (l3 / input_dim) ** (1.0 / 3.0) #(out / in)**(1/n), same rule as the mlp hidden dims
    #clipped to the dst layer, a fanout can never exceed the neurons available to receive it
    return tuple(int(np.clip(round(ratio * dst), 1, dst)) for dst in (l1, l2, l3))

def auto_layers(total):
    base, rem = divmod(total, 3)
    l1 = base + (rem > 0) #first spare to l1
    l2 = base + (rem > 1) #second spare to l2
    l3 = base
    return l1, l2, l3 #sums to total exactly, each >= 1 when total >= 3