from .ltc_wiring import NeuralCircuitPolicy as ncp 
from .ltccell import LTCCell
from .ltccell_adaptive import LTCCell_adaptive

#function wrapper to build ltc; non-LoRA  
def build_ltc_prop(r1=3, r2=5, r3=2, 
                   input_fanout=3, r1_fanout=2, r2_fanout=4, self_connections=6, 
                   seed=24573471, input_dim=40, input_mapping='affine', output_mapping='affine', 
                   ode_unfolds=6, epsilon=1e-4, project_output=False,
                   print_parameters=True):
    
    #add assertion statements
    assert input_fanout <= r1, "input fanout should be less than or equal to r1 amount"
    assert r1_fanout <= r2, "r1 fanout should be less than or equal to r2 amount"
    assert r2_fanout <= r3, "r2 fanout should be less than or equal to r2 amount"

    #build neural wire first
    wire = ncp(R1_count=r1, R2_count=r2, R3_count=r3, 
               input_fanout=input_fanout, R1_fanout=r1_fanout, R2_fanout=r2_fanout, 
               recurrent_connections=self_connections, seed=seed)
    
    #build/initialise
    wire.build(input_dim=input_dim)

    #build ltc
    ltc =  LTCCell(neural_wiring=wire, 
                   input_mapping=input_mapping, output_mapping=output_mapping, 
                   ode_unfolds=ode_unfolds, epsilon=epsilon, 
                   project_output=project_output)
    
    if print_parameters:
        pc = ltc.get_parameter_counts() #list [trainable, non-trainable, total]

        print(f"propagator (ltc) parameter count:",
              f"trainable: {pc[0]} | non-trainable: {pc[1]} | total: {pc[2]}")
    
    return ltc

def build_ltc_adaptive_prop(r1=3, r2=5, r3=2, 
                   input_fanout=3, r1_fanout=2, r2_fanout=4, self_connections=6, 
                   seed=24573471, input_dim=40, input_mapping='affine', output_mapping='affine', 
                   ode_unfolds=6, epsilon=1e-4, project_output=False,
                   print_parameters=True):
    
    #add assertion statements
    assert input_fanout <= r1, "input fanout should be less than or equal to r1 amount"
    assert r1_fanout <= r2, "r1 fanout should be less than or equal to r2 amount"
    assert r2_fanout <= r3, "r2 fanout should be less than or equal to r2 amount"

    #build neural wire first
    wire = ncp(R1_count=r1, R2_count=r2, R3_count=r3, 
               input_fanout=input_fanout, R1_fanout=r1_fanout, R2_fanout=r2_fanout, 
               recurrent_connections=self_connections, seed=seed)
    
    #build/initialise
    wire.build(input_dim=input_dim)

    #build ltc
    ltc =  LTCCell_adaptive(neural_wiring=wire, 
                   input_mapping=input_mapping, output_mapping=output_mapping, 
                   ode_unfolds=ode_unfolds, epsilon=epsilon, 
                   project_output=project_output)
    
    if print_parameters:
        pc = ltc.get_parameter_counts() #list [trainable, non-trainable, total]

        print(f"propagator (ltc) parameter count:",
              f"trainable: {pc[0]} | non-trainable: {pc[1]} | total: {pc[2]}")
    
    return ltc