from .ltc_wiring import NeuralCircuitPolicy as ncp 
from .ltccell import LTCCell

def build_propagator(r1, r2, r3, in_fanout, r1_fanout, r2_fanout, recurrent, seed, input_dim,
                          input_mapping, output_mapping, ode_unfolds, epsilon, project_output):
        
        #build wire
    wire = ncp(
            R1_count=r1,
            R2_count=r2,
            R3_count=r3,
            input_fanout=in_fanout,
            R1_fanout=r1_fanout,
            R2_fanout=r2_fanout,
            recurrent_connections=recurrent,
            seed=seed
        )
    wire.build(input_dim=input_dim) #initialise

        #build propagator
    propagator = LTCCell(
            neural_wiring=wire,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon,
            project_output=project_output
    )

    return propagator