import torch

#the four input ablations, selected once before the epoch loop rather than branched per step
INPUT_MODES = ("coeffs", "coeffs_residual", "electrodes", "noise")

#windows passed as shape B x num_windows x nchns x F
def compute_seq_coeffs(windows, solver):
    return torch.einsum("ce, bwef -> bwcf", solver, windows) #B x num_windows x coeffs x F

def compute_seq_residual(windows, res_op):
    return torch.einsum("ne, bwef -> bwnf", res_op, windows) #B x num_windows x nchns x F

#returns (x, residual) for one batch under the chosen mode; residual is None unless the
#residual branch is being trained
def build_inputs(mode, windows, solver, residual_operator, noise_shape, clip):
    if mode == "noise": #for noise ablations
        #keeps batch and sequence order, replaces the data itself
        shape = (*windows.shape[:2], *noise_shape)
        return torch.randn(shape, device=windows.device), None

    windows = torch.minimum(windows, clip) #clip outliers before operations or input

    if mode == "electrodes":
        return windows, None #already B x num_windows x nchns x F, no solver involved

    with torch.no_grad():
        coeffs = compute_seq_coeffs(windows, solver) #B x num_windows x coeffs x F

        if mode == "coeffs_residual":
            return coeffs, compute_seq_residual(windows, residual_operator)

        return coeffs, None