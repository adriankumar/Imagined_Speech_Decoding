import json
import numpy as np 

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_labels_matrix(label_encoding, labels):
    #BCE withlogitloss needs float targets
    table = np.zeros((len(label_encoding), len(labels)), dtype=np.float32) 
    for cls, bnry_cls in label_encoding.items():
        table[cls] = bnry_cls #the tuple item
    return table #shape dataset_labels x model_class

#a partial window keeps its label only if the label sits equal to
#or above the specified threshold (i.e arbitrary window splitting doesn't cleanly separate 
#trial/epoch boundaries as paper shows, so drop or keep the label based on a hyperparameter)
def resolve_partial_label(classes, window_frac, threshold):
    resolved = classes.copy()
    resolved[window_frac < threshold] = 0
    return resolved

#truncate recording sequence (in its windowed form already from the cache) as either
#independent windows for FFN based networks, or to retain an equal number of sequential steps
#of full recording length for RNNs
def truncate_recording(runs, seq_len=None):
    #checks all sequence lengths of every recording is the same length
    lengths = {run["n_rows"] for run in runs}
    assert len(lengths) == 1, f"runs are not equal length, got {sorted(lengths)}" 
    n_rows = lengths.pop() #remove dict

    #sequence size for arranging windows
    seq_size = n_rows if seq_len is None else seq_len 

    #ensure truncated sequence length yields equally across divided parts (i.e splitting recording of 30min into segments/chunks of 60s recordings as 'independent' samples)
    assert 0 < seq_size <= n_rows, f"seq_len {seq_size} does not fit a run of {n_rows} windows"
    assert n_rows % seq_size == 0, f"seq_len {seq_size} does not divide a run of {n_rows} windows"

    starts, run_index = [], []
    for i, run in enumerate(runs): #for each recording
        for nxt_window in range(0, n_rows, seq_size): #steps by the sequence size, so size=1 yields one chunk per window and seq_len=None yields one chunk per recording
            starts.append(run["start_row"] + nxt_window) #append the starting window and the succeeding windows to include as part of the 'sequence'
            run_index.append(i)

    #return the sequences, their corresponding real run index and the truncated sequence size
    return np.asarray(starts, dtype=np.int64), np.asarray(run_index, dtype=np.int64), seq_size
