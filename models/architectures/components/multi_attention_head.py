import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

#valid aggregation modes for the forward output
AGG_MODES = ('concat', 'list', 'dict')

#example head_specs, each declared as (seq, dim) or (dim,) for a single vector
#dims are extracted from the shapes, only num_heads is specified
MAH_EXAMPLE_SPECS = {
    'env_state':      {'q': (16, 84), 'k': (16, 84), 'v': (16, 84), 'num_heads': 4},
    'imagined_state': {'q': (16, 84), 'k': (16, 84), 'v': (16, 84), 'num_heads': 4},
    'semantic_acc':   {'q': (8, 64),  'k': (8, 56),  'v': (8, 54),  'num_heads': 4},
}


#pure helper, split a declared shape into (seq, dim); a single-element (dim,) sets seq to 1
def resolve_shape(shape, label, name):
    if isinstance(shape, int):
        return 1, shape #bare int is a single vector, same as (dim,)
    if len(shape) == 2:
        seq, dim = shape
        return int(seq), int(dim)
    if len(shape) == 1:
        return 1, int(shape[0])
    raise ValueError(f"[{label}] '{name}' shape must be (seq, dim), (dim,), or dim, got {shape}")

#pure helper, accept (batch, dim) or (batch, seq, dim), add a single seq axis when missing
#returns the tensor and whether a seq axis was inserted, so masks can be checked against real shapes
def to_batch_seq_dim(tensor, label, name):
    if tensor.dim() == 2:
        return tensor.unsqueeze(1), True #(batch, dim) -> (batch, 1, dim)
    if tensor.dim() == 3:
        return tensor, False
    raise ValueError(f"[{label}] '{name}' must be (batch, dim) or (batch, seq, dim), got {tuple(tensor.shape)}")


class MAH(nn.Module):
    def __init__(self, head_specs, use_dense=True, dense_out_dim=None, dropout=0.0):
        super().__init__()

        assert isinstance(head_specs, dict) and len(head_specs) > 0, "head_specs must be a non-empty dict"

        self.use_dense = use_dense
        self.dense_out_dim = dense_out_dim if use_dense else None
        if use_dense:
            assert dense_out_dim is not None, "use_dense=True requires dense_out_dim"

        self.dropout = dropout
        self._specs = {} #resolved per-head state, surfaced via property

        self._build_attention_heads(head_specs)
        self.registered_labels = list(self._specs.keys())
        self.attention_amnt = len(self.registered_labels)
        self._att_weights = {label: None for label in self.registered_labels} #diagnostic cache

    def _build_attention_heads(self, head_specs):
        self.attention_heads = nn.ModuleDict()
        if self.use_dense:
            self.dense_projections = nn.ModuleDict()

        for label, spec in head_specs.items():
            num_heads = spec['num_heads']

            #extract expected seq/dim per stream from the declared shapes
            q_seq, query_dim = resolve_shape(spec['q'], label, 'q')
            k_seq, key_dim = resolve_shape(spec['k'], label, 'k')
            v_seq, value_dim = resolve_shape(spec['v'], label, 'v')

            #build-time validation, key and value share a sequence length (one value per key)
            if k_seq != v_seq:
                raise ValueError(f"[{label}] key seq {k_seq} and value seq {v_seq} must match")
            if query_dim % num_heads != 0:
                raise ValueError(f"[{label}] query_dim {query_dim} not divisible by num_heads {num_heads}")

            #kdim/vdim only passed when they differ from query_dim, else left as None
            self.attention_heads[label] = nn.MultiheadAttention(
                embed_dim=query_dim,
                num_heads=num_heads,
                dropout=self.dropout,
                batch_first=True,
                kdim=key_dim if key_dim != query_dim else None,
                vdim=value_dim if value_dim != query_dim else None,
            )

            #attention output is always query_dim wide, projection shape known at build
            if self.use_dense:
                self.dense_projections[label] = nn.Linear(query_dim, self.dense_out_dim)

            #store resolved state for inspection and forward-time seq checks
            self._specs[label] = {
                'query_dim': query_dim, 'key_dim': key_dim, 'value_dim': value_dim,
                'num_heads': num_heads,
                'q_seq': q_seq, 'k_seq': k_seq, 'v_seq': v_seq,
            }

    #forward validation 
    def _check_labels(self, inputs):
        incoming = set(inputs.keys())
        registered = set(self.registered_labels)

        unknown = incoming - registered
        assert not unknown, f"unknown attention labels {unknown}, registered are {registered}"

        missing = registered - incoming
        assert not missing, f"missing inputs for attention heads {missing}"

    #check incoming tensor against the head's declared expectation
    #dim mismatch is an error, seq mismatch is a suppressible warning
    def _check_dim_dec(self, label, name, tensor, expected_dim, expected_seq, suppress_seq_warning):
        actual_seq, actual_dim = tensor.shape[1], tensor.shape[2]

        if actual_dim != expected_dim:
            raise ValueError(
                f"[{label}] '{name}' feature dim {actual_dim} does not match declared {expected_dim}")

        if actual_seq != expected_seq and not suppress_seq_warning:
            warnings.warn(
                f"[{label}] '{name}' seq {actual_seq} differs from declared {expected_seq}, proceeding; "
                f"pass suppress_seq_warning=True to silence for variable-length use")

    #normalise a head's q/k/v to (batch, seq, dim), validate, guard mask against auto-unsqueezed inputs
    def _prepare_head_input(self, label, head_input, suppress_seq_warning):
        spec = self._specs[label]

        q, q_unsq = to_batch_seq_dim(head_input['q'], label, 'q')
        k, k_unsq = to_batch_seq_dim(head_input['k'], label, 'k')
        v, v_unsq = to_batch_seq_dim(head_input['v'], label, 'v')

        #compare post-normalisation shapes against declared expectations
        self._check_dim_dec(label, 'q', q, spec['query_dim'], spec['q_seq'], suppress_seq_warning)
        self._check_dim_dec(label, 'k', k, spec['key_dim'], spec['k_seq'], suppress_seq_warning)
        self._check_dim_dec(label, 'v', v, spec['value_dim'], spec['v_seq'], suppress_seq_warning)

        mask = head_input.get('attn_mask', None)

        #a mask's shape is tied to the real seq lengths, auto-unsqueezing a bare vector hides that
        #so a mask with any auto-unsqueezed input is a hard error, not a warning
        if mask is not None and (q_unsq or k_unsq or v_unsq):
            raise ValueError(
                f"[{label}] attn_mask provided with a (batch, dim) input; "
                f"pass q/k/v as explicit (batch, seq, dim) so the mask shape is unambiguous")

        return q, k, v, mask

    #inputs: dict[label -> {'q':, 'k':, 'v':, 'attn_mask': optional}]
    def forward(self, inputs, aggregate='concat', return_weights=False, suppress_seq_warning=False):
        assert isinstance(inputs, dict), "inputs must be a dict[label -> {q,k,v,attn_mask}]"
        assert aggregate in AGG_MODES, f"aggregate must be one of {AGG_MODES}"
        self._check_labels(inputs)

        outputs = {}

        #iterate registered order for deterministic concat, dict lookup is order-free
        for label in self.registered_labels:
            q, k, v, mask = self._prepare_head_input(label, inputs[label], suppress_seq_warning)

            attended, weights = self.attention_heads[label](
                query=q,
                key=k,
                value=v,
                need_weights=return_weights,
                attn_mask=mask,
            )

            if self.use_dense:
                attended = F.silu(self.dense_projections[label](attended))

            outputs[label] = attended

            #overwrite cache each weighted pass, detached to cpu so it is inert
            if return_weights:
                self._att_weights[label] = weights.detach().cpu()

        return self._aggregate(outputs, aggregate)

    def _aggregate(self, outputs, mode):
        #dict: labelled streams kept distinct
        if mode == 'dict':
            return outputs

        #list: outputs in registered order
        if mode == 'list':
            return [outputs[label] for label in self.registered_labels]

        #concat: merge over sequence dim then layer-norm, the single-vector path
        #only dimensionally safe when heads share a feature dim (guaranteed under use_dense -> dense_out_dim)
        merged = torch.cat([outputs[label] for label in self.registered_labels], dim=1)
        return F.layer_norm(merged, [merged.shape[-1]])

    #resolved per-head config, dims and expected seqs, for inspection
    @property
    def head_specs(self):
        return self._specs

    #diagnostic property, most recent averaged weights per label or None
    @property
    def current_att_weight(self):
        return self._att_weights

    #manual clear back to all-None
    def reset_att_weight(self):
        self._att_weights = {label: None for label in self.registered_labels}

    @property
    def attention_types(self):
        return ' '.join(self.registered_labels)

    def get_parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'non_trainable': total - trainable}

    def print_param_count(self):
        for p_type, count in self.get_parameter_counts().items():
            print(f"{p_type.lower()} parameters: {count}")