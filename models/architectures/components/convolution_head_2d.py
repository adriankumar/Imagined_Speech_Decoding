import torch 
import torch.nn.functional as F 
import torch.nn as nn
from global_lvl import EPSILON

#allowed activations for cnn head
activation_fns = {
    'relu': F.relu,
    'silu': F.silu,
    'leaky-relu': lambda x: F.leaky_relu(x, 0.2)
}

#(num_filters, kernel_size, stride)
DEFAULT_CONV = [(32, 8, 5), (24, 4, 3), (8, 3, 1)]

#pure helpers, no state --------------------------------------------------
def to_pair(v):
    #int -> (v, v); 2-sequence -> (h, w)
    if isinstance(v, int):
        return (v, v)
    h, w = v
    return (int(h), int(w))

def conv_out_dim(size, kernel, stride, padding):
    #output length along one spatial axis for a conv
    return (size + 2 * padding - kernel) // stride + 1

#cnn attention visualisation --------------------------------------------------
#implementation take from visualbackprop: https://arxiv.org/pdf/1611.05418;
#but for activation functions with negative values, the attention just takes the 
#its absolute value
#strip a singleton batch dim, block real batches; accept (c,h,w) or (1,c,h,w)
def to_chw(x):
    if x.dim() == 4:
        if x.shape[0] != 1:
            raise ValueError(f"expected a single image, got a batch of {x.shape[0]}; pass one sample")
        x = x[0]
    if x.dim() != 3:
        raise ValueError(f"expected (c, h, w) or (1, c, h, w), got shape {tuple(x.shape)}")
    return x

#all-ones transpose conv lifting a (1,1,h,w) map to target resolution
#output_padding resolves the stride ambiguity so the result lands exactly on target
def deconv_upsample(single_map, kernel, stride, padding, target_hw):
    kh, kw = kernel
    sh, sw = stride
    ph, pw = padding
    th, tw = target_hw

    #size a plain transpose conv would produce, before output_padding
    base_h = (single_map.shape[-2] - 1) * sh - 2 * ph + kh
    base_w = (single_map.shape[-1] - 1) * sw - 2 * pw + kw

    weight = torch.ones(1, 1, kh, kw, device=single_map.device, dtype=single_map.dtype)
    return F.conv_transpose2d(single_map, weight, stride=(sh, sw),
                              padding=(ph, pw), output_padding=(th - base_h, tw - base_w))

#object-oriented matplotlib render, headless-safe, no pyplot global state
def render_attention(base_grey, mask, save_path, cmap='inferno'):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(11, 5))
    FigureCanvasAgg(fig)

    #raw base on the left for comparison
    ax_raw = fig.add_subplot(1, 2, 1)
    ax_raw.imshow(base_grey, cmap='gray', interpolation='nearest')
    ax_raw.set_xlabel('width')
    ax_raw.set_ylabel('height')
    ax_raw.set_title('input (channel mean)')

    #attention washed over the same base on the right
    ax_heat = fig.add_subplot(1, 2, 2)
    ax_heat.imshow(base_grey, cmap='gray', interpolation='nearest')
    heat = ax_heat.imshow(mask, cmap=cmap, alpha=0.6, interpolation='bilinear')
    ax_heat.set_xlabel('width')
    ax_heat.set_ylabel('height')
    ax_heat.set_title('conv attention (abs, all layers)')
    fig.colorbar(heat, ax=ax_heat, fraction=0.046, pad=0.04, label='normalised attention')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')

def init_module(module, activation):
    #match weight init to the activation, zero the bias
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if activation == 'silu':
            nn.init.xavier_uniform_(module.weight)
        elif activation == 'relu':
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        else:  #leaky-relu
            nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


#should add drop out layers
class ConvHead_2D(nn.Module):
    def __init__(self, conv_layers=DEFAULT_CONV, #conv layers specified as a list [(filters, kernel size, stride),...n_layers]
                                         #kernel and stride can be specified as (h, w) if non-square
                 input_dim=(1, 64, 64), #channels x h x w
                 out_dim_per_fl=8, activation='silu',
                 use_bias=True): #features per filter at the last conv layer for dense projections
        
        super().__init__()

        #validate upfront
        if conv_layers is None or len(conv_layers) == 0:
            raise ValueError(f"conv_spec must contain at least one layer, got: {conv_layers}")
        
        if len(input_dim) != 3:
            raise ValueError(f"input_shape must be (channels, height, width), got {input_dim}")

        if activation not in activation_fns:
            raise ValueError(f"activation {activation} must be one of {tuple(activation_fns)}")
        
        #store conv layers; expand square inputs for kernel and stride into h, w for compatability
        self.conv_config = [(f, to_pair(k), to_pair(s)) for f, k, s in conv_layers] 
        self.num_dense = self.conv_config[-1][0] #final number of filters in conv config
        self.use_bias = use_bias
        self.act_type = activation

        self._build(input_dim, out_dim_per_fl)

        #initialise parameters of each module (for m in self.modules()) depending on the activation type
        self.apply(lambda m: init_module(m, self.act_type)) 

    def _build(self, input_dim, out_pfl):
        self.input_dim = tuple(input_dim)
        self.out_dim_pfl = out_pfl #output dim after dense layer on each filter in the final conv layer (self.conv_config[-1][0]) 
        self.act_fn = activation_fns[self.act_type]

        #final conv layer info
        self.conv_h_out, self.conv_w_out = self._get_conv_out_dims()
        self.conv_layers = self._build_conv_layers()
        self.dense_proj = self._build_dense_projection()

    def _build_dense_projection(self):
        dense_projection = nn.ModuleList() #single dense projection into feature vector; not a deep CNN

        #flatten final conv output into vector for dense layers
        flattened = self.conv_h_out * self.conv_w_out

        for _ in range(self.num_dense): #for each filter in the final layer
            dense_projection.append(
                nn.Linear(
                    in_features=flattened,
                    out_features=self.out_dim_pfl, #out dim per dense layer
                    bias=self.use_bias
                )
            )
        
        return dense_projection

    def _build_conv_layers(self):
        layers = nn.ModuleList()
        in_chns = self.input_dim[0] #incoming channel amount; changes at each conv layer

        for num_filters, (kh, kw), (sh, sw) in self.conv_config:
            layers.append(nn.Conv2d(in_channels=in_chns, 
                                    out_channels=num_filters, 
                                    kernel_size=(kh, kw), 
                                    stride=(sh, sw),
                                    padding=(kh//2, kw//2),
                                    bias=self.use_bias))
            
            in_chns = num_filters #update next incoming channel amount
        
        return layers

    def _get_conv_out_dims(self):
        _, h_out, w_out = self.input_dim 

        #walk the stack and raise any window collapses
        for i, (_, (kh, kw), (sh, sw)) in enumerate(self.conv_config):
            #print current
            # print(f"conv layer {i}; current h_out: {h_out} | current w_out: {w_out}")    
            #next dim_out layers
            h_out = conv_out_dim(size=h_out, kernel=kh, stride=sh, padding=(kh // 2))
            w_out = conv_out_dim(size=w_out, kernel=kw, stride=sw, padding=(kw // 2))

            if h_out <= 0 or w_out <= 0:
                raise ValueError(
                    f"conv layer {i} collapses the map to ({h_out}, {w_out});"
                    f"reduce stride/kernel or raise input size"
                )
        
        return h_out, w_out
    
    def forward(self, x, return_list=False):
        #per-sample standardise across channel + spatial dims
        #(normalise input)
        x = (x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + EPSILON)

        for conv in self.conv_layers:
            x = self.act_fn(conv(x)) #forward convolve input across conv layers
        
        #flatten h,w dims and dense project each filter into a feature output
        #x is shape b x chns (filters in final layer) x h x w; arg flattens all dims from specified starting index dim
        x_flat = x.flatten(start_dim=2) #now shape b x chns x (h*w)
        features = [self.act_fn(proj(x_flat[:, i])) for i, proj in enumerate(self.dense_proj)]

        if return_list:
            return features #list of b x out_dim_pfl, with self.num_dense elements
        
        #concate features into a feature vector, not stack across new dim
        return torch.cat(features, dim=1) #shape batch x self.output_dim
    
    def get_parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'non_trainable': total - trainable}

    def print_param_count(self):
        for p_type, count in self.get_parameter_counts().items():
            print(f"{p_type.lower()} parameters: {count}")
    
    #no-grad diagnostic forward: capture each layer's response, chain deep-to-shallow into one mask
    #each unfold here is average -> upscale -> multiply, not the training forward pass
    @torch.no_grad()
    def compute_attention(self, x):
        image = to_chw(x).unsqueeze(0) #(1,c,h,w)
        input_hw = (image.shape[-2], image.shape[-1])

        #same per-sample standardise the real forward uses
        z = (image - image.mean(dim=(1, 2, 3), keepdim=True)) / (image.std(dim=(1, 2, 3), keepdim=True) + 1e-5)

        #forward through convs, keeping the abs-averaged map after each activation
        #abs before averaging turns signed activations into response strength with no cancellation
        maps = []
        geometry = []
        for conv in self.conv_layers:
            z = self.act_fn(conv(z))
            maps.append(z.abs().mean(dim=1, keepdim=True)) #(1,1,h,w)
            geometry.append((conv.kernel_size, conv.stride, conv.padding))

        #start at the deepest map, walk back toward the input
        mask = maps[-1]
        for i in range(len(maps) - 1, 0, -1):
            target_hw = (maps[i - 1].shape[-2], maps[i - 1].shape[-1])
            kernel, stride, padding = geometry[i] #layer i's own geometry lifts its map up
            mask = deconv_upsample(mask, kernel, stride, padding, target_hw)
            mask = mask * maps[i - 1] #gate by the shallower, higher-resolution map

        #final lift from the first layer's resolution up to the input image
        kernel, stride, padding = geometry[0]
        mask = deconv_upsample(mask, kernel, stride, padding, input_hw)

        mask = mask[0, 0] #(h,w)
        lo, hi = mask.min(), mask.max()
        mask = (mask - lo) / (hi - lo + 1e-8) #normalise to [0,1] for display
        return mask.cpu().numpy()

    #diagnostic entry point: compute the mask and save an overlay image
    def view_conv_attention(self, x, save_path, cmap='inferno'):
        image = to_chw(x) #(c,h,w)
        mask = self.compute_attention(image)

        #channel-mean grey base, contrast-stretched for display only (eeg values are not image range)
        base_grey = image.detach().cpu().float().mean(dim=0).numpy()
        b_lo, b_hi = base_grey.min(), base_grey.max()
        base_grey = (base_grey - b_lo) / (b_hi - b_lo + 1e-8)

        render_attention(base_grey, mask, save_path, cmap=cmap)
        return save_path

    @property
    def output_dim(self): #output dim of expected feature vector
        return (self.out_dim_pfl * self.num_dense)


