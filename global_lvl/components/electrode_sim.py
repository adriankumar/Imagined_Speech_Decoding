
from ..constants import MNE_MONTAGES
from ..helpers import (classify_chns, get_3d_pos, get_2d_pos, build_chn_name_order,
                       build_img_interpolation, build_topo_mask, compute_spherical_angles)

#stateless-class handling the electrode-space of things
class ElectrodeSim:
    def __init__(self, src_chn_names=None, num_chns=None, montage="standard_1005", 
                 img_dims=(64, 64), img_margin=0.75, print_channel_resolve=True):

        assert montage in MNE_MONTAGES, f"Montage: {montage} is not recognised by MNE, use default standard_1005 or check which ones resolve the most channels"
        #the montage the current source uses or resolves the most channels against MNE
        #or to use for the general electrode sim
        self._montage = montage 

        #explicit check
        if src_chn_names is not None and num_chns is None:
            assert isinstance(src_chn_names, list), f"source channel names must be a list of strings, got {type(src_chn_names)}"
            self._resolved_chns = None #if any channel names deviate from a standard, exclude it from use
            self._excluded_chns = None 
            self._original_chns = src_chn_names 
            self._for_general = False #if channel names provided, Electrode sim is used for a source input

        elif num_chns is not None and src_chn_names is None: 
            assert isinstance(num_chns, int) and num_chns > 0, f"number of channels should be a positive integer"
            self._n_schns = num_chns #number of simulated channels
            self._for_general = True

        else:
            raise ValueError("Source channel names and Number of channels can't both be None or both be specified. Either provide a list of channel names to build for a specific source input, and leave num_chns blank; or a number of arbitrary channels for an EEG simulation, leaving src_chn_names blank")

        self._img_dims = img_dims 
        self._interpol_margin = img_margin

        self._build(print_chns_resolve=print_channel_resolve)

    def _resolve_3d_pos(self, printout):
        if self._for_general: #if for general sim, just get arbitrary positions
            return get_3d_pos(montage=self._montage,
                              chns_list=None, #dont provide a channel list
                              num_chns=self._n_schns)
        
        else: #otherwise get the specific channel data for the current source
            self._resolved_chns, self._excluded_chns = classify_chns(montage=self._montage,
                                                                    original_chns=self._original_chns,
                                                                    printout=printout)
            
            self._chns_order = build_chn_name_order(original_chns=self._original_chns,
                                                    resolved_chns=self._resolved_chns)
            
            return get_3d_pos(montage=self._montage,
                              chns_list=self._resolved_chns, #only resolved channels
                              num_chns=None) #no number of channels
        
    def _build(self, print_chns_resolve):

        #get channel positions
        self._pos_3d = self._resolve_3d_pos(printout=print_chns_resolve)
        thetas, phis = compute_spherical_angles(pos_3d=self._pos_3d)
        self._pos_2d = get_2d_pos(theta=thetas, phi=phis)

        #build image interpol operator and the pixel-space grid/positions it uses
        self._M, pix_2d, grid_2d = build_img_interpolation(electrode_pos_2d=self._pos_2d,
                                                           img_res=self._img_dims,
                                                           margin=self._interpol_margin)
        
        #mask uses M's own pixel space so its edge sits on the interpolated alignment
        self._topo_mask = build_topo_mask(electrode_pix_2d=pix_2d,
                                          grid_2d=grid_2d,
                                          img_dims=self._img_dims)
        
        self._sh_args = {
            "thetas": thetas,
            "phis": phis
        }

    #re-build electrode geometry dependencies
    #only meaningful for a source sim, where arbitrary eeg input recordings come in and have different settings, 
    #the simulated route has no channel names to swap since it's built with num_chns
    def rebuild(self, src_chn_names=None, montage=None, img_margin=None, print_channel_resolve=False):
        assert not self._for_general, "a simulated electrode sim has no source to change"

        if montage is not None:
            assert montage in MNE_MONTAGES, f"Montage: {montage} is not recognised by MNE"
            self._montage = montage

        if src_chn_names is not None:
            assert isinstance(src_chn_names, list), f"source channel names must be a list of strings, got {type(src_chn_names)}"
            self._original_chns = src_chn_names

        if img_margin is not None:
            self._interpol_margin = img_margin

        self._build(print_chns_resolve=print_channel_resolve)

    @property 
    def montage(self):
        return self._montage

    @property 
    def original_channels(self):
        return None if self._for_general else self._original_chns
    
    @property 
    def resolved_channels(self):
        return None if self._for_general else self._resolved_chns

    @property
    def excluded_channels(self):
        return None if self._for_general else self._excluded_chns

    #only for resolved channels
    @property 
    def num_channels(self):
        return self._n_schns if self._for_general else len(self.resolved_channels)

    @property
    def is_general(self):
        return self._for_general

    @property 
    def spherical_coords(self):
        return self._sh_args

    @property 
    def img_size(self):
        return self._img_dims

    @property 
    def img_margin(self):
        return self._interpol_margin

    @property
    def electrode_3d_pos(self):
        return self._pos_3d

    @property
    def electrode_2d_pos(self):
        return self._pos_2d

    @property 
    def img_transform(self):
        return self._M 

    @property
    def electrode_mask(self):
        return self._topo_mask

    @property 
    def channels_order(self):
        return None if self._for_general else self._chns_order