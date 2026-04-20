from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config_x4 import Config
import math

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    print("causal_conv1d import fails")
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None


#----------------------------------Pixhel Shuffle ------------------------------------------#

#input shape: (BC, F, H, W) 

class PixelShuffle(torch.nn.Module):
   
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        
        batch_size, long_channel_len, short_height, short_width = x.shape
        
        short_channel_len = long_channel_len // ( self.upscale_factor *  self.upscale_factor)
        long_width = self.upscale_factor * short_width
        long_height = self.upscale_factor * short_height

        x = x.contiguous().view([batch_size, short_channel_len, self.upscale_factor, self.upscale_factor, short_height, short_width])
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(batch_size, short_channel_len, long_height, long_width)

        return x

#output shape: (B, C/k, k*W, k*H)

#----------------------------------Channel Attention Block------------------------------------------#
class ChannelGate(nn.Module):
    def __init__(self, gate_channels,out_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Conv1d(gate_channels, gate_channels // reduction_ratio, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv1d(gate_channels // reduction_ratio, out_channels, 1, padding=0, bias=True),

            )
        self.pool_types = pool_types
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # INPUT (BH, F, W)
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = self.avg_pool( x )
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = self.max_pool( x )
                channel_att_raw = self.mlp( max_pool )
            
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = self.sigmoid( channel_att_sum ) #.unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
#----------------------------------First Convolutional Layer------------------------------------------#

class FirstConvLayer(nn.Module):
    def __init__(self, in_channels, out_features, kernel_size=3, stride=1, padding=1):
        super(FirstConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_features, kernel_size=kernel_size, stride=stride, padding=padding , padding_mode="reflect")

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        x = rearrange(x, "B C H W -> B H C W ")
        x = rearrange(x, "B H C W -> (B H) C W")
        x = self.conv(x)  
        
        # x shape: (B H) F W
        
        return x

#----------------------------------NafNet Block------------------------------------------#

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.2):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv1d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv1d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True, padding_mode="reflect")
        self.conv3 = nn.Conv1d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv1d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv1d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = nn.LayerNorm([c])
        self.norm2 = nn.LayerNorm([c])

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1)), requires_grad=True)
        

    def forward(self, inp, eval= False, inference_params=None, index_initialize_states=0):
        x = inp
        x = rearrange(x, "B F W -> B W F")        
        x = self.norm1(x)
        x = rearrange(x, "B W F -> B F W")
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        
        x = rearrange(y, "B F W -> B W F")        
        x = self.norm2(x)
        x = rearrange(x, "B W F -> B F W")
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma

#----------------------------------Upsampler------------------------------------------#

class Upsampler(nn.Module):
    def __init__(self, scale, in_channels, out_channels, kernel_size=3, bias=True, bn=False, act='relu'):
        
        super(Upsampler, self).__init__()
        
        self.F = 64
        self.conv_1 = nn.Conv1d(in_channels, self.F * 16, kernel_size, padding=(kernel_size//2), bias=bias)        
        self.PixelShuf_1 = PixelShuffle(4)
        
    def forward(self, x, HH = 32):
        x = self.conv_1(x)
        
        x = rearrange(x, "(B H) F W -> B H F W", H=HH)
        x = rearrange(x, "B H F W -> B F H W")
        x = self.PixelShuf_1(x)
        x = rearrange(x, "B F H W -> B H F W", H=HH*4)
        x = rearrange(x, "B H F W -> (B H) F W", H = HH*4)
                
        return x        

#----------------------------------Mamba------------------------------------------#

class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None, index_initialize_states=0):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
#            print("Inference Params")
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch, initialize_states= index_initialize_states)
#            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
            out, _, _= self.step(hidden_states, conv_state, ssm_state)
            return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
    #            if conv_state is not None:
    #                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
    #                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
    #                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])

            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
            
            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
                # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
                y = selective_state_update(
                    ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
                )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
                
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=0):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
                        
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states == 0 :
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

class InferenceParams:
    def __init__(self):
        self.key_value_memory_dict = {}
#----------------------------------Model------------------------------------------# 

class LineParams:
    def __init__(self):
        self.key_value_memory_dict = []

class Model(nn.Module):

    def __init__(self,) -> None:
        super().__init__()
        
        ##############################
        self.num_of_features  = 280 
        self.in_channels= 128 
        ##############################
        
        self.config = Config()
                
        self.width = self.config.width        
        self.height = self.config.height        
        self.batch_size = self.config.batch_size        
        self.upscaling = self.config.upscaling        
               
        self.conv_1 = FirstConvLayer(in_channels=self.in_channels, out_features=self.num_of_features )        
        self.NafBlock_1 = NAFBlock(self.num_of_features )        
        self.NafBlock_2 = NAFBlock(self.num_of_features )
        self.Mamba_1 = Mamba(d_model=self.num_of_features, d_state=16, d_conv=4, expand=1, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, conv_bias=True, bias=False, use_fast_path=True, layer_idx=1, device=None, dtype=None)        
        self.Mamba_2 = Mamba(d_model=self.num_of_features, d_state=16, d_conv=4, expand=1, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, conv_bias=True, bias=False, use_fast_path=True, layer_idx=2, device=None, dtype=None)
        self.silu = nn.SiLU()
        self.silu_1 = nn.SiLU()
        
        self.layer_norm1 = nn.LayerNorm([self.num_of_features])
        
        self.channel_attention = ChannelGate(self.num_of_features, self.num_of_features)
        
        self.layer_norm3 = nn.LayerNorm([ self.num_of_features])

        self.conv_finale = nn.Conv1d(in_channels= 64 , out_channels=self.in_channels, kernel_size=3, stride=1, padding=1)

        self.layer_norm5 = nn.LayerNorm([self.num_of_features])
      
        self.sg = SimpleGate()
        
        self.upsampler = Upsampler(scale=4, in_channels=self.num_of_features, out_channels=self.num_of_features)
        
        self.inference_params_mamba_1 = InferenceParams()
        self.inference_params_mamba_2 = InferenceParams()
        
        self.line_params = LineParams()
        
        print("Model initialized")
    
    def get_line_from_cache(self, inference_params, initialize_states=0, batch_size=1, height=1,  channels = 128, width = 32):
            if initialize_states == 0:
                line_state = torch.zeros(
                    batch_size,
                    channels,
                    height,
                    width,
                    device=self.conv_finale.weight.device,
                    dtype=self.conv_finale.weight.dtype,
                
                )

                inference_params.key_value_memory_dict = [line_state]
            else: 
                line_state = inference_params.key_value_memory_dict[0]
            return line_state
        
    def forward(self, xx, inference=False, index_initialize_states=0):   
        #------------------------------------------------First Feat. Extractor Layer------------------------------------------------#        
        # Input shape: (B, C, H, W)
        
        if torch.is_tensor(inference):
            inference = bool(inference.item())        
        batch, channels, height, width = xx.shape
        
        xz = self.conv_1(xx)

        xz = rearrange(xz, "B F W -> B W F").contiguous()        
        xz = self.layer_norm1(xz)
        y = rearrange(xz, "B W F -> B F W").contiguous()
       
        y = self.silu_1(y)
        x = self.channel_attention(y)

        #------------------------------------------------First NafBlock------------------------------------------------#
        q = self.NafBlock_1(x)
        
        q1q = rearrange(q, "B F W -> B W F").contiguous()        
        q1q = self.layer_norm5(q1q)
        q1q = rearrange(q1q, "B W F -> B F W").contiguous()  
        q1q = rearrange(q1q, "(B H) F W -> B H F W", H=height)
        q1q = rearrange(q1q, "B H F W -> B W H F", H=height)
        q1q = rearrange(q1q, "B W H F -> (B W) H F")
        
        #------------------------------------------------First Mamba------------------------------------------------#
        '''
        y = (B, l, d_in)
        '''
        if inference is False:
            q1q = self.Mamba_1(q1q)
        else: 
            q1q = self.Mamba_1(q1q, inference_params=self.inference_params_mamba_1, index_initialize_states=index_initialize_states)
        
        # Input shape: (B, H, F=64, W, C)
        q1q = rearrange(q1q, "(B W) H F -> B W H F", W=width, B=batch).contiguous()        
        q1q = rearrange(q1q, " B W H F -> B H F W").contiguous()
        q1q = rearrange(q1q, "B H F W -> (B H) F W", H = height).contiguous()
        
        q = q + q1q
        
        #------------<-----------------------------------Second NafBlock------------------------------------------------#

        q = self.NafBlock_2(q)
        
        q1q = rearrange(q, "B F W -> B W F").contiguous()        
        q1q = self.layer_norm3(q1q)
        q1q = rearrange(q1q, "B W F -> B F W").contiguous()
            
        q1q = rearrange(q1q, "(B H) F W -> B H F W", H=height)
        q1q = rearrange(q1q, "B H F W -> B W H F", H=height)
        q1q = rearrange(q1q, "B W H F -> (B W) H F")
        
        #------------------------------------------------Second Mamba------------------------------------------------#
        '''
        y = (B, l, d_in)
        '''
        if inference is False:
            q1q = self.Mamba_2(q1q)
        else:
            q1q = self.Mamba_2(q1q, inference_params=self.inference_params_mamba_2, index_initialize_states=index_initialize_states)
        
        # Input shape: (B, H, F=64, W, C)
        q1q = rearrange(q1q, "(B W) H F -> B W H F", W=width, B=batch).contiguous()        
        q1q = rearrange(q1q, " B W H F -> B H F W").contiguous()
        q1q = rearrange(q1q, "B H F W -> (B H) F W", H = height).contiguous()
        q = q + q1q
       
        #------------<-----------------------------------UP------------------------------------------------#

        q_z = self.upsampler(q, HH = height)

        #Bilinear upsampling on each 2 consecutive lines of the input. H_input=32 --> H_after_bilinear_upsampling = 31 * 4 
        if inference is False:
            upsampled_slices = []
            # There are 31 contiguous slices of height 2 (from rows 0–1, 1–2, …, 30–31)
            for i in range(height - 1):
                # Extract 2 contiguous rows: shape (B, C, 2, W)
                slice_2 = xx[:, :, i:i+2, :]
                # Upsample the slice
                upsampled_1=F.interpolate(slice_2,size=(5,width*4), align_corners=False, mode='bilinear')
                upsampled = upsampled_1[:,:,:4,:]
                upsampled_slices.append(upsampled)

            # Concatenate along the height (dim=2) to preserve the original order
            bilinear_upsampling = torch.cat(upsampled_slices, dim=2)  # shape: (B, C, 31*4, W*4)
            
        if inference is True:
            # X has dim (1, 128, 1, 32)
            upsampled_slices = []
            slice_2 = self.get_line_from_cache(self.line_params, initialize_states=index_initialize_states)
            upsampled_slices.append(slice_2)
            upsampled_slices.append(xx)
            two_lines = torch.cat(upsampled_slices, dim=2)
            upsampled_1=F.interpolate(two_lines,size=(5,width*4), align_corners=False, mode='bilinear')
            bilinear_upsampling = upsampled_1[:,:,:4,:]
            self.line_params.key_value_memory_dict[0] = xx

        #----------------------------------------------Final Conv kernel 1x1x1------------------------------------------------#
        r1 = self.conv_finale(q_z) 
        r1 = rearrange(r1, "(B H) C W -> B H C W", B=batch).contiguous()        
        r1 = rearrange(r1, "B H C W -> B C H W").contiguous()
        # x shape: (B, C, H, W)
        if inference is False:        
            r1 = r1[:, :, 4:, :]  # Remove the first 4 rows to match bilinear upsampling size
        output = r1 + bilinear_upsampling 
        return output
