import torch
import torch.nn as nn
from networks.utils import *
from typing import Tuple
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
import numpy as np
import cv2
import math
from einops import rearrange


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LaplacianPyramid(nn.Module):
    def __init__(self, in_channels=64, pyramid_levels=3):
        """
        Constructs a Laplacian pyramid from an input tensor.

        Args:
            in_channels    (int): Number of input channels.
            pyramid_levels (int): Number of pyramid levels.
        
        Input: 
            x : (B, in_channels, H, W)
        Output:
            Fused frequency attention map : (B, in_channels, in_channels)
        """
        super().__init__()
        self.in_channels = in_channels
        self.pyramid_levels = pyramid_levels
        sigma = 1.6
        s_value = 2 ** (1/3)

        self.sigma_kernels = [
            self.get_gaussian_kernel(2*i + 3, sigma * s_value ** i)
            for i in range(pyramid_levels)
        ]

    def get_gaussian_kernel(self, kernel_size, sigma):
        kernel_weights = cv2.getGaussianKernel(ksize=kernel_size, sigma=sigma)
        kernel_weights = kernel_weights * kernel_weights.T
        kernel_weights = np.repeat(kernel_weights[None, ...], self.in_channels, axis=0)[:, None, ...]

        return torch.from_numpy(kernel_weights).float().to(device)

    def forward(self, x):
        G = x
        
        # Level 1
        L0 = Rearrange('b d h w -> b d (h w)')(G)
        L0_att= F.softmax(L0, dim=2) @ L0.transpose(1, 2)  # L_k * L_v
        L0_att = F.softmax(L0_att, dim=-1)
        
        # Next Levels
        attention_maps = [L0_att]
        pyramid = [G]
        
        for kernel in self.sigma_kernels:
            G = F.conv2d(input=G, weight=kernel, bias=None, padding='same', groups=self.in_channels)
            pyramid.append(G)
        
        for i in range(1, self.pyramid_levels):
            L = torch.sub(pyramid[i - 1], pyramid[i])
            L = Rearrange('b d h w -> b d (h w)')(L)
            L_att= F.softmax(L, dim=2) @ L.transpose(1, 2) 
            L_att = F.softmax(L_att, dim=-1) # Check it????
            attention_maps.append(L_att)

        return sum(attention_maps)


class DES(nn.Module):
    """
    Diversity-Enhanced Shortcut (DES) based on: "Gu et al.,
    Multi-Scale High-Resolution Vision Transformer for Semantic Segmentation.
    https://github.com/facebookresearch/HRViT
    """
    def __init__(self, in_features, out_features, bias=True, act_func: nn.Module = nn.GELU):
        super().__init__()
        _, self.p = self._decompose(min(in_features, out_features))
        self.k_out = out_features // self.p
        self.k_in = in_features // self.p
        self.proj_right = nn.Linear(self.p, self.p, bias=bias)
        self.act = act_func()
        self.proj_left = nn.Linear(self.k_in, self.k_out, bias=bias)

    def _decompose(self, n):
        assert n % 2 == 0, f"Feature dimension has to be a multiple of 2, but got {n}"
        e = int(math.log2(n))
        e1 = e // 2
        e2 = e - e // 2
        return 2 ** e1, 2 ** e2

    def forward(self, x):
        B = x.shape[:-1]
        x = x.view(*B, self.k_in, self.p)
        x = self.proj_right(x).transpose(-1, -2)
        
        if self.act is not None:
            x = self.act(x)
            
        x = self.proj_left(x).transpose(-1, -2).flatten(-2, -1)

        return x



class EfficientFrequencyAttention(nn.Module):
    """
    args:
        in_channels:    (int) : Embedding Dimension.
        key_channels:   (int) : Key Embedding Dimension,   Best: (in_channels).
        value_channels: (int) : Value Embedding Dimension, Best: (in_channels or in_channels//2). 
        pyramid_levels  (int) : Number of pyramid levels.
    input:
        x : [B, D, H, W]
    output:
        Efficient Attention : [B, D, H, W]
    
    """
    
    def __init__(self, in_channels, key_channels, value_channels, pyramid_levels=3):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1) 
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)
        
        # Build a laplacian pyramid
        self.freq_attention = LaplacianPyramid(in_channels=in_channels, pyramid_levels=pyramid_levels) 
        
        self.conv_dw = nn.Conv3d(in_channels, in_channels, kernel_size=(2, 1, 1), bias=False, groups=in_channels)
                
        
    def forward(self, x):
        n, _, h, w = x.size()
        
        # Efficient Attention
        keys = F.softmax(self.keys(x).reshape((n, self.key_channels, h * w)), dim=2)
        queries = F.softmax(self.queries(x).reshape(n, self.key_channels, h * w), dim=1)
        values = self.values(x).reshape((n, self.value_channels, h * w))          
        context = keys @ values.transpose(1, 2) # dk*dv            
        attended_value = (context.transpose(1, 2) @ queries).reshape(n, self.value_channels, h, w) # n*dv
        eff_attention  = self.reprojection(attended_value)

        # Freqency Attention
        freq_context = self.freq_attention(x)
        freq_attention =  (freq_context.transpose(1, 2) @ queries).reshape(n, self.value_channels , h, w) 
        
        # Attention Aggregation: Efficient Frequency Attention (EF-Att) Block
        attention = torch.cat([eff_attention[:, :, None, ...], freq_attention[:, :, None, ...]], dim=2)
        attention = self.conv_dw(attention)[:, :, 0, ...] 

        return attention


class FrequencyTransformerBlock(nn.Module):
    """
        Input:
            x : [b, (H*W), d], H, W
            
        Output:
            mx : [b, (H*W), d]
    """
    def __init__(self, in_dim, key_dim, value_dim, pyramid_levels=3, token_mlp='mix'):
        super().__init__()
        
        self.in_dim = in_dim 
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientFrequencyAttention(in_channels=in_dim, key_channels=key_dim, value_channels=value_dim,
                                                pyramid_levels=pyramid_levels)
        
        self.norm2 = nn.LayerNorm(in_dim)
        if token_mlp=='mix':
            self.mlp = MixFFN(in_dim, int(in_dim*4))  
        elif token_mlp=='mix_skip':
            self.mlp = MixFFN_skip(in_dim, int(in_dim*4)) 
        else:
            self.mlp = MLP_FFN(in_dim, int(in_dim*4))
        
        self.des = DES(in_features=in_dim, out_features=in_dim, bias=True, act_func=nn.GELU)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        norm_1 = self.norm1(x)
        norm_1 = Rearrange('b (h w) d -> b d h w', h=H, w=W)(norm_1)
        
        attn = self.attn(norm_1)
        attn = Rearrange('b d h w -> b (h w) d')(attn)
        
        # DES Shortcut
        shortcut = self.des(x.reshape(x.shape[0], self.in_dim, -1).permute(0, 2, 1))
                
        tx = x + attn + shortcut
        mx = tx + self.mlp(self.norm2(tx), H, W)
        
        return mx


class Encoder(nn.Module):
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers, pyramid_levels=3, token_mlp='mix_skip'):
        super().__init__()

        patch_specs = [
            (7, 4, 3),
            (3, 2, 1),
            (3, 2, 1),
            (3, 2, 1)
        ]

        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(len(patch_specs)):
            patch_size, stride, padding = patch_specs[i]
            in_channels = in_dim[i - 1] if i > 0 else 3  # Input channels for the first patch_embed
            out_channels = in_dim[i]

            # Patch Embedding
            patch_embed = OverlapPatchEmbeddings(image_size // (2 ** i), patch_size, stride, padding,
                                                 in_channels, out_channels)
            self.patch_embeds.append(patch_embed)

            # Transformer Blocks
            transformer_block = nn.ModuleList([
                FrequencyTransformerBlock(out_channels, key_dim[i], value_dim[i], pyramid_levels, token_mlp)
                for _ in range(layers[i])
            ])
            self.blocks.append(transformer_block)

            # Layer Normalization
            norm = nn.LayerNorm(out_channels)
            self.norms.append(norm)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(len(self.patch_embeds)):
            x, H, W = self.patch_embeds[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            x = self.norms[i](x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


class EfficientAttentionScore(nn.Module):
    """
    args:
        in_channels:    int -> Embedding Dimension 
        key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
        value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2) 
        
    input:
        x -> [B, D, H, W]
    output:
        x -> [B, D, D]
    """
    
    def __init__(self, in_channels, key_channels, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1) 
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        
    def forward(self, x, h, w):
        n, _, h, w = x.size()
        
        keys = F.softmax(self.keys(x).reshape((n, self.key_channels, h * w)), dim=2)
        values = self.values(x).reshape((n, self.value_channels, h * w))
        context = keys @ values.transpose(1, 2) # dk*dv                        

        return context


class SkipConnection(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.att_levels = nn.ModuleList([
            EfficientAttentionScore(in_dim[i], in_dim[0], in_dim[0])
            for i in range(4)
        ])

        self.mlps = nn.ModuleList([
            MixFFN_skip(in_dim[i], int(in_dim[i] * 4))
            for i in range(4)
        ])

        self.norm_mlps = nn.ModuleList([
            nn.LayerNorm(in_dim[i])
            for i in range(4)
        ])

        self.query_convs = nn.ModuleList([
            nn.Conv2d(in_dim[i], in_dim[i], 1)
            for i in range(4)
        ])

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        B, C, _, _ = c1.shape

        att_scores = [att(c, c.shape[2], c.shape[3]) for att, c in zip(self.att_levels, [c1, c2, c3, c4])]

        sum_attentions = sum(att_scores)

        enhanced_features = []

        for i, (query_conv, norm_mlp, mlp) in enumerate(zip(self.query_convs, self.norm_mlps, self.mlps)):
            query = F.softmax(query_conv(inputs[i]).reshape(B, C, -1), dim=1)
            enhanced = inputs[i].reshape(B, C, -1) + (sum_attentions.transpose(1, 2) @ query)
            enhanced = enhanced.reshape(B, inputs[i].shape[1], -1).permute(0, 2, 1) if i > 0 else enhanced.permute(0, 2, 1)
            out = enhanced + mlp(norm_mlp(enhanced), inputs[i].shape[2], inputs[i].shape[3])
            out = Rearrange(f'b (h w) c -> b h w c', h=inputs[i].shape[2])(out)
            enhanced_features.append(out)

        return enhanced_features



class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale == 2 else nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim if dim_scale == 4 else dim//dim_scale
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        if self.dim_scale == 2:
            x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1 w p2) c', p1=2, p2=2, c=C//4)
        else:
            x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1 w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
            
        x = self.norm(x.clone())

        return x


class MyDecoderLayer(nn.Module):
    def __init__(self, input_size, in_out_chan, token_mlp_mode, n_class=9, pyramid_levels=3,
                 norm_layer=nn.LayerNorm, is_last=False, is_first=False):
        """
        Custom decoder layer for a neural network.

        Args:
            input_size (int): The input resolution size.
            in_out_chan (tuple): Tuple containing input, output, key, and value channel dimensions.
            token_mlp_mode: Mode for the token-level MLPs in the transformer blocks.
            n_class (int): Number of output classes (for the last layer).
            norm_layer: Normalization layer (e.g., nn.LayerNorm).
            is_last (bool): Indicates if this is the last layer.
        """
        super().__init__()
        
        dims, out_dim, key_dim, value_dim = in_out_chan
        
        self.concat_linear = None if is_first else nn.Linear(dims * (4 if is_last else 2), out_dim)
        self.expansion_layer = PatchExpand(input_resolution=input_size, dim=out_dim, 
                                           dim_scale=2 if not is_last else 4, norm_layer=norm_layer)
        self.last_layer = nn.Conv2d(out_dim, n_class, 1) if is_last else None
        self.layer_former = nn.ModuleList([FrequencyTransformerBlock(out_dim, key_dim, value_dim, pyramid_levels,
                                                                     token_mlp_mode) for _ in range(2)])  

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x1, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            cat_x = torch.cat([x1, x2], dim=-1)
            cat_linear_x = self.concat_linear(cat_x)
            tran_layers = [cat_linear_x]
            for layer in self.layer_former:
                tran_layers.append(layer(tran_layers[-1], h, w))
    
            if self.last_layer:
                return self.last_layer(self.expansion_layer(tran_layers[-1]).view(b, 4*h, 4*w, -1).permute(0, 3, 1, 2))
            else:
                return self.expansion_layer(tran_layers[-1])
        else:
            return self.expansion_layer(x1)



class LaplacianFormer(nn.Module):
    def __init__(self, num_classes=9, n_skip_bridge=1, pyramid_levels=3, token_mlp_mode="mix_skip"):
        super().__init__()
    
        self.n_skip_bridge = n_skip_bridge
        
        # Encoder configurations
        params = [[64, 128, 320, 512],  # dims
                  [64, 128, 320, 512],  # key_dim
                  [64, 128, 320, 512],  # value_dim
                  [2, 2, 2, 2]]        # layers
        
        self.encoder = Encoder(image_size=224, in_dim=params[0], key_dim=params[1], value_dim=params[2],
                               layers=params[3], pyramid_levels=pyramid_levels, token_mlp=token_mlp_mode)
        
        
        # Skip Connection
        self.skip_bridges = nn.ModuleList([SkipConnection(params[0]) for _ in range(n_skip_bridge)]) 
        
        
        # Decoder configurations
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [[32, 64, 64, 64],     # [dim, out_dim, key_dim, value_dim]
                       [144, 128, 128, 128], 
                       [288, 320, 320, 320], 
                       [512, 512, 512, 512]] 

        self.decoders = nn.ModuleList()
        for i in range(4):
            in_dim = d_base_feat_size * 2**i
            decoder = MyDecoderLayer((in_dim, in_dim), in_out_chan[3-i], token_mlp_mode,
                                     n_class=num_classes, pyramid_levels=pyramid_levels, is_last=(i==3), is_first=(i==0))
            self.decoders.append(decoder)
        
    def forward(self, x):
        # Encoder
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc = self.encoder(x)
                
        # Skip connections
        skip_outputs = []
        for i, skip_bridge in enumerate(self.skip_bridges):
            output_enc = skip_bridge(output_enc)
            skip_outputs.append(output_enc) 
            output_enc = [y.permute(0, 3, 1 ,2) for y in output_enc]
     
        
        # Decoder
        output_enc = skip_outputs[-1]
        b, _, _, c = output_enc[3].shape
        out = self.decoders[0](output_enc[3].view(b,-1,c))        
        out = self.decoders[1](out, output_enc[2])
        out = self.decoders[2](out, output_enc[1])
        out = self.decoders[3](out, output_enc[0])
                
        return out