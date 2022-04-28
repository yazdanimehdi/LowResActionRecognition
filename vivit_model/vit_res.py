
from functools import partial
from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import Block
from resnet import generate_model
'''
H =img height
W = img width
T = video time
tw = tubelet width
th = tubelet height
tt = tubelet time
h = H/th
w = W/tw # h*w: numner of tubelets with unique spatial index
nb = T/tt #number of blocks or tubelets with unique temporal index
'''


class ViViT_FE(nn.Module):
    def __init__(self, spatial_embed_dim=1024, sdepth=4, tdepth=4, vid_dim=(128, 128, 100),
                 num_heads=16, mlp_ratio=2., qkv_bias=True, qk_scale=None, spat_op='cls', tubelet_dim=(3, 4, 4, 4),
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=None, num_classes=26):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            in_chans (int): number of input channels, RGB videos have 3 chanels
            spatial_embed_dim (int): spatial patch embedding dimension
            sdepth (int): depth of spatial transformer
            tdepth(int):depth of temporal transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            spat_op(string): Spatial Transformer output type - pool(Global avg pooling of encded features) or cls(Just CLS token)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            tubelet_dim(tuple): tubelet size (ch,tt,th,tw)
            vid_dim: Original video (H , W, T)
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        temporal_embed_dim = spatial_embed_dim  #### one temporal token embedding dimension is equal to one spatial patch embedding dim
        print("Spatial embed dimension", spatial_embed_dim)
        print("Temporal embed dim:", temporal_embed_dim)
        print("Drop Rate: ", drop_rate)
        print("Attn drop rate: ", attn_drop_rate)
        print("Drop path rate: ", drop_path_rate)
        print("Tubelet dim: ", tubelet_dim)

        c, tt, th, tw = tubelet_dim

        self.Spatial_patch_to_embedding = generate_model(1)
        num_spat_tokens = (vid_dim[0] // th) * (vid_dim[1] // tw)
        self.Spatial_pos_embed = nn.Parameter(
            torch.zeros(1, num_spat_tokens + 1, spatial_embed_dim))  # num joints + 1 for cls token
        self.spatial_cls_token = nn.Parameter(torch.zeros(1, 1, spatial_embed_dim))  # spatial cls token patch embed
        self.spat_op = spat_op

        num_temp_tokens = vid_dim[-1] // tt
        self.Temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_temp_tokens + 1, temporal_embed_dim))  # additional pos embedding zero for class token
        self.temporal_cls_token = nn.Parameter(torch.zeros(1, 1,
                                                           temporal_embed_dim))  # temporal class token patch embed - this token is used for final classification!
        self.pos_drop = nn.Dropout(p=drop_rate)

        sdpr = [x.item() for x in torch.linspace(0, drop_path_rate, sdepth)]  # stochastic depth decay rule
        tdpr = [x.item() for x in torch.linspace(0, drop_path_rate, tdepth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=spatial_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=sdpr[i], norm_layer=norm_layer)
            for i in range(sdepth)])

        self.blocks = nn.ModuleList([
            Block(
                dim=temporal_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=tdpr[i], norm_layer=norm_layer)
            for i in range(tdepth)])

        self.Spatial_norm = norm_layer(spatial_embed_dim)
        self.Temporal_norm = norm_layer(temporal_embed_dim)

        # Classification head
        self.class_head = nn.Sequential(
            nn.LayerNorm(spatial_embed_dim),
            nn.Linear(temporal_embed_dim, num_classes)
        )

    def Spatial_forward_features(self, x, spat_op='cls'):
        # spat_op: 'cls' output is CLS token, otherwise global average pool of attention encoded spatial features

        # Input shape: batch x num_clips x H x W x (tube tempo dim * 3)
        b, nc, ch, H, W, t = x.shape
        x = rearrange(x, 'b nc ch H W t  -> (b nc) ch H W t', )  # for spatial transformer, batch size if b*f
        x = self.Spatial_patch_to_embedding(x)  # all input spatial tokens, op: (b nc) x H/h x W/w x Se

        # Reshape input to pass through encoder blocks
        _, Se, h, w, _ = x.shape
        x = torch.reshape(x, (b * nc, -1, Se))  # batch x num_spatial_tokens(s) x spat_embed_dim
        _, s, _ = x.shape

        class_token = torch.tile(self.spatial_cls_token, (b * nc, 1, 1))  # (B*nc,1,1)
        x = torch.cat((x, class_token), dim=1)  # (B*nc,s+1,spatial_embed)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        # Pass through transformer blocks
        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)

        ###Extract Class token head from the output
        cls_token = x[:, -1, :]
        cls_token = torch.reshape(cls_token, (b, nc, Se))

        # Determine the output type from Spatial transformer
        if spat_op == 'cls':
            return cls_token  # b x nc x Se
        else:
            x = x[:, :s, :]
            x = rearrange(x, '(b nc) s Se -> (b nc) Se s')
            x = F.avg_pool1d(x, x.shape[-1], stride=x.shape[-1])  # (b*nc) x Se x 1
            x = torch.reshape(x, (b, nc, Se))
            return x  # b x nc x Se

    def Temporal_forward_features(self, x):

        b = x.shape[0]
        class_token = torch.tile(self.temporal_cls_token, (b, 1, 1))  # (B,1,temp_embed_dim)
        x = torch.cat((x, class_token), dim=1)  # (B,F+1,temp_embed_dim)

        x += self.Temporal_pos_embed

        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)

        ###Extract Class token head from the output
        x = x[:, -1, :]
        x = x.view(b, -1)  # (Batch_size, class_embedding_size)
        return x

    def forward(self, x):
        x = x.permute(0, 1, 2, 4, 5, 3)
        # Input x: batch x num_clips x num_chans x img_height x img_width x tubelet_time
        # nc should be T/tt
        b, nc, ch, H, W, t = x.shape

        # Reshape input to pass through Conv3D patch embedding
        x = self.Spatial_forward_features(x, self.spat_op)  # b x nc x Se
        x = self.Temporal_forward_features(x)

        return x


class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        out_features = 26
        hidden_features = 4096
        self.fc1 = nn.Linear(1024, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, training=True):
        x = self.fc1(x)
        x = self.act(x)
        x = F.dropout(x, training=training, p=0.2)
        x = self.fc2(x)
        # x = self.drop(x)
        return x
