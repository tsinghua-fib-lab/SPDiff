import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .vnns import *

def get_graph_features(x, context , relative=False):
    """
        x:[B, N, 2]
        context:
            [0] neigh_ped_mask[B, N, k]
            [1] near_ped_idx[B, N, k]
        
    """
    neigh_ped_mask = context[0]
    near_ped_idx = context[1]
    x_neigh = torch.gather(x[:, None, :, :].expand(x.shape[0], x.shape[-2], x.shape[-2], x.shape[-1]), 2, near_ped_idx[...,None].repeat(1,1,1, x.shape[-1])) # B, N, k, 2
    x_neigh[~neigh_ped_mask.bool()] = 0 # B, N, k, 2'
    x_neigh = x_neigh.unsqueeze(-2) # B, N, k, 1, 2'
    x = x.unsqueeze(-2).unsqueeze(-2).repeat(1,1,x_neigh.shape[2],1,1) # B, N, k, 1, 2'
    if relative:
        feature = torch.cat((x_neigh-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous() # [B, 2, 2', N, k]
    else:
        feature = torch.cat((x_neigh, x), dim=3).permute(0, 3, 4, 1, 2).contiguous() # [B, 2, 2', N, k]

    return feature

class diff_feat_encoder(nn.Module):
    def __init__(self, config, dim_in=2, hid_dim=64, out_dim=64, mlp_layers=3, batchnorm=False):
        super().__init__()
        self.config = config
        self.batchnorm = batchnorm
        self.dim_in = dim_in
        self.neigh_lin = VNLinearLeakyReLU(in_channels=dim_in, out_channels=hid_dim//2, dim=5, negative_slope=0.0)
        if config.pooling == 'max':
            self.pool = VNMaxPool(hid_dim//2)
        elif config.pooling == 'mean':
            self.pool = mean_pool
        if mlp_layers==3:
            self.mlp = nn.ModuleList([
                VNLinearLeakyReLU(in_channels=hid_dim//2, out_channels=2*hid_dim//2, dim=4, negative_slope=0.0),
                VNLinearLeakyReLU(in_channels=2*hid_dim//2, out_channels=2*hid_dim//2, dim=4, negative_slope=0.0),
                VNLinearLeakyReLU(in_channels=2*hid_dim//2, out_channels=hid_dim//2, dim=4, negative_slope=0.0)]
            )
        else:
            raise NotImplementedError
        
        self.out_lin = VNLinearLeakyReLU(in_channels=hid_dim//2, out_channels=out_dim, dim=4, negative_slope=0.0)
        if batchnorm:
            self.bn = VNBatchNorm(out_dim)
    
    def forward(self, x, context):
        """
        x:[B, N, 2]
        context:
            [0] neigh_ped_mask[B, N, k]
            [1] near_ped_idx[B, N, k]
        
        """
        assert x.dim()==3
        feat = get_graph_features(x,context)  # [B, 2, 2', N, k]
        assert feat.dim()==5 and feat.shape[2]==2 and feat.shape[1]==self.dim_in
        feat = self.neigh_lin(feat) # [B, dim, 2', N, k]
        feat = self.pool(feat)  # [B, dim, 2', N]
        for layer in self.mlp:
            feat = layer(feat)
        if self.batchnorm:
            output = self.bn(self.out_lin(feat))
        else:
            output = self.out_lin(feat)  # [B, outdim, 2', N]
        
        return output

# class AdaptiveFusion_ver5(Module):
#     def __init__(self, dim_in, dim_out, dim_ctx, dim_time=3):
#         super().__init__()
#         self._layer1 = Linear(dim_in, dim_out)
#         self._layer2 = Linear(dim_ctx + dim_time , dim_out)
#         self._hyper_gate1 = Linear(dim_time + dim_ctx + dim_in, dim_out)
#         self._hyper_gate2 = Linear(dim_time + dim_ctx + dim_in, dim_out)

#     def forward(self, ctx, x, timeemb):
#         assert ctx.shape[:-1]==x.shape[:-1]==timeemb.shape[:-1]
#         in1 = torch.cat((x,ctx,timeemb),dim=-1)
#         ctx_t = torch.cat((ctx,timeemb),dim=-1)
#         gate1 = torch.sigmoid(self._hyper_gate1(in1))

#         gate2 = torch.sigmoid(self._hyper_gate2(in1))

#         ret = self._layer1(x) * gate1 + self._layer2(ctx_t) * gate2
#         return ret


class AF_vnn(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super().__init__()
        self.layer1 = VNLinear(dim_in, dim_out)
        self.layer2 = VNLinear(dim_ctx, dim_out)
        self.gate1 = VNStdFeature_mod(dim_in + dim_ctx, dim_out, negative_slope=0.0)
        # self.gate1_out = VNLinear(dim_in + dim_ctx, dim_out)
        self.gate2 = VNStdFeature_mod(dim_in + dim_ctx, dim_out, negative_slope=0.0)

    def forward(self, ctx, x):
        """
            ctx,x:[B, dim, 2, N]
        """
        assert ctx.dim()==x.dim()==4
        inp = torch.cat((ctx, x), dim=1)
        gate1 = torch.sigmoid(self.gate1(inp)) # B, dim, 1, N
        gate2 = torch.sigmoid(self.gate2(inp)) # B, dim, 1, N

        x_emb = self.layer1(x)
        ctx_emb = self.layer2(ctx)
        assert x_emb.shape[:2]==ctx_emb.shape[:2]==gate1.shape[:2]==gate2.shape[:2]

        ret = x_emb * gate1 + ctx_emb * gate2
        return ret
    
class AF_vnn_inp_hid(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx, dim_hid):
        super().__init__()
        self.layer1 = VNLinear(dim_in, dim_out)
        self.layer2 = VNLinear(dim_ctx, dim_out)
        # self.gate1 = VNStdFeature_mod(dim_in + dim_ctx, dim_out, negative_slope=0.0)
        # self.gate1_out = VNLinear(dim_in + dim_ctx, dim_out)
        # self.gate2 = VNStdFeature_mod(dim_in + dim_ctx, dim_out, negative_slope=0.0)
        self.gate1 = nn.Linear(dim_hid, dim_out)
        self.gate2 = nn.Linear(dim_hid, dim_out)
        

    def forward(self, ctx, x, hid):
        """
            ctx,x:[B, dim, 2, N]
            hid:[B,N, dim_h]
        """
        assert ctx.dim()==x.dim()==4
        inp = torch.cat((ctx, x), dim=1)
        gate1 = torch.sigmoid(self.gate1(hid)).permute(0,2,1).unsqueeze(-2) # B, dim_out, 1, N
        gate2 = torch.sigmoid(self.gate2(hid)).permute(0,2,1).unsqueeze(-2) # B, dim_out, 1, N

        x_emb = self.layer1(x)
        ctx_emb = self.layer2(ctx)
        assert x_emb.shape[:2]==ctx_emb.shape[:2]==gate1.shape[:2]==gate2.shape[:2]

        ret = x_emb * gate1 + ctx_emb * gate2
        return ret

