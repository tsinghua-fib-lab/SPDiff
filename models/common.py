import torch
from torch.nn import Module, Linear
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import torch.nn as nn
import math

def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'approx_gelu':
            self.activation = nn.GELU(approximate="tanh")
        else:
            raise NotImplementedError

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
            else:
                x = self.activation(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_dim, hidden_units, activation, dropout=-1, use_bn=False):
        super(ResBlock, self).__init__()
        self.use_bn = use_bn
        self.activation = activation

        self.lin = MLP(in_dim, hidden_units[-1], hidden_units[:-1], activation, dropout=dropout)
        if self.use_bn:
            raise NotImplementedError('bn in resblock has not been implemented!')

    def forward(self, x):
        return self.lin(x) + x

class ResDNN(nn.Module):
    """The Multi Layer Percetron with Residuals
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most
          common situation would be a 2D input with shape
          ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``.
          For instance, for a 2D input with shape ``(batch_size, input_dim)``,
          the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **hidden_units**:list of list, which contains the layer number and
          units in each layer.
            - e.g., [[5], [5,5], [5,5]]
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied
          to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation.
    """

    def __init__(self, input_dim, hidden_units, activation='relu', dropout=-1, use_bn=False):
        super(ResDNN, self).__init__()
        if input_dim != hidden_units[0][0]:
            raise ValueError('In ResBlock, the feature size must be equal to the hidden \
                size! input_dim:{}, hidden_size: {}'.format(input_dim, hidden_units[0]))
        self.dropout = nn.Dropout(dropout if dropout>0 else 0)
        self.use_bn = use_bn
        self.hidden_units = hidden_units
        self.hidden_units[0] = [input_dim] + self.hidden_units[0]
        self.resnet = nn.ModuleList(
            [ResBlock(h[0], h[1:], activation, use_bn) for h in hidden_units])

    def forward(self, x):
        for i in range(len(self.hidden_units)):
            out = self.resnet[i](x)
            out = self.dropout(out)
        return out

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret

gatex_mean = torch.tensor([])
gatec_mean = torch.tensor([])
  
class AdaptiveFusion(Module):
    def __init__(self, dim_in, dim_out, dim_ctx, dim_time=3):
        super().__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate1 = Linear(dim_time, dim_out)
        self._hyper_gate2 = Linear(dim_time, dim_out)

    def forward(self, ctx, x, timeemb):
        gate1 = torch.sigmoid(self._hyper_gate1(timeemb))
        global gatex_mean
        gatex_mean = torch.mean(gate1).unsqueeze(0)
        gate2 = torch.sigmoid(self._hyper_gate2(timeemb))
        global gatec_mean
        gatec_mean = torch.mean(gate2).unsqueeze(0)
        
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate1 + self._hyper_bias(ctx) * gate2
        return ret

class AdaptiveFusion_ver4(Module):
    def __init__(self, dim_in, dim_out, dim_ctx, dim_time=3):
        super().__init__()
        self._layer1 = Linear(dim_in, dim_out)
        self._layer2 = Linear(dim_ctx + dim_time , dim_out)
        self._hyper_gate1 = Linear(dim_time + dim_ctx, dim_out)
        self._hyper_gate2 = Linear(dim_in, dim_out)

    def forward(self, ctx, x, timeemb):
        assert ctx.shape[:-1]==x.shape[:-1]==timeemb.shape[:-1]
        ctx_t = torch.cat((ctx,timeemb),dim=-1)
        gate1 = torch.sigmoid(self._hyper_gate1(ctx_t))
        global gatex_mean
        gatex_mean = torch.mean(gate1).unsqueeze(0)
        gate2 = torch.sigmoid(self._hyper_gate2(x))
        global gatec_mean
        gatec_mean = torch.mean(gate2).unsqueeze(0)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer1(x) * gate1 + self._layer2(ctx_t) * gate2
        return ret

class AdaptiveFusion_ver4_nogateed(Module):
    def __init__(self, dim_in, dim_out, dim_ctx, dim_time=3):
        super().__init__()
        self._layer1 = Linear(dim_in, dim_out)
        self._layer2 = Linear(dim_ctx + dim_time , dim_out)
        # self._hyper_gate1 = Linear(dim_time + dim_ctx, dim_out)
        self._hyper_gate2 = Linear(dim_in, dim_out)

    def forward(self, ctx, x, timeemb):
        assert ctx.shape[:-1]==x.shape[:-1]==timeemb.shape[:-1]
        ctx_t = torch.cat((ctx,timeemb),dim=-1)
        # gate1 = torch.sigmoid(self._hyper_gate1(ctx_t))
        global gatex_mean
        # gatex_mean = torch.mean(gate1).unsqueeze(0)
        gate2 = torch.sigmoid(self._hyper_gate2(x))
        global gatec_mean
        gatec_mean = torch.mean(gate2).unsqueeze(0)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer1(x) + self._layer2(ctx_t) * gate2
        return ret


class AdaptiveFusion_ver5(Module):
    def __init__(self, dim_in, dim_out, dim_ctx, dim_time=3):
        super().__init__()
        self._layer1 = Linear(dim_in, dim_out)
        self._layer2 = Linear(dim_ctx + dim_time , dim_out)
        self._hyper_gate1 = Linear(dim_time + dim_ctx + dim_in, dim_out)
        self._hyper_gate2 = Linear(dim_time + dim_ctx + dim_in, dim_out)

    def forward(self, ctx, x, timeemb):
        assert ctx.shape[:-1]==x.shape[:-1]==timeemb.shape[:-1]
        in1 = torch.cat((x,ctx,timeemb),dim=-1)
        ctx_t = torch.cat((ctx,timeemb),dim=-1)
        gate1 = torch.sigmoid(self._hyper_gate1(in1))
        global gatex_mean
        gatex_mean = torch.mean(gate1).unsqueeze(0)
        gate2 = torch.sigmoid(self._hyper_gate2(in1))
        global gatec_mean
        gatec_mean = torch.mean(gate2).unsqueeze(0)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer1(x) * gate1 + self._layer2(ctx_t) * gate2
        return ret

class AdaptiveFusion_ver5_nogateed(Module):
    def __init__(self, dim_in, dim_out, dim_ctx, dim_time=3):
        super().__init__()
        self._layer1 = Linear(dim_in, dim_out)
        self._layer2 = Linear(dim_ctx + dim_time , dim_out)
        self._hyper_gate1 = Linear(dim_time + dim_ctx + dim_in, dim_out)
        self._hyper_gate2 = Linear(dim_time + dim_ctx + dim_in, dim_out)

    def forward(self, ctx, x, timeemb):
        assert ctx.shape[:-1]==x.shape[:-1]==timeemb.shape[:-1]
        in1 = torch.cat((x,ctx,timeemb),dim=-1)
        ctx_t = torch.cat((ctx,timeemb),dim=-1)
        gate1 = torch.sigmoid(self._hyper_gate1(in1))
        global gatex_mean
        gatex_mean = torch.mean(gate1).unsqueeze(0)
        gate2 = torch.sigmoid(self._hyper_gate2(in1))
        global gatec_mean
        gatec_mean = torch.mean(gate2).unsqueeze(0)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer1(x) + self._layer2(ctx_t) * gate2
        return ret

class AdaptiveFusion_ver1(Module):
    def __init__(self, dim_in, dim_out, dim_ctx, dim_time=3):
        super().__init__()
        self._layer1 = Linear(dim_in, dim_out)
        self._layer2 = Linear(dim_ctx, dim_out)
        self._hyper_gate1 = Linear(dim_time + dim_ctx, dim_out)
        self._hyper_gate2 = Linear(dim_time + dim_in, dim_out)

    def forward(self, ctx, x, timeemb):
        assert ctx.shape[:-1]==x.shape[:-1]==timeemb.shape[:-1]
        in1 = torch.cat((ctx,timeemb),dim=-1)
        in2 = torch.cat((x,timeemb),dim=-1)
        gate1 = torch.sigmoid(self._hyper_gate1(in1))
        gate2 = torch.sigmoid(self._hyper_gate2(in2))
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer1(x) * gate1 + self._layer2(ctx) * gate2
        return ret

class AdaptiveFusion_ver1_2(Module):
    def __init__(self, dim_in, dim_out, dim_ctx, dim_time=3):
        super().__init__()
        self._layer1 = Linear(dim_in, dim_out)
        self._layer2 = Linear(dim_ctx, dim_out)
        self._hyper_gate1 = Linear(dim_time + dim_ctx + dim_in, dim_out)
        self._hyper_gate2 = Linear(dim_time + dim_in + dim_ctx, dim_out)

    def forward(self, ctx, x, timeemb):
        assert ctx.shape[:-1]==x.shape[:-1]==timeemb.shape[:-1]
        in1 = torch.cat((x,ctx,timeemb),dim=-1)

        gate1 = torch.sigmoid(self._hyper_gate1(in1))
        gate2 = torch.sigmoid(self._hyper_gate2(in1))
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer1(x) * gate1 + self._layer2(ctx) * gate2
        return ret

class AdaptiveFusion_ver1_3(Module):
    def __init__(self, dim_in, dim_out, dim_ctx, dim_time=3):
        super().__init__()
        self._layer1 = Linear(dim_in, dim_out)
        self._layer2 = Linear(dim_ctx, dim_out)
        self._hyper_gate1 = Linear(dim_time + dim_ctx + dim_in, dim_out)
        # self._hyper_gate2 = Linear(dim_time + dim_in + dim_ctx, dim_out)

    def forward(self, ctx, x, timeemb):
        assert ctx.shape[:-1]==x.shape[:-1]==timeemb.shape[:-1]
        in1 = torch.cat((x,ctx,timeemb),dim=-1)

        gate1 = torch.sigmoid(self._hyper_gate1(in1))
        gate2 = 1 - gate1
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer1(x) * gate1 + self._layer2(ctx) * gate2
        return ret

class AdaptiveFusion2(Module):
    def __init__(self, dim_in, dim_out, dim_ctx, dim_time=3):
        super(AdaptiveFusion2, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out)
        self._hyper_gate1 = Linear(dim_time, dim_out)
        self._hyper_gate2 = Linear(dim_time, dim_out)

    def forward(self, ctx, x, timeemb):
        gate1 = torch.sigmoid(self._hyper_gate1(timeemb))
        gate2 = torch.ones_like(gate1)
        gate2 = gate2-gate1
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate1 + self._hyper_bias(ctx) * gate2
        return ret
from einops import *
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=1, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class DITBlock(nn.Module):
    def __init__(self, hidden_size, cond_dim, num_heads, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.crossattn = CrossAttention(hidden_size, cond_dim, heads = num_heads, dim_head=64)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # approx_gelu = nn.GELU(approximate="tanh")
        self.mlp = MLP(input_dim=hidden_size, output_dim=hidden_size, hidden_size=(mlp_hidden_dim,),activation='relu')
    def forward(self, cond, x_in):
        assert x_in.ndim==3 and cond.ndim==3,'dim wrong in DIT'
        x = self.norm1(x_in).permute(1,0,2)
        x = self.attn(query=x,key=x,value=x)[0].permute(1,0,2)+x_in
        x = self.crossattn(self.norm2(x),cond)+x
        x = self.mlp(self.norm3(x))+x
        return x
        
        
class ConcatTransformerLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatTransformerLinear, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_in, nhead=8)
        #self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        # x: (B*12*2)
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self.encoder_layer(x) * gate + bias
        return ret

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1-frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr
    return LambdaLR(optimizer, lr_lambda=lr_func)

def lr_func(epoch):
    if epoch <= start_epoch:
        return 1.0
    elif epoch <= end_epoch:
        total = end_epoch - start_epoch
        delta = epoch - start_epoch
        frac = delta / total
        return (1-frac) * 1.0 + frac * (end_lr / start_lr)
    else:
        return end_lr / start_lr
