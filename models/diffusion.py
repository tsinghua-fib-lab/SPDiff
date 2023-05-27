import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import torch.nn as nn
from .common import *
import pdb
# from utils.TrajectoryDS import anorm
from utils.utils import mask_mse_func
from tqdm.auto import tqdm
from .egnn import *
from .vnn_models import *

plot_beta_gate = False
beta_global = torch.tensor([])
gatex_mean1_global = torch.tensor([])
gatec_mean1_global = torch.tensor([])
gatex_mean2_global = torch.tensor([])
gatec_mean2_global = torch.tensor([])
class VarianceSchedule(Module):

    def __init__(self, num_steps, mode='linear',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i] # posterior variance
        sigmas_inflex = torch.sqrt(sigmas_inflex)
        posterior_mean_coef1 = torch.zeros_like(sigmas_flex)
        posterior_mean_coef2 = torch.zeros_like(sigmas_flex)
        for i in range(1, posterior_mean_coef1.size(0)):
            posterior_mean_coef1[i] = torch.sqrt(alpha_bars[i-1]) * betas[i] / (1 - alpha_bars[i])
        posterior_mean_coef1[0] = 1.0
        for i in range(1, posterior_mean_coef2.size(0)):
            posterior_mean_coef2[i] = torch.sqrt(alphas[i]) * (1 - alpha_bars[i-1]) / (1 - alpha_bars[i])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex) 
        self.register_buffer('sigmas_inflex', sigmas_inflex)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas

class TrajNet(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(2, 128, context_dim+3),
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3),
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, 2, context_dim+3),

        ])

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        #pdb.set_trace()
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


class TransformerConcatLinear(Module):

    def __init__(self, point_dim, context_dim, tf_layer, residual):
        super().__init__()
        self.residual = residual
        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
        self.concat1 = ConcatSquashLinear(2,2*context_dim,context_dim+3)
        self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=4, dim_feedforward=4*context_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
        self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
        self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
        self.linear = ConcatSquashLinear(context_dim//2, 2, context_dim+3)
        #self.linear = nn.Linear(128,2)


    def forward(self, x, beta, context):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
        x = self.concat1(ctx_emb,x)
        final_emb = x.permute(1,0,2)
        final_emb = self.pos_emb(final_emb)


        trans = self.transformer_encoder(final_emb).permute(1,0,2)
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)

# net 2 use
class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.nhead = nhead
        self.src_mask = None
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

    def forward(self, src, mask=None):
        # n_mask = mask 
        # n_mask = n_mask.float().masked_fill(n_mask == 0., float(-1e20)).masked_fill(n_mask == 1., float(0.0))
        src = src.permute(1,0,2)
        if mask is not None:
            assert mask.dim()==3
            mask = mask.repeat([self.nhead, 1, 1])
            mask = mask.float().masked_fill(mask == 0., float(-1e20)).masked_fill(mask == 1., float(0.0))
        output = self.transformer_encoder(src, mask) #TODO
        output = output.permute(1,0,2)
        return output
    
class TransformerModel2(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.3):
        super(TransformerModel2, self).__init__()
        self.model_type = 'Transformer'
        self.nhead = nhead
        self.src_mask = None
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

    def forward(self, src, mask=None):
        # n_mask = mask 
        # n_mask = n_mask.float().masked_fill(n_mask == 0., float(-1e20)).masked_fill(n_mask == 1., float(0.0))
        src = src.permute(1,0,2)
        if mask is not None:
            assert mask.dim()==2 # BS, seq_len
            
        output = self.transformer_encoder(src, src_key_padding_mask=mask) #TODO
        output = output.permute(1,0,2)
        return output

class SpatialTransformer(Module):
    def __init__(self, config):
        super().__init__()
        self.spatial_encoder=TransformerModel(config.spatial_emsize, config.spatial_encoder_head,
                                              config.spatial_emsize, config.spatial_encoder_layers,
                                              config.dropout)
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        self.concat1 = ConcatSquashLinear(config.spatial_emsize, 2*config.spatial_emsize, config.context_dim+3)
        self.concat2 = ConcatSquashLinear(2*config.spatial_emsize, config.spatial_emsize, config.context_dim+3)
        self.concat3 = ConcatSquashLinear(config.spatial_emsize, 2, config.context_dim+3)
        self.decode1 = nn.Sequential(
            nn.Linear(2*config.spatial_emsize, config.spatial_emsize),nn.ReLU(),
            nn.Linear(config.spatial_emsize, config.spatial_emsize),nn.ReLU(),
            nn.Linear(config.spatial_emsize, 2)
        )
        
        # context [bs, obs_len+1(destination), N, 2]
        # history encoder
        self.history_encoder = nn.Sequential(nn.Linear(config.history_dim, config.history_emsize),
                                          nn.ReLU())
        #                                 #   nn.Dropout(config.dropout))
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm)
        # destination encoder
        self.dest_encoder = nn.Sequential(nn.Linear(2, config.dest_emsize),
                                          nn.ReLU(),
                                          nn.Dropout(config.dropout))
                                        #   nn.Linear(config.dest_emsize, config.dest_emsize),
                                        #   nn.Sigmoid())
        # self.dest_encoder = nn.Linear(2, config.dest_emsize)
        assert config.dest_emsize+config.history_lstm ==config.context_dim
        
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.history_encoder(hist_feature) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        hist_embedded,(_,_) = self.history_LSTM(hist_embedded)
        hist_embedded = hist_embedded.view(*origin_shape[:-1],-1)[...,-1,:] # bs,N,his_len,embsize, use the end of seq
        # hist_embedded = self.lstm_output(hist_embedded)
        
        dest_feature = context[1]
        dest_feature = self.dest_encoder(dest_feature) # bs,N,embsize
        
        context_emb0 = torch.cat((dest_feature, hist_embedded),dim=-1) #bs,N,embsize_sum
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list)
        # spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)[-1]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,context_emb0.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        assert time_emb.shape[:-1]==context_emb0.shape[:-1], 'context emb shape not match !'
        context_emb = torch.cat((context_emb0, time_emb),dim=-1) # (B,N,F+3)
        spatial_input_embedded = self.concat1(context_emb, spatial_input_embedded)
        output = self.decode1(spatial_input_embedded)
        # spatial_input_embedded=self.concat2(context_emb, spatial_input_embedded)
        # output = self.concat3(context_emb, spatial_input_embedded)
        assert output.shape[-1]==2, 'wrong code!'
        return output
# end net 2 use

class SpatialTransformer2(Module):
    def __init__(self, config):
        super().__init__()
        self.spatial_encoder=TransformerModel(config.spatial_emsize, config.spatial_encoder_head,
                                              config.spatial_emsize, config.spatial_encoder_layers,
                                              config.dropout)
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        self.concat1 = ConcatSquashLinear(config.spatial_emsize, 2*config.spatial_emsize, config.context_dim+3)
        self.concat2 = ConcatSquashLinear(2*config.spatial_emsize, config.spatial_emsize, config.context_dim+3)
        self.concat3 = ConcatSquashLinear(config.spatial_emsize, 2, config.context_dim+3)
        self.decode1 = nn.Sequential(
            nn.Linear(2*config.spatial_emsize, config.spatial_emsize),nn.ReLU(),
            nn.Linear(config.spatial_emsize, config.spatial_emsize),nn.ReLU(),
            nn.Linear(config.spatial_emsize, 2)
        )
        
        # context [bs, obs_len+1(destination), N, 2]
        # history encoder
        self.history_encoder = nn.Sequential(nn.Linear(config.history_dim, config.history_emsize),
                                          nn.ReLU())
        #                                 #   nn.Dropout(config.dropout))
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm)
        # destination encoder
        # self.dest_encoder = nn.Sequential(nn.Linear(2, config.dest_emsize),
        #                                   nn.ReLU(),
        #                                   nn.Dropout(config.dropout))
                                        #   nn.Linear(config.dest_emsize, config.dest_emsize),
                                        #   nn.Sigmoid())
        self.dest_encoder = MLP(input_dim=2, output_dim=config.dest_emsize, hidden_size=config.dest_hidden)
        assert config.dest_emsize+config.history_lstm ==config.context_dim
        
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.history_encoder(hist_feature) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        hist_embedded,(_,_) = self.history_LSTM(hist_embedded)
        hist_embedded = hist_embedded.view(*origin_shape[:-1],-1)[...,-1,:] # bs,N,his_len,embsize, use the end of seq
        hist_embedded = self.lstm_output(hist_embedded)
        
        dest_feature = context[1]
        dest_feature = self.dest_encoder(dest_feature) # bs,N,embsize
        
        context_emb0 = torch.cat((dest_feature, hist_embedded),dim=-1) #bs,N,embsize_sum
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list)
        # spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)[-1]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,context_emb0.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        assert time_emb.shape[:-1]==context_emb0.shape[:-1], 'context emb shape not match !'
        context_emb = torch.cat((context_emb0, time_emb),dim=-1) # (B,N,F+3)
        spatial_input_embedded = self.concat1(context_emb, spatial_input_embedded)
        output = self.decode1(spatial_input_embedded)
        # spatial_input_embedded=self.concat2(context_emb, spatial_input_embedded)
        # output = self.concat3(context_emb, spatial_input_embedded)
        assert output.shape[-1]==2, 'wrong code!'
        return output
# end net 2 use

class SpatialTransformer3(Module):
    def __init__(self, config):
        super().__init__()
        self.spatial_encoder=TransformerModel(config.spatial_emsize, config.spatial_encoder_head,
                                              config.spatial_emsize, config.spatial_encoder_layers,
                                              config.dropout)
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        self.concat1 = ConcatSquashLinear(config.spatial_emsize, 2*config.spatial_emsize, config.context_dim+3)
        self.concat2 = ConcatSquashLinear(2*config.spatial_emsize, config.spatial_emsize, config.context_dim+3)
        self.concat3 = ConcatSquashLinear(config.spatial_emsize, 2, config.context_dim+3)
        self.decode1 = nn.Sequential(
            nn.Linear(2*config.spatial_emsize, config.spatial_emsize),nn.ReLU(),
            nn.Linear(config.spatial_emsize, config.spatial_emsize),nn.ReLU(),
            nn.Linear(config.spatial_emsize, 2)
        )
        
        # context [bs, obs_len+1(destination), N, 2]
        # history encoder
        self.history_encoder = nn.Sequential(nn.Linear(config.history_dim, config.history_emsize),
                                          nn.ReLU())
        #                                 #   nn.Dropout(config.dropout))
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm)
        # destination encoder
        # self.dest_encoder = nn.Sequential(nn.Linear(2, config.dest_emsize),
        #                                   nn.ReLU(),
        #                                   nn.Dropout(config.dropout))
                                        #   nn.Linear(config.dest_emsize, config.dest_emsize),
                                        #   nn.Sigmoid())
        self.dest_encoder = MLP(input_dim=2, output_dim=config.dest_emsize, hidden_size=config.dest_hidden)
        assert config.dest_emsize+config.history_lstm ==config.context_dim
        
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.history_encoder(hist_feature) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        hist_embedded,(_,_) = self.history_LSTM(hist_embedded)
        hist_embedded = hist_embedded.view(*origin_shape[:-1],-1)[...,-1,:] # bs,N,his_len,embsize, use the end of seq
        # hist_embedded = self.lstm_output(hist_embedded)
        
        dest_feature = context[1]
        dest_feature = self.dest_encoder(dest_feature) # bs,N,embsize
        
        context_emb0 = torch.cat((dest_feature, hist_embedded),dim=-1) #bs,N,embsize_sum
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list)
        # spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)[-1]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,context_emb0.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        assert time_emb.shape[:-1]==context_emb0.shape[:-1], 'context emb shape not match !'
        context_emb = torch.cat((context_emb0, time_emb),dim=-1) # (B,N,F+3)
        spatial_input_embedded = self.concat1(context_emb, spatial_input_embedded)
        output = self.decode1(spatial_input_embedded)
        # spatial_input_embedded=self.concat2(context_emb, spatial_input_embedded)
        # output = self.concat3(context_emb, spatial_input_embedded)
        assert output.shape[-1]==2, 'wrong code!'
        return output
# end net 2 use

class SpatialTransformer4(Module):
    def __init__(self, config):
        super().__init__()
        config.context_dim = config.dest_emsize + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat1 = ConcatSquashLinear(2, 2*config.spatial_emsize, config.context_dim+3)
        self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        self.concat3 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize//2, config.context_dim)
        self.decode1 = AdaptiveFusion(config.spatial_emsize//2, 2, config.context_dim)
        
        
        # context [bs, obs_len+1(destination), N, 2]
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        # destination encoder
        # self.dest_encoder = nn.Sequential(nn.Linear(2, config.dest_emsize),
        #                                   nn.ReLU(),
        #                                   nn.Dropout(config.dropout))
                                        #   nn.Linear(config.dest_emsize, config.dest_emsize),
                                        #   nn.Sigmoid())
        # self.dest_encoder = MLP(input_dim=2, output_dim=config.dest_emsize, hidden_size=config.dest_hidden)
        assert config.dest_emsize+config.history_lstm_out ==config.context_dim
        
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize, use the end of seq
        hist_embedded = self.lstm_output(hist_embedded)
        
        # dest_feature = context[1]
        # dest_feature = self.dest_encoder(dest_feature) # bs,N,embsize
        
        # context_emb0 = torch.cat((dest_feature, hist_embedded),dim=-1) #bs,N,embsize_sum
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        
        # spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)[-1]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        # context_emb = torch.cat((hist_embedded, time_emb),dim=-1) # (B,N,F+3)
        
        # spatial_input_embedded = self.concat1(context_emb, x)
        spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list)
        spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        spatial_input_embedded = self.concat3(hist_embedded, spatial_input_embedded, time_emb)
        output = self.decode1(hist_embedded, spatial_input_embedded, time_emb)
        # spatial_input_embedded=self.concat2(context_emb, spatial_input_embedded)
        # output = self.concat3(context_emb, spatial_input_embedded)
        assert output.shape[-1]==2, 'wrong code!'
        return output
# end net 2 use

class SpatialTransformer_ped_inter_2(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        self.concat_ped1 = CrossAttention(query_dim = config.spatial_emsize,
                                          context_dim = config.context_dim+3,
                                          dim_head = config.spatial_emsize,
                                          dropout = config.dropout) #
        # self.decode_ped1 = AdaptiveFusion(config.spatial_emsize, 2, config.context_dim)
        # self.decode_ped1 = nn.Sequential(nn.Linear(config.spatial_emsize, config.spatial_emsize//2),
        #                                  nn.ReLU(),
        #                                  nn.Linear(config.spatial_emsize//2, 2))
        self.decode_ped1 = nn.Linear(config.spatial_emsize, 2)
        # self.decode_ped2 = AdaptiveFusion(config.spatial_emsize//2, 2, config.context_dim)
        # self.decode1 = Linear(config.spatial_emsize//2, 2)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        spatial_input_embedded = self.concat_ped1(spatial_input_embedded, torch.cat((context_ped, time_emb),dim=-1))
        # output_ped = self.decode_ped1(context_ped, spatial_input_embedded, time_emb)
        output_ped = self.decode_ped1(spatial_input_embedded)
        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        self.concat_ped1 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize//2, config.context_dim)
        self.decode_ped1 = AdaptiveFusion(config.spatial_emsize//2, 2, config.context_dim)
        # self.decode1 = Linear(config.spatial_emsize//2, 2)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)

        spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB

        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)
        output_ped = self.decode_ped1(context_ped, spatial_input_embedded, time_emb)
        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_notrans(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        self.concat_ped1 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize//2, config.context_dim)
        self.decode_ped1 = AdaptiveFusion(config.spatial_emsize//2, 2, config.context_dim)
        # self.decode1 = Linear(config.spatial_emsize//2, 2)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        global beta_global
        beta_global = torch.cat((beta_global,beta.cpu()),dim=-1) #for debug
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)
        from .common import gatec_mean as gatec_mean1
        from .common import gatex_mean as gatex_mean1
        global gatec_mean1_global
        global gatex_mean1_global
        gatec_mean1_global = torch.cat((gatec_mean1_global,gatec_mean1.cpu()),dim=-1)
        gatex_mean1_global = torch.cat((gatex_mean1_global,gatex_mean1.cpu()),dim=-1)
        output_ped = self.decode_ped1(context_ped, spatial_input_embedded, time_emb)
        from .common import gatec_mean as gatec_mean2
        from .common import gatex_mean as gatex_mean2
        global gatec_mean2_global
        global gatex_mean2_global
        gatec_mean2_global = torch.cat((gatec_mean2_global,gatec_mean2.cpu()),dim=-1)
        gatex_mean2_global = torch.cat((gatex_mean2_global,gatex_mean2.cpu()),dim=-1)
        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_notrans_wohistory_ver2(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        config.context_dim = config.ped_encode_dim2
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        self.concat_ped1 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize//2, config.context_dim)
        # self.decode_ped1 = AdaptiveFusion(config.spatial_emsize//2, 2, config.context_dim)
        self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        # self.decode1 = Linear(config.spatial_emsize//2, 2)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        # hist_feature = context[0]
        # hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        # origin_shape = hist_embedded.shape
        # hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        # _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        # hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        # hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        # context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        context_ped = ped_emb
        
        # global beta_global
        # beta_global = torch.cat((beta_global,beta.cpu()),dim=-1) #for debug
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)
        
        output_ped = self.decode_ped1(spatial_input_embedded)
        
        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_notrans_ver2(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        self.concat_ped1 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize//2, config.context_dim)
        self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        # self.decode1 = Linear(config.spatial_emsize//2, 2)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        global beta_global
        beta_global = torch.cat((beta_global,beta.cpu()),dim=-1) #for debug
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)
        from .common import gatec_mean as gatec_mean1
        from .common import gatex_mean as gatex_mean1
        global gatec_mean1_global
        global gatex_mean1_global
        gatec_mean1_global = torch.cat((gatec_mean1_global,gatec_mean1.cpu()),dim=-1)
        gatex_mean1_global = torch.cat((gatex_mean1_global,gatex_mean1.cpu()),dim=-1)
        output_ped = self.decode_ped1(spatial_input_embedded)
        from .common import gatec_mean as gatec_mean2
        from .common import gatex_mean as gatex_mean2
        global gatec_mean2_global
        global gatex_mean2_global
        gatec_mean2_global = torch.cat((gatec_mean2_global,gatec_mean2.cpu()),dim=-1)
        gatex_mean2_global = torch.cat((gatex_mean2_global,gatex_mean2.cpu()),dim=-1)
        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_obs_inter_notrans_ver2(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
            self.obs_encode_flag=False
            config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        else:
            self.tau=2
            self.obs_encode_flag=True
            config.context_dim = config.ped_encode_dim2 + config.history_lstm_out + config.obs_encode_dim2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out + config.obs_encode_dim2
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        self.concat_ped1 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize//2, config.context_dim)
        self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        # self.decode1 = Linear(config.spatial_emsize//2, 2)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        # obs interaction encoder
        if self.obs_encode_flag:
            self.obs_encoder1 = MLP(input_dim=6, output_dim=config.obs_encode_dim1, hidden_size=config.obs_encode_hid1)
            self.obs_encoder2 = ResDNN(input_dim=config.obs_encode_dim1, hidden_units=[[config.obs_encode_dim2]]+[[config.obs_encode_dim2]*2]*(config.obs_encode2_layers-1))
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        if self.obs_encode_flag:
            obs_features = context[3] #bs, N, k, 6
            obs_emb = self.obs_encoder1(obs_features)
            obs_emb = self.obs_encoder2(obs_emb) #bs, N, k, dim
            obs_emb = torch.sum(obs_emb, dim=-2) #bs, N, dim
            context_ped = torch.cat((hist_embedded,ped_emb,obs_emb), dim=-1)
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)

        output_ped = self.decode_ped1(spatial_input_embedded)

        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_notrans_ver2_csql(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        self.concat_ped1 = ConcatSquashLinear(config.spatial_emsize, config.spatial_emsize//2, config.context_dim+3)
        self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        # self.decode1 = Linear(config.spatial_emsize//2, 2)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        global beta_global
        beta_global = torch.cat((beta_global,beta.cpu()),dim=-1) #for debug
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
        spatial_input_embedded = self.concat_ped1(torch.cat((context_ped,time_emb),dim=-1), spatial_input_embedded)
        # output_ped = spatial_input_embedded
        output_ped = self.decode_ped1(spatial_input_embedded)

        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_notrans_ver2_csql2(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        self.concat_ped1 = ConcatSquashLinear(config.spatial_emsize, config.spatial_emsize//2, config.context_dim+3)
        self.decode_ped1 = ConcatSquashLinear(config.spatial_emsize//2, 2, config.context_dim+3)
        
        # self.decode1 = Linear(config.spatial_emsize//2, 2)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        global beta_global
        beta_global = torch.cat((beta_global,beta.cpu()),dim=-1) #for debug
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
        spatial_input_embedded = self.concat_ped1(torch.cat((context_ped,time_emb),dim=-1), spatial_input_embedded)
        # output_ped = spatial_input_embedded
        output_ped = self.decode_ped1(torch.cat((context_ped,time_emb),dim=-1), spatial_input_embedded)

        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_notrans_ver3(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        # self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        # self.relu = nn.ReLU()
        # self.dropout_in = nn.Dropout(config.dropout)
        self.diffusedSampleEnc = MLP(input_dim=2, output_dim=config.spatial_emsize, 
                                     hidden_size=[config.spatial_emsize]*(config.yt_encoder_layers-1))
        
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        self.concat_ped1 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize//2, config.context_dim)
        self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        # self.decode1 = Linear(config.spatial_emsize//2, 2)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        global beta_global
        beta_global = torch.cat((beta_global,beta.cpu()),dim=-1) #for debug
        
        # spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        spatial_input_embedded = self.diffusedSampleEnc(x)
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)
        from .common import gatec_mean as gatec_mean1
        from .common import gatex_mean as gatex_mean1
        global gatec_mean1_global
        global gatex_mean1_global
        gatec_mean1_global = torch.cat((gatec_mean1_global,gatec_mean1.cpu()),dim=-1)
        gatex_mean1_global = torch.cat((gatex_mean1_global,gatex_mean1.cpu()),dim=-1)
        output_ped = self.decode_ped1(spatial_input_embedded)
        from .common import gatec_mean as gatec_mean2
        from .common import gatex_mean as gatex_mean2
        global gatec_mean2_global
        global gatex_mean2_global
        gatec_mean2_global = torch.cat((gatec_mean2_global,gatec_mean2.cpu()),dim=-1)
        gatex_mean2_global = torch.cat((gatex_mean2_global,gatex_mean2.cpu()),dim=-1)
        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_notrans_ver3_wohistory(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        # self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        # self.relu = nn.ReLU()
        # self.dropout_in = nn.Dropout(config.dropout)
        self.diffusedSampleEnc = MLP(input_dim=2, output_dim=config.spatial_emsize, 
                                     hidden_size=[config.spatial_emsize]*(config.yt_encoder_layers-1))
        
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        self.concat_ped1 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize//2, config.context_dim)
        self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        # self.decode1 = Linear(config.spatial_emsize//2, 2)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        # hist_feature = context[0]
        # hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        # origin_shape = hist_embedded.shape
        # hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        # _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        # hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        # hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = ped_emb
        global beta_global
        beta_global = torch.cat((beta_global,beta.cpu()),dim=-1) #for debug
        
        # spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        spatial_input_embedded = self.diffusedSampleEnc(x)
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)
        from .common import gatec_mean as gatec_mean1
        from .common import gatex_mean as gatex_mean1
        global gatec_mean1_global
        global gatex_mean1_global
        gatec_mean1_global = torch.cat((gatec_mean1_global,gatec_mean1.cpu()),dim=-1)
        gatex_mean1_global = torch.cat((gatex_mean1_global,gatex_mean1.cpu()),dim=-1)
        output_ped = self.decode_ped1(spatial_input_embedded)
        from .common import gatec_mean as gatec_mean2
        from .common import gatex_mean as gatex_mean2
        global gatec_mean2_global
        global gatex_mean2_global
        gatec_mean2_global = torch.cat((gatec_mean2_global,gatec_mean2.cpu()),dim=-1)
        gatex_mean2_global = torch.cat((gatex_mean2_global,gatex_mean2.cpu()),dim=-1)
        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_notrans_ver2_klearnedenc(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        self.kencoder = nn.Sequential(
            nn.Linear(1, config.kenc_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        self.concat_ped1 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize//2, config.context_dim, dim_time=config.kenc_dim)
        self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        # self.decode1 = Linear(config.spatial_emsize//2, 2)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        global beta_global
        beta_global = torch.cat((beta_global,beta.cpu()),dim=-1) #for debug
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        # time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        time_emb = self.kencoder(beta)
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)
        
        output_ped = self.decode_ped1(spatial_input_embedded)
        
        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_notrans_ver2_ksinuenc(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize//2, config.context_dim, dim_time=config.kenc_dim)
        self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        # self.decode1 = Linear(config.spatial_emsize//2, 2)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list ,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        global beta_global
        beta_global = torch.cat((beta_global,beta.cpu()),dim=-1) #for debug
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        t = torch.Tensor([t]).view(-1).cuda()
        time_emb = self.k_emb(t)
        time_emb = time_emb.view(time_emb.shape[0], 1, time_emb.shape[-1]).repeat([1,x.shape[-2],1])
        # time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)
        output_ped = self.decode_ped1(spatial_input_embedded)
        
        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_notrans_ver2_ksinuenc_afver1(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AdaptiveFusion_ver1(config.spatial_emsize, config.spatial_emsize//2, config.context_dim, dim_time=config.kenc_dim)
        self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        # self.decode1 = Linear(config.spatial_emsize//2, 2)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list ,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        global beta_global
        beta_global = torch.cat((beta_global,beta.cpu()),dim=-1) #for debug
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        t = torch.Tensor([t]).view(-1).cuda()
        time_emb = self.k_emb(t)
        time_emb = time_emb.view(time_emb.shape[0], 1, time_emb.shape[-1]).repeat([1,x.shape[-2],1])
        # time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)

        output_ped = self.decode_ped1(spatial_input_embedded)

        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_notrans_ver2_ksinuenc_afver1_3(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AdaptiveFusion_ver1_3(config.spatial_emsize, config.spatial_emsize//2, config.context_dim, dim_time=config.kenc_dim)
        self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        # self.decode1 = Linear(config.spatial_emsize//2, 2)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list ,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        global beta_global
        beta_global = torch.cat((beta_global,beta.cpu()),dim=-1) #for debug
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        t = torch.Tensor([t]).view(-1).cuda()
        time_emb = self.k_emb(t)
        time_emb = time_emb.view(time_emb.shape[0], 1, time_emb.shape[-1]).repeat([1,x.shape[-2],1])
        # time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)

        output_ped = self.decode_ped1(spatial_input_embedded)

        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest



class SpatialTransformer_ped_inter_notrans_ver2_ksinuenc_afver1_2(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AdaptiveFusion_ver1_2(config.spatial_emsize, config.spatial_emsize//2, config.context_dim, dim_time=config.kenc_dim)
        self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        # self.decode1 = Linear(config.spatial_emsize//2, 2)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list ,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        global beta_global
        beta_global = torch.cat((beta_global,beta.cpu()),dim=-1) #for debug
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        t = torch.Tensor([t]).view(-1).cuda()
        time_emb = self.k_emb(t)
        time_emb = time_emb.view(time_emb.shape[0], 1, time_emb.shape[-1]).repeat([1,x.shape[-2],1])
        # time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)

        output_ped = self.decode_ped1(spatial_input_embedded)

        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_notrans_ver2_layer2(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize//2, config.context_dim, dim_time=3)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        self.decode_ped1 = AdaptiveFusion(config.spatial_emsize//2, 2, config.context_dim, dim_time=3)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list ,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        global beta_global
        beta_global = torch.cat((beta_global,beta.cpu()),dim=-1) #for debug
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        t = torch.Tensor([t]).view(-1).cuda()
        # time_emb = self.k_emb(t)
        # time_emb = time_emb.view(time_emb.shape[0], 1, time_emb.shape[-1]).repeat([1,x.shape[-2],1])
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)

        output_ped = self.decode_ped1(context_ped, spatial_input_embedded, time_emb)

        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_notrans_ver4(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AdaptiveFusion_ver4_nogateed(config.spatial_emsize, config.spatial_emsize//2, config.context_dim, dim_time=3)
        self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        # self.decode_ped1 = AdaptiveFusion(config.spatial_emsize//2, 2, config.context_dim, dim_time=3)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list ,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        t = torch.Tensor([t]).view(-1).cuda()
        global beta_global
        beta_global = torch.cat((beta_global,t.cpu()),dim=-1) #for debug
        # time_emb = self.k_emb(t)
        # time_emb = time_emb.view(time_emb.shape[0], 1, time_emb.shape[-1]).repeat([1,x.shape[-2],1])
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
        # spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)
        # output_ped = self.decode_ped1(spatial_input_embedded)
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)
        from .common import gatec_mean as gatec_mean1
        from .common import gatex_mean as gatex_mean1
        global gatec_mean1_global
        global gatex_mean1_global
        gatec_mean1_global = torch.cat((gatec_mean1_global,gatec_mean1.cpu()),dim=-1)
        gatex_mean1_global = torch.cat((gatex_mean1_global,gatex_mean1.cpu()),dim=-1)
        output_ped = self.decode_ped1(spatial_input_embedded)
        from .common import gatec_mean as gatec_mean2
        from .common import gatex_mean as gatex_mean2
        global gatec_mean2_global
        global gatex_mean2_global
        gatec_mean2_global = torch.cat((gatec_mean2_global,gatec_mean2.cpu()),dim=-1)
        gatex_mean2_global = torch.cat((gatex_mean2_global,gatex_mean2.cpu()),dim=-1)
        # output_ped = self.decode_ped1(context_ped, spatial_input_embedded, time_emb)

        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

# class SpatialTransformer_ped_inter_notrans_ver4(Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config=config
#         if 'ucy' in config.data_dict_path:
#             self.tau=5/6
#         else:
#             self.tau=2
#         config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
#         self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
#                                               2*config.spatial_emsize, config.spatial_encoder_layers)
        
#         self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
#         self.relu = nn.ReLU()
#         self.dropout_in = nn.Dropout(config.dropout)
#         # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
#         # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
#         self.concat_ped1 = AdaptiveFusion_ver4(config.spatial_emsize, config.spatial_emsize//2, config.context_dim, dim_time=3)
#         self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
#         # self.decode_ped1 = AdaptiveFusion(config.spatial_emsize//2, 2, config.context_dim, dim_time=3)
        
#         # history encoder
#         self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(config.dropout)
#         # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
#         self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
#         self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
#         # ped interaction encoder
#         self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
#         self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
#     def forward(self, x, beta, context:tuple, nei_list ,t):
#         # context encoding
#         hist_feature = context[0]
#         hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
#         origin_shape = hist_embedded.shape
#         hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
#         _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
#         hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
#         hist_embedded = self.lstm_output(hist_embedded)
        
#         self_features = context[2]
#         desired_speed = self_features[..., -1].unsqueeze(-1)
#         temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
#         temp_ = temp.clone()
#         temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
#         dest_direction = self_features[..., :2] / temp_ #des,direction
#         pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
#         ped_features = context[1] #bs, N, k, 6
#         ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
#         ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
#         ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
#         context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        
        
#         spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
#         t = torch.Tensor([t]).view(-1).cuda()
#         global beta_global
#         beta_global = torch.cat((beta_global,t.cpu()),dim=-1) #for debug
#         # time_emb = self.k_emb(t)
#         # time_emb = time_emb.view(time_emb.shape[0], 1, time_emb.shape[-1]).repeat([1,x.shape[-2],1])
#         beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
#         time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        
        
#         # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
#         # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
#         # spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)
#         # output_ped = self.decode_ped1(spatial_input_embedded)
#         spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)
#         from .common import gatec_mean as gatec_mean1
#         from .common import gatex_mean as gatex_mean1
#         global gatec_mean1_global
#         global gatex_mean1_global
#         gatec_mean1_global = torch.cat((gatec_mean1_global,gatec_mean1.cpu()),dim=-1)
#         gatex_mean1_global = torch.cat((gatex_mean1_global,gatex_mean1.cpu()),dim=-1)
#         output_ped = self.decode_ped1(spatial_input_embedded)
#         from .common import gatec_mean as gatec_mean2
#         from .common import gatex_mean as gatex_mean2
#         global gatec_mean2_global
#         global gatex_mean2_global
#         gatec_mean2_global = torch.cat((gatec_mean2_global,gatec_mean2.cpu()),dim=-1)
#         gatex_mean2_global = torch.cat((gatex_mean2_global,gatex_mean2.cpu()),dim=-1)
#         # output_ped = self.decode_ped1(context_ped, spatial_input_embedded, time_emb)

#         assert output_ped.shape[-1]==2, 'wrong code!'
#         return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_notrans_ver4_layer2(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AdaptiveFusion_ver4(config.spatial_emsize, config.spatial_emsize//2, config.context_dim, dim_time=3)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        self.decode_ped1 = AdaptiveFusion_ver4(config.spatial_emsize//2, 2, config.context_dim, dim_time=3)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list ,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        t = torch.Tensor([t]).view(-1).cuda()
        global beta_global
        beta_global = torch.cat((beta_global,t.cpu()),dim=-1) #for debug
        # time_emb = self.k_emb(t)
        # time_emb = time_emb.view(time_emb.shape[0], 1, time_emb.shape[-1]).repeat([1,x.shape[-2],1])
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
        # spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)

        # output_ped = self.decode_ped1(context_ped, spatial_input_embedded, time_emb)
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)
        from .common import gatec_mean as gatec_mean1
        from .common import gatex_mean as gatex_mean1
        global gatec_mean1_global
        global gatex_mean1_global
        gatec_mean1_global = torch.cat((gatec_mean1_global,gatec_mean1.cpu()),dim=-1)
        gatex_mean1_global = torch.cat((gatex_mean1_global,gatex_mean1.cpu()),dim=-1)
        output_ped = self.decode_ped1(context_ped, spatial_input_embedded, time_emb)
        from .common import gatec_mean as gatec_mean2
        from .common import gatex_mean as gatex_mean2
        global gatec_mean2_global
        global gatex_mean2_global
        gatec_mean2_global = torch.cat((gatec_mean2_global,gatec_mean2.cpu()),dim=-1)
        gatex_mean2_global = torch.cat((gatex_mean2_global,gatex_mean2.cpu()),dim=-1)
        
        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_notrans_ver5(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AdaptiveFusion_ver5_nogateed(config.spatial_emsize, config.spatial_emsize//2, config.context_dim, dim_time=3)
        self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        # self.decode_ped1 = AdaptiveFusion(config.spatial_emsize//2, 2, config.context_dim, dim_time=3)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list ,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        t = torch.Tensor([t]).view(-1).cuda()
        
        global beta_global
        beta_global = torch.cat((beta_global,t.cpu()),dim=-1) #for debug
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        # time_emb = self.k_emb(t)
        # time_emb = time_emb.view(time_emb.shape[0], 1, time_emb.shape[-1]).repeat([1,x.shape[-2],1])
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)
        from .common import gatec_mean as gatec_mean1
        from .common import gatex_mean as gatex_mean1
        global gatec_mean1_global
        global gatex_mean1_global
        gatec_mean1_global = torch.cat((gatec_mean1_global,gatec_mean1.cpu()),dim=-1)
        gatex_mean1_global = torch.cat((gatex_mean1_global,gatex_mean1.cpu()),dim=-1)
        output_ped = self.decode_ped1(spatial_input_embedded)
        from .common import gatec_mean as gatec_mean2
        from .common import gatex_mean as gatex_mean2
        global gatec_mean2_global
        global gatex_mean2_global
        gatec_mean2_global = torch.cat((gatec_mean2_global,gatec_mean2.cpu()),dim=-1)
        gatex_mean2_global = torch.cat((gatex_mean2_global,gatex_mean2.cpu()),dim=-1)
        # output_ped = self.decode_ped1(context_ped, spatial_input_embedded, time_emb)

        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_obs_inter_notrans_ver5_layer2(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
            self.obs_encode_flag=False
            config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        else:
            self.tau=2
            self.obs_encode_flag=True
            config.context_dim = config.ped_encode_dim2 + config.history_lstm_out + config.obs_encode_dim2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AdaptiveFusion_ver5(config.spatial_emsize, config.spatial_emsize//2, config.context_dim, dim_time=3)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        self.decode_ped1 = AdaptiveFusion_ver5(config.spatial_emsize//2, 2, config.context_dim, dim_time=3)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        # obs interaction encoder
        if self.obs_encode_flag:
            self.obs_encoder1 = MLP(input_dim=6, output_dim=config.obs_encode_dim1, hidden_size=config.obs_encode_hid1)
            self.obs_encoder2 = ResDNN(input_dim=config.obs_encode_dim1, hidden_units=[[config.obs_encode_dim2]]+[[config.obs_encode_dim2]*2]*(config.obs_encode2_layers-1))
        
    def forward(self, x, beta, context:tuple, nei_list ,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        t = torch.Tensor([t]).view(-1).cuda()
        global beta_global
        beta_global = torch.cat((beta_global,t.cpu()),dim=-1) #for debug
        if self.obs_encode_flag:
            obs_features = context[3] #bs, N, k, 6
            obs_emb = self.obs_encoder1(obs_features)
            obs_emb = self.obs_encoder2(obs_emb) #bs, N, k, dim
            obs_emb = torch.sum(obs_emb, dim=-2) #bs, N, dim
            context_ped = torch.cat((hist_embedded,ped_emb,obs_emb), dim=-1)
            
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        
        # time_emb = self.k_emb(t)
        # time_emb = time_emb.view(time_emb.shape[0], 1, time_emb.shape[-1]).repeat([1,x.shape[-2],1])
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)


        output_ped = self.decode_ped1(context_ped, spatial_input_embedded, time_emb)

        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_notrans_ver5_layer2(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        # self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
        #                                       2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AdaptiveFusion_ver5(config.spatial_emsize, config.spatial_emsize//2, config.context_dim, dim_time=3)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        self.decode_ped1 = AdaptiveFusion_ver5(config.spatial_emsize//2, 2, config.context_dim, dim_time=3)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list ,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        t = torch.Tensor([t]).view(-1).cuda()
        global beta_global
        beta_global = torch.cat((beta_global,t.cpu()),dim=-1) #for debug
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        
        # time_emb = self.k_emb(t)
        # time_emb = time_emb.view(time_emb.shape[0], 1, time_emb.shape[-1]).repeat([1,x.shape[-2],1])
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)
        from .common import gatec_mean as gatec_mean1
        from .common import gatex_mean as gatex_mean1
        global gatec_mean1_global
        global gatex_mean1_global
        gatec_mean1_global = torch.cat((gatec_mean1_global,gatec_mean1.cpu()),dim=-1)
        gatex_mean1_global = torch.cat((gatex_mean1_global,gatex_mean1.cpu()),dim=-1)
        output_ped = self.decode_ped1(context_ped, spatial_input_embedded, time_emb)
        from .common import gatec_mean as gatec_mean2
        from .common import gatex_mean as gatex_mean2
        global gatec_mean2_global
        global gatex_mean2_global
        gatec_mean2_global = torch.cat((gatec_mean2_global,gatec_mean2.cpu()),dim=-1)
        gatex_mean2_global = torch.cat((gatex_mean2_global,gatex_mean2.cpu()),dim=-1)
        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_notrans_ver2_crossattn(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        dit_depth = config.dit_depth
        self.concat_ped1 = nn.ModuleList([
            DITBlock(hidden_size=config.spatial_emsize, cond_dim=config.context_dim+3 ,num_heads=1) for _ in range(dit_depth)])
        # self.concat_ped1 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize//2, config.context_dim)
        self.decode_ped1 = nn.Sequential(nn.LayerNorm(config.spatial_emsize, elementwise_affine=False, eps=1e-6),
                                        nn.Linear(config.spatial_emsize, 2, bias=True))
        
        # self.decode1 = Linear(config.spatial_emsize//2, 2)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        global beta_global
        beta_global = torch.cat((beta_global,beta.cpu()),dim=-1) #for debug
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        for block in self.concat_ped1:
            spatial_input_embedded = block(torch.cat((context_ped, time_emb),dim=-1), spatial_input_embedded)
        
        output_ped = self.decode_ped1(spatial_input_embedded)

        
        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_wo_history(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        config.context_dim = config.ped_encode_dim2
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        self.concat_ped1 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize//2, config.context_dim)
        self.decode_ped1 = AdaptiveFusion(config.spatial_emsize//2, 2, config.context_dim)
        # self.decode1 = Linear(config.spatial_emsize//2, 2)
        
        # history encoder
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        # self.relu1 = nn.ReLU()
        # self.dropout1 = nn.Dropout(config.dropout)
        # # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        # self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        # self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        # hist_feature = context[0]
        # hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        # origin_shape = hist_embedded.shape
        # hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        # _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        # hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        # hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features) # TODO:内存增长点：40MB
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = ped_emb
        # context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list) #TODO：内存增长150MB
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)
        output_ped = self.decode_ped1(context_ped, spatial_input_embedded, time_emb)
        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_geometric(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        config.context_dim = config.ped_encode_dim2
        self.egnn = NetEGNN()
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        # hist_feature = context[0]
        # hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        # origin_shape = hist_embedded.shape
        # hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        # _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        # hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        # hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        
        out1 = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)
        output_ped = 0.
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_geometric_cond(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        config.context_dim = config.egnn_hid_dim
        #config.context_dim = 2
        self.egnn = NetEGNN(hid_dim = config.egnn_hid_dim, n_layers = config.egnn_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AdaptiveFusion_ver5(config.spatial_emsize, config.spatial_emsize//2, config.context_dim, dim_time=3)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        self.decode_ped1 = AdaptiveFusion_ver5(config.spatial_emsize//2, 2, config.context_dim, dim_time=3)
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        # hist_feature = context[4]
        # hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        # origin_shape = hist_embedded.shape
        # hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        # _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        # hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        # hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        obstacles = context[5]
        acce_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx, obstacles], time_emb)
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        
        
        spatial_input_embedded = self.concat_ped1(acce_emb, spatial_input_embedded, time_emb)
        output_ped = self.decode_ped1(acce_emb, spatial_input_embedded, time_emb)
        
        
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_geometric_cond_w_history(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        config.context_dim = config.egnn_hid_dim + config.history_lstm_out
        #config.context_dim = 2
        self.egnn = NetEGNN_hid2(hid_dim = config.egnn_hid_dim, n_layers = config.egnn_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        self.concat_ped1 = AdaptiveFusion_ver5(config.spatial_emsize, config.spatial_emsize//2, config.context_dim, dim_time=3)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        self.decode_ped1 = AdaptiveFusion_ver5(config.spatial_emsize//2, 2, config.context_dim, dim_time=3)
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[4]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded) # bs,N,embsize_out
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        acce_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        
        context_emb = torch.cat((acce_emb, hist_embedded), dim=-1)

        spatial_input_embedded = self.concat_ped1(context_emb, spatial_input_embedded, time_emb)
        output_ped = self.decode_ped1(context_emb, spatial_input_embedded, time_emb)
        
        
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_geometric_cond_w_history_wo_af(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        config.context_dim = config.egnn_hid_dim + config.history_lstm_out
        #config.context_dim = 2
        self.egnn = NetEGNN_hid2(hid_dim = config.egnn_hid_dim, n_layers = config.egnn_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        self.concat_ped1 = nn.Linear(config.spatial_emsize+config.context_dim+3, config.spatial_emsize//2)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        self.relu2 = nn.ReLU()
        self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2)
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[4]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded) # bs,N,embsize_out
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        acce_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        
        context_emb = torch.cat((acce_emb, hist_embedded), dim=-1)

        spatial_input_embedded = self.concat_ped1(torch.cat((context_emb, spatial_input_embedded, time_emb), dim=-1))
        output_ped = self.decode_ped1(self.relu2(spatial_input_embedded))
        
        
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_geometric_cond_w_obs_w_history(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        
        # config.context_dim = 2
        config.context_dim = config.egnn_hid_dim + config.history_lstm_out
        self.egnn = NetEGNN_hid2(hid_dim = config.egnn_hid_dim, n_layers = config.egnn_layers)
        self.has_obstacles = config.has_obstacles
        if config.has_obstacles == True:
            self.egnn_obs = NetEGNN_hid_obs2(hid_dim = config.egnn_hid_dim_obs, n_layers = config.egnn_layers_obs)
            config.context_dim = config.egnn_hid_dim + config.egnn_hid_dim_obs + config.history_lstm_out
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)

        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AdaptiveFusion_ver5(config.spatial_emsize, config.spatial_emsize//2, config.context_dim, dim_time=3)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        self.decode_ped1 = AdaptiveFusion_ver5(config.spatial_emsize//2, 2, config.context_dim, dim_time=3)
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[4]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        ped_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)
        
        if self.has_obstacles:
          obs_features = context[5]
          near_obstacle_idx = context[6]
          neigh_obs_mask = context[7]
          obs_emb = self.egnn_obs([ped_features, obs_features, neigh_obs_mask, near_obstacle_idx], time_emb)
          ctx_emb = torch.cat((hist_embedded, ped_emb, obs_emb), dim=-1)
        else:
          ctx_emb = torch.cat((hist_embedded, ped_emb), dim=-1)


        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        
        
        spatial_input_embedded = self.concat_ped1(ctx_emb, spatial_input_embedded, time_emb)
        output_ped = self.decode_ped1(ctx_emb, spatial_input_embedded, time_emb)
        
        
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_geometric_cond_w_obs_w_history_wo_af(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        
        # config.context_dim = 2
        config.context_dim = config.egnn_hid_dim + config.history_lstm_out
        self.egnn = NetEGNN_hid2(hid_dim = config.egnn_hid_dim, n_layers = config.egnn_layers)
        self.has_obstacles = config.has_obstacles
        if config.has_obstacles == True:
            self.egnn_obs = NetEGNN_hid_obs2(hid_dim = config.egnn_hid_dim_obs, n_layers = config.egnn_layers_obs)
            config.context_dim = config.egnn_hid_dim + config.egnn_hid_dim_obs + config.history_lstm_out
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)

        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = nn.Linear(config.spatial_emsize+config.context_dim+3, config.spatial_emsize//2)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        self.relu2 = nn.ReLU()
        self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2)
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[4]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        ped_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)
        
        if self.has_obstacles:
          obs_features = context[5]
          near_obstacle_idx = context[6]
          neigh_obs_mask = context[7]
          obs_emb = self.egnn_obs([ped_features, obs_features, neigh_obs_mask, near_obstacle_idx], time_emb)
          ctx_emb = torch.cat((hist_embedded, ped_emb, obs_emb), dim=-1)
        else:
          ctx_emb = torch.cat((hist_embedded, ped_emb), dim=-1)


        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        
        
        spatial_input_embedded = self.concat_ped1(torch.cat((ctx_emb, spatial_input_embedded, time_emb), dim=-1))
        output_ped = self.decode_ped1(self.relu2(spatial_input_embedded))
        
        
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_geometric_cond_w_obs_wo_history(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        
        # config.context_dim = 2
        config.context_dim = config.egnn_hid_dim
        self.egnn = NetEGNN_hid2(hid_dim = config.egnn_hid_dim, n_layers = config.egnn_layers)
        self.has_obstacles = config.has_obstacles
        if config.has_obstacles == True:
            self.egnn_obs = NetEGNN_hid_obs2(hid_dim = config.egnn_hid_dim_obs, n_layers = config.egnn_layers_obs)
            config.context_dim = config.egnn_hid_dim + config.egnn_hid_dim_obs
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        # self.relu1 = nn.ReLU()
        # self.dropout1 = nn.Dropout(config.dropout)
        # # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        # self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        # self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)

        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AdaptiveFusion_ver5(config.spatial_emsize, config.spatial_emsize//2, config.context_dim, dim_time=3)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        self.decode_ped1 = AdaptiveFusion_ver5(config.spatial_emsize//2, 2, config.context_dim, dim_time=3)
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        # hist_feature = context[4]
        # hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        # origin_shape = hist_embedded.shape
        # hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        # _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        # hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        # hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        ped_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)
        
        if self.has_obstacles:
          obs_features = context[5]
          near_obstacle_idx = context[6]
          neigh_obs_mask = context[7]
          obs_emb = self.egnn_obs([ped_features, obs_features, neigh_obs_mask, near_obstacle_idx], time_emb)
          ctx_emb = torch.cat((ped_emb, obs_emb), dim=-1)
        else:
          ctx_emb = ped_emb


        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        
        
        spatial_input_embedded = self.concat_ped1(ctx_emb, spatial_input_embedded, time_emb)
        output_ped = self.decode_ped1(ctx_emb, spatial_input_embedded, time_emb)
        
        
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_geometric_cond_w_obs_w_history2(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        
        # config.context_dim = 2
        config.context_dim = config.egnn_hid_dim + config.history_lstm_out
        self.egnn = NetEGNN_hid2(hid_dim = config.egnn_hid_dim, n_layers = config.egnn_layers)
        self.has_obstacles = config.has_obstacles
        if config.has_obstacles == True:
            self.egnn = NetEGNN_hid_ped_obs2(hid_dim = config.egnn_hid_dim, hid_dim_obs = config.egnn_hid_dim_obs, n_layers = config.egnn_layers, n_layers_obs = config.egnn_layers_obs)
            config.context_dim = config.egnn_hid_dim + config.egnn_hid_dim_obs + config.history_lstm_out
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)

        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AdaptiveFusion_ver5(config.spatial_emsize, config.spatial_emsize//2, config.context_dim, dim_time=3)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        self.decode_ped1 = AdaptiveFusion_ver5(config.spatial_emsize//2, 2, config.context_dim, dim_time=3)
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[4]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        
        if self.has_obstacles:
            obs_features = context[5]
            near_obstacle_idx = context[6]
            neigh_obs_mask = context[7]
            ctx_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx, obs_features, neigh_obs_mask, near_obstacle_idx], time_emb)
            ctx_emb = torch.cat((hist_embedded, *ctx_emb), dim=-1)
        else:
            ped_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)

            ctx_emb = torch.cat((hist_embedded, ped_emb), dim=-1) 


        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        
        
        spatial_input_embedded = self.concat_ped1(ctx_emb, spatial_input_embedded, time_emb)
        output_ped = self.decode_ped1(ctx_emb, spatial_input_embedded, time_emb)
        
        
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_geometric_cond_w_obs_w_history_acce_ver3(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        config.context_dim = config.egnn_hid_dim + config.egnn_hid_dim_obs + config.history_lstm_out
        # config.context_dim = 2
        self.egnn = NetEGNN_hid(hid_dim = config.egnn_hid_dim, n_layers = config.egnn_layers)
        self.has_obstacles = config.has_obstacles
        if config.has_obstacles == True:
            self.egnn_obs = NetEGNN_hid_obs(hid_dim = config.egnn_hid_dim_obs, n_layers = config.egnn_layers_obs)
        
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)

        # self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        # self.relu = nn.ReLU()
        # self.dropout_in = nn.Dropout(config.dropout)
        self.acce_enc = VNLinearLeakyReLU(1, config.diff_emb_out, dim=4)

        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AdaptiveFusion_ver5(config.diff_emb_out*2, config.spatial_emsize//2, config.context_dim, dim_time=3)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        self.decode_ped1 = AdaptiveFusion_ver5(config.spatial_emsize//2, 2, config.context_dim, dim_time=3)
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[4]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        ped_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)
        
        obs_features = context[5]
        near_obstacle_idx = context[6]
        neigh_obs_mask = context[7]
        obs_emb = self.egnn_obs([ped_features, obs_features, neigh_obs_mask, near_obstacle_idx], time_emb)
        ctx_emb = torch.cat((hist_embedded, ped_emb, obs_emb), dim=-1)


        # spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        spatial_input_embedded = self.acce_enc(x.permute(0,2,1).unsqueeze(1)).permute(0,3,1,2)
        spatial_input_embedded = spatial_input_embedded.reshape(*spatial_input_embedded.shape[:2], -1)
        
        spatial_input_embedded = self.concat_ped1(ctx_emb, spatial_input_embedded, time_emb)
        output_ped = self.decode_ped1(ctx_emb, spatial_input_embedded, time_emb)
        
        
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_geometric_cond_acce(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        config.context_dim = config.egnn_hid_dim
        # config.context_dim = 2
        self.egnn = NetEGNN_hid(hid_dim = config.egnn_hid_dim, n_layers = config.egnn_layers)
        
        self.acce_enc = diff_feat_encoder(config, dim_in=2, hid_dim=config.diff_emb_hid, out_dim=1, mlp_layers=3, batchnorm=False)

        # self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        # self.relu = nn.ReLU()
        # self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AdaptiveFusion_ver5(2, config.spatial_emsize, config.context_dim, dim_time=3)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        self.decode_ped1 = AdaptiveFusion_ver5(config.spatial_emsize, 2, config.context_dim, dim_time=3)
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        """
            x:[B, N, 2]
            beta:[B]
        """
        # context encoding
        # hist_feature = context[0]
        # hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        # origin_shape = hist_embedded.shape
        # hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        # _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        # hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        # hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        egnn_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)
        
        spatial_input_embedded = self.acce_enc(x, [neigh_ped_mask, near_ped_idx])
        spatial_input_embedded = spatial_input_embedded.squeeze(1).permute(0,2,1) # B,N,2
        assert spatial_input_embedded.shape==x.shape
        # spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        
        
        spatial_input_embedded = self.concat_ped1(egnn_emb, spatial_input_embedded, time_emb)
        output_ped = self.decode_ped1(egnn_emb, spatial_input_embedded, time_emb)
        
        
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_geometric_cond_acce_ver2(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        config.context_dim = config.egnn_hid_dim
        # config.context_dim = 2
        self.egnn = NetEGNN_hid(hid_dim = config.egnn_hid_dim, n_layers = config.egnn_layers)
        
        self.acce_enc = diff_feat_encoder(config, dim_in=2, hid_dim=config.diff_emb_hid, out_dim=config.diff_emb_out, mlp_layers=3, batchnorm=False)

        # self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        # self.relu = nn.ReLU()
        # self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AdaptiveFusion_ver5(config.diff_emb_out*2, config.spatial_emsize, config.context_dim, dim_time=3)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        self.decode_ped1 = AdaptiveFusion_ver5(config.spatial_emsize, 2, config.context_dim, dim_time=3)
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        """
            x:[B, N, 2]
            beta:[B]
        """
        # context encoding
        # hist_feature = context[0]
        # hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        # origin_shape = hist_embedded.shape
        # hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        # _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        # hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        # hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        egnn_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)
        
        spatial_input_embedded = self.acce_enc(x, [neigh_ped_mask, near_ped_idx]).permute(0,3,1,2)
        spatial_input_embedded = spatial_input_embedded.reshape(*x.shape[:2], -1)

        # spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        
        
        spatial_input_embedded = self.concat_ped1(egnn_emb, spatial_input_embedded, time_emb)
        output_ped = self.decode_ped1(egnn_emb, spatial_input_embedded, time_emb)
        
        
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_geometric_cond_acce_ver3(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        config.context_dim = config.egnn_hid_dim
        # config.context_dim = 2
        self.egnn = NetEGNN_hid(hid_dim = config.egnn_hid_dim, n_layers = config.egnn_layers)
        
        self.acce_enc = VNLinearLeakyReLU(1, config.diff_emb_out, dim=4)

        # self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        # self.relu = nn.ReLU()
        # self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AdaptiveFusion_ver5(config.diff_emb_out*2, config.spatial_emsize, config.context_dim, dim_time=3)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        self.decode_ped1 = AdaptiveFusion_ver5(config.spatial_emsize, 2, config.context_dim, dim_time=3)
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        """
            x:[B, N, 2]
            beta:[B]
        """
        # context encoding
        # hist_feature = context[0]
        # hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        # origin_shape = hist_embedded.shape
        # hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        # _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        # hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        # hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        egnn_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)
        
        spatial_input_embedded = self.acce_enc(x.permute(0,2,1).unsqueeze(1)).permute(0,3,1,2)
        spatial_input_embedded = spatial_input_embedded.reshape(*spatial_input_embedded.shape[:2], -1)

        # spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        
        
        spatial_input_embedded = self.concat_ped1(egnn_emb, spatial_input_embedded, time_emb)
        output_ped = self.decode_ped1(egnn_emb, spatial_input_embedded, time_emb)
        
        
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_geometric_cond_acce_ver4(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        config.context_dim = config.egnn_hid_dim
        # config.context_dim = 2
        self.egnn = NetEGNN_hid(hid_dim = config.egnn_hid_dim, n_layers = config.egnn_layers)
        
        # self.acce_enc = VNLinearLeakyReLU(1, config.diff_emb_out, dim=4)

        # self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        # self.relu = nn.ReLU()
        # self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AdaptiveFusion_ver5(2, config.spatial_emsize, config.context_dim, dim_time=3)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        self.decode_ped1 = AdaptiveFusion_ver5(config.spatial_emsize, 2, config.context_dim, dim_time=3)
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        """
            x:[B, N, 2]
            beta:[B]
        """
        # context encoding
        # hist_feature = context[0]
        # hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        # origin_shape = hist_embedded.shape
        # hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        # _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        # hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        # hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        egnn_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)
        
        # spatial_input_embedded = self.acce_enc(x.permute(0,2,1).unsqueeze(1)).permute(0,3,1,2)
        # spatial_input_embedded = spatial_input_embedded.view(*spatial_input_embedded.shape[:2], -1)
        spatial_input_embedded = x
        # spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        
        
        spatial_input_embedded = self.concat_ped1(egnn_emb, spatial_input_embedded, time_emb)
        output_ped = self.decode_ped1(egnn_emb, spatial_input_embedded, time_emb)
        
        
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_geometric_all(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        # config.context_dim = config.egnn_hid_dim
        #config.context_dim = 2
        self.egnn = NetEGNN_acce(hid_dim = config.egnn_hid_dim, n_layers = config.egnn_layers)
        
        self.acce_enc = diff_feat_encoder(config, dim_in=2, hid_dim=config.diff_emb_hid, out_dim=config.diff_emb_out, mlp_layers=3, batchnorm=False)

        # self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        # self.relu = nn.ReLU()
        # self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AF_vnn(dim_in=config.diff_emb_out, dim_out=config.spatial_emsize, dim_ctx=1)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        self.decode_ped1 = AF_vnn(dim_in=config.spatial_emsize, dim_out=1, dim_ctx=1)
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        """
            x:[B, N, 2]
            beta:[B]
        """
        # context encoding
        # hist_feature = context[0]
        # hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        # origin_shape = hist_embedded.shape
        # hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        # _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        # hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        # hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        egnn_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)
        egnn_emb = egnn_emb.permute(0,2,1).unsqueeze(1) # B, 1, 2, N
        
        spatial_input_embedded = self.acce_enc(x, [neigh_ped_mask, near_ped_idx])
        # spatial_input_embedded = spatial_input_embedded.squeeze(1).permute(0,2,1) # B,N,2
        # spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        
        
        spatial_input_embedded = self.concat_ped1(egnn_emb, spatial_input_embedded)
        output_ped = self.decode_ped1(egnn_emb, spatial_input_embedded).squeeze(1).permute(0,2,1)
        
        
        
        return output_ped + pred_acc_dest

class SpatialTransformer_ped_inter_geometric_all_ver2(Module):
    """
        use AF_vnn_inp_hid
    """
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        # config.context_dim = config.egnn_hid_dim
        #config.context_dim = 2
        self.egnn = NetEGNN_acce_hid(hid_dim = config.egnn_hid_dim, n_layers = config.egnn_layers)
        
        self.acce_enc = diff_feat_encoder(config, dim_in=2, hid_dim=config.diff_emb_hid, out_dim=config.diff_emb_out, mlp_layers=3, batchnorm=False)

        # self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        # self.relu = nn.ReLU()
        # self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)
        self.concat_ped1 = AF_vnn_inp_hid(dim_in=config.diff_emb_out, dim_out=config.spatial_emsize, dim_ctx=1, dim_hid=config.egnn_hid_dim)
        # self.decode_ped1 = nn.Linear(config.spatial_emsize//2, 2, bias=True)
        
        self.decode_ped1 = AF_vnn_inp_hid(dim_in=config.spatial_emsize, dim_out=1, dim_ctx=1, dim_hid=config.egnn_hid_dim)
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        """
            x:[B, N, 2]
            beta:[B]
        """
        # context encoding
        # hist_feature = context[0]
        # hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        # origin_shape = hist_embedded.shape
        # hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        # _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        # hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO:内存增长点
        # hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        acce_emb, h_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)
        acce_emb = acce_emb.permute(0,2,1).unsqueeze(1) # B, 1, 2, N

        
        spatial_input_embedded = self.acce_enc(x, [neigh_ped_mask, near_ped_idx])
        # spatial_input_embedded = spatial_input_embedded.squeeze(1).permute(0,2,1) # B,N,2
        # spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        
        spatial_input_embedded = self.concat_ped1(acce_emb, spatial_input_embedded, h_emb)
        output_ped = self.decode_ped1(acce_emb, spatial_input_embedded, h_emb).squeeze(1).permute(0,2,1)

        return output_ped + pred_acc_dest

class SpatialTransformer_ped_obs_inter(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
            self.obs_encode_flag=False
            config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        else:
            self.tau=2
            self.obs_encode_flag=True
            config.context_dim = config.ped_encode_dim2 + config.history_lstm_out + config.obs_encode_dim2
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        self.concat_ped1 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize//2, config.context_dim)
        self.decode_ped1 = AdaptiveFusion(config.spatial_emsize//2, 2, config.context_dim)
        # self.decode1 = Linear(config.spatial_emsize//2, 2)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        if self.obs_encode_flag:
            self.obs_encoder1 = MLP(input_dim=6, output_dim=config.obs_encode_dim1, hidden_size=config.obs_encode_hid1)
            self.obs_encoder2 = ResDNN(input_dim=config.obs_encode_dim1, hidden_units=[[config.obs_encode_dim2]]+[[config.obs_encode_dim2]*2]*(config.obs_encode2_layers-1))
            
        
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        hist_feature = context[0]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        ped_features = context[1] #bs, N, k, 6
        ped_emb = self.ped_encoder1(ped_features)
        ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        if self.obs_encode_flag:
            obs_features = context[3] #bs, N, k, 6
            obs_emb = self.obs_encoder1(obs_features)
            obs_emb = self.obs_encoder2(obs_emb) #bs, N, k, dim
            obs_emb = torch.sum(obs_emb, dim=-2) #bs, N, dim
            context_ped = torch.cat((hist_embedded,ped_emb,obs_emb), dim=-1)  
        
        
        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list)
        # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)
        output_ped = self.decode_ped1(context_ped, spatial_input_embedded, time_emb)
        assert output_ped.shape[-1]==2, 'wrong code!'
        return output_ped + pred_acc_dest

class SpatialTransformer_dest_force(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        self.spatial_encoder=TransformerModel2(config.spatial_emsize, config.spatial_encoder_head,
                                              2*config.spatial_emsize, config.spatial_encoder_layers)
        
        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        self.concat_ped1 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize//2, config.context_dim)
        self.decode_ped1 = AdaptiveFusion(config.spatial_emsize//2, 2, config.context_dim)
        # self.decode1 = Linear(config.spatial_emsize//2, 2)
        
        # history encoder
        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        
        # ped interaction encoder
        self.ped_encoder1 = MLP(input_dim=6, output_dim=config.ped_encode_dim1, hidden_size=config.ped_encode_hid1)
        self.ped_encoder2 = ResDNN(input_dim=config.ped_encode_dim1, hidden_units=[[config.ped_encode_dim2]]+[[config.ped_encode_dim2]*2]*(config.ped_encode2_layers-1))
        
        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        # hist_feature = context[0]
        # hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        # origin_shape = hist_embedded.shape
        # hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        # _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        # hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize
        # hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau
        
        # ped_features = context[1] #bs, N, k, 6
        # ped_emb = self.ped_encoder1(ped_features)
        # ped_emb = self.ped_encoder2(ped_emb) #bs, N, k, dim
        # ped_emb = torch.sum(ped_emb, dim=-2) #bs, N, dim
        # context_ped = torch.cat((hist_embedded,ped_emb), dim=-1)
        
        
        
        # spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        # beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        # time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        
        # spatial_input_embedded = self.spatial_encoder(spatial_input_embedded, nei_list)
        # # spatial_input_embedded = self.concat2(hist_embedded, spatial_input_embedded, time_emb)
        # spatial_input_embedded = self.concat_ped1(context_ped, spatial_input_embedded, time_emb)
        # output_ped = self.decode_ped1(context_ped, spatial_input_embedded, time_emb)
        # assert output_ped.shape[-1]==2, 'wrong code!'
        return pred_acc_dest



class TransformerLinear(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.residual = residual

        self.pos_emb = PositionalEncoding(d_model=128, dropout=0.1, max_len=24)
        self.y_up = nn.Linear(2, 128)
        self.ctx_up = nn.Linear(context_dim+3, 128)
        self.layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=3)
        self.linear = nn.Linear(128, point_dim)

    def forward(self, x, beta, context):

        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        ctx_emb = self.ctx_up(ctx_emb)
        emb = self.y_up(x)
        final_emb = torch.cat([ctx_emb, emb], dim=1).permute(1,0,2)
        #pdb.set_trace()
        final_emb = self.pos_emb(final_emb)

        trans = self.transformer_encoder(final_emb)  # 13 * b * 128
        trans = trans[1:].permute(1,0,2)   # B * 12 * 128, drop the first one which is the z
        return self.linear(trans)







class LinearDecoder(Module):
    def __init__(self):
            super().__init__()
            self.act = F.leaky_relu
            self.layers = ModuleList([
                #nn.Linear(2, 64),
                nn.Linear(32, 64),
                nn.Linear(64, 128),
                nn.Linear(128, 256),
                nn.Linear(256, 512),
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 12)
                #nn.Linear(2, 64),
                #nn.Linear(2, 64),
            ])
    def forward(self, code):

        out = code
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.act(out)
        return out



class DiffusionTraj(Module):

    def __init__(self, net, var_sched:VarianceSchedule, config):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.config = config
    def denoise_fn(self, x_0, curr=None, context=None, timestep=0.08, t=None, mask=None):
        batch_size = x_0.shape[0]
        # point_dim = x_0.shape[-1]
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)

        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t].cuda()

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).cuda()       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).cuda()   # (B, 1, 1)
        
        

        e_rand = torch.randn_like(x_0).cuda()  # (B, N, d),这里的N是未来轨迹的数量，对应原文大T
        x_t = c0 * x_0 + c1 * e_rand
        nei_list = None
        p0 = curr[...,:2] # B, N, 2
        # if self.config.nei_attn_mask==True:
        #     nei_list = self.cur_mask(p0)
        if self.config.nei_padding_mask==True:
            nei_list = self.curr_padding(p0)
        # if curr != None:
        #     p0 = p0.masked_fill((p0!=p0),0) # zero_padding
        #     p2 = (curr[...,4:6]*timestep+curr[...,2:4])*timestep + curr[...,2:4]*timestep + p0 # p2 = p0+v0*t+v1*t, v1=a0*t+v0
        #     x_t = (x_t*timestep+(curr[...,4:6]*timestep+curr[...,2:4]))*timestep + p2 # p3 = p2+v2*t, v2 = a1*t+v1
        #     assert torch.all(~x_t.isnan())
        x_0_hat = self.net(x_t, beta=beta, context=context, nei_list = nei_list, t=t)
        # x_0_hat = self.net(x_t, beta=torch.tensor(t,device=x_t.device), context=context, nei_list = nei_list)
        return x_0_hat
        
    def get_loss(self, x_0, curr=None, context=None, timestep=0.08, t=None, mask=None):

        # batch_size = x_0.shape[0]
        point_dim = x_0.shape[-1]
        x_0_hat = self.denoise_fn(x_0, curr=curr, context=context, timestep=timestep, t=t, mask=mask)
        # if t == None:
        #     t = self.var_sched.uniform_sample_t(batch_size)

        # alpha_bar = self.var_sched.alpha_bars[t]
        # beta = self.var_sched.betas[t].cuda()

        # c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).cuda()       # (B, 1, 1)
        # c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).cuda()   # (B, 1, 1)

        # e_rand = torch.randn_like(x_0).cuda()  # (B, N, d),这里的N是未来轨迹的数量，对应原文大T
        # x_t = c0 * x_0 + c1 * e_rand
        # nei_list = None
        # p0 = curr[...,:2] # B, N, 2
        # if self.config.nei_attn_mask==True:
        #     nei_list = self.cur_mask(p0)
        # if self.config.nei_padding_mask==True:
        #     nei_list = self.curr_padding(p0)
        # if curr != None:
        #     p0 = p0.masked_fill((p0!=p0),0) # zero_padding
        #     p2 = (curr[...,4:6]*timestep+curr[...,2:4])*timestep + curr[...,2:4]*timestep + p0 # p2 = p0+v0*t+v1*t, v1=a0*t+v0
        #     x_t = (x_t*timestep+(curr[...,4:6]*timestep+curr[...,2:4]))*timestep + p2 # p3 = p2+v2*t, v2 = a1*t+v1
        #     
        # x_0_hat = self.net(x_t, beta=beta, context=context, nei_list = nei_list)
        
        if mask==None:
            loss = F.mse_loss(x_0_hat.view(-1,point_dim), x_0.continuous().view(-1,point_dim), reduction='mean')
        else:
            loss = mask_mse_func(x_0_hat, x_0, mask)       
        return loss
    
    def cur_mask(self, p0):
        assert p0.dim()==3
        attn_mask = torch.tensor((),device=p0.device)
        for i in range(p0.shape[0]):
            isnan = torch.isnan(p0[i,:,0]) # N
            index = torch.where(~isnan) 
            # 构造y
            y = torch.zeros((p0.shape[1], p0.shape[1]), device=p0.device) # N, N
            index = torch.meshgrid(index[0],index[0])
            y[index[0], index[1]] = 1
            y = y.unsqueeze(0) # 1, N, N
            attn_mask = torch.cat((attn_mask,y), dim=0)
        return attn_mask
        
    def curr_padding(self, p0):
        return p0[...,0].isnan()
    # def _node_to_nei(self, nodes):
    #     """use nodes positions to generate mask(adjacent matrix)

    #     Args:
    #         nodes (_type_): B,N,2

    #     Returns:
    #         _type_: _description_
    #     """
    #     seq_len = nodes.shape[0]
    #     max_nodes = nodes.shape[1]
    #     Ab = np.zeros((seq_len, max_nodes,max_nodes))    #neig_mat
        
    #     for s in range(seq_len):
    #         step_ = nodes[s,:,:]
    #         for h in range(len(step_)): 
    #             Ab[s,h,h] = 1   # neig_mat
    #             for k in range(h+1,len(step_)):
    #                 l2_norm = anorm(step_[h],step_[k])
    #                 Ab[s,h,k] = int((1/l2_norm)<self.config.neighbor_thred if l2_norm!=0 else 0)
    #                 Ab[s,k,h] = int((1/l2_norm)<self.config.neighbor_thred if l2_norm!=0 else 0)
                
    #     return torch.from_numpy(Ab).type(torch.float)

    def sample(self, context, curr = None, bestof=True, point_dim=2, timestep=0.08, ret_traj=False):
        """_summary_

        Args:
            num_points (_type_): _description_
            context (_type_): [bs(sample), obs_len+1(destination),N,2]
            sample (_type_): _description_
            bestof (_type_): _description_
            point_dim (int, optional): _description_. Defaults to 2.
            flexibility (float, optional): _description_. Defaults to 0.0.
            ret_traj (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        traj_list = []
        history = context[0]
        ped_features = context[1]
        assert context[0].shape[0]==context[1].shape[0]
        if curr!=None:
            # assert curr.shape[0]==ped_features.shape[0]
            assert len(curr.shape)==3 # t, n, 6
        batch_size = history.shape[0]
        num_ped = context[1].shape[1]
        if bestof:
            x_T = torch.randn([batch_size, num_ped, point_dim]).to('cuda')
        else:
            x_T = torch.zeros([batch_size, num_ped, point_dim]).to('cuda')
        traj = {self.var_sched.num_steps: x_T}
        # pbar = tqdm(range(self.var_sched.num_steps, 0, -1))
        pbar = range(self.var_sched.num_steps, 0, -1)
        nei_list = None
        p0 = curr[...,:2] # B, N, 2
        # if self.config.nei_attn_mask==True:
        #     nei_list = self.cur_mask(p0)
        if self.config.nei_padding_mask==True:
            nei_list = self.curr_padding(p0)
        for t in pbar:
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            # alpha = self.var_sched.alphas[t]
            # alpha_ = self.var_sched.alphas[t-1]
            # alpha_bar = self.var_sched.alpha_bars[t]
            alpha_bar_ = self.var_sched.alpha_bars[t-1]
            # if t == self.var_sched.num_steps:
            #     print('alpha_bar:',alpha_bar_)
            # sigma = self.var_sched.get_sigmas(t, flexibility)
            # sigma_ = self.var_sched.get_sigmas(t-1, flexibility)

            # c0 = 1.0 / torch.sqrt(alpha)
            # c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            c0 = torch.sqrt(alpha_bar_).view(-1, 1, 1).cuda()       # (B, 1, 1)
            c1 = torch.sqrt(1 - alpha_bar_).view(-1, 1, 1).cuda()

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]
            
            # if curr != None:
            #     p0[p0.isnan()]=0 # zero_padding
            #     p2 = (curr[...,4:6]*timestep+curr[...,2:4])*timestep + curr[...,2:4]*timestep + p0 # p2 = p0+v0*t+v1*t, v1=a0*t+v0
            #     x_t = (x_t*timestep+(curr[...,4:6]*timestep+curr[...,2:4]))*timestep + p2 # p3 = p2+v2*t, v2 = a1*t+v1
            #     assert torch.all(~x_t.isnan())
            x_0_hat = self.net(x_t, beta=beta, context=context ,nei_list=nei_list, t=t)
            # x_0_hat = self.net(x_t, beta=torch.tensor([t],device=x_t.device), context=context, nei_list = nei_list)
            mean, var = self.p_mean_variance(x_0_hat,x_t,t)
            x_next = mean + torch.sqrt(var)*z
            assert x_next.shape == x_t.shape
            # x_next = c0 * x_0_hat + c1 * z
            traj[t-1] = x_next
            # traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            
            # if not ret_traj:
            #    del traj[t]

        if ret_traj:
            traj_list.append(traj)
        else:
            traj_list.append(traj[0])
        global plot_beta_gate
        if plot_beta_gate:
            global beta_global
            global gatec_mean1_global
            global gatex_mean1_global
            global gatec_mean2_global
            global gatex_mean2_global
            import matplotlib.pyplot as plt
            fig,ax = plt.subplots(2,2)
            
            ax[0][0].scatter(beta_global, gatec_mean1_global,s=2)
            ax[0][0].set_title('layer1_gateec_mean')
            ax[0][1].scatter(beta_global, gatex_mean1_global,s=2)
            ax[0][1].set_title('layer1_gateed_mean')
            ax[1][0].scatter(beta_global, gatec_mean2_global,s=2)
            ax[1][0].set_title('layer2_gateec_mean')
            ax[1][1].scatter(beta_global, gatex_mean2_global,s=2)
            ax[1][1].set_title('layer2_gateed_mean')
            plt.tight_layout()
            import time
            plt.savefig('2gate_beta_plots'+'ver4_layer2'+'ep0'+'.png')
            plt.close()
            plot_beta_gate = False
        return torch.stack(traj_list).squeeze()

    def p_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self.var_sched.posterior_mean_coef1[t] * x_start +
            self.var_sched.posterior_mean_coef2[t]  * x_t
        )
        posterior_variance = self.var_sched.sigmas_inflex[t].view(x_start.shape[0],*[1]*(x_start.ndim-1))
        
        assert (posterior_mean.shape[0] == posterior_variance.shape[0]  ==
                x_start.shape[0])
        return posterior_mean, posterior_variance
