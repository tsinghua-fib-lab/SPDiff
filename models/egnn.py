import time
import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import data.data as DATA


def index_sum(agg_size, source, idx, cuda):
    """
        source is N x hid_dim [float]
        idx    is N           [int]
        
        Sums the rows source[.] with the same idx[.];
    """
    tmp = torch.zeros((agg_size, source.shape[1]))
    tmp = tmp.cuda() if cuda else tmp
    res = torch.index_add(tmp, 0, idx, source)
    return res

class ConvEGNN(nn.Module):
    def __init__(self, in_dim, hid_dim, cuda=True):
        super().__init__()
        self.hid_dim=hid_dim
        self.cuda = cuda
        
        # computes messages based on hidden representations -> [0, 1]
        self.f_e = nn.Sequential(
            nn.Linear(in_dim*2+1, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim), nn.SiLU())
        
        # update acceleration
        self.f_x = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, 2), nn.SiLU())
        
        self.f_a = nn.Sequential(
            nn.Linear(hid_dim, 2), nn.SiLU())
        
        # updates hidden representations -> [0, 1]
        self.f_h = nn.Sequential(
            nn.Linear(in_dim+hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim))
    
    def forward(self, ped_features, h_st, h_neigh, rela_features, neigh_mask):
        """
            ped_features: [bs, N, 6]
            h_st: [bs, N, dim_h]
            h_neigh: [bs, N, k, dim_h]
            rela_features: [bs, N, k, 6]
            neigh_index: [bs, N, k]
        """
        dists = torch.norm(rela_features[..., :2], dim=-1) # bs, N, k
        neigh_num = neigh_mask.sum(dim=-1) # bs, N
        
        # compute messages
        tmp = torch.cat([h_st.unsqueeze(-2).repeat((1, 1, dists.shape[-1], 1)), 
                            h_neigh, dists.unsqueeze(-1)], dim=-1) #bs, N, k, dim_h*2+1
        m_ij = self.f_e(tmp) #bs, N, k, dim_h
        
        # predict edges
        agg = rela_features[..., :2] * self.f_x(m_ij) # bs, N, k, 2
        agg[~neigh_mask.bool()] = 0. # bs, N, k, 2
        agg = 1/(neigh_num.unsqueeze(-1) + 1e-6) * agg.sum(dim=-2) # bs, N, 2
        a_new = self.f_a(h_st) * ped_features[...,4: ] + agg
        v_new = ped_features[...,2:4] + a_new
        x_new = ped_features[..., :2] + v_new # bs, N, 2
        
        # average e_ij-weighted messages  
        # m_i is num_nodes x hid_dim
        m_i = m_ij.sum(dim=-2) # bs, N, dim_h
        
        # update hidden representations (with residual)
        h_st = h_st + self.f_h(torch.cat([h_st, m_i], dim=-1))

        return torch.cat([x_new, v_new, a_new] ,dim=-1), h_st

class ConvEGNN2(nn.Module):
    """
        change log: modified NN structures
    """
    def __init__(self, in_dim, hid_dim, cuda=True):
        super().__init__()
        self.hid_dim=hid_dim
        self.cuda = cuda
        
        # computes messages based on hidden representations -> [0, 1]
        self.f_e = nn.Sequential(
            nn.Linear(in_dim*2+1, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim), nn.SiLU())
        
        # update acceleration
        self.f_x = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, 1))
        
        self.f_a = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, 1))
        
        # updates hidden representations -> [0, 1]
        self.f_h = nn.Sequential(
            nn.Linear(in_dim+hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim))
    
    def forward(self, ped_features, h_st, h_neigh, rela_features, neigh_mask):
        """
            ped_features: [bs, N, 6]
            h_st: [bs, N, dim_h]
            h_neigh: [bs, N, k, dim_h]
            rela_features: [bs, N, k, 6]
            neigh_index: [bs, N, k]
        """
        dists = torch.norm(rela_features[..., :2], dim=-1) # bs, N, k
        neigh_num = neigh_mask.sum(dim=-1) # bs, N
        
        # compute messages
        tmp = torch.cat([h_st.unsqueeze(-2).repeat((1, 1, dists.shape[-1], 1)), 
                            h_neigh, dists.unsqueeze(-1)], dim=-1) #bs, N, k, dim_h*2+1
        m_ij = self.f_e(tmp) #bs, N, k, dim_h
        
        # predict edges
        agg = rela_features[..., :2] * self.f_x(m_ij) # bs, N, k, 2
        agg[~neigh_mask.bool()] = 0. # bs, N, k, 2
        agg = 1/(neigh_num.unsqueeze(-1) + 1e-6) * agg.sum(dim=-2) # bs, N, 2
        a_new = self.f_a(h_st) * ped_features[...,4: ] + agg
        v_new = ped_features[...,2:4] + a_new
        x_new = ped_features[..., :2] + v_new # bs, N, 2
        
        # average e_ij-weighted messages  
        # m_i is num_nodes x hid_dim
        m_i = m_ij.sum(dim=-2) # bs, N, dim_h
        
        # update hidden representations (with residual)
        h_st = h_st + self.f_h(torch.cat([h_st, m_i], dim=-1))

        return torch.cat([x_new, v_new, a_new] ,dim=-1), h_st
    
class ConvEGNN3(nn.Module):
    """
        change log: modified NN structures
        see update in ver3
    """
    def __init__(self, in_dim, hid_dim, cuda=True):
        super().__init__()
        self.hid_dim=hid_dim
        self.cuda = cuda
        
        # computes messages based on hidden representations -> [0, 1]
        self.f_e = nn.Sequential(
            nn.Linear(in_dim*2+1, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim), nn.SiLU())
        
        # update acceleration
        self.f_x = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, 1))
        
        self.f_a = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, 1))
        
        # updates hidden representations -> [0, 1]
        self.f_h = nn.Sequential(
            nn.Linear(in_dim+hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim))
    
    def forward(self, ped_features, h_st, h_neigh, rela_features, neigh_mask):
        """
            ped_features: [bs, N, 6]
            h_st: [bs, N, dim_h]
            h_neigh: [bs, N, k, dim_h]
            rela_features: [bs, N, k, 6]
            neigh_index: [bs, N, k]
        """
        dists = torch.norm(rela_features[..., :2], dim=-1) # bs, N, k
        neigh_num = neigh_mask.sum(dim=-1) # bs, N
        
        # compute messages
        tmp = torch.cat([h_st.unsqueeze(-2).repeat((1, 1, dists.shape[-1], 1)), 
                            h_neigh, dists.unsqueeze(-1)], dim=-1) #bs, N, k, dim_h*2+1
        m_ij = self.f_e(tmp) #bs, N, k, dim_h

        # update in ver3
        m_ij[~neigh_mask.bool()] = 0.
        
        # predict edges
        agg = rela_features[..., :2] * self.f_x(m_ij) # bs, N, k, 2
        # agg[~neigh_mask.bool()] = 0. # bs, N, k, 2 # deleted in ver3
        agg = 1/(neigh_num.unsqueeze(-1) + 1e-6) * agg.sum(dim=-2) # bs, N, 2
        a_new = self.f_a(h_st) * ped_features[...,4: ] + agg
        v_new = ped_features[...,2:4] + a_new
        x_new = ped_features[..., :2] + v_new # bs, N, 2
        
        # average e_ij-weighted messages  
        # m_i is num_nodes x hid_dim
        m_i = m_ij.sum(dim=-2) # bs, N, dim_h
        
        # update hidden representations (with residual)
        h_st = h_st + self.f_h(torch.cat([h_st, m_i], dim=-1))

        return torch.cat([x_new, v_new, a_new] ,dim=-1), h_st

class ConvEGNN3_obs(nn.Module):
    """
        change log: modified NN structures
        see update in ver3
    """
    def __init__(self, in_dim, hid_dim, cuda=True):
        super().__init__()
        self.hid_dim=hid_dim
        self.cuda = cuda
        
        # computes messages based on hidden representations -> [0, 1]
        self.f_e = nn.Sequential(
            nn.Linear(in_dim+1, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim), nn.SiLU())
        
        # update acceleration
        self.f_x = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, 1))
        
        self.f_a = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, 1))
        
        # updates hidden representations -> [0, 1]
        self.f_h = nn.Sequential(
            nn.Linear(in_dim+hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim))
    
    def forward(self, ped_features, h_st_ped, rela_features, neigh_mask):
        """
            ped_features: [bs, N, 6]
            h_st_ped: [bs, N, dim_h]
            rela_features: [bs, N, k, 4]
            neigh_mask: [bs, N, k]
        """
        dists = torch.norm(rela_features[..., :2], dim=-1) # bs, N, k
        neigh_num = neigh_mask.sum(dim=-1) # bs, N
        
        # compute messages
        tmp = torch.cat([h_st_ped.unsqueeze(-2).repeat((1, 1, dists.shape[-1], 1)), dists.unsqueeze(-1)], dim=-1) #bs, N, k, dim_h*2+1
        m_ij = self.f_e(tmp) #bs, N, k, dim_h

        # update in ver3
        m_ij[~neigh_mask.bool()] = 0.
        
        # predict edges
        agg = rela_features[..., :2] * self.f_x(m_ij) # bs, N, k, 2
        # agg[~neigh_mask.bool()] = 0. # bs, N, k, 2
        agg = 1/(neigh_num.unsqueeze(-1) + 1e-6) * agg.sum(dim=-2) # bs, N, 2
        a_new = self.f_a(h_st_ped) * ped_features[...,4: ] + agg
        v_new = ped_features[...,2:4] + a_new
        x_new = ped_features[..., :2] + v_new # bs, N, 2
        
        # average e_ij-weighted messages  
        # m_i is num_nodes x hid_dim
        m_i = m_ij.sum(dim=-2) # bs, N, dim_h
        
        # update hidden representations (with residual)
        h_st_ped = h_st_ped + self.f_h(torch.cat([h_st_ped, m_i], dim=-1))

        return torch.cat([x_new, v_new, a_new] ,dim=-1), h_st_ped

class ConvEGNN4(nn.Module):
    """

        add relative speed
    """
    def __init__(self, in_dim, hid_dim, cuda=True):
        super().__init__()
        self.hid_dim=hid_dim
        self.cuda = cuda
        
        # computes messages based on hidden representations -> [0, 1]
        self.f_e = nn.Sequential(
            nn.Linear(in_dim*2+2, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim), nn.SiLU())
        
        # update acceleration
        self.f_x = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, 1))
        
        self.f_a = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, 1))
        
        # updates hidden representations -> [0, 1]
        self.f_h = nn.Sequential(
            nn.Linear(in_dim+hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim))
    
    def forward(self, ped_features, h_st, h_neigh, rela_features, neigh_mask):
        """
            ped_features: [bs, N, 6]
            h_st: [bs, N, dim_h]
            h_neigh: [bs, N, k, dim_h]
            rela_features: [bs, N, k, 6]
            neigh_index: [bs, N, k]
        """
        dists = torch.norm(rela_features[..., :2], dim=-1) # bs, N, k
        rela_speed = torch.norm(rela_features[..., 2:4], dim=-1)
        neigh_num = neigh_mask.sum(dim=-1) # bs, N
        
        # compute messages
        tmp = torch.cat([h_st.unsqueeze(-2).repeat((1, 1, dists.shape[-1], 1)), 
                            h_neigh, dists.unsqueeze(-1), rela_speed.unsqueeze(-1)], dim=-1) #bs, N, k, dim_h*2+1
        m_ij = self.f_e(tmp) #bs, N, k, dim_h

        m_ij[~neigh_mask.bool()] = 0.
        
        # predict edges
        agg = rela_features[..., :2] * self.f_x(m_ij) # bs, N, k, 2
        # agg[~neigh_mask.bool()] = 0. # bs, N, k, 2 # deleted in ver3

        agg = 1/(neigh_num.unsqueeze(-1) + 1e-6) * agg.sum(dim=-2) # bs, N, 2

        a_new = self.f_a(h_st) * ped_features[...,4: ] + agg 

        v_new = ped_features[...,2:4] + a_new
        x_new = ped_features[..., :2] + v_new # bs, N, 2
        
        # average e_ij-weighted messages  
        # m_i is num_nodes x hid_dim
        m_i = m_ij.sum(dim=-2) # bs, N, dim_h
        
        # update hidden representations (with residual)
        h_st = h_st + self.f_h(torch.cat([h_st, m_i], dim=-1))

        return torch.cat([x_new, v_new, a_new] ,dim=-1), h_st

class ConvEGNN4_obs(nn.Module):
    """
        change log: modified NN structures
        see update in ver3
    """
    def __init__(self, in_dim, hid_dim, cuda=True):
        super().__init__()
        self.hid_dim=hid_dim
        self.cuda = cuda
        
        # computes messages based on hidden representations -> [0, 1]
        self.f_e = nn.Sequential(
            nn.Linear(in_dim+2, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim), nn.SiLU())
        
        # update acceleration
        self.f_x = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, 1))
        
        self.f_a = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, 1))
        
        # updates hidden representations -> [0, 1]
        self.f_h = nn.Sequential(
            nn.Linear(in_dim+hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim))
    
    def forward(self, ped_features, h_st_ped, rela_features, neigh_mask):
        """
            ped_features: [bs, N, 6]
            h_st_ped: [bs, N, dim_h]
            rela_features: [bs, N, k, 4]
            neigh_mask: [bs, N, k]
        """
        rela_vel = torch.norm(rela_features[...,2:4], dim=-1)
        dists = torch.norm(rela_features[..., :2], dim=-1) # bs, N, k
        neigh_num = neigh_mask.sum(dim=-1) # bs, N
        
        # compute messages
        tmp = torch.cat([h_st_ped.unsqueeze(-2).repeat((1, 1, dists.shape[-1], 1)), dists.unsqueeze(-1), rela_vel.unsqueeze(-1)], dim=-1) #bs, N, k, dim_h*2+1
        m_ij = self.f_e(tmp) #bs, N, k, dim_h

        # update in ver3
        m_ij[~neigh_mask.bool()] = 0.
        
        # predict edges
        agg = rela_features[..., :2] * self.f_x(m_ij) # bs, N, k, 2
        # agg[~neigh_mask.bool()] = 0. # bs, N, k, 2
        agg = 1/(neigh_num.unsqueeze(-1) + 1e-6) * agg.sum(dim=-2) # bs, N, 2
        a_new = self.f_a(h_st_ped) * ped_features[...,4: ] + agg
        v_new = ped_features[...,2:4] + a_new
        x_new = ped_features[..., :2] + v_new # bs, N, 2
        
        # average e_ij-weighted messages  
        # m_i is num_nodes x hid_dim
        m_i = m_ij.sum(dim=-2) # bs, N, dim_h
        
        # update hidden representations (with residual)
        h_st_ped = h_st_ped + self.f_h(torch.cat([h_st_ped, m_i], dim=-1))

        return torch.cat([x_new, v_new, a_new] ,dim=-1), h_st_ped

class NetEGNN_acce(nn.Module, DATA.Pedestrians):
    """
        use convegnn4
    """
    def __init__(self, in_dim=3+8+8, hid_dim=64, out_dim=1, n_layers=3, cuda=True):
        super().__init__()
        self.hid_dim=hid_dim
        self.k_dim = 3
        self.encode_v = nn.Linear(1, 8) 
        self.encode_a = nn.Linear(1, 8) 
        self.emb = nn.Linear(in_dim, hid_dim) 

        self.gnn = nn.ModuleList(ConvEGNN4(hid_dim, hid_dim, cuda=cuda) for _ in range(n_layers))
        # self.gnn = nn.Sequential(*self.gnn)
        
        # self.pre_mlp = nn.Sequential(
        #     nn.Linear(hid_dim, hid_dim), nn.SiLU(),
        #     nn.Linear(hid_dim, hid_dim))
        
        # self.post_mlp = nn.Sequential(
        #     nn.Dropout(0.4),
        #     nn.Linear(hid_dim, hid_dim), nn.SiLU(),
        #     nn.Linear(hid_dim, out_dim))

        if cuda: self.cuda()
        self.cuda = cuda
    
    def forward(self, context, k_emb):
        """
            context: (list), 
                [0]ped_features (bs, N, 6), 
                [1]neigh_mask (bs, N, k), 
                [2]neigh_index (bs, N, k)
            k_emb: [bs, N, 3]
        """
        ped_features = context[0]
        neigh_mask = context[1]
        neigh_index = context[2]
        # obstacles = context[3]

        # torch.set_printoptions(threshold=np.inf)
        # print('velocity\n',torch.norm(ped_features[...,2:4],dim=-1))
        # print(torch.norm(ped_features[...,2:4],dim=-1).isnan().nonzero())
        # print(torch.norm(ped_features[...,2:4],dim=-1).isinf().nonzero())

        # print('acce\n',torch.norm(ped_features[...,4: ],dim=-1))
        # print(torch.norm(ped_features[...,4: ],dim=-1).isnan().nonzero())
        # print(torch.norm(ped_features[...,4: ],dim=-1).isinf().nonzero())


        h_initial = torch.cat((self.encode_v(torch.norm(ped_features[...,2:4],dim=-1, keepdim=True)),
                               self.encode_a(torch.norm(ped_features[...,4: ],dim=-1, keepdim=True)),
                               k_emb), dim=-1) # bs, N, 19 
        # print(h_initial.isnan().nonzero())
        # print(h_initial.isinf().nonzero())
        h_st = self.emb(h_initial) # bs, N, dim_h
        # print(h_st.isnan().nonzero())
        # print(h_st.isinf().nonzero())
        h_neigh = torch.zeros(h_st.shape[:2]+neigh_index.shape[-1:]+h_st.shape[-1:])
        # acce = ped_features[..., 4:].clone()

        for i, model in enumerate(self.gnn):
            # h_neigh = torch.zeros(neigh_index.shape+[h_st.shape[-1]]) # bs, N, k, dim_h
            # relative_features = torch.zeros(neigh_index.shape+[ped_features.shape[-1]]) # bs, N, k, 6

            h_neigh = torch.gather(h_st.unsqueeze(1).expand(h_st.shape[:2]+h_st.shape[1:]), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h
            # h_neigh = torch.gather(h_st.unsqueeze(1).repeat(1,h_st.shape[1],1,1), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h


            relative_features = self.get_relative_quantity(ped_features, ped_features) # bs, N, N, 6
            dim = neigh_index.dim()
            neigh_index2 = neigh_index.unsqueeze(-1).repeat(*([1]*dim + [relative_features.shape[-1]]))  # bs,n,k,6
            relative_features = torch.gather(relative_features, -2, neigh_index2)  # bs,n,k,6
            
            h_neigh[~neigh_mask.bool()]=0.
            relative_features[~neigh_mask.bool()]=0.
            #print('layer:',i)
            #print('h_neigh',h_neigh.isnan().sum())
            #print('relative_features',relative_features.isnan().sum())
            
            ped_features, h_st = model(ped_features, h_st, h_neigh, relative_features, neigh_mask)
            #print('acce',ped_features[...,4:].isnan().sum())
            #print('h_st',h_st.isnan().sum())

        output = ped_features[...,4:]
        # output = h_st

        return output # bs, N, 6; bs, N, dim_hd
    
class NetEGNN_acce2(nn.Module, DATA.Pedestrians):
    """
        use convegnn2
        remove encode v & a
    """
    def __init__(self, in_dim=3+1+1, hid_dim=64, out_dim=1, n_layers=3, cuda=True):
        super().__init__()
        self.hid_dim=hid_dim
        self.k_dim = 3
        # self.encode_v = nn.Linear(1, 8) 
        # self.encode_a = nn.Linear(1, 8) 
        self.emb = nn.Linear(in_dim, hid_dim) 

        self.gnn = nn.ModuleList(ConvEGNN2(hid_dim, hid_dim, cuda=cuda) for _ in range(n_layers))
        # self.gnn = nn.Sequential(*self.gnn)
        
        # self.pre_mlp = nn.Sequential(
        #     nn.Linear(hid_dim, hid_dim), nn.SiLU(),
        #     nn.Linear(hid_dim, hid_dim))
        
        # self.post_mlp = nn.Sequential(
        #     nn.Dropout(0.4),
        #     nn.Linear(hid_dim, hid_dim), nn.SiLU(),
        #     nn.Linear(hid_dim, out_dim))
        if cuda: self.cuda()
        self.cuda = cuda
    
    def forward(self, context, k_emb):
        """
            context: (list), 
                [0]ped_features (bs, N, 6), 
                [1]neigh_mask (bs, N, k), 
                [2]neigh_index (bs, N, k)
            k_emb: [bs, N, 3]
        """
        ped_features = context[0]
        neigh_mask = context[1]
        neigh_index = context[2]
        # obstacles = context[3]

        # torch.set_printoptions(threshold=np.inf)
        print('velocity\n',torch.norm(ped_features[...,2:4],dim=-1))
        print(torch.norm(ped_features[...,2:4],dim=-1).isnan().nonzero())
        print(torch.norm(ped_features[...,2:4],dim=-1).isinf().nonzero())

        print('acce\n',torch.norm(ped_features[...,4: ],dim=-1))
        print(torch.norm(ped_features[...,4: ],dim=-1).isnan().nonzero())
        print(torch.norm(ped_features[...,4: ],dim=-1).isinf().nonzero())


        h_initial = torch.cat((torch.norm(ped_features[...,2:4],dim=-1, keepdim=True),
                               torch.norm(ped_features[...,4: ],dim=-1, keepdim=True),
                               k_emb), dim=-1) # bs, N, 19 
        print(h_initial.isnan().nonzero())
        print(h_initial.isinf().nonzero())
        h_st = self.emb(h_initial) # bs, N, dim_h
        print(h_st.isnan().nonzero())
        print(h_st.isinf().nonzero())
        h_neigh = torch.zeros(h_st.shape[:2]+neigh_index.shape[-1:]+h_st.shape[-1:])
        acce = ped_features[..., 4:].clone()

        for i, model in enumerate(self.gnn):
            # h_neigh = torch.zeros(neigh_index.shape+[h_st.shape[-1]]) # bs, N, k, dim_h
            # relative_features = torch.zeros(neigh_index.shape+[ped_features.shape[-1]]) # bs, N, k, 6

            h_neigh = torch.gather(h_st.unsqueeze(1).expand(h_st.shape[:2]+h_st.shape[1:]), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h
            # h_neigh = torch.gather(h_st.unsqueeze(1).repeat(1,h_st.shape[1],1,1), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h


            relative_features = self.get_relative_quantity(ped_features, ped_features) # bs, N, N, 6
            dim = neigh_index.dim()
            neigh_index2 = neigh_index.unsqueeze(-1).repeat(*([1]*dim + [relative_features.shape[-1]]))  # bs,n,k,6
            relative_features = torch.gather(relative_features, -2, neigh_index2)  # bs,n,k,6
            
            h_neigh[~neigh_mask.bool()]=0.
            relative_features[~neigh_mask.bool()]=0.
            #print('layer:',i)
            #print('h_neigh',h_neigh.isnan().sum())
            #print('relative_features',relative_features.isnan().sum())
            
            ped_features, h_st = model(torch.cat((ped_features[...,:4], acce), dim=-1), h_st, h_neigh, relative_features, neigh_mask)
            # ped_features, h_st = model(ped_features, h_st, h_neigh, relative_features, neigh_mask)
            
            #print('acce',ped_features[...,4:].isnan().sum())
            #print('h_st',h_st.isnan().sum())

        output = ped_features[...,4:]
        # output = h_st

        return output # bs, N, 6; bs, N, dim_hd

class NetEGNN_acce_hid(nn.Module, DATA.Pedestrians):
    """
        use convegnn2
        remove encode v & a
        output acce and hid emb
    """
    def __init__(self, in_dim=3+8+8, hid_dim=64, out_dim=1, n_layers=3, cuda=True):
        super().__init__()
        self.hid_dim=hid_dim
        self.k_dim = 3
        self.encode_v = nn.Linear(1, 8) 
        self.encode_a = nn.Linear(1, 8) 
        self.emb = nn.Linear(in_dim, hid_dim) 

        self.gnn = nn.ModuleList(ConvEGNN3(hid_dim, hid_dim, cuda=cuda) for _ in range(n_layers))
        # self.gnn = nn.Sequential(*self.gnn)
        
        # self.pre_mlp = nn.Sequential(
        #     nn.Linear(hid_dim, hid_dim), nn.SiLU(),
        #     nn.Linear(hid_dim, hid_dim))
        
        # self.post_mlp = nn.Sequential(
        #     nn.Dropout(0.4),
        #     nn.Linear(hid_dim, hid_dim), nn.SiLU(),
        #     nn.Linear(hid_dim, out_dim))
        if cuda: self.cuda()
        self.cuda = cuda
    
    def forward(self, context, k_emb):
        """
            context: (list), 
                [0]ped_features (bs, N, 6), 
                [1]neigh_mask (bs, N, k), 
                [2]neigh_index (bs, N, k)
            k_emb: [bs, N, 3]
        """
        ped_features = context[0]
        neigh_mask = context[1]
        neigh_index = context[2]
        # obstacles = context[3]

        # torch.set_printoptions(threshold=np.inf)
        # print('velocity\n',torch.norm(ped_features[...,2:4],dim=-1))
        # print(torch.norm(ped_features[...,2:4],dim=-1).isnan().nonzero())
        # print(torch.norm(ped_features[...,2:4],dim=-1).isinf().nonzero())

        # print('acce\n',torch.norm(ped_features[...,4: ],dim=-1))
        # print(torch.norm(ped_features[...,4: ],dim=-1).isnan().nonzero())
        # print(torch.norm(ped_features[...,4: ],dim=-1).isinf().nonzero())


        h_initial = torch.cat((self.encode_v(torch.norm(ped_features[...,2:4],dim=-1, keepdim=True)),
                               self.encode_a(torch.norm(ped_features[...,4: ],dim=-1, keepdim=True)),
                               k_emb), dim=-1) # bs, N, 19 
        # print(h_initial.isnan().nonzero())
        # print(h_initial.isinf().nonzero())
        h_st = self.emb(h_initial) # bs, N, dim_h
        # print(h_st.isnan().nonzero())
        # print(h_st.isinf().nonzero())
        h_neigh = torch.zeros(h_st.shape[:2]+neigh_index.shape[-1:]+h_st.shape[-1:])
        # acce = ped_features[..., 4:].clone()

        for i, model in enumerate(self.gnn):
            # if h_st.isnan().any() or h_st.isinf().any() or ped_features[...,2:].isnan().any() or ped_features[...,2:].isinf().any():
            #     pdb.set_trace()
            # h_neigh = torch.zeros(neigh_index.shape+[h_st.shape[-1]]) # bs, N, k, dim_h
            # relative_features = torch.zeros(neigh_index.shape+[ped_features.shape[-1]]) # bs, N, k, 6

            h_neigh = torch.gather(h_st.unsqueeze(1).expand(h_st.shape[:2]+h_st.shape[1:]), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h
            # h_neigh = torch.gather(h_st.unsqueeze(1).repeat(1,h_st.shape[1],1,1), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h


            relative_features = self.get_relative_quantity(ped_features, ped_features) # bs, N, N, 6
            dim = neigh_index.dim()
            neigh_index2 = neigh_index.unsqueeze(-1).repeat(*([1]*dim + [relative_features.shape[-1]]))  # bs,n,k,6
            relative_features = torch.gather(relative_features, -2, neigh_index2)  # bs,n,k,6
            
            h_neigh[~neigh_mask.bool()]=0.
            relative_features[~neigh_mask.bool()]=0.
            #print('layer:',i)
            #print('h_neigh',h_neigh.isnan().sum())
            #print('relative_features',relative_features.isnan().sum())
            
            # ped_features, h_st = model(torch.cat((ped_features[...,:4], acce), dim=-1), h_st, h_neigh, relative_features, neigh_mask)
            ped_features, h_st = model(ped_features, h_st, h_neigh, relative_features, neigh_mask)
            
            #print('acce',ped_features[...,4:].isnan().sum())
            #print('h_st',h_st.isnan().sum())

        output = [ped_features[...,4:] , h_st]
        # output = h_st

        return output # bs, N, 6; bs, N, dim_hd

class NetEGNN_hid(nn.Module, DATA.Pedestrians):
    def __init__(self, in_dim=3+8+8, hid_dim=64, out_dim=1, n_layers=3, cuda=True):
        super().__init__()
        self.hid_dim=hid_dim
        self.k_dim = 3
        self.encode_v = nn.Linear(1, 8) 
        self.encode_a = nn.Linear(1, 8) 
        self.emb = nn.Linear(in_dim, hid_dim) 

        self.gnn = nn.ModuleList(ConvEGNN3(hid_dim, hid_dim, cuda=cuda) for _ in range(n_layers))
        # self.gnn = nn.Sequential(*self.gnn)
        
        # self.pre_mlp = nn.Sequential(
        #     nn.Linear(hid_dim, hid_dim), nn.SiLU(),
        #     nn.Linear(hid_dim, hid_dim))
        
        # self.post_mlp = nn.Sequential(
        #     nn.Dropout(0.4),
        #     nn.Linear(hid_dim, hid_dim), nn.SiLU(),
        #     nn.Linear(hid_dim, out_dim))

        if cuda: self.cuda()
        self.cuda = cuda
    
    def forward(self, context, k_emb):
        """
            context: (list), 
                [0]ped_features (bs, N, 6), 
                [1]neigh_mask (bs, N, k), 
                [2]neigh_index (bs, N, k)
            k_emb: [bs, N, 3]
        """
        ped_features = context[0]
        neigh_mask = context[1]
        neigh_index = context[2]
        
        
        h_initial = torch.cat((self.encode_v(torch.norm(ped_features[...,2:4],dim=-1, keepdim=True)),
                               self.encode_a(torch.norm(ped_features[...,4: ],dim=-1, keepdim=True)),
                               k_emb), dim=-1) # bs, N, 19 
        h_st = self.emb(h_initial) # bs, N, dim_h
        h_neigh = torch.zeros(h_st.shape[:2]+neigh_index.shape[-1:]+h_st.shape[-1:])
        for i, model in enumerate(self.gnn):
            # h_neigh = torch.zeros(neigh_index.shape+[h_st.shape[-1]]) # bs, N, k, dim_h
            # relative_features = torch.zeros(neigh_index.shape+[ped_features.shape[-1]]) # bs, N, k, 6

            h_neigh = torch.gather(h_st.unsqueeze(1).expand(h_st.shape[:2]+h_st.shape[1:]), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h
            # h_neigh = torch.gather(h_st.unsqueeze(1).repeat(1,h_st.shape[1],1,1), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h


            relative_features = self.get_relative_quantity(ped_features, ped_features) # bs, N, N, 6
            dim = neigh_index.dim()
            neigh_index2 = neigh_index.unsqueeze(-1).repeat(*([1]*dim + [relative_features.shape[-1]]))  # bs,n,k,6
            relative_features = torch.gather(relative_features, -2, neigh_index2)  # bs,n,k,6
            
            h_neigh[~neigh_mask.bool()]=0.
            relative_features[~neigh_mask.bool()]=0.
            #print('layer:',i)
            #print('h_neigh',h_neigh.isnan().sum())
            #print('relative_features',relative_features.isnan().sum())
            
            ped_features, h_st = model(ped_features, h_st, h_neigh, relative_features, neigh_mask)
            #print('acce',ped_features[...,4:].isnan().sum())
            #print('h_st',h_st.isnan().sum())

        # output = ped_features[...,4:]
        output = h_st

        return output # bs, N, 6; bs, N, dim_hd



class NetEGNN_hid_obs(nn.Module, DATA.Pedestrians):
    def __init__(self, in_dim=3+8+8, hid_dim=64, out_dim=1, n_layers=3, cuda=True):
        super().__init__()
        self.hid_dim=hid_dim
        self.k_dim = 3
        self.encode_v = nn.Linear(1, 8) 
        self.encode_a = nn.Linear(1, 8) 
        self.emb = nn.Linear(in_dim, hid_dim) 

        self.gnn = nn.ModuleList(ConvEGNN3_obs(hid_dim, hid_dim, cuda=cuda) for _ in range(n_layers))
        # self.gnn = nn.Sequential(*self.gnn)
        
        # self.pre_mlp = nn.Sequential(
        #     nn.Linear(hid_dim, hid_dim), nn.SiLU(),
        #     nn.Linear(hid_dim, hid_dim))
        
        # self.post_mlp = nn.Sequential(
        #     nn.Dropout(0.4),
        #     nn.Linear(hid_dim, hid_dim), nn.SiLU(),
        #     nn.Linear(hid_dim, out_dim))

        if cuda: self.cuda()
        self.cuda = cuda
    
    def forward(self, context, k_emb):
        """
            context: (list), 
                [0]ped_features (bs, N, 6), 
                [1]obs_features (bs, N, 2), 
                [2]neigh_mask (bs, N, k), 
                [3]neigh_index (bs, N, k)
            k_emb: [bs, N, 3]
        """
        ped_features = context[0]
        obs_features = context[1]
        neigh_mask = context[2]
        neigh_index = context[3]
        
        # print('before gnn velo',ped_features[...,2:4].isnan().sum())
        # print('before gnn acce',ped_features[...,4: ].isnan().sum())
        # print('before gnn kemb',k_emb.isnan().sum())

        
        h_initial_ped = torch.cat((self.encode_v(torch.norm(ped_features[...,2:4],dim=-1, keepdim=True)),
                               self.encode_a(torch.norm(ped_features[...,4: ],dim=-1, keepdim=True)),
                               k_emb), dim=-1) # bs, N, 19 
        # print('before gnn h_initial',h_initial.isnan().sum())
        h_st_ped = self.emb(h_initial_ped) # bs, N, dim_h
        # print('emb layer weight', list(self.emb.parameters())[0].isnan().sum())
        # print('emb layer bias', list(self.emb.parameters())[1].isnan().sum())
        # print('before gnn h_st',h_st.isnan().sum())
        # h_neigh = torch.zeros(h_st.shape[:2]+neigh_index.shape[-1:]+h_st.shape[-1:])
        obs_features = torch.cat((obs_features, torch.zeros(obs_features.shape, device=obs_features.device)), dim=-1)
        

        for i, model in enumerate(self.gnn):
            # h_neigh = torch.zeros(neigh_index.shape+[h_st.shape[-1]]) # bs, N, k, dim_h
            # relative_features = torch.zeros(neigh_index.shape+[ped_features.shape[-1]]) # bs, N, k, 6
            # print('layer:',i)

            # print('h_st',h_st.isnan().sum())

            # h_neigh = torch.gather(h_st.unsqueeze(1).expand(h_st.shape[:2]+h_st.shape[1:]), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h
            # h_neigh = torch.gather(h_st.unsqueeze(1).repeat(1,h_st.shape[1],1,1), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h
            # pdb.set_trace()
            assert obs_features.shape[0] == ped_features.shape[0]
            relative_features = self.get_relative_quantity(ped_features[...,:4] , obs_features) # bs, N, M, 4
            dim = neigh_index.dim()
            neigh_index2 = neigh_index.unsqueeze(-1).repeat(*([1]*dim + [relative_features.shape[-1]]))  # bs,n,k,6
            relative_features = torch.gather(relative_features, -2, neigh_index2)  # bs,n,k,6
            
            # h_neigh[~neigh_mask.bool()]=0.
            relative_features[~neigh_mask.bool()]=0.
            # print('h_neigh',h_neigh.isnan().sum())
            #print('relative_features',relative_features.isnan().sum())
            
            ped_features, h_st_ped = model(ped_features, h_st_ped, relative_features, neigh_mask)
            #print('acce',ped_features[...,4:].isnan().sum())
            #print('h_st',h_st.isnan().sum())

        # output = ped_features[...,4:]
        output = h_st_ped

        return output # bs, N, 6; bs, N, dim_hd

class NetEGNN_hid2(nn.Module, DATA.Pedestrians):
    def __init__(self, in_dim=3+8+8, hid_dim=64, out_dim=1, n_layers=3, cuda=True):
        super().__init__()
        self.hid_dim=hid_dim
        self.k_dim = 3
        self.encode_v = nn.Linear(1, 8) 
        self.encode_a = nn.Linear(1, 8) 
        self.emb = nn.Linear(in_dim, hid_dim) 

        self.gnn = nn.ModuleList(ConvEGNN3(hid_dim, hid_dim, cuda=cuda) for _ in range(n_layers))
        # self.gnn = nn.Sequential(*self.gnn)
        
        # self.pre_mlp = nn.Sequential(
        #     nn.Linear(hid_dim, hid_dim), nn.SiLU(),
        #     nn.Linear(hid_dim, hid_dim))
        
        # self.post_mlp = nn.Sequential(
        #     nn.Dropout(0.4),
        #     nn.Linear(hid_dim, hid_dim), nn.SiLU(),
        #     nn.Linear(hid_dim, out_dim))

        if cuda: self.cuda()
        self.cuda = cuda
    
    def forward(self, context, k_emb):
        """
            context: (list), 
                [0]ped_features (bs, N, 6), 
                [1]neigh_mask (bs, N, k), 
                [2]neigh_index (bs, N, k)
            k_emb: [bs, N, 3]
        """
        ped_features = context[0]
        neigh_mask = context[1]
        neigh_index = context[2]
        
        # print('before gnn velo',ped_features[...,2:4].isnan().sum())
        # print('before gnn acce',ped_features[...,4: ].isnan().sum())
        # print('before gnn kemb',k_emb.isnan().sum())

        
        h_initial = torch.cat((self.encode_v(torch.norm(ped_features[...,2:4],dim=-1, keepdim=True)),
                               self.encode_a(torch.norm(ped_features[...,4: ],dim=-1, keepdim=True)),
                               k_emb), dim=-1) # bs, N, 19 
        # print('before gnn h_initial',h_initial.isnan().sum())
        h_st = self.emb(h_initial) # bs, N, dim_h
        # print('emb layer weight', list(self.emb.parameters())[0].isnan().sum())
        # print('emb layer bias', list(self.emb.parameters())[1].isnan().sum())
        # print('before gnn h_st',h_st.isnan().sum())
        h_neigh = torch.zeros(h_st.shape[:2]+neigh_index.shape[-1:]+h_st.shape[-1:])
        acce = ped_features[..., 4:].clone()

        for i, model in enumerate(self.gnn):
            # h_neigh = torch.zeros(neigh_index.shape+[h_st.shape[-1]]) # bs, N, k, dim_h
            # relative_features = torch.zeros(neigh_index.shape+[ped_features.shape[-1]]) # bs, N, k, 6
            # print('layer:',i)

            # print('h_st',h_st.isnan().sum())

            h_neigh = torch.gather(h_st.unsqueeze(1).expand(h_st.shape[:2]+h_st.shape[1:]), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h
            # h_neigh = torch.gather(h_st.unsqueeze(1).repeat(1,h_st.shape[1],1,1), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h


            relative_features = self.get_relative_quantity(ped_features, ped_features) # bs, N, N, 6
            dim = neigh_index.dim()
            neigh_index2 = neigh_index.unsqueeze(-1).repeat(*([1]*dim + [relative_features.shape[-1]]))  # bs,n,k,6
            relative_features = torch.gather(relative_features, -2, neigh_index2)  # bs,n,k,6
            
            h_neigh[~neigh_mask.bool()]=0.
            relative_features[~neigh_mask.bool()]=0.
            # print('h_neigh',h_neigh.isnan().sum())
            #print('relative_features',relative_features.isnan().sum())
            
            #ped_features, h_st = model(ped_features, h_st, h_neigh, relative_features, neigh_mask)
            ped_features, h_st = model(torch.cat((ped_features[...,:4], acce), dim=-1), h_st, h_neigh, relative_features, neigh_mask)
            #print('acce',ped_features[...,4:].isnan().sum())
            #print('h_st',h_st.isnan().sum())

        # output = ped_features[...,4:]
        output = h_st

        return output # bs, N, 6; bs, N, dim_hd

class NetEGNN_hid_obs2(nn.Module, DATA.Pedestrians):
    def __init__(self, in_dim=3+8+8, hid_dim=64, out_dim=1, n_layers=3, cuda=True):
        super().__init__()
        self.hid_dim=hid_dim
        self.k_dim = 3
        self.encode_v = nn.Linear(1, 8) 
        self.encode_a = nn.Linear(1, 8) 
        self.emb = nn.Linear(in_dim, hid_dim) 

        self.gnn = nn.ModuleList(ConvEGNN3_obs(hid_dim, hid_dim, cuda=cuda) for _ in range(n_layers))
        # self.gnn = nn.Sequential(*self.gnn)
        
        # self.pre_mlp = nn.Sequential(
        #     nn.Linear(hid_dim, hid_dim), nn.SiLU(),
        #     nn.Linear(hid_dim, hid_dim))
        
        # self.post_mlp = nn.Sequential(
        #     nn.Dropout(0.4),
        #     nn.Linear(hid_dim, hid_dim), nn.SiLU(),
        #     nn.Linear(hid_dim, out_dim))

        if cuda: self.cuda()
        self.cuda = cuda
    
    def forward(self, context, k_emb):
        """
            context: (list), 
                [0]ped_features (bs, N, 6), 
                [1]obs_features (bs, N, 2), 
                [2]neigh_mask (bs, N, k), 
                [3]neigh_index (bs, N, k)
            k_emb: [bs, N, 3]
        """
        ped_features = context[0]
        obs_features = context[1]
        neigh_mask = context[2]
        neigh_index = context[3]
        
        # print('before gnn velo',ped_features[...,2:4].isnan().sum())
        # print('before gnn acce',ped_features[...,4: ].isnan().sum())
        # print('before gnn kemb',k_emb.isnan().sum())

        
        h_initial_ped = torch.cat((self.encode_v(torch.norm(ped_features[...,2:4],dim=-1, keepdim=True)),
                               self.encode_a(torch.norm(ped_features[...,4: ],dim=-1, keepdim=True)),
                               k_emb), dim=-1) # bs, N, 19 
        # print('before gnn h_initial',h_initial.isnan().sum())
        h_st_ped = self.emb(h_initial_ped) # bs, N, dim_h
        # print('emb layer weight', list(self.emb.parameters())[0].isnan().sum())
        # print('emb layer bias', list(self.emb.parameters())[1].isnan().sum())
        # print('before gnn h_st',h_st.isnan().sum())
        # h_neigh = torch.zeros(h_st.shape[:2]+neigh_index.shape[-1:]+h_st.shape[-1:])
        obs_features = torch.cat((obs_features, torch.zeros(obs_features.shape, device=obs_features.device)), dim=-1)
        acce = ped_features[..., 4:].clone()
        

        for i, model in enumerate(self.gnn):
            # h_neigh = torch.zeros(neigh_index.shape+[h_st.shape[-1]]) # bs, N, k, dim_h
            # relative_features = torch.zeros(neigh_index.shape+[ped_features.shape[-1]]) # bs, N, k, 6
            # print('layer:',i)

            # print('h_st',h_st.isnan().sum())

            # h_neigh = torch.gather(h_st.unsqueeze(1).expand(h_st.shape[:2]+h_st.shape[1:]), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h
            # h_neigh = torch.gather(h_st.unsqueeze(1).repeat(1,h_st.shape[1],1,1), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h
            # pdb.set_trace()
            assert obs_features.shape[0] == ped_features.shape[0]
            relative_features = self.get_relative_quantity(ped_features[...,:4] , obs_features) # bs, N, M, 4
            dim = neigh_index.dim()
            neigh_index2 = neigh_index.unsqueeze(-1).repeat(*([1]*dim + [relative_features.shape[-1]]))  # bs,n,k,6
            relative_features = torch.gather(relative_features, -2, neigh_index2)  # bs,n,k,6
            
            # h_neigh[~neigh_mask.bool()]=0.
            relative_features[~neigh_mask.bool()]=0.
            # print('h_neigh',h_neigh.isnan().sum())
            #print('relative_features',relative_features.isnan().sum())
            
            ped_features, h_st_ped = model(torch.cat((ped_features[...,:4], acce), dim=-1), h_st_ped, relative_features, neigh_mask)
            #print('acce',ped_features[...,4:].isnan().sum())
            #print('h_st',h_st.isnan().sum())

        # output = ped_features[...,4:]
        output = h_st_ped

        return output # bs, N, 6; bs, N, dim_hd


class NetEGNN_hid_ped_obs2(nn.Module, DATA.Pedestrians):
    def __init__(self, in_dim=3+8+8, hid_dim=64, hid_dim_obs=64, n_layers=3, n_layers_obs=1, cuda=True):
        super().__init__()
        self.hid_dim=hid_dim
        self.k_dim = 3
        self.encode_v = nn.Linear(1, 8) 
        self.encode_a = nn.Linear(1, 8) 
        self.emb = nn.Linear(in_dim, hid_dim) 

        self.gnn = nn.ModuleList(ConvEGNN3(hid_dim, hid_dim, cuda=cuda) for _ in range(n_layers))
        self.obs_gnn = nn.ModuleList(ConvEGNN3_obs(hid_dim, hid_dim, cuda=cuda) for _ in range(n_layers))

        # self.gnn = nn.Sequential(*self.gnn)
        
        # self.pre_mlp = nn.Sequential(
        #     nn.Linear(hid_dim, hid_dim), nn.SiLU(),
        #     nn.Linear(hid_dim, hid_dim))
        
        # self.post_mlp = nn.Sequential(
        #     nn.Dropout(0.4),
        #     nn.Linear(hid_dim, hid_dim), nn.SiLU(),
        #     nn.Linear(hid_dim, out_dim))

        if cuda: self.cuda()
        self.cuda = cuda
    
    def forward(self, context, k_emb):
        """
            context: (list), 
                [0]ped_features (bs, N, 6), 
                [1]neigh_mask (bs, N, k), 
                [2]neigh_index (bs, N, k)
            k_emb: [bs, N, 3]
        """
        ped_features = context[0]
        neigh_mask = context[1]
        neigh_index = context[2]
        obs_features = context[3]
        neigh_mask_obs = context[4]
        neigh_index_obs = context[5]
        
        # print('before gnn velo',ped_features[...,2:4].isnan().sum())
        # print('before gnn acce',ped_features[...,4: ].isnan().sum())
        # print('before gnn kemb',k_emb.isnan().sum())

        
        h_initial = torch.cat((self.encode_v(torch.norm(ped_features[...,2:4],dim=-1, keepdim=True)),
                               self.encode_a(torch.norm(ped_features[...,4: ],dim=-1, keepdim=True)),
                               k_emb), dim=-1) # bs, N, 19 
        # print('before gnn h_initial',h_initial.isnan().sum())
        h_st = self.emb(h_initial) # bs, N, dim_h

        # print('emb layer weight', list(self.emb.parameters())[0].isnan().sum())
        # print('emb layer bias', list(self.emb.parameters())[1].isnan().sum())
        # print('before gnn h_st',h_st.isnan().sum())
        # h_neigh = torch.zeros(h_st.shape[:2]+neigh_index.shape[-1:]+h_st.shape[-1:])
        obs_features = torch.cat((obs_features, torch.zeros(obs_features.shape, device=obs_features.device)), dim=-1)

        acce = ped_features[..., 4:].clone()
        ped_features_obs = ped_features.clone()
        h_st_obs = h_st

        for i, model in enumerate(self.gnn):
            # h_neigh = torch.zeros(neigh_index.shape+[h_st.shape[-1]]) # bs, N, k, dim_h
            # relative_features = torch.zeros(neigh_index.shape+[ped_features.shape[-1]]) # bs, N, k, 6
            # print('layer:',i)

            # print('h_st',h_st.isnan().sum())

            h_neigh = torch.gather(h_st.unsqueeze(1).expand(h_st.shape[:2]+h_st.shape[1:]), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h
            # h_neigh = torch.gather(h_st.unsqueeze(1).repeat(1,h_st.shape[1],1,1), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h


            relative_features = self.get_relative_quantity(ped_features, ped_features) # bs, N, N, 6
            dim = neigh_index.dim()
            neigh_index2 = neigh_index.unsqueeze(-1).repeat(*([1]*dim + [relative_features.shape[-1]]))  # bs,n,k,6
            relative_features = torch.gather(relative_features, -2, neigh_index2)  # bs,n,k,6
            
            h_neigh[~neigh_mask.bool()]=0.
            relative_features[~neigh_mask.bool()]=0.
            # print('h_neigh',h_neigh.isnan().sum())
            #print('relative_features',relative_features.isnan().sum())
            
            #ped_features, h_st = model(ped_features, h_st, h_neigh, relative_features, neigh_mask)
            ped_features, h_st = model(torch.cat((ped_features[...,:4], acce), dim=-1), h_st, h_neigh, relative_features, neigh_mask)
            #print('acce',ped_features[...,4:].isnan().sum())
            #print('h_st',h_st.isnan().sum())

        # output = ped_features[...,4:]
        output1 = h_st

        for i, model in enumerate(self.obs_gnn):
            # h_neigh = torch.zeros(neigh_index.shape+[h_st.shape[-1]]) # bs, N, k, dim_h
            # relative_features = torch.zeros(neigh_index.shape+[ped_features.shape[-1]]) # bs, N, k, 6
            # print('layer:',i)

            # print('h_st',h_st.isnan().sum())

            # h_neigh = torch.gather(h_st.unsqueeze(1).expand(h_st.shape[:2]+h_st.shape[1:]), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h
            # h_neigh = torch.gather(h_st.unsqueeze(1).repeat(1,h_st.shape[1],1,1), 2, neigh_index.unsqueeze(-1).repeat((1,1,1,self.hid_dim))) # bs, N, k, dim_h
            # pdb.set_trace()
            assert obs_features.shape[0] == ped_features.shape[0]
            relative_features_obs = self.get_relative_quantity(ped_features_obs[...,:4] , obs_features) # bs, N, M, 4
            dim = neigh_index_obs.dim()
            neigh_index2_obs = neigh_index_obs.unsqueeze(-1).repeat(*([1]*dim + [relative_features_obs.shape[-1]]))  # bs,n,k,6
            relative_features_obs = torch.gather(relative_features_obs, -2, neigh_index2_obs)  # bs,n,k,6
            
            # h_neigh[~neigh_mask.bool()]=0.
            relative_features_obs[~neigh_mask_obs.bool()]=0.
            # print('h_neigh',h_neigh.isnan().sum())
            #print('relative_features',relative_features.isnan().sum())
            
            ped_features_obs, h_st_obs = model(torch.cat((ped_features_obs[...,:4], acce), dim=-1), h_st_obs, relative_features_obs, neigh_mask_obs)
            #print('acce',ped_features[...,4:].isnan().sum())
            #print('h_st',h_st.isnan().sum())

        # output = ped_features[...,4:]
        output2 = h_st_obs


        return output1, output2 # bs, N, 6; bs, N, dim_hd