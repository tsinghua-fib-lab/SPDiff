"""
    trajectory dataset code from Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction
"""

import os
import math
import sys
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import dill
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time
from scipy.interpolate import interp1d

M = np.array([[ 2.84217540e-02,  2.97335273e-03,  6.02821031e+00],
                [-1.67162992e-03,  4.40195878e-02,  7.29109248e+00],
                [-9.83343172e-05,  5.42377797e-04,  1.00000000e+00]])

neighbor_thred = 1e5

def anorm(p1,p2): 
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)
                
def seq_to_graph(seq_,seq_rel,norm_lap_matr = True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    
    V = np.zeros((seq_len,max_nodes,2))
    A = np.zeros((seq_len,max_nodes,max_nodes))
    Ab = np.zeros((seq_len,max_nodes,max_nodes))    #neig_mat
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        step_rel = seq_rel[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_rel[h]
            A[s,h,h] = 1
            Ab[s,h,h] = 1   # neig_mat
            for k in range(h+1,len(step_)):
                l2_norm = anorm(step_[h],step_[k])
                A[s,h,k] = l2_norm
                A[s,k,h] = l2_norm
                Ab[s,h,k] = int((1/l2_norm)<neighbor_thred if l2_norm!=0 else 0)
                Ab[s,k,h] = int((1/l2_norm)<neighbor_thred if l2_norm!=0 else 0)
        if norm_lap_matr: 
            G = nx.from_numpy_matrix(A[s,:,:])
            A[s,:,:] = nx.normalized_laplacian_matrix(G).toarray()
            
    return torch.from_numpy(V).type(torch.float),\
           torch.from_numpy(A).type(torch.float), torch.from_numpy(Ab).type(torch.int)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
def read_file(_path, delim='\t', normalize = False):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    dataf = pd.DataFrame(data,columns=['frame','id','pos_x','pos_y'])
    dataf_new = pd.DataFrame()
    ids = dataf.id.unique()
    time_unit=1/12.5
    for _id in ids:
        dataf_i = dataf[dataf.id==_id]
        begin_f,end_f = dataf_i.frame.min(),dataf_i.frame.max()
        if end_f-begin_f<30:
            continue
        sample_f = np.arange(begin_f, end_f+1, 25*time_unit)
        # image 2 world coord
        image_coordination = np.concatenate((dataf_i.pos_x.values[:,np.newaxis], dataf_i.pos_y.values[:,np.newaxis], np.ones((len(dataf_i.pos_x),1))), axis=1)
        world_coordination = np.einsum('ij,nj->ni', M, image_coordination)
        interp_x = interp1d(dataf_i.frame.values, world_coordination[:,0]/world_coordination[:, 2], kind='cubic')(sample_f)
        interp_y = interp1d(dataf_i.frame.values, world_coordination[:,1]/world_coordination[:, 2], kind='cubic')(sample_f)
            
        dataf_new=pd.concat([dataf_new,pd.DataFrame([sample_f,_id*np.ones(len(sample_f)),interp_x,interp_y]).T])
    dataf_new.columns=['frame','id','inter_x','inter_y']
    dataf_new.frame=dataf_new.frame.astype(int)
    dataf_new.id=dataf_new.id.astype(int)
    dataf_new.sort_values('frame',inplace=True)
    if normalize:
        dataf_new.inter_x = dataf_new.inter_x-dataf_new.inter_x.mean()
        dataf_new.inter_y = dataf_new.inter_y-dataf_new.inter_y.mean()
    return dataf_new.to_numpy()

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
        min_ped=1, delim='\t',norm_lap_matr = True, normalize=False):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        vel_list = []
        acc_list = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim, normalize)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_vel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_acc = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                flag=0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=6)  # round 6 decimals
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        flag=1
                        break
                        # continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    
                    vel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    vel_curr_ped_seq[:, 1:] = rel_curr_ped_seq[:, 1:] / (1/12.5)
                    acc_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    acc_curr_ped_seq[:, 2:] = \
                        (vel_curr_ped_seq[:, 2:] - vel_curr_ped_seq[:, 1:-1])/(1/12.5)
                    curr_vel[_idx, :, pad_front:pad_end] = vel_curr_ped_seq
                    curr_acc[_idx, :, pad_front:pad_end] = acc_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    # _non_linear_ped.append(
                    #     poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped and flag==0:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    vel_list.append(curr_vel[:num_peds_considered])
                    acc_list.append(curr_acc[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        vel_list = np.concatenate(vel_list, axis=0)
        acc_list = np.concatenate(acc_list, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = seq_list[:, :, :self.obs_len]
        self.pred_traj = seq_list[:, :, self.obs_len:]
        self.obs_traj_rel = seq_list_rel[:, :, :self.obs_len]
        self.pred_traj_rel = seq_list_rel[:, :, self.obs_len:]
        
        self.obs_traj_vel = vel_list[:, :, :self.obs_len]
        self.pred_traj_vel = vel_list[:, :, self.obs_len:]
        self.obs_traj_acc = acc_list[:, :, :self.obs_len]
        self.pred_traj_acc = acc_list[:, :, self.obs_len:]
        
        self.loss_mask = loss_mask_list
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        #Convert to Graphs 
        self.v_obs = [] 
        self.A_obs = [] 
        self.Nei_obs = []
        self.v_pred = [] 
        self.A_pred = []
        self.Nei_pred = [] 
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]

            v_,a_,ab_ = seq_to_graph(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            self.Nei_obs.append(ab_.clone())
            v_,a_,ab_=seq_to_graph(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
            self.Nei_pred.append(ab_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.obs_traj_vel[start:end, :],self.pred_traj_vel[start:end, :],
            self.obs_traj_acc[start:end, :],self.pred_traj_acc[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index], self.Nei_obs[index],
            self.v_pred[index], self.A_pred[index],self.Nei_pred[index]

        ]
        return out
    
if __name__=="__main__":
    dataset='eth'
    assert dataset in ['eth','hotel','univ','zara1','zara2']
    data_set = './raw_data/'+dataset+'/'
    obs_seq_len=8
    pred_seq_len=12
    data_folder_name = 'processed_data_sf'
    os.makedirs(data_folder_name,exist_ok=True)
    
    dset_train = TrajectoryDataset(
        data_set+'train/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1,norm_lap_matr=False,normalize=True)
    data_dict_path = os.path.join(data_folder_name, '_'.join([dataset, 'train']) + '.pkl')
    
    with open(data_dict_path, 'wb') as f:
        dill.dump(dset_train, f, protocol=dill.HIGHEST_PROTOCOL)
        # read use:
        # with open(self.train_data_path, 'rb') as f:
        #    self.train_env = dill.load(f, encoding='latin1')
        
    dset_eval = TrajectoryDataset(
        data_set+'val/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=10,norm_lap_matr=False,normalize=True)
    data_dict_path = os.path.join(data_folder_name, '_'.join([dataset, 'eval']) + '.pkl')
    
    with open(data_dict_path, 'wb') as f:
        dill.dump(dset_train, f, protocol=dill.HIGHEST_PROTOCOL)
        
    dset_eval = TrajectoryDataset(
        data_set+'test/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=10,norm_lap_matr=False,normalize=True)
    data_dict_path = os.path.join(data_folder_name, '_'.join([dataset, 'test']) + '.pkl')
    
    with open(data_dict_path, 'wb') as f:
        dill.dump(dset_train, f, protocol=dill.HIGHEST_PROTOCOL)