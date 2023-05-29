# -*- coding: utf-8 -*-
"""
***

"""
import numpy as np
import torch
import math
from data.data import RawData

def args_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    for arg in vars(args):
        print(arg, getattr(args, arg))


def save_exp_configs_default(args):
    file_path = '../saved_configs/config_' + args.model_name_suffix + '.npy'
    np.save(file_path, np.array([args]))


def load_exp_configs_default(model_name_suffix):
    file_path = '../saved_configs/config_' + model_name_suffix + '.npy'
    data = np.load(file_path, allow_pickle=True)
    return data[0]


def calc_acceleration(relative_data: torch.tensor, equation_version: str = 'v0', dataset: str = 'gc1560',
                      eps=1e-6) -> torch.tensor:
    """
    calculate acceleration from relative position and relative velocity.
    relative_data should be shaped like (batch_size, N, M, 4) or (N, M, 4), where:
        (..., N, M, 0:2) means the position of M relative to N.
        (..., N, M, 2:4) means the velocity of M relative to N.
    equation_version can be:
        'v0': A*exp(B*r), in r direction
        'v1': A*exp(B*r + C*cos(theta)), in r direction
        'v2': A*exp(B*r + C*cos(theta) + D*r*cos(theta)), in r direction with theta bias.
    dataset can be 'gc1560', 'gc2344', 'ucy', to choose the corresponding hyperparameter.
    
    return acceleration tensor shaped like (..., N, M, 0:2), means the acceleration of N generated by M.
    """
    if equation_version == 'v0':
        if dataset == 'gc1560':
            A, B = 8.75, -2.5
        elif dataset == 'gc2344':
            A, B = 8.75, -2.5
        elif dataset == 'ucy':
            A, B = 10.67, -3.33
        dr = relative_data[..., 0:2]
        r = torch.linalg.norm(dr, ord=2, dim=-1, keepdim=True)
        r += eps
        acc = A * torch.exp(B * r)
        dir = dr / r
        return -acc * dir
    if equation_version == 'v1':
        if dataset == 'gc1560':
            A, B, C = 8.75, -2.5, 0
        elif dataset == 'gc2344':
            A, B, C = 8.75, -2.5, 0
        elif dataset == 'ucy':
            A, B, C = 10.67, -3.33, 0
        dr = relative_data[..., 0:2]
        dv = relative_data[..., 0:2]
        r = torch.linalg.norm(dr, ord=2, dim=-1, keepdim=True)
        v = torch.linalg.norm(dv, ord=2, dim=-1, keepdim=True)
        r += eps
        v += eps
        cos = torch.sum(dr * dv, dim=-1, keepdim=True) / r / v
        acc = A * torch.exp(B * r + C * cos)
        dir = dr / r
        return -acc * dir
    elif equation_version == 'v2':
        if dataset == 'gc1560':
            raise NotImplementedError
        elif dataset == 'gc2344':
            A, B, C, D, theta = 9.00, -2.75, 0.06, -0.3, 10 *3.1415 / 180
        elif dataset == 'ucy':
            raise NotImplementedError
        dr = relative_data[..., 0:2]
        dv = relative_data[..., 0:2]
        r = torch.linalg.norm(dr, ord=2, dim=-1, keepdim=True)
        v = torch.linalg.norm(dv, ord=2, dim=-1, keepdim=True)
        r += eps
        v += eps
        cos = torch.sum(dr * dv, dim=-1, keepdim=True) / r / v
        acc = A * torch.exp(B * r + C * cos + D * r * cos)
        dir = dr / r
        rotate_mat = torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
                                  device=relative_data.device)
        if len(dir.shape) == 3:
            dir_with_bias = torch.einsum('ij,nmj->nmi', rotate_mat, dir)
        elif len(dir.shape) == 4:
            dir_with_bias = torch.einsum('ij,bnmj->bnmi', rotate_mat, dir)
        else:
            raise ValueError
        return -acc * dir_with_bias

def rollout_MAE(label:RawData, preq:RawData, mask_p_preq, split=list(range(0, 12, 2))):
    from functions.metrics import mae_with_time_mask
    split_f = [int(t / label.meta_data['time_unit']) for t in split]
    begin_f = []
    end_f = []
    for p in range(label.num_pedestrians):
        simulate_duration = torch.nonzero(mask_p_preq[:, p])
        if(simulate_duration.numel() > 0):
            begin_f.append(simulate_duration[0])
            end_f.append(simulate_duration[-1])
        else:
            begin_f.append(0)
            end_f.append(0)
    valid = (torch.tensor(end_f) - torch.tensor(begin_f) >= split_f[-1])
    print(f'{torch.sum(valid)}/{label.num_pedestrians} pedestrians have simulation data for more than {split[-1]}s.')
    maes = []
    for n in range(1, len(split_f)):
        tmp_mask_p_preq = torch.zeros_like(mask_p_preq)
        for ped in range(label.num_pedestrians):
            if(valid[ped] == 1):
                tmp_mask_p_preq[begin_f[ped] + split_f[n - 1]:begin_f[ped] + split_f[n], ped] = 1
        mae = mae_with_time_mask(label.position, preq.position, tmp_mask_p_preq, reduction='mean')
        maes.append(mae)
    maes = np.array(maes)
    a, b = np.polyfit(np.log(split[1:]), np.log(maes), 1)
    print(maes)
    print(f'MAE = {np.exp(b):.3f}*t^{a:.3f}')


def cross_dot_z(a:torch.tensor, b:torch.tensor)->torch.tensor:
    """
    calculate (a cross b) dot z,
    a & b should have the shape as [1, 2] or [N, 2],
    return a tensor has the shape as [1] or [N]
    """
    # (a cross b) dot z = (b cross z) dot a
    b_cross_z = torch.stack((b[:, 1], -b[:, 0]), dim=1)
    return torch.sum(b_cross_z * a, dim=1)

def route(od:torch.tensor, obs:torch.tensor)->torch.tensor:
    """
    generate a route shaped like tensor.shape(N, 2) from od[0, :] to od[1, :]
    """
    o = od[(0,), :] # 1, 2
    d = od[(1,), :] # 1, 2
    r = d.clone()
    while True:
        A = r - o  # 1, 2
        B = torch.diff(obs, dim=0)  # N-1, 2
        C = obs[:-1, :] - o  # N-1, 2
        det = cross_dot_z(B, A)  # N-1
        alpha = cross_dot_z(B, C) / det  # N-1
        beta = cross_dot_z(A, C) / det  # N-1
        collision = (0 < alpha) * (alpha < 1) * (0 < beta) * (beta < 1)  # N-1
        # collision = torch.concat((collision, torch.tensor([0], device=collision.device)), dim=0)  # N
        if not torch.any(collision):
            break
        indexes = torch.nonzero(collision)
        index = indexes[torch.argmin(alpha[indexes])]
        cross = alpha[index] * r + (1 - alpha[index]) * o
        normal_direc = -cross_dot_z(A, B[index, :]) * torch.stack((A[:, 1], -A[:, 0]), dim=1)  # 1, 2
        normal_direc = normal_direc / torch.linalg.norm(normal_direc, dim=1, keepdim=True)  # 1, 2
        r = cross + 2 * normal_direc
    return torch.stack((o, r, d), dim=0)  # 3, 1, 2

def clear_nan(tensor:torch.Tensor):
    mask = tensor!=tensor
    tensor = tensor.masked_fill(mask,0)
    return tensor

# def mask_mse_func(input1, input2, mask):
#     assert input1.shape==input2.shape, 'mask_mse inputs should have same dims'
#     assert mask.shape==input1.shape[:-1]
#     fn = torch.nn.MSELoss(reduction='none')
#     input1 = clear_nan(input1)
#     input2 = clear_nan(input2)
#     input1[~mask.bool()]=0
#     input2[~mask.bool()]=0
#     loss = fn(input1, input2)
#     divisor = mask.sum(dim=-1,keepdim=True)
#     divisor = torch.where(divisor==0, torch.tensor(1,dtype=divisor.dtype).cuda(), divisor)
#     return (loss.sum(dim=-1)/divisor).sum()/loss.shape[0]

def mask_mse_func(input1, input2, mask):
    fn = torch.nn.MSELoss(reduction='mean')
    return fn(input1[mask==1], input2[mask==1])

def post_process(data, pred_data, pred_mask_p, mask_p):
        """
        把到达终点以后的人置为终点
        t, n, 2
        """
        waypoints = data.waypoints  # *c, d, n, 2
        if waypoints.dim() > 3:
            dest_num = data.dest_num.unsqueeze(0).repeat(waypoints.shape[0], 1)
        else:
            dest_num = data.dest_num
        dest_idx = dest_num - 1  # *c, n

        dest_idx_ = dest_idx.unsqueeze(-2).unsqueeze(-1)  # *c, 1, n, 1
        dest_idx_ = dest_idx_.repeat(*([1] * (dest_idx_.dim() - 1) + [2]))
        dest = torch.gather(waypoints, -3, dest_idx_)  # *c, 1, n, 2
        tmp_arg = [1] * dest.dim()
        tmp_arg[-3] = pred_data.shape[-3]
        dest = dest.repeat(*tmp_arg)  # *c, t, n, 2

        pred_data[(mask_p == 1) & (pred_mask_p == 0)] = dest[(mask_p == 1) & (pred_mask_p == 0)]
        return pred_data