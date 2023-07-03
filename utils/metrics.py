# -*- coding: utf-8 -*-
"""
***

"""
import torch
import torch.nn as nn
import numpy as np

import sys

sys.path.append('..')
import data.data as DATA


def collision_count(position, threshold, real_position=None, reduction=None):
    collisions = DATA.Pedestrians.collision_detection(position, threshold, real_position)
    if reduction == 'sum':
        out = torch.sum(collisions).item()
    elif reduction == 'mean':
        out = torch.mean(collisions).item()
    elif reduction is None:
        out = collisions
    else:
        raise NotImplementedError
    return out


def mae_with_time_mask(p, q, mask, reduction=None):
    """
    Args:
        p: (*c, t, N, feature_dim)
        q: (*c, t, N, feature_dim)
        mask: (t, N)
    """
    with torch.no_grad():
        mae = torch.norm(p[mask == 1] - q[mask == 1], p=2, dim=-1)
        if reduction == 'sum':
            out = torch.sum(mae).item()
        elif reduction == 'mean':
            out = torch.mean(mae).item()
    return out


def ot_with_time_mask(p, q, mask, eps=0.1, max_iter=100, reduction=None, dvs='cpu'):
    """
    Args:
        p: (*c, t, N, feature_dim)
        q: (*c, t, N, feature_dim)
        mask: (*c, t, N)
    """
    sinkhorn = SinkhornDistance(eps, max_iter, reduction, dvs=dvs)
    out = []
    for t in range(mask.shape[-2]):
        if torch.sum(mask[..., t, :]) > 1:
            ot, _, _ = sinkhorn(
                p[..., t, :, :][mask[..., t, :] == 1], q[..., t, :, :][mask[..., t, :] == 1])
            ot = ot.tolist()
            if type(ot) == list:
                out += ot
            else:
                out += [ot]
    if reduction == 'sum':
        out = np.sum(out)
    elif reduction == 'mean':
        out = np.mean(out)
    return out


def mmd_with_time_mask(p, q, mask, kernel_mul=2.0, kernel_num=5, fix_sigma=None, reduction=None):
    """
    Args:
        p: (*c, t, N, feature_dim)
        q: (*c, t, N, feature_dim)
        mask: (*c, t, N)
    """
    MMD = MaximumMeanDiscrepancy()
    if mask.dim() > 2:
        mask = mask.reshape(-1, mask.shape[-1])
        p = p.reshape(mask.shape[0], p.shape[-2], p.shape[-1])
        q = q.reshape(mask.shape[0], q.shape[-2], q.shape[-1])
    out = []
    for t in range(mask.shape[-2]):
        if torch.sum(mask[t, :]) > 1:
            mmd = MMD(p[t, :, :][mask[t, :] == 1], q[t, :, :][mask[t, :] == 1])
            out.append(mmd.item())
    if reduction == 'sum':
        out = np.sum(out)
    elif reduction == 'mean':
        out = np.mean(out)
    return out


def wasserstein_distance_2d(distribution_p, distribution_q, eps=0.1, max_iter=100, reduction=None):
    sinkhorn = SinkhornDistance(eps, max_iter, reduction)
    dist, P, C = sinkhorn(distribution_p, distribution_q)
    return dist, P, C


def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    MMD = MaximumMeanDiscrepancy()
    return MMD(source, target, kernel_mul, kernel_num, fix_sigma)


# Merged from https://github.com/dfdazac/wassdistance/blob/master/layers.py
# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction='none', dvs='cpu'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.dvs = dvs

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        mu = mu.to(self.dvs)
        nu = nu.to(self.dvs)
        u = u.to(self.dvs)
        v = v.to(self.dvs)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) -
                            torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * \
                (torch.log(nu + 1e-8) -
                 torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        # "Modified cost for logarithmic updates"
        # "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        # "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        # "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


# adapted from https://zhuanlan.zhihu.com/p/163839117
class MaximumMeanDiscrepancy(object):
    """docstring for MaximumMeanDiscrepancy"""

    def __init__(self):
        super(MaximumMeanDiscrepancy, self).__init__()

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        ""**Gra**
        source: sample_size_1 * feature_size**
        target: sample_size_2 * feature_size**
        kernel_mul:****bandwith
        kernel_num:**
        fix_sigma:**
            return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2**
                           ****:
                            [   K_ss K_st
                                K_ts K_tt ]
        """
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)  #**

        total0 = total.unsqueeze(0).expand(int(total.size(0)),
                                           int(total.size(0)),
                                           int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)),
                                           int(total.size(0)),
                                           int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)  #**|x-y|

        #**bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]

        #**，exp(-|x-y|/bandwith)
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for
                      bandwidth_temp in bandwidth_list]

        return sum(kernel_val)  #**

    def __call__(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        # source.dim=2 
        n = int(source.size()[0])
        m = int(target.size()[0])

        kernels = self.guassian_kernel(
            source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:n, :n]
        YY = kernels[n:, n:]
        XY = kernels[:n, n:]
        YX = kernels[n:, :n]

        # K_s**，Source<->Source
        XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)
        # K_s**，Source<->Target
        XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)

        # K_t**,Target<->Source
        YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)
        # K_t**,Target<->Target
        YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)

        loss = (XX + XY).sum() + (YX + YY).sum()
        return loss


if __name__ == "__main__":
    import numpy as np

    mmd = MaximumMeanDiscrepancy()
    # data_1 = torch.tensor(np.random.normal(loc=0,scale=10,size=(100,50)))
    # data_2 = torch.tensor(np.random.normal(loc=10,scale=10,size=(90,50)))
    data_1 = torch.zeros((3, 2))
    data_2 = torch.ones((3, 2))
    print("MMD Loss:", mmd(data_1, data_2))

    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, dvs='cpu')
    data_1 = torch.zeros((3, 3, 2))
    data_2 = torch.ones((3, 3, 2))
    ot, _, _ = sinkhorn(data_1, data_2)
    ot = ot.tolist()
    print(type(ot))
    print('ot: ', ot)

    data_1 = torch.zeros((2, 3, 3, 2))
    data_2 = torch.ones((2, 3, 3, 2))
    mask = torch.ones((2, 3, 3))
    mmd = mmd_with_time_mask(data_1, data_2, mask)
    print(mmd)

    print(mae_with_time_mask(data_1, data_2, mask, reduction='mean'))


def fde_at_label_end(p_pred, labels, reduction='mean'):
    nonzero1 = torch.nonzero(~torch.isnan(labels[...,0]))
    last_step_labels = []
    for i in range(labels.shape[1]):
        last_step_labels.append(nonzero1[nonzero1[:,1]==i][:,0].max().item())
        
    nonzero2 = torch.nonzero(~torch.isnan(p_pred[...,0]))
    last_step_pred = []
    for i in range(labels.shape[1]):
        last_step_pred.append(nonzero2[nonzero2[:,1]==i][:,0].max().item())
    last_step = []
    for i in range(len(last_step_pred)):
        last_step.append(last_step_labels[i] if last_step_labels[i]<=last_step_pred[i] else last_step_pred[i])
    m = torch.zeros(labels.shape[:2], device=labels.device)
    mask_fde = m.scatter(0, torch.tensor(last_step, device=labels.device).view(1,-1), 1)
    error = torch.norm(labels[mask_fde==1]-p_pred[mask_fde==1], p=2, dim=-1)
    if reduction=='mean':
        fde = torch.mean(error).item()
    elif reduction=='sum':
        fde = torch.sum(error).item()
    return fde



    
    
def dtw(x, y, dist_func=None, return_path=False):
    """
   **DT**

    Args:
        x: shape (m, d)**
        y: shape (n, d)**
        dist_func:****
        return_path:**DT**

    Returns:
        DT****）
    """
    if dist_func is None:
        #**
        dist_func = lambda a, b: torch.norm(a - b)

    #**
    D = torch.zeros((len(x)+1, len(y)+1)) + float('inf')
    D[0, 0] = 0
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            D[i, j] = dist_func(x[i-1], y[j-1])

    #**DT**
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            D[i, j] += torch.min(torch.stack([D[i-1, j], D[i, j-1], D[i-1, j-1]]))

    dtw_dist = D[-1, -1]

    if return_path:
        #**DT**
        dtw_path = [(len(x)-1, len(y)-1)]
        i, j = len(x), len(y)
        while i > 1 or j > 1:
            if i == 1:
                j -= 1
            elif j == 1:
                i -= 1
            else:
                if D[i-1, j] == torch.min(torch.stack([D[i-1, j], D[i, j-1], D[i-1, j-1]])):
                    i -= 1
                elif D[i, j-1] == torch.min(torch.stack([D[i-1, j], D[i, j-1], D[i-1, j-1]])):
                    j -= 1
                else:
                    i -= 1
                    j -= 1
            dtw_path.append((i-1, j-1))
        dtw_path.reverse()

        return dtw_dist, dtw_path
    else:
        return dtw_dist

def lcss(X, Y):
    m = len(X)
    n = len(Y)

    #**
    l = [[None]*(n+1) for i in range(m+1)]

    #**l[i][j]******LCS**
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                l[i][j] = 0
            elif X[i-1] == Y[j-1]:
                l[i][j] = l[i-1][j-1] + 1
            else:
                l[i][j] = max(l[i-1][j], l[i][j-1])

    # l[m][n**
    return l[m][n]



def dtw_tensor(tensor_x, tensor_y, mask_x, mask_y, disc_func=None, return_path=False, reduction='mean', norm=True):
    """
   ****DT**

    Args:
        x: shape (m, N, d)**
        y: shape (m, N, d)** N**
        mask_x: shape (m, N)，tensor_**mask
        mask_y: shape (m, N)，tensor_**mask
        dist_func:****
        return_path:**DT**

    Returns:
        DT****）
    """
    assert tensor_x.shape==tensor_y.shape
    assert tensor_x.shape[:2]==mask_x.shape==mask_y.shape
    mask_x = mask_x.bool()
    mask_y = mask_y.bool()
    N = tensor_x.shape[1]
    dtws = []
    for i in range(N):
        x_masked = tensor_x[mask_x[:,i],i] # mask_len_x, d
        y_masked = tensor_y[mask_y[:,i],i] # mask_len_y, d
        dtw_dist = dtw(x_masked, y_masked, dist_func=disc_func, return_path=return_path)
        if norm:
            dtw_dist = dtw_dist/(min(mask_x[:,i].sum().item(), mask_y[:,i].sum().item())+1e-6)
        dtws.append(dtw_dist.item())
    if reduction=='mean':
        result = np.mean(dtws)
    elif reduction=='sum':
        result = np.sum(dtws)
    else:
        raise NotImplementedError
    return result

def inter_ped_dis(x, label, mask_pred, reduction='mean', applied_func=None):
    """calculate the inter pedestrian distance

    Args:
        x (T, N, 2): pred sample
        label (T, N, 2): ground truth
        mask_pred (T, N): gt pedestrian mask
    """
    assert x.shape[:-1]==label.shape[:-1]==mask_pred.shape
    # N = x.shape[1]
    T = x.shape[0]
    dists = []
    for i in range(T):
        x_masked = x[i, mask_pred[i,:]==1]
        label_masked = label[i, mask_pred[i,:]==1]
        x_dist_mat = torch.cdist(x_masked.unsqueeze(0), x_masked.unsqueeze(0), p=2).squeeze(0)
        label_dist_mat = torch.cdist(label_masked.unsqueeze(0), label_masked.unsqueeze(0), p=2).squeeze(0)
        if applied_func is not None:
            x_dist_mat = applied_func(x_dist_mat)
            label_dist_mat = applied_func(label_dist_mat)
        dist_diff = torch.triu(x_dist_mat, diagonal=0).sum() - torch.triu(label_dist_mat, diagonal=0).sum()
        dists.append(dist_diff.abs().item())
    if reduction == 'mean':
        ret = np.mean(dists)
    elif reduction =='sum':
        ret = np.sum(dists)
    else:
         raise NotImplementedError
    return ret

def get_nearby_distance_mmd(data_pred_pos, data_pred_velo, label_pos, label_velo, mask, dist_threshold, near_k, reduction):
    """_summary_

    Args:
        data_pred (_type_): [T,N,...]
        label (_type_): [T,N,...]
        dist_threshold (_type_): scalar
    """
    T = mask.shape[0]
    N = mask.shape[1]
    MMD = MaximumMeanDiscrepancy()
    Pedestrian = DATA.Pedestrians()
    out = []
    for t in range(T):
        if torch.sum(mask[t,:]) > 1:
            mmd = get_dist_mmd(data_pred_pos[t, :, :][mask[t, :] == 1], 
                         data_pred_velo[t, :, :][mask[t, :] == 1],
                         label_pos[t, :, :][mask[t, :] == 1],
                         label_velo[t, :, :][mask[t, :] == 1], dist_threshold, near_k=near_k, MMD=MMD, func_class = Pedestrian)
            out.append(mmd.item())
    if reduction == 'sum':
        out = np.sum(out)
    elif reduction == 'mean':
        out = np.mean(out)
    return out

def get_dist_mmd(pred_pos, pred_velo, label_pos, label_velo, dist_threshold, near_k, MMD, func_class):
    heading_direction = func_class.get_heading_direction(pred_velo)
    near_ped_dist, near_ped_idx = func_class.get_nearby_obj_in_sight(
            pred_pos, pred_pos, heading_direction, near_k, angle_threshold=180)**top k=6
    near_ped_dist = near_ped_dist[(near_ped_dist < dist_threshold) &
                                (near_ped_dist > 0)]
    near_ped_dist = near_ped_dist.flatten().unsqueeze(-1)
    
    heading_direction_gt = func_class.get_heading_direction(label_velo)
    near_ped_dist_gt, near_ped_idx_gt = func_class.get_nearby_obj_in_sight(
            label_pos, label_pos, heading_direction_gt, near_k, 180)**top k=6
    near_ped_dist_gt = near_ped_dist_gt[(near_ped_dist_gt < dist_threshold) &
                                (near_ped_dist_gt > 0)]
    near_ped_dist_gt = near_ped_dist_gt.flatten().unsqueeze(-1)
    
    mmd = MMD(near_ped_dist, near_ped_dist_gt)
    return mmd
