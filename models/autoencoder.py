import torch
from torch.nn import Module
import torch.nn as nn
# from .encoders.trajectron import Trajectron
# from .encoders import dynamics as dynamic_module
import models.diffusion as diffusion
from models.diffusion import DiffusionTraj,VarianceSchedule
import pdb
import numpy as np
import data.data as DATA
from tqdm.auto import tqdm

def clear_nan(tensor:torch.Tensor):
    tensor[tensor.isnan()]=0
    return tensor

class AutoEncoder(Module, DATA.Pedestrians):

    def __init__(self, config, encoder=None):
        super().__init__()
        self.config = config
        self.encoder = encoder # 
        self.diffnet = getattr(diffusion, config.diffnet) # diffusion decod****DiffusionTra**

        self.diffusion = DiffusionTraj( 
            # net = self.diffnet(point_dim=2, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
            net = self.diffnet(config),
            var_sched = VarianceSchedule(
                num_steps=config.diffusion_steps,
                beta_T=5e-2,
                mode=config.variance_mode #'linear', 'cosine'

            ),
            config=config
        )
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2

    def encode(self, batch,node_type):
        z = self.encoder.get_latent(batch, node_type)
        return z

    def generate(self, batch, node_type, num_points, sample, bestof,flexibility=0.0, ret_traj=False):

        dynamics = self.encoder.node_models_dict[node_type].dynamic
        encoded_x = self.encoder.get_latent(batch, node_type)
        predicted_y_vel =  self.diffusion.sample(num_points, encoded_x,sample,bestof, flexibility=flexibility, ret_traj=ret_traj)
        predicted_y_pos = dynamics.integrate_samples(predicted_y_vel)
        return predicted_y_pos.cpu().detach().numpy()
    def generate2(self, batch, node_type, sample:int,bestof,flexibility=0.0, ret_traj=False):
        if self.config.train_mode == 'origin':
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,\
            obs_vel, pred_vel_gt, obs_acc, pred_acc_gt, non_linear_ped,\
            loss_mask,V_obs,A_obs,Nei_obs,V_tr,A_tr,Nei_tr = batch
            obs_traj, pred_traj_gt, pred_traj_gt_rel = obs_traj.squeeze().cuda(), pred_traj_gt.squeeze().cuda(), pred_traj_gt_rel.squeeze().cuda()
            sample_outputs = torch.Tensor().cuda()
            
            for i in range(self.config.pred_seq_len):
                if i==0:
                    context = obs_traj[...,-1]
                    context = torch.stack([context]*sample, dim=0)
                else:
                    context = context + sample_outputs[:,-1,...].squeeze()
                
                pred_traj = self.diffusion.sample(context.type(torch.float32).cuda(),bestof, flexibility=flexibility, ret_traj=ret_traj)
                sample_outputs = torch.cat((sample_outputs, pred_traj.unsqueeze(1)),dim=1)
        
        elif self.config.train_mode=='multi':
            ped_features,obs_features,self_features, labels, self_hist_features = batch
            self_features = self_features[0] # N,dim
            self_hist_features = self_hist_features[0,:,:-1,:] # N, len, dim(6)
            sample_outputs = torch.Tensor().cuda()
            for i in range(self.config.pred_seq_len):

                if self.config.esti_goal == 'acce': #TODO
                    context = torch.tensor([]).cuda()
                    pred_traj = self.diffusion.sample(context) 
                elif self.config.esti_goal == 'pos':
                    if i == 0:
                        contexts = self_hist_features[...,:-1,:2].unsqueeze(0).repeat(sample,1,1,1) # sample, N, len, 2
                        contexts = contexts.permute(0,2,1,3)
                        past_traj = contexts.clone()
                        mask = contexts[:,-1,:,:]!=contexts[:,-1,:,:]
                        mask = mask[...,0] # bs, N (mask[...,0]==mask[...,1]TODO?)
                        assert len(mask.shape)==2
                    else:
                        contexts = past_traj[:,-self.config.obs_seq_len:,:,:]
                    assert self.config.obs_seq_len==contexts.shape[1]
                    dest = self_features[...,:2].unsqueeze(-3)
                    contexts = torch.cat((contexts,dest), dim=-3) # bs, obs_len+1, N, 2
                    contexts = clear_nan(contexts)
                    pred_traj = self.diffusion.sample(contexts) #bs, N, 2
                    past_traj = torch.cat((past_traj,pred_traj.unsqueeze(-3)), dim=-3)
                    sample_outputs = torch.cat((sample_outputs, pred_traj.unsqueeze(-3)),dim=-3)
        return sample_outputs.cpu().detach().numpy() # sample, pred_len, N, 2

    def generate_multistep(self, data: DATA.TimeIndexedPedData, t_start=0, load_model=True):
        """
        Args:
            data:
            t_start: rollout starts from frame #t
        """
        args = self.config
        # if load_model:
        #     self.load_model(args, set_model=False, finetune_flag=self.finetune_flag)

        destination = data.destination
        waypoints = data.waypoints
        obstacles = data.obstacles
        mask_p_ = data.mask_p_pred.clone().long()  # *c, t, n

        desired_speed = data.self_features[...,t_start,:,-1].unsqueeze(-1)  # *c, n, 1

        if self.config.esti_goal=='acce':
            history_features = data.self_hist_features[...,t_start, :, :, :]
            history_features = clear_nan(history_features)
        elif self.config.esti_goal=='pos': 
            raise NotImplementedError
            # hist_pos = data.self_hist_features[...,t_start, :, :, :2]
            # hist_vel = torch.zeros_like(hist_pos, device=hist_pos.device)
            # hist_acce = torch.zeros_like(hist_pos, device=hist_pos.device)
            # hist_vel[:,1:,:] = data.self_hist_features[...,t_start, :, :-1, 2:4]
            # hist_acce[:,2:,:] = data.self_hist_features[...,t_start, :, :-2, 4:6]
            # history_features = torch.cat((hist_pos, hist_vel, hist_acce), dim=-1)
            # history_features = clear_nan(history_features)
        ped_features = data.ped_features[..., t_start, :, :, :]
        obs_features = data.obs_features[..., t_start, :, :, :]
        self_feature = data.self_features[...,t_start, :, :]
        
        near_ped_idx = data.near_ped_idx[...,t_start, :, :]
        neigh_ped_mask = data.neigh_ped_mask[...,t_start, :, :]
        near_obstacle_idx = data.near_obstacle_idx[...,t_start, :, :]
        neigh_obs_mask = data.neigh_obs_mask[...,t_start, :, :]
        
        a_cur = data.acceleration[..., t_start, :, :]  # *c, N, 2
        v_cur = data.velocity[..., t_start, :, :]  # *c, N, 2
        p_cur = data.position[..., t_start, :, :]  # *c, N, 2
        curr = torch.cat((p_cur, v_cur, a_cur), dim=-1) # *c, N, 6
        dest_cur = data.destination[..., t_start, :, :]  # *c, N, 2
        dest_idx_cur = data.dest_idx[..., t_start, :]  # *c, N
        dest_num = data.dest_num

        p_res = torch.zeros(data.position.shape, device=args.device)  # *c, t, n, 2
        v_res = torch.zeros(data.velocity.shape, device=args.device)  # *c, t, n, 2
        a_res = torch.zeros(data.acceleration.shape, device=args.device)  # *c, t, n, 2
        dest_force_res = torch.zeros(data.acceleration.shape, device=args.device)
        ped_force_res = torch.zeros(data.acceleration.shape, device=args.device)
        

        p_res[..., :t_start + 1, :, :] = data.position[..., :t_start + 1, :, :]
        v_res[..., :t_start + 1, :, :] = data.velocity[..., :t_start + 1, :, :]
        a_res[..., :t_start + 1, :, :] = data.acceleration[..., :t_start + 1, :, :]

        #**mask_p
        mask_p_new = torch.zeros(mask_p_.shape, device=mask_p_.device)
        mask_p_new[..., :t_start + 1, :] = data.mask_p[..., :t_start + 1, :].long()

        new_peds_flag = (data.mask_p - data.mask_p_pred).long()  # c, t, n

        for t in tqdm(range(t_start, data.num_frames)):
            p_res[..., t, :, :] = p_cur
            v_res[..., t, :, :] = v_cur
            a_res[..., t, :, :] = a_cur
            # mask_p_new[..., t, ~p_cur[:, 0].isnan()] = 1
            mask_p_new[..., t, :][~p_cur[..., 0].isnan()] = 1

            
            
            # a_next = self.diffusion.sample(*state_features)[0]
            if self.config.esti_goal=='acce':
                a_next = self.diffusion.sample(context = (history_features.unsqueeze(0), 
                                                          ped_features.unsqueeze(0), 
                                                          self_feature.unsqueeze(0),
                                                          obs_features.unsqueeze(0)), 
                                               curr = curr.unsqueeze(0)) 
                # print(a_next.sum())
                
                # dest force part
                self_feature = self_feature
                desired_speed = self_feature[..., -1].unsqueeze(-1)
                temp = torch.norm(self_feature[..., :2], p=2, dim=-1, keepdim=True)
                temp_ = temp.clone()
                temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
                dest_direction = self_feature[..., :2] / temp_ #des,direction
                pred_acc_dest = (desired_speed * dest_direction - self_feature[..., 2:4]) / self.tau
                pred_acc_ped = a_next - pred_acc_dest
                if t < data.num_frames-1:
                    dest_force_res[..., t+1, :, :] = pred_acc_dest
                    ped_force_res[..., t+1, :, :] = pred_acc_ped
                        
                v_next = v_cur + a_cur * data.time_unit
                p_next = p_cur + v_cur * data.time_unit  # *c, n, 2
            elif self.config.esti_goal=='pos':
                raise NotImplementedError
                # if self.config.history_dim==6:
                #     p_next = self.diffusion.sample(history_features.unsqueeze(0), dest_features.unsqueeze(0))
                # elif self.config.history_dim==2:
                #     p_next = self.diffusion.sample(history_features[...,:2].unsqueeze(0), dest_features.unsqueeze(0))
                #     v_next = v_cur
                #     a_next = a_cur

            # update destination & mask_p
            out_of_bound = torch.tensor(float('nan'), device=args.device)
            dis_to_dest = torch.norm(p_cur - dest_cur, p=2, dim=-1)
            dest_idx_cur[dis_to_dest < 0.5] += 1  # *c, n
            # TODO: currently don't delete?
            p_next[dest_idx_cur > dest_num - 1, :] = out_of_bound  # destination arrived 

            dest_idx_cur[dest_idx_cur > dest_num - 1] -= 1 
            dest_idx_cur_ = dest_idx_cur.unsqueeze(-2).unsqueeze(-1)  # *c, 1, n, 1
            dest_idx_cur_ = dest_idx_cur_.repeat(*([1] * (dest_idx_cur_.dim() - 1) + [2]))
            dest_cur = torch.gather(waypoints, -3, dest_idx_cur_).squeeze()  # *c, n, 2

            # update everyone's state
            p_cur = p_next  # *c, n, 2
            v_cur = v_next
            a_cur = a_next
            curr = torch.cat((p_cur, v_cur, a_cur), dim=-1)
            # update hist_v
            hist_v = self_feature[..., :, 2:-3]  # *c, n, 2*x
            hist_v[..., :, :-2] = hist_v[..., :, 2:]
            hist_v[..., :, -2:] = v_cur

            # add newly joined pedestrians
            if t < data.num_frames - 1:
                new_idx = new_peds_flag[..., t + 1, :]  # c, n
                if torch.sum(new_idx) > 0:
                    p_cur[new_idx == 1] = data.position[..., t + 1, :, :][new_idx == 1, :]
                    v_cur[new_idx == 1] = data.velocity[..., t + 1, :, :][new_idx == 1, :]
                    a_cur[new_idx == 1] = data.acceleration[..., t + 1, :, :][new_idx == 1, :]
                    dest_cur[new_idx == 1] = data.destination[..., t + 1, :, :][new_idx == 1, :]
                    dest_idx_cur[new_idx == 1] = data.dest_idx[..., t + 1, :][new_idx == 1]

                    # update hist_v
                    hist_v[new_idx == 1] = data.self_features[..., t + 1, :, 2:-3][new_idx == 1]

            # update hist_features
            if self.config.esti_goal=='acce':
                history_features[..., :-1, :] = history_features[..., 1:,:].clone()  # history_features: n, len, 6
                history_features[..., -1, :2] = p_cur.clone()
                history_features[..., -1, 2:4] = v_cur.clone()
                history_features[..., -1, 4:6] = a_cur.clone()
                history_features = clear_nan(history_features)
                
            elif self.config.esti_goal=='pos':
                if self.config.history_dim==2:
                    history_features[..., :-1, :] = history_features[:,1:,:].clone()  # history_features: n, len, 6
                    history_features[..., -1, :2] = p_cur.clone()
            
            # calculate features
            if self.config.esti_goal=='acce':
                ped_features, obs_features, dest_features,\
                near_ped_idx, neigh_ped_mask, near_obstacle_idx, neigh_obs_mask= self.get_relative_features(
                    p_cur.unsqueeze(-3), v_cur.unsqueeze(-3), a_cur.unsqueeze(-3),
                    dest_cur.unsqueeze(-3), obstacles, args.topk_ped, args.sight_angle_ped,
                    args.dist_threshold_ped, args.topk_obs,
                    args.sight_angle_obs, args.dist_threshold_obs)
                ped_features = ped_features.squeeze()
                obs_features = obs_features.squeeze()
                dest_features = dest_features.squeeze() 
                self_feature = torch.cat((dest_features, hist_v, a_cur, desired_speed), dim=-1)
            elif self.config.esti_goal=='pos':
                raise NotImplementedError
                dest_features = dest_cur - p_cur
                dest_features[dest_features.isnan()] = 0.

            

        # todo:**mask_v** mask_a;**mask_p
        output = DATA.RawData(p_res, v_res, a_res, destination, destination, obstacles,
                                mask_p_new, meta_data=data.meta_data)
        return output, dest_force_res, ped_force_res
    
    def generate_multistep_geo(self, data: DATA.TimeIndexedPedData, t_start=0, load_model=True):
        """
        Args:
            data:
            t_start: rollout starts from frame #t
        """
        args = self.config
        # if load_model:
        #     self.load_model(args, set_model=False, finetune_flag=self.finetune_flag)

        destination = data.destination
        waypoints = data.waypoints
        obstacles = data.obstacles
        mask_p_ = data.mask_p_pred.clone().long()  # *c, t, n

        desired_speed = data.self_features[...,t_start,:,-1].unsqueeze(-1)  # *c, n, 1

        if self.config.esti_goal=='acce':
            history_features = data.self_hist_features[...,t_start, :, :, :]
            history_features = clear_nan(history_features)
        elif self.config.esti_goal=='pos': 
            raise NotImplementedError
            # hist_pos = data.self_hist_features[...,t_start, :, :, :2]
            # hist_vel = torch.zeros_like(hist_pos, device=hist_pos.device)
            # hist_acce = torch.zeros_like(hist_pos, device=hist_pos.device)
            # hist_vel[:,1:,:] = data.self_hist_features[...,t_start, :, :-1, 2:4]
            # hist_acce[:,2:,:] = data.self_hist_features[...,t_start, :, :-2, 4:6]
            # history_features = torch.cat((hist_pos, hist_vel, hist_acce), dim=-1)
            # history_features = clear_nan(history_features)
        ped_features = data.ped_features[..., t_start, :, :, :]
        obs_features = data.obs_features[..., t_start, :, :, :]
        self_feature = data.self_features[...,t_start, :, :]
        
        near_ped_idx = data.near_ped_idx[...,t_start, :, :]
        neigh_ped_mask = data.neigh_ped_mask[...,t_start, :, :]
        near_obstacle_idx = data.near_obstacle_idx[...,t_start, :, :]
        neigh_obs_mask = data.neigh_obs_mask[...,t_start, :, :]
        
        a_cur = data.acceleration[..., t_start, :, :]  # *c, N, 2
        v_cur = data.velocity[..., t_start, :, :]  # *c, N, 2
        p_cur = data.position[..., t_start, :, :]  # *c, N, 2
        curr = torch.cat((p_cur, v_cur, a_cur), dim=-1) # *c, N, 6
        dest_cur = data.destination[..., t_start, :, :]  # *c, N, 2
        dest_idx_cur = data.dest_idx[..., t_start, :]  # *c, N
        dest_num = data.dest_num

        p_res = torch.zeros(data.position.shape, device=args.device)  # *c, t, n, 2
        v_res = torch.zeros(data.velocity.shape, device=args.device)  # *c, t, n, 2
        a_res = torch.zeros(data.acceleration.shape, device=args.device)  # *c, t, n, 2
        dest_force_res = torch.zeros(data.acceleration.shape, device=args.device)
        ped_force_res = torch.zeros(data.acceleration.shape, device=args.device)
        

        p_res[..., :t_start + 1, :, :] = data.position[..., :t_start + 1, :, :]
        v_res[..., :t_start + 1, :, :] = data.velocity[..., :t_start + 1, :, :]
        a_res[..., :t_start + 1, :, :] = data.acceleration[..., :t_start + 1, :, :]

        #**mask_p
        mask_p_new = torch.zeros(mask_p_.shape, device=mask_p_.device)
        mask_p_new[..., :t_start + 1, :] = data.mask_p[..., :t_start + 1, :].long()

        new_peds_flag = (data.mask_p - data.mask_p_pred).long()  # c, t, n

        for t in tqdm(range(t_start, data.num_frames)):
            p_res[..., t, :, :] = p_cur
            v_res[..., t, :, :] = v_cur
            a_res[..., t, :, :] = a_cur
            # mask_p_new[..., t, ~p_cur[:, 0].isnan()] = 1
            mask_p_new[..., t, :][~p_cur[..., 0].isnan()] = 1



            # a_next = self.diffusion.sample(*state_features)[0]
            if self.config.esti_goal=='acce':
                # pdb.set_trace()
                a_next = self.diffusion.sample(context = (curr.unsqueeze(0), 
                                                          neigh_ped_mask.unsqueeze(0), 
                                                          self_feature.unsqueeze(0),
                                                          near_ped_idx.unsqueeze(0),
                                                          history_features.unsqueeze(0),
                                                          obstacles.unsqueeze(0),
                                                          near_obstacle_idx.unsqueeze(0),
                                                          neigh_obs_mask.unsqueeze(0)), 
                                               curr = curr.unsqueeze(0)) 
                # print(a_next.sum())
                # if a_next[mask_p_[t]].max()>15 or a_next[mask_p_[t]].isnan().any():
                #     pdb.set_trace()
                # dest force part
                self_feature = self_feature
                desired_speed = self_feature[..., -1].unsqueeze(-1)
                temp = torch.norm(self_feature[..., :2], p=2, dim=-1, keepdim=True)
                temp_ = temp.clone()
                temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
                dest_direction = self_feature[..., :2] / temp_ #des,direction
                pred_acc_dest = (desired_speed * dest_direction - self_feature[..., 2:4]) / self.tau
                pred_acc_ped = a_next - pred_acc_dest
                if t < data.num_frames-1:
                    dest_force_res[..., t+1, :, :] = pred_acc_dest
                    ped_force_res[..., t+1, :, :] = pred_acc_ped
                        
                v_next = v_cur + a_cur * data.time_unit
                p_next = p_cur + v_cur * data.time_unit  # *c, n, 2
            elif self.config.esti_goal=='pos':
                raise NotImplementedError
                # if self.config.history_dim==6:
                #     p_next = self.diffusion.sample(history_features.unsqueeze(0), dest_features.unsqueeze(0))
                # elif self.config.history_dim==2:
                #     p_next = self.diffusion.sample(history_features[...,:2].unsqueeze(0), dest_features.unsqueeze(0))
                #     v_next = v_cur
                #     a_next = a_cur

            # update destination & mask_p
            out_of_bound = torch.tensor(float('nan'), device=args.device)
            dis_to_dest = torch.norm(p_cur - dest_cur, p=2, dim=-1)
            dest_idx_cur[dis_to_dest < 0.5] += 1  # *c, n
            # TODO: currently don't delete?
            p_next[dest_idx_cur > dest_num - 1, :] = out_of_bound  # destination arrived 

            dest_idx_cur[dest_idx_cur > dest_num - 1] -= 1 
            dest_idx_cur_ = dest_idx_cur.unsqueeze(-2).unsqueeze(-1)  # *c, 1, n, 1
            dest_idx_cur_ = dest_idx_cur_.repeat(*([1] * (dest_idx_cur_.dim() - 1) + [2]))
            dest_cur = torch.gather(waypoints, -3, dest_idx_cur_).squeeze()  # *c, n, 2

            # update everyone's state
            p_cur = p_next  # *c, n, 2
            v_cur = v_next
            a_cur = a_next
            # curr = torch.cat((p_cur, v_cur, a_cur), dim=-1)
            # update hist_v
            hist_v = self_feature[..., :, 2:-3]  # *c, n, 2*x
            hist_v[..., :, :-2] = hist_v[..., :, 2:]
            hist_v[..., :, -2:] = v_cur

            # add newly joined pedestrians
            if t < data.num_frames - 1:
                new_idx = new_peds_flag[..., t + 1, :]  # c, n
                if torch.sum(new_idx) > 0:
                    p_cur[new_idx == 1] = data.position[..., t + 1, :, :][new_idx == 1, :]
                    v_cur[new_idx == 1] = data.velocity[..., t + 1, :, :][new_idx == 1, :]
                    a_cur[new_idx == 1] = data.acceleration[..., t + 1, :, :][new_idx == 1, :]
                    dest_cur[new_idx == 1] = data.destination[..., t + 1, :, :][new_idx == 1, :]
                    dest_idx_cur[new_idx == 1] = data.dest_idx[..., t + 1, :][new_idx == 1]

                    # update hist_v
                    hist_v[new_idx == 1] = data.self_features[..., t + 1, :, 2:-3][new_idx == 1]
            
            curr = torch.cat((p_cur, v_cur, a_cur), dim=-1) # *c, N, 6


            # update hist_features
            if self.config.esti_goal=='acce':
                history_features[..., :-1, :] = history_features[..., 1:,:].clone()  # history_features: n, len, 6
                history_features[..., -1, :2] = p_cur.clone()
                history_features[..., -1, 2:4] = v_cur.clone()
                history_features[..., -1, 4:6] = a_cur.clone()
                history_features = clear_nan(history_features)
                
            elif self.config.esti_goal=='pos':
                if self.config.history_dim==2:
                    history_features[..., :-1, :] = history_features[:,1:,:].clone()  # history_features: n, len, 6
                    history_features[..., -1, :2] = p_cur.clone()
            
            # calculate features
            if self.config.esti_goal=='acce':
                ped_features, obs_features, dest_features,\
                near_ped_idx, neigh_ped_mask, near_obstacle_idx, neigh_obs_mask= self.get_relative_features(
                    p_cur.unsqueeze(-3), v_cur.unsqueeze(-3), a_cur.unsqueeze(-3),
                    dest_cur.unsqueeze(-3), obstacles, args.topk_ped, args.sight_angle_ped,
                    args.dist_threshold_ped, args.topk_obs,
                    args.sight_angle_obs, args.dist_threshold_obs)
                ped_features = ped_features.squeeze()
                obs_features = obs_features.squeeze()
                dest_features = dest_features.squeeze() 
                near_ped_idx = near_ped_idx.squeeze() 
                neigh_ped_mask = neigh_ped_mask.squeeze() 
                near_obstacle_idx = near_obstacle_idx.squeeze() 
                neigh_obs_mask =  neigh_obs_mask.squeeze() 
                
                self_feature = torch.cat((dest_features, hist_v, a_cur, desired_speed), dim=-1)
            elif self.config.esti_goal=='pos':
                raise NotImplementedError
                dest_features = dest_cur - p_cur
                dest_features[dest_features.isnan()] = 0.

            

        # todo:**mask_v** mask_a;**mask_p
        output = DATA.RawData(p_res, v_res, a_res, destination, destination, obstacles,
                                mask_p_new, meta_data=data.meta_data)
        return output, dest_force_res, ped_force_res
    
    
    def generate_onestep(self, context:tuple, curr=None):
        history = context[0]
        ped_features = context[1]
        assert len(history.shape)==4 and len(ped_features.shape)==4 
        history = clear_nan(history)
        context = list(context)
        context[0] = history
        context = tuple(context)
        with torch.no_grad():
            if self.config.esti_goal =='pos':
                pred_traj=self.diffusion.sample(context = context) # t, N, 2
            elif self.config.esti_goal =='acce':
                pred_traj=self.diffusion.sample(context = context, curr = curr) # t, N, 2
        return pred_traj
        
        
    def get_loss(self, batch, node_type=None, timestep=0.08):
        if self.config.train_mode == 'origin':
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,\
            obs_vel, pred_vel_gt, obs_acc, pred_acc_gt, non_linear_ped,\
            loss_mask,V_obs,A_obs,Nei_obs,V_tr,A_tr,Nei_tr = batch # x_t**************）；y_t**1******encode**latent variable

            # feat_x_encoded = self.encode(batch,node_type) # B * 64
            loss = self._teacher_loss(obs_traj,pred_traj_gt,pred_traj_gt_rel)
            
        elif self.config.train_mode == 'multi':
            ped_features,obs_features,self_features, labels, self_hist_features,\
            mask_p_pred, mask_v_pred, mask_a_pred = batch #**mask**********）
            if self.config.esti_goal == 'acce':
                y_t = labels[1:,:,4:6]
                # y_t = labels[:-1,:,4:6]
                y_t = clear_nan(y_t)
                curr = labels[:-1,:,:6] #****

                history = self_hist_features[:-1,:,:,:6] #bs, N, obs_len, 6
                ped_features = ped_features[:-1] #bs, N, k_near, 6
                obs_features = obs_features[:-1]
                self_feature = self_features[:-1,:,:]
                history = clear_nan(history)
                mask = mask_a_pred[:-1]
                loss = self.diffusion.get_loss(y_t, curr=curr,context=(history, ped_features, self_feature, obs_features),timestep=timestep,mask = mask)
            elif self.config.esti_goal == 'pos':
                raise NotImplementedError
                y_t = labels[1:,:,:2] #**timestep—**
                y_t = clear_nan(y_t)
                hist_pos = self_hist_features[:-1, :, :, :2] #**timestep**label）
                hist_vel = torch.zeros_like(hist_pos, device=hist_pos.device)
                hist_acce = torch.zeros_like(hist_pos, device=hist_pos.device)
                hist_vel[:,:,1:,:] = self_hist_features[:-1, :, :-1, 2:4]
                hist_acce[:,:,2:,:] = self_hist_features[:-1, :, :-2, 4:6]
                history = torch.cat((hist_pos, hist_vel, hist_acce), dim=-1)
                # mask = contexts[:,-1,:,:]!=contexts[:,-1,:,:] #**mas**mask**** TODO
                # mask = ~mask[...,0] # bs, N 
                dest = self_features[:-1,:,:2]
                mask = dest.abs().sum(dim=-1)==0
                mask = ~mask
                # mask = mask_v_pred[:-1]
                history = clear_nan(history)
                if self.config.history_dim==6:
                    loss = self.diffusion.get_loss(y_t, curr=None, context = (history, dest), mask = mask)
                elif self.config.history_dim==2:
                    loss = self.diffusion.get_loss(y_t, curr=None, context = (history[..., :2], dest), mask = mask)
        else:
            raise NotImplementedError
        return loss
    
    def _teacher_loss(self, obs_traj,pred_traj_gt,pred_traj_gt_rel):
        """do teacher force training

        Args:
            obs_traj (np.array): (N,2,obs_len)
            pred_traj_gt (np.array): (N,2,pred_len)
            pred_traj_gt_rel (np.array): (N,2,pred_len)
        """
        obs_traj, pred_traj_gt, pred_traj_gt_rel = obs_traj.squeeze(), pred_traj_gt.squeeze(), pred_traj_gt_rel.squeeze()
        contexts = np.zeros([self.config.pred_seq_len, obs_traj.shape[0],obs_traj.shape[1]]) # (pred_len,N,2)
        contexts[0] = obs_traj[...,-1]
        contexts[1:] = pred_traj_gt[...,:-1].permute(2,0,1)
        contexts = torch.from_numpy(contexts).type(torch.float32)
        y_t = pred_traj_gt_rel.permute(2,0,1) # (pred_len,N,2)
        # torch.set_default_dtype(torch.float64)
        loss = self.diffusion.get_loss(y_t.type(torch.float32).cuda())
        return loss

    def test_multiple_rollouts_for_training(self, data: DATA.TimeIndexedPedData, t_start=0):
        """
       **，dynamic weighting;
       **
        Args:
            data:
            t_start:

        Returns:

        """
        args = self.config

        # destination = data.destination
        waypoints = data.waypoints
        obstacles = data.obstacles
        mask_p_ = data.mask_p_pred.clone().long()  # *c, t, n
        mask_a_ = data.mask_a_pred.clone().long()
        
        desired_speed = data.self_features[...,t_start,:,-1].unsqueeze(-1)  # *c, n, 1
        history_features = data.self_hist_features[...,t_start, :, :, :]
        history_features[history_features.isnan()]=0
        ped_features = data.ped_features[..., t_start, :, :, :]
        obs_features = data.obs_features[..., t_start, :, :, :]
        self_features = data.self_features[...,t_start, :, :]
        a_cur = data.acceleration[..., t_start, :, :]  # *c, N, 2
        v_cur = data.velocity[..., t_start, :, :]  # *c, N, 2
        p_cur = data.position[..., t_start, :, :]  # *c, N, 2
        collisions = torch.zeros(mask_p_.shape, device=p_cur.device)  # c, t, n
        label_collisions = torch.zeros(mask_p_.shape, device=p_cur.device)  # c, t, n
        p_cur_nonan = clear_nan(p_cur.clone())
        curr = torch.cat((p_cur, v_cur, a_cur), dim=-1) # *c, N, 6
        dest_cur = data.destination[..., t_start, :, :]  # *c, N, 2
        dest_idx_cur = data.dest_idx[..., t_start, :]  # *c, N
        dest_num = data.dest_num

        new_peds_flag = (data.mask_p - data.mask_p_pred).long()  # c, t, n

        # loss = torch.tensor(0., requires_grad=True, device=p_cur.device)
        p_res = torch.zeros(data.position.shape, device=p_cur.device)
        
        a_res = torch.zeros(data.acceleration.shape, device=p_cur.device)
        
    
        for t in range(t_start, data.num_frames):
            
            collision = self.collision_detection(p_cur.clone().detach(), args.collision_threshold)  # c, n, n
            collision = torch.sum(collision, dim=-1)  # c, n
            collisions[:, t, :] = collision
            label_collision = self.collision_detection(data.labels[:, t, :, :2],
                                                           args.collision_threshold)  # c, n, n
            label_collision = torch.sum(label_collision, dim=-1)  # c, n
            label_collisions[:, t, :] = label_collision
            # a_next = self.diffusion.sample(context = (history_features.detach(), 
            #                                               ped_features.detach(), 
            #                                               self_features.detach()), 
            #                                    curr = curr.detach()).view(*a_cur.shape) 
            if t < data.num_frames - 1 + t_start:
                a_next = self.diffusion.denoise_fn(x_0 = data.acceleration[..., t+1, :, :] , #c, n, 2
                                                context = (history_features.detach(), 
                                                            ped_features.detach(), 
                                                            self_features.detach(),
                                                            obs_features.detach()), 
                                                curr = curr.detach()).view(*a_cur.shape)  #chec**
            # mask = mask_p_[:, t, :]  # c,n

            # if torch.sum(mask) > 0:
            p_res[:, t, ...] = p_cur_nonan
            a_res[:, t, ...] = a_cur

                # if args.reg_weight > 0: # mynote: regularization
                #     reg_loss += self.l1_reg_loss(p_msg, args.reg_weight, 'sum')
                #     loss = loss + reg_loss
                # if len(predictions) > 2:
                #     loss = loss + self.l1_reg_loss(o_msg, args.reg_weight, 'sum')


            v_next = v_cur + a_cur * data.time_unit
            p_next = p_cur + v_cur * data.time_unit  # *c, n, 2
            p_next_nonan = p_cur_nonan + v_cur * data.time_unit
            assert ~a_next.isnan().any(), print('find nan in epoch :', self.epoch, self.batch_idx)

            # update destination, yet do not delete people when they arrive
            dis_to_dest = torch.norm(p_cur.detach() - dest_cur, p=2, dim=-1)
            dest_idx_cur[dis_to_dest < 0.5] = dest_idx_cur[dis_to_dest < 0.5] + 1

            dest_idx_cur[dest_idx_cur > dest_num - 1] = dest_idx_cur[dest_idx_cur > dest_num - 1] - 1
            dest_idx_cur_ = dest_idx_cur.unsqueeze(-2).unsqueeze(-1)  # *c, 1, n, 1
            dest_idx_cur_ = dest_idx_cur_.repeat(*([1] * (dest_idx_cur_.dim() - 1) + [2]))
            dest_cur = torch.gather(waypoints, -3, dest_idx_cur_).squeeze().view(*p_cur.shape)  # *c, n, 2

            # update everyone's state
            p_cur = p_next  # *c, n, 2
            p_cur_nonan = p_next_nonan
            v_cur = v_next
            a_cur = a_next
            del p_next_nonan, p_next, v_next #ceshi
            curr = torch.cat((p_cur_nonan, v_cur, a_cur), dim=-1).detach()
            # update hist_v
            # hist_v = self_features[..., :, 2:-3]  # *c, n, 2*x
            # hist_v[..., :, :-2] = hist_v[..., :, 2:]
            # hist_v[..., :, -2:] = v_cur.detach()

            # add newly joined pedestrians
            if t < data.num_frames - 1:
                new_idx = new_peds_flag[..., t + 1, :]  # c, n
                if torch.sum(new_idx) > 0:
                    p_cur[new_idx == 1] = data.position[..., t + 1, :, :][new_idx == 1, :]
                    p_cur_nonan[new_idx == 1] = data.position[..., t + 1, :, :][new_idx == 1, :]
                    v_cur[new_idx == 1] = data.velocity[..., t + 1, :, :][new_idx == 1, :]
                    a_cur[new_idx == 1] = data.acceleration[..., t + 1, :, :][new_idx == 1, :]
                    dest_cur[new_idx == 1] = data.destination[..., t + 1, :, :][new_idx == 1, :]
                    dest_idx_cur[new_idx == 1] = data.dest_idx[..., t + 1, :][new_idx == 1]
                    # update hist_v
                    # hist_v[new_idx == 1] = data.self_features[..., t + 1, :, 2:-3][new_idx == 1]

            # update hist_features
            if self.config.esti_goal=='acce':
                new_traj = curr.clone().unsqueeze(-2)
                history_features = torch.cat([history_features[...,1:,:], new_traj],dim=-2)
                # history_features[..., :-1, :] = history_features[...,1:,:].detach().clone()  # history_features: n, len, 6
                # history_features[..., -1, :2] = p_cur_nonan.clone().detach()
                # history_features[..., -1, 2:4] = v_cur.clone().detach()
                # history_features[..., -1, 4:6] = a_cur.clone().detach()
                # history_features = clear_nan(history_features)   
                
            elif self.config.esti_goal=='pos':
                if self.config.history_dim==2:
                    history_features[..., :-1, :] = history_features[:,1:,:].clone()  # history_features: n, len, 6
                    history_features[..., -1, :2] = p_cur_nonan.clone()
            ped_features_shape = ped_features.shape
            obs_features_shape = obs_features.shape
            # calculate features
            ped_features, obs_features, dest_features,\
            near_ped_idx, neigh_ped_mask, near_obstacle_idx, neigh_obs_mask= self.get_relative_features(
                p_cur.unsqueeze(-3).detach(), v_cur.unsqueeze(-3).detach(), a_cur.unsqueeze(-3).detach(),
                dest_cur.unsqueeze(-3).detach(), obstacles, args.topk_ped, args.sight_angle_ped,
                args.dist_threshold_ped, args.topk_obs,
                args.sight_angle_obs, args.dist_threshold_obs)  # c,1,n,k,2

            self_features = torch.cat((dest_features.view(*v_cur.shape), v_cur, a_cur, desired_speed), dim=-1)
            ped_features, obs_features = ped_features.view(*ped_features_shape), obs_features.view(*obs_features_shape)
            torch.cuda.empty_cache() #ceshi

        p_res[mask_p_ == 0] = 0.  # delete 'nan'
        data.labels[mask_p_ == 0] = 0.  # delete 'nan'
        a_res[mask_a_==0] = 0.
        
        loss = torch.tensor(0., requires_grad=True, device=p_cur.device)
        # mse_loss = self.multiple_rollout_mse_loss(p_res, data.labels[:, :, :, :2], args.time_decay, reduction='sum')
        # or
        mse_loss = self.multiple_rollout_mse_loss(a_res, data.labels[:, :, :, 4:6], args.time_decay, reduction='sum')

        loss = loss + mse_loss
        if self.config.use_col_focus_loss:
            collision_loss = self.multiple_rollout_collision_loss(
                    p_res, data.labels[:, :, :, :2], 1, args.collision_focus_weight, collisions,
                    reduction='sum')
            loss = loss + collision_loss
        
        return loss
    
    def test_multiple_rollouts_for_training_geo(self, data: DATA.TimeIndexedPedData, t_start=0):
        """
       **，dynamic weighting;
       **
        Args:
            data:
            t_start:

        Returns:

        """
        args = self.config

        # destination = data.destination
        waypoints = data.waypoints
        obstacles = data.obstacles
        mask_p_ = data.mask_p_pred.clone().long()  # *c, t, n
        mask_a_ = data.mask_a_pred.clone().long()
        
        desired_speed = data.self_features[...,t_start,:,-1].unsqueeze(-1)  # *c, n, 1
        history_features = data.self_hist_features[...,t_start, :, :, :]
        history_features[history_features.isnan()]=0
        ped_features = data.ped_features[..., t_start, :, :, :]
        obs_features = data.obs_features[..., t_start, :, :, :]
        self_features = data.self_features[...,t_start, :, :]
        
        near_ped_idx = data.near_ped_idx[...,t_start, :, :]
        neigh_ped_mask = data.neigh_ped_mask[...,t_start, :, :]
        near_obstacle_idx = data.near_obstacle_idx[...,t_start, :, :]
        neigh_obs_mask = data.neigh_obs_mask[...,t_start, :, :]
        
        a_cur = data.acceleration[..., t_start, :, :]  # *c, N, 2
        v_cur = data.velocity[..., t_start, :, :]  # *c, N, 2
        p_cur = data.position[..., t_start, :, :]  # *c, N, 2
        collisions = torch.zeros(mask_p_.shape, device=p_cur.device)  # c, t, n
        label_collisions = torch.zeros(mask_p_.shape, device=p_cur.device)  # c, t, n
        p_cur_nonan = clear_nan(p_cur.clone())
        curr = torch.cat((p_cur, v_cur, a_cur), dim=-1) # *c, N, 6
        dest_cur = data.destination[..., t_start, :, :]  # *c, N, 2
        dest_idx_cur = data.dest_idx[..., t_start, :]  # *c, N
        dest_num = data.dest_num

        new_peds_flag = (data.mask_p - data.mask_p_pred).long()  # c, t, n

        # loss = torch.tensor(0., requires_grad=True, device=p_cur.device)
        p_res = torch.zeros(data.position.shape, device=p_cur.device)
        
        a_res = torch.zeros(data.acceleration.shape, device=p_cur.device)
        obstacles = obstacles.unsqueeze(0).repeat(near_obstacle_idx.shape[0],1,1)
    
        for t in range(t_start, data.num_frames):
            
            collision = self.collision_detection(p_cur.clone().detach(), args.collision_threshold)  # c, n, n
            collision = torch.sum(collision, dim=-1)  # c, n
            collisions[:, t, :] = collision
            label_collision = self.collision_detection(data.labels[:, t, :, :2],
                                                           args.collision_threshold)  # c, n, n
            label_collision = torch.sum(label_collision, dim=-1)  # c, n
            label_collisions[:, t, :] = label_collision
            # a_next = self.diffusion.sample(context = (history_features.detach(), 
            #                                               ped_features.detach(), 
            #                                               self_features.detach()), 
            #                                    curr = curr.detach()).view(*a_cur.shape) 
            if t < data.num_frames - 1 + t_start:
                a_next = self.diffusion.denoise_fn(x_0 = data.acceleration[..., t+1, :, :] , #c, n, 2
                                                context = (curr.detach(), 
                                                            neigh_ped_mask.detach(), 
                                                            self_features.detach(),
                                                            near_ped_idx.detach(),
                                                            history_features.detach(),
                                                            obstacles.detach(),
                                                            near_obstacle_idx.detach(),
                                                            neigh_obs_mask.detach()),  
                                                curr = curr.detach()).view(*a_cur.shape)  #chec**
                # if a_next[mask_a_[:,t]].max()>15 or a_next[mask_a_[:,t]].isnan().any():
                #     pdb.set_trace()
            # mask = mask_p_[:, t, :]  # c,n

            # if torch.sum(mask) > 0:
            p_res[:, t, ...] = p_cur_nonan
            a_res[:, t, ...] = a_cur

                # if args.reg_weight > 0: # mynote: regularization
                #     reg_loss += self.l1_reg_loss(p_msg, args.reg_weight, 'sum')
                #     loss = loss + reg_loss
                # if len(predictions) > 2:
                #     loss = loss + self.l1_reg_loss(o_msg, args.reg_weight, 'sum')


            v_next = v_cur + a_cur * data.time_unit
            p_next = p_cur + v_cur * data.time_unit  # *c, n, 2
            p_next_nonan = p_cur_nonan + v_cur * data.time_unit
            assert ~a_next.isnan().any(), print('find nan in epoch')

            # update destination, yet do not delete people when they arrive
            dis_to_dest = torch.norm(p_cur.detach() - dest_cur, p=2, dim=-1)
            dest_idx_cur[dis_to_dest < 0.5] = dest_idx_cur[dis_to_dest < 0.5] + 1

            dest_idx_cur[dest_idx_cur > dest_num - 1] = dest_idx_cur[dest_idx_cur > dest_num - 1] - 1
            dest_idx_cur_ = dest_idx_cur.unsqueeze(-2).unsqueeze(-1)  # *c, 1, n, 1
            dest_idx_cur_ = dest_idx_cur_.repeat(*([1] * (dest_idx_cur_.dim() - 1) + [2]))
            dest_cur = torch.gather(waypoints, -3, dest_idx_cur_).squeeze().view(*p_cur.shape)  # *c, n, 2

            # update everyone's state
            p_cur = p_next  # *c, n, 2
            p_cur_nonan = p_next_nonan
            v_cur = v_next
            a_cur = a_next
            del p_next_nonan, p_next, v_next #ceshi
            # update hist_v
            # hist_v = self_features[..., :, 2:-3]  # *c, n, 2*x
            # hist_v[..., :, :-2] = hist_v[..., :, 2:]
            # hist_v[..., :, -2:] = v_cur.detach()

            # add newly joined pedestrians
            if t < data.num_frames - 1:
                new_idx = new_peds_flag[..., t + 1, :]  # c, n
                if torch.sum(new_idx) > 0:
                    p_cur[new_idx == 1] = data.position[..., t + 1, :, :][new_idx == 1, :]
                    p_cur_nonan[new_idx == 1] = data.position[..., t + 1, :, :][new_idx == 1, :]
                    v_cur[new_idx == 1] = data.velocity[..., t + 1, :, :][new_idx == 1, :]
                    a_cur[new_idx == 1] = data.acceleration[..., t + 1, :, :][new_idx == 1, :]
                    dest_cur[new_idx == 1] = data.destination[..., t + 1, :, :][new_idx == 1, :]
                    dest_idx_cur[new_idx == 1] = data.dest_idx[..., t + 1, :][new_idx == 1]
                    # update hist_v
                    # hist_v[new_idx == 1] = data.self_features[..., t + 1, :, 2:-3][new_idx == 1]
            
            curr = torch.cat((p_cur, v_cur, a_cur), dim=-1).detach()

            # update hist_features
            if self.config.esti_goal=='acce':
                new_traj = curr.clone().unsqueeze(-2)
                history_features = torch.cat([history_features[...,1:,:], new_traj],dim=-2)
                history_features[history_features.isnan()]=0

                # history_features[..., :-1, :] = history_features[...,1:,:].detach().clone()  # history_features: n, len, 6
                # history_features[..., -1, :2] = p_cur_nonan.clone().detach()
                # history_features[..., -1, 2:4] = v_cur.clone().detach()
                # history_features[..., -1, 4:6] = a_cur.clone().detach()
                # history_features = clear_nan(history_features)   
                
            elif self.config.esti_goal=='pos':
                if self.config.history_dim==2:
                    history_features[..., :-1, :] = history_features[:,1:,:].clone()  # history_features: n, len, 6
                    history_features[..., -1, :2] = p_cur_nonan.clone()
            ped_features_shape = ped_features.shape
            obs_features_shape = obs_features.shape
            near_ped_idx_shape = near_ped_idx.shape
            neigh_ped_mask_shape = neigh_ped_mask.shape
            near_obstacle_idx_shape = near_obstacle_idx.shape
            neigh_obs_mask_shape = neigh_obs_mask.shape
            # calculate features
            ped_features, obs_features, dest_features,\
            near_ped_idx, neigh_ped_mask, near_obstacle_idx, neigh_obs_mask= self.get_relative_features(
                p_cur.unsqueeze(-3).detach(), v_cur.unsqueeze(-3).detach(), a_cur.unsqueeze(-3).detach(),
                dest_cur.unsqueeze(-3).detach(), obstacles, args.topk_ped, args.sight_angle_ped,
                args.dist_threshold_ped, args.topk_obs,
                args.sight_angle_obs, args.dist_threshold_obs)  # c,1,n,k,2
            
            self_features = torch.cat((dest_features.view(*v_cur.shape), v_cur, a_cur, desired_speed), dim=-1)
            ped_features, obs_features = ped_features.view(*ped_features_shape), obs_features.view(*obs_features_shape)
            near_ped_idx, neigh_ped_mask, near_obstacle_idx, neigh_obs_mask = \
                            near_ped_idx.view(*near_ped_idx_shape), \
                            neigh_ped_mask.view(*neigh_ped_mask_shape), \
                            near_obstacle_idx.view(*near_obstacle_idx_shape), \
                            neigh_obs_mask.view(*neigh_obs_mask_shape)

            


        p_res[mask_p_ == 0] = 0.  # delete 'nan'
        data.labels[mask_p_ == 0] = 0.  # delete 'nan'
        a_res[mask_a_==0] = 0.
        
        loss = torch.tensor(0., requires_grad=True, device=p_cur.device)
        # mse_loss = self.multiple_rollout_mse_loss(p_res, data.labels[:, :, :, :2], args.time_decay, reduction='sum')
        # or
        mse_loss = self.multiple_rollout_mse_loss(a_res, data.labels[:, :, :, 4:6], args.time_decay, reduction='sum')

        loss = loss + mse_loss
        if self.config.use_col_focus_loss:
            collision_loss = self.multiple_rollout_collision_loss(
                    p_res, data.labels[:, :, :, :2], 1, args.collision_focus_weight, collisions,
                    reduction='sum')
            loss = loss + collision_loss
        
        return loss
    
    
    def multiple_rollout_mse_loss(self, pred, labels, time_decay, reduction='none', reverse=False):
        """
        multiple rollout training loss with time decay
        Args:
            reverse:
            time_decay:
            pred: c, t, n, 2
            labels:
            reduction:

        Returns:

        """
        loss = (pred - labels) * (pred - labels)
        if not reverse: # mynote: reverse=False stands for the reverse long-term discounted factor in student-force training
            decay = torch.tensor([time_decay ** (pred.shape[1] - t - 1) for t in range(pred.shape[1])],
                                 device=pred.device)
        else:
            decay = torch.tensor([time_decay ** t for t in range(pred.shape[1])], device=pred.device)
        decay = decay.reshape(1, int(pred.shape[1]), 1, 1)
        loss = loss * decay
        return self.reduction(loss, reduction)
    
    @staticmethod
    def reduction(values, mode):
        if mode == 'sum':
            return torch.sum(values)
        elif mode == 'mean':
            return torch.mean(values)
        elif mode == 'none':
            return values
        else:
            raise NotImplementedError
    
    def multiple_rollout_collision_avoidance_loss(self, pred, labels, time_decay, reduction='none'):
        """
        multiple rollout training loss with time decay
        Args:
            weight:
            time_decay:
            pred: c, t, n, 2
            labels: c, t, n, 2
            reduction:

        Returns:

        """
        ni = labels[:, -1:, :, :] - labels[:, 0:1, :, :]
        ni_norm = torch.norm(ni, p=2, dim=-1, keepdim=True)
        ni_norm = ni_norm + 1e-6
        ni = ni / ni_norm  # c,1,n,2

        pred_ = pred - torch.sum(pred * ni, dim=-1, keepdim=True) * ni 
        labels_ = labels - torch.sum(labels * ni, dim=-1, keepdim=True) * ni

        loss = self.multiple_rollout_mse_loss(pred_, labels_, time_decay, reduction='none')
        return self.reduction(loss, reduction)
    
    def multiple_rollout_collision_loss(self, pred, labels, time_decay, coll_focus_weight, collisions,
                                        reduction='none', abnormal_mask=None):
        """
        multiple rollout training loss with time decay
        Args:
            collisions: c, t, n
            time_decay:
            pred: c, t, n, 2
            labels: c, t, n, 2
            reduction:

        Returns:

        """
        collisions = torch.sum(collisions, dim=1)  # c, n
        collisions[collisions > 0] = 1.  #**
        collision_w = collisions
        collision_w = collision_w.unsqueeze(1).repeat(1, pred.shape[1], 1)
        collision_w = collision_w.unsqueeze(-1)  # c, t, n, 1

        # mse_loss = self.multiple_rollout_mse_loss(pred, labels, time_decay, reduction='none')
        collision_focus_loss = self.multiple_rollout_collision_avoidance_loss(pred, labels, time_decay,
                                                                              reduction='none')
        # loss = collision_w * (mse_loss + collision_focus_loss * coll_focus_weight)
        loss = collision_w * collision_focus_loss  # c,t,n,2

        if abnormal_mask is not None:
            abnormal_mask = abnormal_mask.reshape(1, 1, -1, 1)
            loss = loss * abnormal_mask
        
        return self.reduction(loss, reduction)*coll_focus_weight
    

    