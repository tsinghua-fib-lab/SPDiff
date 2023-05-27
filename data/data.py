# -*- coding: utf-8 -*-
"""
@author: zgz

"""
import math
import torch
import torch.utils
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import pdb

class RawData(object):
    """
    Attributes:
        position: (t, N, 2)
        velosity: (t, N, 2)
        acceleration: (t, N, 2)
        destination: (t, N, 2)
        waypoints: (D, N, 2)
        dest_num: (N): the number of waypoints of each pedestrian
        dest_idx: (t, N): the index of the waypoint a user is heading at time t
        obstacles: (M * 2)
        mask_a: (t, N), If someone is not in the frame, mask_a is zero too.
        mask_v: (t, N)
        mask_p: (t, N): 标记了一个人在画面中的时间，在画面中则mask_p为1
        num_steps: total number of time steps
        num_pedestrains: total number of pedestrains
        time_unit
    Notes：
        If an agent is not in the frame, then its position and destination is
        assigned as 'nan'
    """

    def __init__(
            self,
            position=torch.tensor([]),
            velocity=torch.tensor([]),
            acceleration=torch.tensor([]),
            destination=torch.tensor([]),
            waypoints=torch.tensor([]),
            obstacles=torch.tensor([]),
            mask_p=torch.tensor([]),
            mask_v=torch.tensor([]),
            mask_a=torch.tensor([]),
            meta_data=None):
        super(RawData, self).__init__()
        if meta_data is None:
            self.meta_data = dict()
        else:
            self.meta_data = meta_data
            self.time_unit = meta_data['time_unit']
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.destination = destination
        self.waypoints = waypoints
        self.obstacles = obstacles
        self.mask_p = mask_p
        self.mask_v = mask_v
        self.mask_a = mask_a
        if position.shape[-1] > 0:
            self.num_steps = position.shape[0]
            self.num_pedestrians = position.shape[1]
            self.destination_flag = torch.zeros(self.num_pedestrians, dtype=int)
        if waypoints.shape[-1] > 0:
            self.num_destinations = waypoints.shape[0]
            self.dest_idx = self.get_waypoints_index_matrix()
            self.dest_num = self.get_dest_num()

    def to(self, device):
        for u, v in self.__dict__.items():
            if type(v) == torch.Tensor:
                exec('self.' + u + '=' + 'self.' + u + '.to(device)')

    def get_waypoints_index_matrix(self):
        pass

    def get_dest_num(self):
        pass

    def load_trajectory_data(self, data_path):
        """
        Process the raw data to get velocity and acceleration. If an agent is
        not in the frame, then its position is assigned as 'nan'

        Dataset format description: see
        https://tsingroc-wiki.atlassian.net/wiki/spaces/TSINGROC/pages/2261120#%E6%95%B0%E6%8D%AE%E9%9B%86%E5%AD%98%E5%82%A8%E6%A0%BC%E5%BC%8F%E8%AF%B4%E6%98%8E

        Note: 数据需要是连续的，trajectory里不能有断点
        """
        print(f"Loading from '{data_path}'...")
        out_of_bound = torch.tensor(float('nan'))

        data = np.load(data_path, allow_pickle=True)
        assert ('version' in data[0] and data[0]['version'] == 'v2.2'), f"'{data_path}' is out of date."
        meta_data, trajectories, destinations, obstacles = data
        obstacles = torch.tensor(obstacles, dtype=torch.float)

        # todo: 目前为了方便，如果没有obstacles就随便给了一个，之后应该做出处理。
        if obstacles.shape[-1] == 0:
            obstacles = torch.tensor([[1e4, 1e4], [1e4+1, 1e4+1]], dtype=torch.float)

        self.meta_data = meta_data
        self.num_steps = max([u[-1][-1] for u in trajectories]) + 1
        self.num_pedestrians = len(trajectories)
        self.num_destinations = max([len(u) for u in destinations])  # destination d的ts就是轨迹的最后一个ts。
        position = torch.zeros((self.num_steps, self.num_pedestrians, 2))   # ts*num ped的矩阵，每个元素2维
        velocity = torch.zeros((self.num_steps, self.num_pedestrians, 2))
        acceleration = torch.zeros((self.num_steps, self.num_pedestrians, 2))
        mask_p = torch.zeros((self.num_steps, self.num_pedestrians))
        mask_v = torch.zeros((self.num_steps, self.num_pedestrians))
        mask_a = torch.zeros((self.num_steps, self.num_pedestrians))
        for i, traj in enumerate(tqdm(trajectories)):
            for x, y, t in traj:
                position[t, i, :] = torch.tensor([x, y])
                mask_p[t, i] = 1
                mask_v[t, i] = 1
                mask_a[t, i] = 1
            mask_v[t, i] = 0 # mynote: 根据两步轨迹可以计算前一步的速度；根据两步速度可以计算前一步的加速度。也就是轨迹、速度、加速度有效编号依次减1
            mask_a[t, i] = 0
            if t >= 1:
                mask_a[t - 1, i] = 0

        assert (not (position.isnan().any())), "ValueError: Find nan in raw data. Raw data should not contain" \
                                               "any nan values! "

        destination = torch.zeros((self.num_steps, self.num_pedestrians, 2))
        waypoints = torch.zeros((self.num_destinations, self.num_pedestrians, 2)) + out_of_bound # mynote: not all pedestrains have D(num_destinations) destinations
        dest_idx = torch.zeros((self.num_steps, self.num_pedestrians), dtype=torch.long) # MN: 这个index标明了当前ts对应的des是第几个
        dest_num = torch.tensor([len(relays) for relays in destinations])

        for i, relays in enumerate(destinations):
            relays = torch.tensor(relays)
            d = relays[:, 0:2]
            t = relays[:, 2].type(torch.int)
            waypoints[:d.shape[0], i, :] = d
            j = -1  # If len(relays) == 1, the loop below will not be executed, and an error will be reported without this statement.
            for j in range(d.shape[0] - 1):
                destination[t[j]:t[j + 1], i, :] = d[j]
                dest_idx[t[j]:t[j + 1], i] = j
            destination[t[j + 1]:, i, :] = d[j + 1]
            dest_idx[t[j + 1]:, i] = j + 1

        destination[mask_p == 0] = out_of_bound
        # dest_idx[mask_p == 0] = out_of_bound  # in this way we can directly use dest_idx to slice dest
        position[mask_p == 0] = out_of_bound
        velocity = torch.cat(
            (position[1:, :, :], position[-1:, :, :]), 0) - position    # MN：第2个到最后减去第一个到最后（相邻timestep相减）
        velocity /= meta_data['time_unit']
        velocity[mask_v == 0] = 0
        acceleration = torch.cat(
            (velocity[1:, :, :], velocity[-1:, :, :]), 0) - velocity    # MN:故技重施
        acceleration /= meta_data['time_unit']
        acceleration[mask_a == 0] = 0

        assert (not(velocity.isnan().any())), f"find nan in velocity."
        assert (not(acceleration.isnan().any())), f"find nan in acceleration."

        
        self.position, self.velocity = position, velocity
        self.position_desc = [position.masked_fill(mask_p.unsqueeze(-1),0).mean(),
                              position.masked_fill(mask_p.unsqueeze(-1),0).std()
                              ]
        self.velocity_desc = [velocity.masked_fill(mask_v.unsqueeze(-1),0).mean(), #TODO:这里应该用position的mask？或者直接用mask_v,mask_a
                              velocity.masked_fill(mask_v.unsqueeze(-1),0).std()
                              ]
        self.acceleration_desc = [acceleration.masked_fill(mask_a.unsqueeze(-1),0).mean(),
                              acceleration.masked_fill(mask_a.unsqueeze(-1),0).std()
                              ]
        self.acceleration, self.destination = acceleration, destination
        self.waypoints, self.dest_idx, self.dest_num = waypoints, dest_idx, dest_num
        self.obstacles, self.mask_v, self.mask_a = obstacles, mask_v, mask_a
        self.mask_p = mask_p
        self.destination_flag = torch.zeros(self.num_pedestrians, dtype=int)
        self.time_unit = meta_data['time_unit']
        # self.ped_num_each_frame = mask_p.

    def get_frame(self, f: int) -> dict:
        """Get the data of frame f in dictionary format.

        Return:
            - "position": (N, 2)
            - "velocity": (N, 2)
            - "acceleration": (N, 2)
            - "destination": (N, 2), include current destination for ped N
            - "destinations": (R, N, 2), include all R destinations for ped N
            - "destination_flag": (N), include index of current destination for ped N
            - "mask_p": (N)
            - "num_destinations": R
            - "num_pedestrians": N
            - "num_pedestrians": T
            - "meta_data": dict
        """
        frame = {
            "position": self.position[f, :, :],
            "velocity": self.velocity[f, :, :],
            "acceleration": self.acceleration[f, :, :],
            "destination": torch.stack([self.waypoints[int(self.destination_flag[i]), i, :] for i in range(self.num_pedestrians)], dim=0),
            "destinations": self.waypoints,
            "destination_flag": self.destination_flag,
            "num_destinations": self.num_destinations,
            "obstacles": self.obstacles,
            "num_pedestrians": self.num_pedestrians,
            "num_steps": self.num_steps,
            "mask_p": self.mask_p[f, :],
            "meta_data": self.meta_data
        }

        return frame

    def get_current_frame(self) -> dict:
        '''Get the data of current frame in dictionary format.'''
        return self.get_frame(self.num_steps)

    def add_frame(self, frame: dict) -> None:
        """Add a frame discribed in dictionary format to data.

        Input:
            - "position": (N, 2)
            - "velocity": (N, 2)
            - "acceleration": (N, 2)
            - "destinations": (R, N, 2).
            - "destination_flag": (N).
            - "mask_p": (N).
            - "num_destinations": D
            - "num_pedestrians": N+dN
            - "meta_data": dict

            - "add_position": (dN, 2)
            - "add_velocity": (dN, 2)
            - "add_acceleration": (dN, 2)
            - "add_destination": (D', dN, 2)

        In this function, we first add new data of raw N pedestrians(e.g. position, saved in frame["position"]). And then, if dN > 0(i.e. frame['num_pedestrians'] > self.num_pedestrians), call self.add_pedestrians() to add data of new dN pedestrians(saved in frame["add_position"]).
        """
        nan = torch.tensor(float('nan'), device=self.position.device)

        # Add data for all attributes who has a time dimention.
        self.num_steps += 1
        self.position = torch.cat((self.position, frame["position"].unsqueeze(0)), dim=0)
        self.velocity = torch.cat((self.velocity, frame["velocity"].unsqueeze(0)), dim=0)
        self.acceleration = torch.cat((self.acceleration, frame["acceleration"].unsqueeze(0)), dim=0)
        self.destination = torch.cat((self.destination, nan+torch.zeros(1, self.num_pedestrians, 2, device=nan.device)), dim=0)

        # A pedestrian whose current destination is nan or current destination's index equals num_destinations has arrived its final destination, so set its masks to zero.
        arrived_final_destination = torch.tensor([(frame['destination_flag'][i] == frame['num_destinations'] or torch.any(torch.isnan(frame['destinations'][frame['destination_flag'][i], i, :]))) for i in range(self.num_pedestrians)])
        self.destination_flag[arrived_final_destination] = -1
        for i in range(self.destination.shape[1]):
            self.destination[self.num_steps - 1, i, :] = self.waypoints[self.destination_flag[i], i, :]

        mask_ = frame['mask_p'].clone()
        mask_[arrived_final_destination] = 0
        self.mask_p = torch.cat((self.mask_p, mask_.unsqueeze(0)), dim=0)
        self.mask_v = torch.cat((self.mask_v, mask_.unsqueeze(0)), dim=0)
        self.mask_a = torch.cat((self.mask_a, mask_.unsqueeze(0)), dim=0)
        self.position[self.num_steps - 1, mask_ == 0, :] = nan
        self.velocity[self.num_steps - 1, mask_ == 0, :] = 0
        self.acceleration[self.num_steps - 1, mask_ == 0, :] = 0
        self.destination[self.num_steps - 1, mask_ == 0, :] = nan

        # If dN > 0(i.e. frame["num_pedestrians"] > self.num_pedestrians), then it implies that 'frame' has keys like 'add_xxxxx', pass it to self.add_pedestrians to add pedestrians.
        add_num_pedestrians = frame["num_pedestrians"] - self.num_pedestrians
        if (add_num_pedestrians > 0):
            self.add_pedestrians(add_num_pedestrians, **frame)

        return add_num_pedestrians

    def add_pedestrians(self, add_num_pedestrians, add_position, add_destination, add_velocity = torch.tensor([]), add_acceleration = torch.tensor([]), **kwargs):
        '''Add pedestrians with given initial state.

        Input:
             - add_position: (dN, 2)
             - add_destination: (D', dN, 2). If D' > D, we first expand self.destination to (D', N, 2), then concatenate it with add_destination to get (D', N+dN, 2).
             - add_velocity: (dN, 2)
             - add_acceleration: (dN, 2)

        Note: Do NOT change the formal parameters' names, as they should be same with the update_functions defined in data.scenarios.
        '''
        nan = torch.tensor(float('nan'), device=self.position.device)
        self.num_pedestrians += add_num_pedestrians

        self.position = torch.cat((self.position, nan + torch.zeros((self.num_steps, add_num_pedestrians, 2), device=nan.device)), dim=1)
        self.position[-1, -add_num_pedestrians:, :] = add_position

        self.velocity = torch.cat((self.velocity, torch.zeros((self.num_steps, add_num_pedestrians, 2), device=nan.device)), dim=1)
        if(add_velocity.numel()):
            self.velocity[-1, -add_num_pedestrians:, :] = add_velocity

        self.acceleration = torch.cat((self.acceleration, torch.zeros((self.num_steps, add_num_pedestrians, 2), device=nan.device)), dim=1)
        if(add_acceleration.numel()):
            self.acceleration[-1, -add_num_pedestrians:, :] = add_acceleration

        self.mask_p = torch.cat((self.mask_p, torch.zeros((self.num_steps, add_num_pedestrians), device=nan.device)), dim=1)
        self.mask_p[-1, -add_num_pedestrians:] = 1

        self.mask_v = torch.cat((self.mask_v, torch.zeros((self.num_steps, add_num_pedestrians), device=nan.device)), dim=1)
        self.mask_v[-1, -add_num_pedestrians:] = 1

        self.mask_a = torch.cat((self.mask_a, torch.zeros((self.num_steps, add_num_pedestrians), device=nan.device)), dim=1)
        self.mask_a[-1, -add_num_pedestrians:] = 1

        self.waypoints = torch.cat((self.waypoints, nan + torch.zeros((self.num_destinations, add_num_pedestrians, 2), device=nan.device)), dim=1)
        add_num_destinations = add_destination.shape[0] - self.num_destinations
        if(add_num_destinations > 0):
            self.num_destinations += add_num_destinations
            self.waypoints = torch.cat((self.waypoints, nan + torch.zeros((add_num_destinations, self.num_pedestrians, 2), device=nan.device)), dim=0)
        self.waypoints[:add_destination.shape[0], -add_num_pedestrians:, :] = add_destination

        self.destination = torch.cat((self.destination, nan + torch.zeros((self.num_steps, add_num_pedestrians, 2), device=nan.device)), dim=1)
        self.destination[self.num_steps - 1, -add_num_pedestrians:, :] = add_destination[0, :]

        self.destination_flag = torch.cat((self.destination_flag, torch.zeros((add_num_pedestrians), device=nan.device)), dim=0).type(torch.int)

    def to_trajectories(self):
        trajectories = []
        for n in range(self.num_pedestrians):
            trajectory = []
            for f in range(self.num_steps):
                if (self.mask_p[f, n] == 1):
                    trajectory.append((self.position[f, n, 0].item(),
                                       self.position[f, n, 1].item(), f))
            trajectories.append(trajectory)
        return trajectories

    def to_destinations(self):
        destinations = []
        frame_id = torch.arange(self.num_steps)
        for i, relays in enumerate(torch.transpose(self.waypoints, 0, 1)):
            destination = []
            for des in relays:
                if(torch.any(torch.isnan(des))):
                    continue
                tmp = frame_id[torch.norm(des - self.destination[:, i, :], dim=1) < 0.01]
                if(tmp.numel() > 0):
                    t = tmp[0]
                    destination.append((des[0].item(), des[1].item(), t.item()))
                else:
                    break
            if(destination):
                destinations.append(destination)
        return destinations

    def save_data(self, data_path: str):
        self.meta_data["version"] = "v2.2"
        data = np.array((self.meta_data, self.to_trajectories(),
                         self.to_destinations(),
                         self.obstacles.tolist()), dtype=object)
        np.save(data_path, data)
        print(f"Saved to '{data_path}'.")


class Pedestrians(object):
    """
    """

    def __init__(self):
        super(Pedestrians, self).__init__()

    @staticmethod
    def get_heading_direction(velocity):
        """
        todo: 这个function需要完善，现在做的操作其实是对v=0的时候做了补全，v自身就是heading direction
        Function: infer people's heading direction (without normalization);
        Using linear smoothing
        Args:
            velocity: (*c, t, N, 2)
        Return:
            heading_direction: (*c, t, N, 2)
            (没有归一化)
        """
        heading_direction = velocity.clone()
        if heading_direction.dim() == 3:
            for i in range(heading_direction.shape[-2]):
                tmp_direction = torch.tensor([0, 0], dtype=float, device=velocity.device)
                for t in range(heading_direction.shape[-3] - 1, -1, -1): #749~0
                    if torch.norm(heading_direction[t, i, :], p=2, dim=0) == 0:
                        heading_direction[t, i, :] = tmp_direction
                    else:
                        tmp_direction = heading_direction[t, i, :]
                for t in range(heading_direction.shape[-3]):
                    if torch.norm(heading_direction[t, i, :], p=2, dim=0) == 0:
                        heading_direction[t, i, :] = tmp_direction
                    else:
                        tmp_direction = heading_direction[t, i, :]
        elif heading_direction.dim() == 4:
            for j in range(heading_direction.shape[-4]):
                for i in range(heading_direction.shape[-2]):
                    tmp_direction = torch.tensor([0, 0], dtype=float, device=velocity.device)
                    for t in range(heading_direction.shape[-3] - 1, -1, -1):
                        if torch.norm(heading_direction[j, t, i, :], p=2, dim=0) == 0:
                            heading_direction[j, t, i, :] = tmp_direction
                        else:
                            tmp_direction = heading_direction[j, t, i, :]
                    for t in range(heading_direction.shape[-3]):
                        if torch.norm(heading_direction[j, t, i, :], p=2, dim=0) == 0:
                            heading_direction[j, t, i, :] = tmp_direction
                        else:
                            tmp_direction = heading_direction[j, t, i, :]
        # 归一化
        tmp_direction = torch.norm(heading_direction, p=2, dim=-1, keepdim=True)
        tmp_direction_ = tmp_direction.clone()
        tmp_direction_[tmp_direction_ == 0] += 0.1
        heading_direction = heading_direction / tmp_direction_
        return heading_direction

    @staticmethod
    def get_relative_quantity(A, B):
        """
        Function:
            The relative amount among all objects in A and all objects in B at
            each moment get relative vector xj - xi fof each pedestrain i (B - A)
        Args:
            A: (*c, t, N, dim)
            B: (*c, t, M, dim)
        Return:
            relative_A: (*c, t, N, M, dim)
        """
        dim = A.dim()
        A = A.unsqueeze(-2).repeat(*([1]*(dim-1) + [B.shape[-2]] + [1]))  # *c, t, N, M, dim
        B = B.unsqueeze(-3).repeat(*([1]*(dim-2) + [A.shape[-3]] + [1, 1]))
        relative_A = B - A

        return relative_A.contiguous()
    
    @staticmethod
    def get_relative_quantity_expand(A, B):
        """
        Function:
            The relative amount among all objects in A and all objects in B at
            each moment get relative vector xj - xi fof each pedestrain i (B - A)
        Args:
            A: (*c, t, N, dim)
            B: (*c, t, M, dim)
        Return:
            relative_A: (*c, t, N, M, dim)
        """
        dim = A.dim()
        sizeA = A.shape[:-1] + B.shape[-2:-1] + A.shape[-1:]
        
        A = A.unsqueeze(-2).expand(sizeA)  # *c, t, N, M, dim
        sizeB = B.shape[:-2] + A.shape[-3:-2] + B.shape[-2:]
        
        B = B.unsqueeze(-3).expand(sizeB)
        relative_A = B - A

        return relative_A.contiguous()

    def get_nearby_obj_in_sight(self, position, objects, heading_direction, k, angle_threshold):
        """
        Function: get The k closest people's index at time t:
            Calculate the relative position between every two people, and then look
            at the angle between this relative position and the speed direction of the
            current person. If the angle is within the threshold range, then it is
            judged to be in the field of view.
        Args:
            k: get the nearest k persons
            position: (*c, t, N, 2)
            objects: (*c, t, M, 2)
            heading_direction: (*c, t, N, 2)
        Return:
            neighbor_index: (*c, t, N, k), The k closest objects' index at time t
        """

        relative_pos = self.get_relative_quantity(position, objects)  # *c,t,N,M,2
        relative_pos[relative_pos.isnan()] = float('inf')
        distance = torch.norm(relative_pos, p=2, dim=-1)  # *c,t,N,M

        dim = heading_direction.dim()
        heading_direction = heading_direction.unsqueeze(-2).repeat(
            *[[1]*(dim-1) + [relative_pos.shape[-2]] + [1]])
        view_field = torch.cosine_similarity(
            relative_pos, heading_direction, dim=-1)
        view_field[view_field.isnan()] = -1
        distance[view_field < math.cos(
            3.14 * angle_threshold / 180)] = float('inf')

        sorted_dist, indices = torch.sort(distance, dim=-1)

        return sorted_dist[..., :k], indices[..., :k]

    def get_filtered_features(self, features, nearby_idx, nearby_dist, dist_threshold):
        """
        features: (*c, t, N, M, dim)
        nearby_idx: (*c, t, N, k)
        nearby_dist: (*c, t, N, k)
        """
        dim = nearby_idx.dim()
        nearby_idx = nearby_idx.unsqueeze(-1).repeat(*([1]*dim + [features.shape[-1]]))  # t,n,k,dim
        features = torch.gather(features, -2, nearby_idx)  # t,n,k,dim

        dist_filter = torch.ones(features.shape, device=features.device)
        nearby_dist = nearby_dist.unsqueeze(-1).repeat(*([1]*dim + [features.shape[-1]]))
        dist_filter[nearby_dist > dist_threshold] = 0
        features[dist_filter == 0] = 0  # nearest neighbor less than k --> zero padding

        return features, dist_filter[...,0]

    def get_relative_features(
            self, position, velocity, acceleration, destination, obstacles,
            topk_ped, sight_angle_ped, dist_threshold_ped, topk_obs,
            sight_angle_obs, dist_threshold_obs
    ):
        """
            position: *c, t, N, 2
            obstacles: *c, N, 2
            destination: *c, t, N, 2
        Return:
            dest_features: *c, t, N, 2
        Notice:
            返回值中，相对position远，或者自己已经不在frame里了，那么relative
            features的值就会是0，做了zero padding。不会有inf出现。
            c is channel size
        """

        acceleration[acceleration.isnan()] = 0
        velocity[velocity.isnan()] = 0

        num_steps = position.shape[-3]
        heading_direction = self.get_heading_direction(velocity)

        near_ped_dist, near_ped_idx = self.get_nearby_obj_in_sight(
            position, position, heading_direction, topk_ped, sight_angle_ped)#找最近的top k=6
        ped = torch.cat((position, velocity, acceleration), dim=-1)
        ped_features = self.get_relative_quantity(ped, ped)  # *c t N N dim
        ped_features, neigh_ped_mask = self.get_filtered_features(
            ped_features, near_ped_idx, near_ped_dist, dist_threshold_ped)

        dest_features = destination - position
        dest_features[dest_features.isnan()] = 0.

        obs_features = torch.tensor([[] for _ in range(num_steps)], device=obstacles.device)
        if len(obstacles) > 0:
            dim = obstacles.dim()
            obstacles = obstacles.unsqueeze(-3).repeat(
                *([1]*(dim-2) + [num_steps] + [1, 1]))  # *c, t, N, 2
            near_obstacle_dist, near_obstacle_idx = self.get_nearby_obj_in_sight(
                position, obstacles, heading_direction, topk_obs, sight_angle_obs) #找最近的top k=10
            obs = torch.cat((obstacles, torch.zeros(obstacles.shape, device=obstacles.device),
                             torch.zeros(obstacles.shape, device=obstacles.device)), dim=-1)
            obs_features = self.get_relative_quantity(ped, obs)  # t N M dim
            obs_features, neigh_obs_mask= self.get_filtered_features(
                obs_features, near_obstacle_idx, near_obstacle_dist, dist_threshold_obs)

        return ped_features, obs_features, dest_features, \
            near_ped_idx, neigh_ped_mask, near_obstacle_idx, neigh_obs_mask

    @staticmethod
    def calculate_collision_label(ped_features):
        """
        计算1s之内，如果保持当前的相对速度，那么是否会撞
        Args:
            ped_features: ...,k,6: (p,v,a)

        Returns:
            collisions: ...,k
        """
        # 计算之后1s中每0.1s的相对位置，如果小于0.5那就会撞
        with torch.no_grad():
            time = torch.arange(10, device=ped_features.device) * 0.1
            time = time.resize(*([1] * (ped_features.dim()-1)), 10, 1)
            collisions = ped_features[..., :2].unsqueeze(-2) + ped_features[..., 2:4].unsqueeze(-2) * time
            collisions = torch.norm(collisions, p=2, dim=-1)  # c,t,n,k,10
            collisions[collisions >= 0.5] = 0
            collisions[(collisions < 0.5) & (collisions != 0)] = 1
            collisions = torch.sum(collisions, dim=-1)  # c,t,n,k
            collisions[collisions > 0] = 1

        return collisions

    @staticmethod
    def collision_detection(position, threshold, real_position=None):
        """
        注意：输入的position必须是包含nan指代出frame的，否则无法计算平均每个人在一个frame里面的时长
        Args:
            position: t,n,2 / c,t,n,2 注意时间这个维度必须对，才能相应的对朋友进行处理


        Returns:

        """
        # position = position.clone()
        relative_pos = Pedestrians.get_relative_quantity(position, position)  # c,n,n,2
        rel_distance = torch.norm(relative_pos, p=2, dim=-1)  # c,n,n
        collisions = rel_distance.clone()
        collisions[rel_distance < threshold] = 1
        collisions[rel_distance >= threshold] = 0

        # delete self loop
        identical_matrix = torch.eye(collisions.shape[-1], device=collisions.device)
        if collisions.dim() == 3:
            identical_matrix = identical_matrix.unsqueeze(0).repeat(collisions.shape[0], 1, 1)
        elif collisions.dim() == 4:
            identical_matrix = identical_matrix.reshape(1, 1, identical_matrix.shape[-1], -1)
            identical_matrix = identical_matrix.repeat(collisions.shape[0], collisions.shape[1], 1, 1)
        collisions = collisions - identical_matrix

        collisions[collisions.isnan()] = 0  # t,n,n

        # valid_steps = position[..., 0].clone()  # c,t,n
        # valid_steps[~valid_steps.isnan()] = 1
        # valid_steps[valid_steps.isnan()] = 0
        # valid_steps = torch.sum(valid_steps, dim=-2, keepdim=True)  # c,1,n

        # 对朋友进行处理，去掉每一步都在碰撞的人：如果两个人有超过2s的时间都黏在一起，那么就认为这两个人是朋友。
        # todo: 如果两个人相撞的时间是两个人共同出现时间的30%或以上，那么两个人就是朋友
        if real_position is not None:
            # 仅供测试使用，支持输入为3维
            assert real_position.dim() == 3, 'Value Error: real_position only supports 3 dimensional inputs (t,n,2)'
            relative_pos = Pedestrians.get_relative_quantity(real_position, real_position)  # c,n,n,2
            rel_distance = torch.norm(relative_pos, p=2, dim=-1)  # c,n,n
            real_collisions = rel_distance.clone()
            real_collisions[rel_distance < threshold] = 1
            real_collisions[rel_distance >= threshold] = 0
            real_collisions[real_collisions.isnan()] = 0  # t,n,n
            friends = torch.sum(real_collisions, dim=0)  # n,n
            friends[friends <= 25] = 1
            friends[friends > 25] = 0
            friends = friends.unsqueeze(0)
        else:
            if collisions.dim() == 3:
                friends = torch.sum(collisions, dim=0)  # n,n
                friends[friends <= 25] = 1
                friends[friends > 25] = 0
                friends = friends.unsqueeze(0)
            elif collisions.dim() == 4:
                # 训练： 删除原始数据前3个step就黏在一起的人，比如朋友，这种人不算避让
                friends = collisions[:, :4]
                friends = torch.sum(friends, dim=1)
                friends[friends > 0] = 1
                friends = 1 - friends
                friends = friends.unsqueeze(1)
        collisions *= friends

        return collisions


class TimeIndexedPedData(Dataset, Pedestrians):
    """
    Attributes:
        ped_features: t * N * k1 * dim(6): relative position, velocity, acceleration
        obs_features: t * N * k2 * dim(6)
        self_features: t * N * dim(6): dest_features, cur_acc, desired_speed
        labels: t * N * 2
    """

    def __init__(
            self,
            ped_features=torch.tensor([]),
            obs_features=torch.tensor([]),
            self_features=torch.tensor([]),
            desired_speed=torch.tensor([]),
            labels=torch.tensor([]),
            mask_p=torch.tensor([]),
            mask_v=torch.tensor([]),
            mask_a=torch.tensor([])):
        super(TimeIndexedPedData, self).__init__()
        self.ped_features = ped_features
        self.obs_features = obs_features
        self.self_features = self_features
        self.labels = labels
        self.desired_speed = desired_speed
        self.mask_p = mask_p
        self.mask_v = mask_v
        self.mask_a = mask_a
        self.num_frames = ped_features.shape[0]
        self.dataset_len = self.num_frames
        if self.dataset_len:
            self.num_pedestrians = ped_features.shape[1]
            self.ped_feature_dim = ped_features.shape[-1]
            self.obs_feature_dim = obs_features.shape[-1]
            self.self_feature_dim = self_features.shape[-1]
        else:
            self.num_pedestrians = 0
            self.ped_feature_dim = 0
            self.obs_feature_dim = 0
            self.self_feature_dim = 0

        self.mask_p_pred = None
        self.mask_v_pred = None
        self.mask_a_pred = None
        self.meta_data = None

    def __len__(self):

        return self.num_frames

    def __getitem__(self, index):

        if self.num_frames > 0:
            item = [self.ped_features[index], self.obs_features[index],
                    self.self_features[index], self.labels[index], self.self_hist_features[index],
                    self.mask_p_pred[index], self.mask_v_pred[index], self.mask_a_pred[index]]
        else:
            raise ValueError("Haven't load any data yet!")

        return item

    def to_pointwise_data(self):
        tmp_data = PointwisePedData()
        tmp_data.load_from_time_indexed_peddata(self)
        return tmp_data

    def to_channeled_time_index_data(self, stride=25, mode='slice'):
        tmp_data = ChanneledTimeIndexedPedData()
        tmp_data.load_from_time_indexed_peddata(self, stride, mode)
        return tmp_data

    def move_index_matrix(self, idx_matrix, direction='forward', n_steps=1, dim=0):
        """
        比如向后移动一位 [[0,1,1,1],[1,1,0,0]] -> [[0,0,1,1], [0,1,0,0]]

        Args:
            idx_matrix: 0-1 index matrix
            direction: 'forward' or 'backward'
            n_steps: number of steps
            dim: moving dimension

        Returns:
            mask: results
        """
        mask = idx_matrix.clone()
        moving_shape = list(mask.shape)
        moving_shape[dim] = n_steps
        if direction == 'backward': # 选取0~ts-n_steps的范围，整体后移
            mask = mask.index_select(dim, torch.arange(mask.shape[dim] - n_steps, device=idx_matrix.device))
            mask = torch.cat((torch.zeros(moving_shape, device=idx_matrix.device), mask), dim=dim)
        elif direction == 'forward': # 选取n_steps~ts的范围，整体前移
            mask = mask.index_select(dim, torch.arange(n_steps, mask.shape[dim], device=idx_matrix.device))
            mask = torch.cat((mask, torch.zeros(moving_shape, device=idx_matrix.device)), dim=dim)
        mask *= idx_matrix
        return mask

    @staticmethod
    def turn_detection(data: RawData):
        """
        判断一个人是否存在转弯行s为，方法：
            1）连接此人起始点和终止点，看当前任进入的时候他的速度和直连的夹角是否小于一定阈值，15度（目标改变，转弯）
            2）看起止点的距离是否远小于这个人真正走过的距离（回头）
            3）看此人的平均速度，如果非常低也不用看（滞留）
        Args:
            data:

        Returns:
            non_abnormal: N，如果一个人行为异常则为0，否则为1

        """
        position = data.position.clone()
        velocity = data.velocity.clone()
        T, N, _ = position.shape
        position[position.isnan()] = 1e4

        # 找到每个人的起止点和初始速度
        starts = torch.zeros((N, 2), device=position.device) + 1e4
        v_starts = torch.zeros((N, 2), device=position.device) + 1e4
        ends = torch.zeros((N, 2), device=position.device) + 1e4
        for i in range(T):
            v_starts[starts >= 1e4] = velocity[i, starts >= 1e4]
            starts[starts >= 1e4] = position[i, starts >= 1e4]
            ends[ends >= 1e4] = position[T - i - 1, ends >= 1e4]
        dist = torch.norm(ends - starts, p=2, dim=-1) + 1e-6
        norm_v = torch.norm(v_starts, p=2, dim=-1) + 1e-6

        # 判断转弯:
        cos_theta = torch.sum((ends - starts) * v_starts, dim=-1) / dist / norm_v
        cos_theta[cos_theta < np.cos(3.1415 * 20 / 180)] = 0
        cos_theta[cos_theta > 0] = 1

        non_abnormal = cos_theta

        # todo: 判断回头

        # 判断滞留
        mean_velocity = torch.norm(velocity, p=2, dim=-1)  # t,n
        mean_velocity = torch.sum(mean_velocity, dim=0) / torch.sum(data.mask_v, dim=0)

        non_abnormal[mean_velocity < 1.3 * 0.3] = 0

        return non_abnormal

    def make_dataset(self, args, raw_data: RawData):
        """
        Transform a RawData object into a TimeIndexedPedData object
        Inputs:
            raw_data: RawData object
        Outputs:
            ped_features: t * N * k1 * dim(6): relative position, velocity, acceleration
            obs_features: t * N * k2 * dim(6)
            self_features: t * N * dim(2 + 2 + 1): dest_features, cur_acc, desired_speed
            self_hist_features:  t*N*k*dim(6)
                historical velocity: vx0, vy0, vx1, vy1, ..., vxn, vyn
            labels: t * N * 6 Position, speed, acceleration at the t+1 time step
            mask_a: t * N
            mask_p_pred: t * N 对于预测来说，最后一个time step无用，同时由于history features的影响，最初的几步也没用

        Notes:
            空25step才开始计算
        """

        # raw_data.to(args.device)

        ped_features, obs_features, dest_features, \
        near_ped_idx, neigh_ped_mask, near_obstacle_idx, neigh_obs_mask = self.get_relative_features(
            raw_data.position, raw_data.velocity,
            raw_data.acceleration, raw_data.destination,
            raw_data.obstacles, args.topk_ped, args.sight_angle_ped,
            args.dist_threshold_ped, args.topk_obs,
            args.sight_angle_obs, args.dist_threshold_obs)

        raw_data.to(args.device)
        ped_features = ped_features.to(args.device)
        obs_features = obs_features.to(args.device)
        dest_features = dest_features.to(args.device)
        near_ped_idx = near_ped_idx.to(args.device)
        neigh_ped_mask = neigh_ped_mask.to(args.device)
        near_obstacle_idx = near_obstacle_idx.to(args.device)        
        neigh_obs_mask = neigh_obs_mask.to(args.device)
        self.near_ped_idx = near_ped_idx
        self.neigh_ped_mask = neigh_ped_mask
        self.near_obstacle_idx = near_obstacle_idx
        self.neigh_obs_mask = neigh_obs_mask

        # 加入对行人是否有转弯、滞留等行为的判断
        self.abnormal_mask = self.turn_detection(raw_data)

        self.ped_features = ped_features
        if len(obs_features) > 0:
            self.obs_features = obs_features
        else:
            self.obs_features = torch.tensor([[] for _ in range(ped_features.shape[0])], device=ped_features.device)

        # get hist_features
        num_frames = ped_features.shape[0]
        num_peds = ped_features.shape[1]
        hist_features = torch.zeros(list(raw_data.velocity.shape[:-1])+[6], device=ped_features.device)  # t, N, 6
        hist_features = hist_features.unsqueeze(2).repeat(1, 1, args.num_history_feature, 1)  # t, N, k, 6
        for i in range(args.num_history_feature):
            tmp_frame = args.num_history_feature-i-1
            hist_features[tmp_frame:, :, i, 0:2] = raw_data.position[:num_frames-tmp_frame, :, :]
            hist_features[tmp_frame:, :, i, 2:4] = raw_data.velocity[:num_frames-tmp_frame, :, :]
            hist_features[tmp_frame:, :, i, 4:6] = raw_data.acceleration[:num_frames-tmp_frame, :, :]
        # mask = hist_features!=hist_features
        # hist_features = hist_features.masked_fill(mask,0) # aborted: do zero_padding during training/sampling
        
        # hist_velocity = hist_velocity.reshape(num_frames, num_peds, -1)  # t, N, k*2 

        # get hist_velocity features
        num_frames = ped_features.shape[0]
        num_peds = ped_features.shape[1]
        hist_velocity = torch.zeros(raw_data.velocity.shape, device=ped_features.device)  # t, N, 2
        hist_velocity = hist_velocity.unsqueeze(2).repeat(1, 1, args.num_history_velocity, 1)  # t, N, k, 2
        for i in range(args.num_history_velocity):
            tmp_frame = args.num_history_velocity-i-1
            hist_velocity[tmp_frame:, :, i, :] = raw_data.velocity[:num_frames-tmp_frame, :, :]
        hist_velocity = hist_velocity.reshape(num_frames, num_peds, -1)  # t, N, k*2 

        # calculate desired_speed
        skip_frames = args.skip_frames  # settings: 开始空25个frames，然后再预测，这25个frames的速度均值作为desired-speed的预测值
        desired_speed = torch.zeros(num_peds, device=ped_features.device)  # N
        for i in range(num_peds):
            start_idx = 0
            for j in range(num_frames):
                if torch.norm(raw_data.velocity[j, i, :]) > 0:
                    start_idx = j
                    break
            desired_speed[i] = torch.mean(torch.norm(raw_data.velocity[start_idx:start_idx+skip_frames, i, :], p=2, dim=-1))
        desired_speed = desired_speed.unsqueeze(0).unsqueeze(-1)
        desired_speed = desired_speed.repeat(num_frames, 1, 1)
        self.self_hist_features = hist_features
        self.self_features = torch.cat((dest_features, hist_velocity, raw_data.acceleration, desired_speed), dim=-1) # MN:用以计算f_des

        self.labels = torch.cat((raw_data.position, raw_data.velocity, raw_data.acceleration), dim=-1)

        collision_labels = self.calculate_collision_label(self.ped_features)
        self.labels = torch.cat((self.labels, collision_labels), dim=-1)

        # update time steps that are useless for validation
        self.mask_a_pred = self.move_index_matrix(raw_data.mask_a, 'backward', skip_frames-1, dim=0) #腾出了前面25个ts置为0，因为这一部分要skip
        self.mask_v_pred = self.move_index_matrix(raw_data.mask_v, 'backward', skip_frames-1, dim=0)
        self.mask_p_pred = self.move_index_matrix(raw_data.mask_p, 'backward', skip_frames-1, dim=0)

        # the last time step cannot be used for prediction
        self.mask_a_pred = self.move_index_matrix(self.mask_a_pred, 'forward', 1, dim=0)
        # self.mask_v_pred = self.move_index_matrix(self.mask_v_pred, 'forward', 1, dim=0)
        # self.mask_p_pred = self.move_index_matrix(self.mask_p_pred, 'forward', 1, dim=0)

        self.meta_data = raw_data.meta_data
        self.topk_obs = args.topk_obs
        self.num_frames = self.dataset_len = num_frames
        self.num_pedestrians = self.ped_features.shape[1]
        self.ped_feature_dim = self.ped_features.shape[-1]
        self.obs_feature_dim = self.obs_features.shape[-1]
        self.self_feature_dim = self.self_features.shape[-1]

    def to(self, device):
        for u, v in self.__dict__.items():
            if type(v) == torch.Tensor:
                exec('self.' + u + '=' + 'self.' + u + '.to(device)')

    def set_dataset_info(self, dataset, raw_data, slice_idx):
        # pdb.set_trace()
        self.meta_data = raw_data.meta_data
        self.time_unit = raw_data.time_unit
        self.num_frames = self.dataset_len = dataset.num_frames
        self.position = raw_data.position[slice_idx, :, :]
        self.velocity = raw_data.velocity[slice_idx, :, :]
        self.acceleration = raw_data.acceleration[slice_idx, :, :]
        self.position_desc = raw_data.position_desc
        self.velocity_desc = raw_data.velocity_desc
        self.acceleration_desc = raw_data.acceleration_desc
        self.obstacles = raw_data.obstacles
        self.destination = raw_data.destination[slice_idx, :, :]
        self.dest_idx = raw_data.dest_idx[slice_idx, :]
        self.waypoints = raw_data.waypoints
        self.dest_num = raw_data.dest_num

        self.mask_p = raw_data.mask_p[slice_idx, :]
        self.mask_a = raw_data.mask_a[slice_idx, :]
        self.mask_v = raw_data.mask_v[slice_idx, :]

        self.mask_p_pred = dataset.mask_p_pred[slice_idx, :]
        self.mask_v_pred = dataset.mask_v_pred[slice_idx, :]
        self.mask_a_pred = dataset.mask_a_pred[slice_idx, :]
        self.self_feature_dim = dataset.self_feature_dim
        self.ped_feature_dim = dataset.ped_feature_dim
        self.obs_feature_dim = dataset.obs_feature_dim
        self.abnormal_mask = dataset.abnormal_mask


class TimeIndexedPedDataPolarCoor(TimeIndexedPedData):
    """docstring for TimeIndexedPedDataPolarCoor"""
    def __init__(self):
        super(TimeIndexedPedDataPolarCoor, self).__init__()

    @staticmethod
    def cart_to_polar(points, base):
        """
        输入是笛卡尔坐标系的points x,y，输出是以base为极轴的极坐标系坐标 r,theta, r>0, theta in [-pi,pi]
        如果输入坐标为nan，输出也会是nan

        Args:
            points: c, t, n, 2
            base: c, t, n, 2  (需要是归一化的)

        Returns:
            polar_coor: c, t, n, 2
        """
        volume = torch.norm(points, p=2, dim=-1, keepdim=True)
        volume_ = volume.clone()
        volume_[volume_ == 0] += 0.1  # to avoid zero devision

        p = points/volume_
        cos_p = p[..., 0]
        sin_p = p[..., 1]
        cos_b = base[..., 0]
        sin_b = base[..., 1]
        sign_sin_pb = torch.sign(sin_p * cos_b - cos_p * sin_b)
        sign_sin_pb = sign_sin_pb.unsqueeze(-1)

        theta = torch.sum(points * base, dim=-1, keepdim=True) / volume_
        theta = torch.clamp(theta, -1+1e-6, 1-1e-6)
        theta = torch.acos(theta) * sign_sin_pb

        return torch.cat((volume, theta), dim=-1)

    @staticmethod
    def polar_to_cart(points, base):
        """
        输入是以base为极轴的极坐标系的points r, theta，输出是笛卡尔坐标系x,y
        Args:
            points: c, t, n, 2
            base: c, t, n, 2  (需要是归一化的)

        Returns:
            cart_coor: c, t, n, 2
        """
        cart_base = torch.zeros(base.shape, device=points.device)
        cart_base[..., 0] = 1.  # base is (1, 0)
        polar_base = TimeIndexedPedDataPolarCoor.cart_to_polar(base, cart_base)
        polar_base[..., 0] = 0
        points = points + polar_base
        x = points[..., 0] * torch.cos(points[..., 1])
        y = points[..., 0] * torch.sin(points[..., 1])
        return torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), dim=-1)

    def to_polar_system(self):
        """
        需要转极坐标的：ped_features,obs_features, labels, dest features
        Args:
            points: ..., N, 2
            base: ..., N, 2
            ped_features: t * N * k1 * dim(6): relative position, velocity, acceleration in polar coordinates (以速度方向为坐标轴方向，以保持旋转不变性)
            obs_features: t * N * k2 * dim(6):
            self_features: t * N * dim(2 + 2*k + 2 + 1): dest_features, hist_velocity, cur_acc, desired_speed
            labels: t * N * 6 Position, speed, acceleration, acc_polar at the t+1 time step
        """

        velocity = self.self_features[..., -5:-3]  # c, t, n, 2
        n_direction = self.get_heading_direction(velocity)  # c, t, N, 2

        # labels
        # acc_polar = self.cart_to_polar(self.labels[..., -2:], n_direction)
        # self.labels = torch.cat((self.labels, acc_polar), dim=-1)

        n_directions_ = n_direction.unsqueeze(-2)
        n_directions_ = n_directions_.repeat(*[1]*(n_direction.dim() - 1), self.ped_features.shape[-2], 1)
        self.ped_features = torch.cat(
            (self.cart_to_polar(self.ped_features[..., 0:2], n_directions_),
            self.cart_to_polar(self.ped_features[..., 2:4], n_directions_),
            self.cart_to_polar(self.ped_features[..., 4:6], n_directions_)), dim=-1
            )
        n_directions_ = n_direction.unsqueeze(-2)
        n_directions_ = n_directions_.repeat(*[1]*(n_direction.dim() - 1), self.obs_features.shape[-2], 1)
        if self.obs_feature_dim > 0:
            self.obs_features = torch.cat(
                (self.cart_to_polar(self.obs_features[..., 0:2], n_directions_),
                self.cart_to_polar(self.obs_features[..., 2:4], n_directions_),
                self.cart_to_polar(self.obs_features[..., 4:6], n_directions_)), dim=-1
                )


class PointwisePedData(Dataset):
    """
    单步预测，和时间无关，因此直接把时间打散，同时利用filter，直接去掉frame里面
    没有的人，从而对齐矩阵，加速训练
    Attributes:
        ped_features: (N * t) * k1 * dim(6)
        obs_features: (N * t) * k2 * dim(6)
        self_features: (N * t) * dim(6)
        labels: (N * t) * 6
    """

    def __init__(
            self,
            ped_features=torch.tensor([]),
            obs_features=torch.tensor([]),
            self_features=torch.tensor([]),
            labels=torch.tensor([]),
            self_hist_features=torch.tensor([])):
        super(PointwisePedData, self).__init__()
        self.ped_features = ped_features
        self.obs_features = obs_features
        self.self_features = self_features
        self.self_hist_features = self_hist_features
        self.labels = labels
        self.dataset_len = labels.shape[0]
        self.self_feature_dim = self.self_features.shape[-1]
        self.ped_feature_dim = self.ped_features.shape[-1]
        self.obs_feature_dim = self.obs_features.shape[-1]

    def __len__(self):

        return self.dataset_len

    def __getitem__(self, idx):
        item = [self.ped_features[idx], self.obs_features[idx],
                self.self_features[idx], self.labels[idx], self.self_hist_features[idx]]
        return item

    def add(self, other):
        assert(self.time_unit == other.time_unit), "PointwisePedData with different time_unit cannot be merged"
        assert(self.ped_features.shape[-1] == other.ped_features.shape[-1]), "PointwisePedData with different feature shape cannot be merged"

        self.ped_features = torch.cat((self.ped_features, other.ped_features), dim=0)
        self.obs_features = torch.cat((self.obs_features, other.obs_features), dim=0)
        self.self_features = torch.cat((self.self_features, other.self_features), dim=0)
        self.labels = torch.cat((self.labels, other.labels), dim=0)
        self.self_hist_features = torch.cat((self.self_hist_features, other.self_hist_features), dim=0)
        self.dataset_len = self.dataset_len + other.dataset_len

    def make_dataset(self, args, raw_data):
        pass

    def load_from_time_indexed_peddata(self, data: TimeIndexedPedData, slice_idx=None):
        if slice_idx:
            ped_features, obs_features, self_features, labels, self_hist_features = data[slice_idx]
            mask_a_pred = data.mask_a_pred[slice_idx]
        else:
            ped_features, obs_features, self_features, labels, self_hist_features = data[:]
            mask_a_pred = data.mask_a_pred

        filter_idx = mask_a_pred.reshape(-1)
        # move one step forward to calibrate with pairwise training
        labels[:-1, :, :] = labels[1:, :, :].clone()
        labels[-1, :, :] = 0
        labels = labels.reshape(filter_idx.shape[0], -1)
        self.labels = labels[filter_idx > 0]
        self.dataset_len = self.labels.shape[0]

        ped_features = ped_features.reshape(*[[-1] + list(ped_features.shape[2:])])
        self.ped_features = ped_features[filter_idx > 0]
        self.ped_feature_dim = self.ped_features.shape[-1]

        self_features = self_features.reshape(*[[-1] + list(self_features.shape[2:])])
        self.self_features = self_features[filter_idx > 0]
        self.self_feature_dim = self.self_features.shape[-1]
        
        self_hist_features = self_hist_features.reshape(*[[-1]+list(self_hist_features.shape[2:])])
        self.self_hist_features = self_hist_features[filter_idx > 0]
        self.self_hist_feature_dim = self.self_hist_features.shape[-1]

        if obs_features.shape[-1]:
            obs_features = obs_features.reshape(*[[-1] + list(obs_features.shape[2:])])
            self.obs_features = obs_features[filter_idx > 0]
            self.obs_feature_dim = self.obs_features.shape[-1]
        else:
            self.obs_features = torch.zeros((self.ped_features.shape[0], data.topk_obs, self.ped_feature_dim), device=ped_features.device)

        self.time_unit = data.time_unit

    def to(self, device):
        for u, v in self.__dict__.items():
            if type(v) == torch.Tensor:
                exec('self.' + u + '=' + 'self.' + u + '.to(device)')


class ChanneledTimeIndexedPedData(Dataset):
    """
    Attributes:
        ped_features: *c, t, N, k1, dim(6)
        obs_features: *c, t, N, k2, dim(6)
        self_features: *c, t, N, dim(6): dest_features, hist_velocity, cur_acc, desired_speed
        labels: *c, t, N, 2
    """
    def __init__(self):
        super(ChanneledTimeIndexedPedData, self).__init__()
        self.num_frames = 0

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):

        if self.num_frames > 0:
            item = [self.ped_features[index, ...], self.obs_features[index, ...],
                    self.self_features[index, ...], self.labels[index, ...]]
        else:
            raise ValueError("Haven't load any data yet!")

        return item

    def transform(self, matrix: torch.tensor, stride, mode='slice'):
        """
        t, ... --> *c, stride, ...
        """
        num_frames = matrix.shape[0]
        if mode == 'slice':
            dim = matrix.dim()
            matrix = matrix.unsqueeze(0).repeat(stride, *([1]*(dim)))
            for i in range(1, stride):
                matrix[i, :num_frames-i, ...] = matrix[i, i:, ...].clone()  # t, *c, ...
            matrix = matrix[:, :num_frames-stride, ...]
            permute_idx = list(range(matrix.dim()))
            permute_idx[0], permute_idx[1] = permute_idx[1], permute_idx[0]
            matrix = matrix.permute(*permute_idx)
        elif mode == 'split':
            step = int(num_frames / stride)
            matrix = matrix[:stride*step, ...]
            matrix = matrix.reshape(step, stride, *matrix.shape[1:])
        else:
            raise NotImplementedError
        return matrix

    def load_from_time_indexed_peddata(self, data: TimeIndexedPedData, stride=25, mode='slice'):
        """
        mode:
            'slice' - 
            'split' - 分割

        """
        assert (data.num_frames > stride), "ValueError: stride < #total time steps"
        length = data.num_frames-10
        sample=int(length-length*0.10)
        # sample = 0
        self.ped_features = self.transform(data.ped_features, stride, mode)[sample:]
        self.obs_features = self.transform(data.obs_features, stride, mode)[sample:]
        self.self_features = self.transform(data.self_features, stride, mode)[sample:]
        self.self_hist_features = self.transform(data.self_hist_features, stride, mode)[sample:]
        self.near_ped_idx = self.transform(data.near_ped_idx, stride, mode)[sample:]
        self.neigh_ped_mask = self.transform(data.neigh_ped_mask, stride, mode)[sample:]
        self.near_obstacle_idx = self.transform(data.near_obstacle_idx, stride, mode)[sample:]
        self.neigh_obs_mask = self.transform(data.neigh_obs_mask, stride, mode)[sample:]
        self.labels = self.transform(data.labels, stride, mode)[sample:]
        self.mask_p = self.transform(data.mask_p, stride, mode)[sample:]
        self.mask_v = self.transform(data.mask_v, stride, mode)[sample:]
        self.mask_a = self.transform(data.mask_a, stride, mode)[sample:]
        self.mask_a_pred = self.transform(data.mask_a_pred, stride, mode)[sample:]
        self.mask_v_pred = self.transform(data.mask_v_pred, stride, mode)[sample:]
        self.mask_p_pred = self.transform(data.mask_p_pred, stride, mode)[sample:]
        self.position = self.transform(data.position, stride, mode)[sample:]  # *c, t, n, 2
        self.velocity = self.transform(data.velocity, stride, mode)[sample:]
        self.acceleration = self.transform(data.acceleration, stride, mode)[sample:]
        self.destination = self.transform(data.destination, stride, mode)[sample:]
        self.dest_idx = self.transform(data.dest_idx, stride, mode)[sample:]

        self.waypoints = data.waypoints.unsqueeze(0).repeat(self.position.shape[0], *[1]*3)  # *c, d, n, 2

        self.num_frames = stride
        self.dataset_len = self.ped_features.shape[0]

        self.set_static_info_like(data)

    @staticmethod
    def slice(data, slice_idx):
        out = ChanneledTimeIndexedPedData()
        out.ped_features = data.ped_features[slice_idx, ...]
        out.obs_features = data.obs_features[slice_idx, ...]
        out.self_features = data.self_features[slice_idx, ...]
        out.self_hist_features =  data.self_hist_features[slice_idx, ...]
        out.near_ped_idx = data.near_ped_idx[slice_idx, ...]
        out.neigh_ped_mask = data.neigh_ped_mask[slice_idx, ...]
        out.near_obstacle_idx = data.near_obstacle_idx[slice_idx, ...]
        out.neigh_obs_mask = data.neigh_obs_mask[slice_idx, ...]
        out.labels = data.labels[slice_idx, ...]
        out.mask_p = data.mask_p[slice_idx, ...]
        out.mask_v = data.mask_v[slice_idx, ...]
        out.mask_a = data.mask_a[slice_idx, ...]
        out.mask_a_pred = data.mask_a_pred[slice_idx, ...]
        out.mask_v_pred = data.mask_v_pred[slice_idx, ...]
        out.mask_p_pred = data.mask_p_pred[slice_idx, ...]
        out.position = data.position[slice_idx, ...]
        out.velocity = data.velocity[slice_idx, ...]
        out.acceleration = data.acceleration[slice_idx, ...]
        out.destination = data.destination[slice_idx, ...]
        out.dest_idx = data.dest_idx[slice_idx, ...]
        out.waypoints = data.waypoints[slice_idx, ...]
        out.num_frames = data.num_frames
        out.dataset_len = data.dataset_len
        out.set_static_info_like(data)

        return out

    def set_static_info_like(self, data):
        self.obstacles = data.obstacles
        self.dest_num = data.dest_num  # n
        self.topk_obs = data.topk_obs
        self.meta_data = data.meta_data
        self.time_unit = data.time_unit
        self.num_pedestrians = data.num_pedestrians
        self.ped_feature_dim = data.ped_feature_dim
        self.obs_feature_dim = data.obs_feature_dim
        self.self_feature_dim = data.self_feature_dim
        self.abnormal_mask = data.abnormal_mask


class SocialForceData(RawData):
    '''
        Dataset for models.socialforce.simulator use

    Attributes:
        - num_steps: int. The number of frames
        - num_pedestrians: int. The number of pedestrians
        - position: (t, N, 2).
        - velocity: (t, N, 2).
        - acceleration: (t, N, 2).
        - destination: (D, N, 2).
        - arrived: (t, N).
        - tau: (N). Reflects the magnitude of the force exerted by the desire to reach the destination, i.e. F = (desired_speed * unit_vector(destination - position) - velocity) / tau.
        - desired_speed: (N).
        - obstacles: (M, 2).
        - avaliable: (t, N). Set to 0 before the pedestrian appears or after the pedestrian leaves.
    '''

    def __init__(self,
                 position=torch.tensor([]),
                 velocity=torch.tensor([]),
                 acceleration=torch.tensor([]),
                 destination=torch.tensor([]),
                 obstacles=torch.tensor([]),
                 mask_p=torch.tensor([]),
                 mask_v=torch.tensor([]),
                 mask_a=torch.tensor([]),
                 desired_speed=torch.tensor([]),
                 meta_data=dict(),
                 default_tau=0.5,
                 use_mask_preq=False,
                 dynamic_desired_speed=False):
        """
        Input:
            ...(same as super())
            "desired_speed": (N),
            "mask_preq": (t, N),
            "duration": (N),
        """
        super().__init__(position=position, velocity=velocity,
                         acceleration=acceleration, waypoints=destination,
                         obstacles=obstacles, mask_p=mask_p, mask_v=mask_v,
                         mask_a=mask_a, meta_data=meta_data)
        if (not self.velocity.numel()):
            self.velocity = torch.zeros_like(self.position)
        if (not self.acceleration.numel()):
            self.acceleration = torch.zeros_like(self.position)
        if (not self.mask_p.numel()):
            self.mask_p = torch.ones((self.num_steps, self.num_pedestrians))
        if (not self.mask_v.numel()):
            self.mask_v = torch.ones((self.num_steps, self.num_pedestrians))
        if (not self.mask_a.numel()):
            self.mask_a = torch.ones((self.num_steps, self.num_pedestrians))

        self.destination = torch.zeros((self.num_steps, self.num_pedestrians, 2), device=self.position.device) + torch.tensor(float('nan'), device=self.position.device)
        self.destination[self.num_steps - 1, :, :] = self.waypoints[0, :, :]
        self.destination_flag = torch.zeros(self.num_pedestrians, dtype=int, device=self.position.device)
        self.dynamic_desired_speed = dynamic_desired_speed
        self.desired_speed = desired_speed
        self.default_tau = default_tau
        self.tau = default_tau * torch.ones(self.num_pedestrians, device=self.position.device)
        if use_mask_preq:
            self.mask_preq = torch.ones((self.num_steps, self.num_pedestrians), device=self.position.device)
        else:
            self.mask_preq = None
        self.duration = torch.zeros(self.num_pedestrians, device=self.position.device)

    def get_frame(self, f: int) -> dict:
        """Get f-th frame in dict format."""
        frame = super().get_frame(f)
        frame["desired_speed"] = self.desired_speed
        frame["tau"] = self.tau
        if self.mask_preq is not None:
            frame["mask_preq"] = self.mask_preq[f, :]
        frame["duration"] = self.duration

        return frame

    def get_current_frame(self):
        """Get last frame in dict format."""
        return self.get_frame(self.num_steps - 1)

    def add_frame(self, frame: dict):
        """Add a frame to dataset."""
        super().add_frame(frame)
        if (self.mask_preq != None):
            self.mask_preq = torch.cat((self.mask_preq, frame['mask_preq'].unsqueeze(0)), dim=0)
        if (self.dynamic_desired_speed):
            self.desired_speed = frame['desired_speed']
        self.duration[self.mask_p[-1, :] == 1] += 1

    def add_pedestrians(self, add_num_pedestrians, add_desired_speed, add_tau=torch.tensor([]), **kwargs):
        super().add_pedestrians(add_num_pedestrians, **kwargs)
        self.desired_speed = torch.cat((self.desired_speed, add_desired_speed), dim=0)
        self.tau = torch.cat((self.tau, self.default_tau * torch.ones(add_num_pedestrians, device=self.position.device)), dim=0)
        if (self.mask_preq != None):
            self.mask_preq = torch.cat((self.mask_preq, torch.zeros((self.num_steps - 1, add_num_pedestrians), device=self.position.device)), dim=1)
        if (add_tau.numel() > 0):
            self.tau[-add_num_pedestrians:] = add_tau
        self.duration = torch.cat((self.duration, torch.zeros(add_num_pedestrians, device=self.position.device)), dim=0)

    def save_data(self, data_path: str):
        super().save_data(data_path)

    def to(self, device):
        for u, v in self.__dict__.items():
            if type(v) == torch.Tensor:
                exec('self.' + u + '=' + 'self.' + u + '.to(device)')
