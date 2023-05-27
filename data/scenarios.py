from typing import Tuple
from data.data import RawData, SocialForceData
import torch
import math
import numpy as np
import random
from utils.utils import route

def crosswalk(length:float=20.0, width:float=7.0, 
    num_ped1:int=10, num_ped2:int=10, time_unit = 0.08,
    uniform_desired_speed=False):
    '''
    Crosswalk scenario:
        num_ped1/2 pedestrians move from left/right to right/left, cross a 
        crosswalk {length} long and {width} wide, and randomly turn left/right 
        after crossing the crosswalk.

        Pedestrians' initial position is uniform distribution in 
            {length}/2 <= abs(x) <= {length}/2+3m, 
            -{width}/2 <= y <= +{width}/2, 
        and the initial velocity and acceleration of every pedestrian are 
        all zero.
        
    Input:
        - length: Length of the crosswalk.
        - width: Width of the crosswalk.
        - num_ped1: The number of pedestrians move from left to right.
        - num_ped2: The number of pedestrians move from right to left.

    Return:
        (data, update_function)
        - data: The data contains only the first frame.
        - update_destination: The function to update each pedestrian's         
            destination, used in models.socialforce.simulators
    '''

    def generate(num_ped):
        side_x = 2 * torch.randint(0, 2, (num_ped,)) - 1
        side_y = 2 * torch.randint(0, 2, (num_ped,)) - 1
        pos_x = side_x * (length / 2 + 3 * torch.rand([num_ped]))
        pos_y = width / 2 * side_y
        pos = torch.stack((pos_x, pos_y), dim=1)
        
        desired_spd = 1.34 * torch.ones([num_ped])
        if not uniform_desired_speed:
            desired_spd += 0.26**(1/2) * torch.randn([num_ped])
        
        vel_y = -side_y * desired_spd.clone()
        vel_x = torch.zeros_like(vel_y)
        vel = torch.stack((vel_x, vel_y), dim=1)

        acc = torch.zeros((num_ped, 2))
        
        des_x1 = -side_x * length / 2
        des_y1 = -width / 2 + width * torch.randint(0, 2, [num_ped])
        des_x2 = des_x1
        des_y2 = des_y1 * 3
        des = torch.stack((torch.stack((des_x1, des_y1), dim=1),
            torch.stack((des_x2, des_y2), dim=1)), dim=0)
        
        msk = torch.ones([num_ped])
        return pos, vel, acc, des, desired_spd, msk

    pos, vel, acc, des, desired_spd, msk = generate(num_ped1 + num_ped2)

    def update(frame):
        dis2des = torch.norm(frame['position'] - frame['destination'], dim=1)
        frame['destination_flag'][dis2des < 1] += 1
        
        add_num_ped = int(torch.poisson(torch.tensor(5 * 0.08)).item()) # 5 ped/s * 0.08 s/f
        if(add_num_ped > 0):
            pos, vel, acc, des, desired_spd, msk = generate(add_num_ped)
            frame['num_pedestrians'] += add_num_ped
            frame['add_position'] = pos
            frame['add_velocity'] = vel
            frame['add_destination'] = des
            frame['add_desired_speed'] = desired_spd
        return frame

    data = SocialForceData(position=pos.unsqueeze(0), 
                           destination=des, 
                           velocity=vel.unsqueeze(0), 
                           desired_speed=desired_spd,
                           meta_data={"time_unit":time_unit})
    return data, update

def four_directional_square(block_length:float=20.0, peds_density:int=5, 
    uniform_desired_speed=True):
    '''

    '''
    num_ped = 4 * peds_density**2
    grid = torch.arange(1 - peds_density, peds_density + 1, 2
        ) * block_length / 2 / peds_density
    grid_x, grid_y = torch.meshgrid([grid, grid], indexing=None)
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)
    pos_1 = torch.stack((grid_x - block_length, grid_y), dim=1) # N,2
    pos_2 = torch.stack((grid_x + block_length, grid_y), dim=1) # N,2
    pos_3 = torch.stack((grid_x, grid_y - block_length), dim=1) # N,2
    pos_4 = torch.stack((grid_x, grid_y + block_length), dim=1) # N,2
    pos = torch.cat((pos_1, pos_2, pos_3, pos_4), dim=0).unsqueeze(0) # f,N,2

    shuffle = torch.randperm(num_ped // 4)
    # shuffle = torch.arange(num_ped // 4)

    des_1 = torch.stack((
        grid_x[shuffle] + block_length, grid_y[shuffle]), dim=1) # N,2
    des_2 = torch.stack((
        grid_x[shuffle] - block_length, grid_y[shuffle]), dim=1) # N,2
    des_3 = torch.stack((
        grid_x[shuffle], grid_y[shuffle] + block_length), dim=1) # N,2
    des_4 = torch.stack((
        grid_x[shuffle], grid_y[shuffle] - block_length), dim=1) # N,2
    des = torch.cat((des_1, des_2, des_3, des_4), dim=0).unsqueeze(0) # f,N,2

    desired_spd = 1.34 * torch.ones(num_ped)
    desired_spd += 0 if uniform_desired_speed else 0.26**(1/2) * torch.randn(
        num_ped)

    obs_angle = torch.linspace(-torch.pi, +torch.pi, 128)
    obs_x = 5 * torch.cos(obs_angle)
    obs_y = 5 * torch.sin(obs_angle)
    obs = torch.stack((obs_x, obs_y), dim=1) # M,2

    def update_destination(frame):
        dis2des = torch.norm(frame['position'] - frame['destination'], dim=1)
        frame['destination_flag'][dis2des < 1] += 1
        return frame

    data = SocialForceData(position=pos, desired_speed=desired_spd, \
        destination=des, obstacles=obs, meta_data={'time_unit': 0.08})

    return data, update_destination


def basic_unit1(length:float=20.0, width:float=10.0, time_unit = 0.08, poisson_lambda:float=5,\
    during:float=40, uniform_desired_speed=True):
    '''

    '''
    def generate(num_ped):
        posx = torch.zeros(num_ped)
        posy = width * torch.rand(num_ped)
        pos = torch.stack((posx, posy), dim=1)
        desx = length * torch.ones(num_ped)
        desy = posy + (2 * torch.rand(num_ped) - 1)
        des = torch.stack((desx, desy), dim=1).unsqueeze(0)
        desired_spd = 1.14 * torch.ones(num_ped)
        if not uniform_desired_speed:
            desired_spd += 0.1**(1/2) * torch.randn(num_ped)
            desired_spd[desired_spd < 0.8] = 0.8
        velx = desired_spd.clone()
        vely = torch.zeros(num_ped)
        vel = torch.stack((velx, vely), dim=1)
        acc = torch.zeros_like(vel)
        msk = torch.ones([num_ped])
        return pos, vel, acc, des, desired_spd, msk
    pos, vel, acc, des, desired_spd, msk = generate(1)
    
    def update(frame):
        arrived = (frame['position'][:, 0] > length)
        frame['mask_p'][arrived] = 0

        num_ped = int(torch.poisson(torch.tensor(poisson_lambda * time_unit)).item())
        if(num_ped > 0):
            pos, vel, acc, des, desired_spd, msk = generate(num_ped)
            frame['num_pedestrians'] += num_ped
            frame['add_position'] = pos
            frame['add_velocity'] = vel
            frame['add_destination'] = des
            frame['add_desired_speed'] = desired_spd
        return frame

    data = SocialForceData(position=pos.unsqueeze(0), 
                        destination=des,
                        velocity=vel.unsqueeze(0), 
                        desired_speed=desired_spd,
                        meta_data={"time_unit":time_unit, "source":"basic unit", "scene": 1})

    return data, update

def basic_unit2(length:float=20.0, width:float=10.0, time_unit = 0.08, poisson_lambda:float=5,\
    side_ratio=0.3, direction_ratio=0.5, during:float=40, \
    uniform_desired_speed=True):
    '''
    side_ratio: The ratio of pedestrians whose initial position is in the left half side.
    direction_ratio: The ratio of pedestrians move from right to left.
    '''
    def generate(num_pedestrians):
        left_side = (torch.rand(num_pedestrians) < side_ratio)
        right_to_left = (torch.rand(num_pedestrians) < direction_ratio)

        posx = torch.zeros(num_pedestrians)
        posy = width/2 * torch.rand(num_pedestrians)
        posy[left_side] += width/2
        posx[right_to_left] = length
        posy[right_to_left] = width - posy[right_to_left]
        pos = torch.stack((posx, posy), dim=1)

        desx = length * torch.ones(num_pedestrians)
        desy = posy + (2 * torch.rand(num_pedestrians) - 1)
        desx[right_to_left] = 0
        des = torch.stack((desx, desy), dim=1).unsqueeze(0)

        desired_spd = 1.14 * torch.ones(num_pedestrians)
        if not uniform_desired_speed:
            desired_spd += 0.1**(1/2) * torch.randn(num_pedestrians)
            desired_spd[desired_spd < 0.8] = 0.8

        velx = desired_spd.clone()
        vely = torch.zeros(num_pedestrians)
        velx[right_to_left] = -velx[right_to_left]
        vel = torch.stack((velx, vely), dim=1)

        acc = torch.zeros_like(vel)
        msk = torch.ones([num_pedestrians])
        return pos, vel, acc, des, desired_spd, msk

    pos, vel, acc, des, desired_spd, msk = generate(1)

    def update(frame):
        dis2des = abs(frame['position'][:, 0] - frame['destination'][:, 0])
        frame['destination_flag'][dis2des < 0.05] += 1

        num_ped = int(torch.poisson(torch.tensor(poisson_lambda * time_unit)).item())
        if(num_ped > 0):
            pos, vel, acc, des, desired_spd, msk = generate(num_ped)
            frame['num_pedestrians'] += num_ped
            frame['add_position'] = pos
            frame['add_velocity'] = vel
            frame['add_destination'] = des
            frame['add_desired_speed'] = desired_spd
        return frame

    data = SocialForceData(position=pos.unsqueeze(0), 
                           destination=des,
                           velocity=vel.unsqueeze(0), 
                           desired_speed=desired_spd, 
                           meta_data={"time_unit":time_unit, "source":"basic unit", "scene": 2})

    return data, update

def basic_unit3(length:float=20.0, width:float=10.0, time_unit = 0.08, poisson_lambda:float=5,\
    poisson_lambda2:float=1, during:float=40, \
    uniform_desired_speed=True):
    '''

    '''
    def generate(num_pedestrians1, num_pedestrians2):
        posx = torch.zeros(num_pedestrians1)
        posy = width * torch.rand(num_pedestrians1)
        posx2 = length * torch.rand(num_pedestrians2)
        posy2 = torch.zeros(num_pedestrians2)
        pos = torch.stack((
            torch.cat((posx, posx2), dim=0), 
            torch.cat((posy, posy2), dim=0)
            ), dim=1)

        desx = length * torch.ones(num_pedestrians1)
        desy = posy + (2 * torch.rand(num_pedestrians1) - 1)
        desx2 = posx2 + (2 * torch.rand(num_pedestrians2) - 1)
        desy2 = width * torch.ones(num_pedestrians2)
        des = torch.stack((
            torch.cat((desx, desx2), dim=0), 
            torch.cat((desy, desy2), dim=0)
        ), dim=1).unsqueeze(0)

        desired_spd = 1.14 * torch.ones(num_pedestrians1 + num_pedestrians2)
        if not uniform_desired_speed:
            desired_spd += 0.1**(1/2) * torch.randn(num_pedestrians1 + num_pedestrians2)
            desired_spd[desired_spd < 0.8] = 0.8

        velx = desired_spd[:num_pedestrians1].clone()
        vely = torch.zeros(num_pedestrians1)
        velx2 = torch.zeros(num_pedestrians2)
        vely2 = desired_spd[num_pedestrians1:].clone()
        vel = torch.stack((
            torch.cat((velx, velx2), dim=0), 
            torch.cat((vely, vely2), dim=0),
        ), dim=1)

        acc = torch.zeros_like(vel)
        msk = torch.ones([num_pedestrians1 + num_pedestrians2]) # dN
        return pos, vel, acc, des, desired_spd, msk

    pos, vel, acc, des, desired_spd, msk = generate(1, 0)

    def update(frame):
        dis2des = torch.norm(frame['position'] - frame['destination'], dim=1)
        frame['destination_flag'][dis2des < 1] += 1

        num_ped1 = int(torch.poisson(torch.tensor(poisson_lambda * time_unit)).item())
        num_ped2 = int(torch.poisson(torch.tensor(poisson_lambda2 * time_unit)).item())
        if(num_ped1 + num_ped2 > 0):
            pos, vel, acc, des, desired_spd, msk = generate(num_ped1, num_ped2)
            frame['num_pedestrians'] += (num_ped1 + num_ped2)
            frame['add_position'] = pos
            frame['add_velocity'] = vel
            frame['add_destination'] = des
            frame['add_desired_speed'] = desired_spd
        return frame

    data = SocialForceData(position=pos.unsqueeze(0), 
                           destination=des, 
                           velocity=vel.unsqueeze(0), 
                           desired_speed=desired_spd,
                           meta_data={"time_unit":time_unit, "source":"basic unit", "scene": 3})

    return data, update


def GC(time_unit=0.08, uniform_desired_speed=False, device='cpu'):
    '''

    '''
    length = 35
    width = 30
    # R = 0.14667 * width / 2
    R = 2.75
    theta = torch.linspace(0, 2*np.pi, 100, device=device)
    # obstacles = torch.stack((R * torch.cos(theta) + 0.45333*width, R * torch.sin(theta) + 0.28974*length), axis=1)
    wall_node = torch.tensor([
        [0, 0], [0, 5.63], [-5, 5.63], [-5, 16.01], [0, 16.01], [0, 35],
        [0, 40], [5.93, 40], [5.93, 35], [21.43, 35], [21.43, 40], [30, 40], [30, 35],
        [35, 35], [35, 29.48], [30, 29.48], [30, 25.62], [35, 25.62], [35, 18.99], [30, 18.99], [30, 14.79], [35, 14.79], [35, 7.07], [30, 7.07], [30, 0],
        [30, -5], [0, -5], [0, 0]
    ], device=device)
    wall_length = torch.linalg.norm(torch.diff(wall_node, dim=0), dim=1)
    wall = []
    for node in range(wall_node.shape[0] - 1):
        x = torch.linspace(wall_node[node, 0], wall_node[node + 1, 0], int(wall_length[node] / 0.05))
        y = torch.linspace(wall_node[node, 1], wall_node[node + 1, 1], int(wall_length[node] / 0.05))
        wall.append(torch.stack((x, y), dim=1))

    obstacles = [
        torch.concat(wall, dim=0).to(device),
        torch.stack((R * torch.cos(theta) + 13.52, R * torch.sin(theta) + 10.71), dim=1)
    ]

    entry = [
        torch.stack((torch.zeros(100, device=device), torch.linspace(5.63+1, 16.01-1, 100, device=device)), dim=1), # Left
        torch.stack((torch.linspace(0+1, 5.93-1, 100, device=device), 35*torch.ones(100, device=device)), dim=1), # Top(1)
        torch.stack((torch.linspace(21.43+1, 30-1, 100, device=device), 35*torch.ones(100, device=device)), dim=1), # Top(2)
        torch.stack((30*torch.ones(100, device=device), torch.linspace(29.48+1, 35-1, 100, device=device)), dim=1), # Right(1)
        torch.stack((30 * torch.ones(100, device=device), torch.linspace(18.99+1, 25.62-1, 100, device=device)), dim=1), # Right(2)
        torch.stack((30 * torch.ones(100, device=device), torch.linspace(7.07+1, 14.79-1, 100, device=device)), dim=1), # Right(3)
        torch.stack((torch.linspace(0+1, 30-1, 100, device=device), torch.zeros(100, device=device)), dim=1), # Bottom
    ]

    def generate(num_ped):
        def get_od():
            o, d = random.sample(entry, 2)
            o = o[random.choice(range(o.shape[0])), :].reshape(1, 2) + torch.rand((1, 2), device=device) * 0.8
            d = d[random.choice(range(d.shape[0])), :].reshape(1, 2) + torch.rand((1, 2), device=device) * 0.8
            od = route(torch.concat((o, d), dim=-2), obstacles[1])
            return (od[0, :, :], od[1:, :, :]) # o: 1, 2; d: D=2, 1, 2

        od_list = [get_od() for _ in range(num_ped)]
        pos = torch.concat([o for o, d in od_list], dim=-2) # N, 2
        des = torch.concat([d for o, d in od_list], dim=-2) # D=2, N, 2

        desired_spd = 1.34 * torch.ones([num_ped])
        if not uniform_desired_speed:
            desired_spd += 0.26 ** (1 / 2) * torch.randn([num_ped])
            desired_spd[desired_spd < 0.7] = 0.7

        vel = torch.zeros_like(pos)
        acc = torch.zeros_like(pos)

        msk = torch.ones([num_ped])
        return pos, vel, acc, des, desired_spd, msk

    pos, vel, acc, des, desired_spd, msk = generate(20)

    def update(frame):
        # des2exit = torch.zeros((frame['num_pedestrian']), device=frame['position'].device)
        # for p in range(frame['num_pedestrian']):
        exit = torch.argmin(torch.stack([torch.min(torch.norm(frame['destination'].reshape(-1, 1, 2) - e.reshape(1, -1, 2), dim=-1), dim=1).values for e in entry], dim=1), dim=1) # N

        dis2exit = torch.concat([torch.min(torch.norm(frame['position'][(p,), :] - entry[int(exit[p])], dim=-1)).reshape(1) for p in range(frame['num_pedestrians'])], dim=0)
        dis2des = torch.norm(frame['position'] - frame['destination'], dim=1)
        frame['destination_flag'][(dis2des < 1) + (dis2exit < 1)] += 1

        add_num_ped = int(torch.poisson(torch.tensor(5 * 0.08)).item())  # 5 ped/s * 0.08 s/f
        if (add_num_ped > 0):
            pos, vel, acc, des, desired_spd, msk = generate(add_num_ped)
            frame['num_pedestrians'] += add_num_ped
            frame['add_position'] = pos.to(frame['position'].device)
            frame['add_velocity'] = vel.to(frame['position'].device)
            frame['add_destination'] = des.to(frame['position'].device)
            frame['add_desired_speed'] = desired_spd.to(frame['position'].device)
        return frame

    data = SocialForceData(position=pos.unsqueeze(0),
                           destination=des,
                           velocity=vel.unsqueeze(0),
                           desired_speed=desired_spd,
                           meta_data={"time_unit": time_unit})
    data.to(device)
    return data, update, obstacles
