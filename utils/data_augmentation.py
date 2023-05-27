import torch
import math
import copy

import sys
sys.path.append('..')

from data.data import RawData


def rotate_augmentation(data: RawData, theta: float) -> RawData:
    """
    Rotate point in 'data' to augment data.
    Input:
        - data: Raw data to augment.
        - theta: Angle to rotate, in degrees unit(0~360), anti-clockwise direction.
    Return:
        - augmentation_data: Data after rotated.
    """
    print('Data Augmentation - Rotation: Angle {}'.format(theta))

    theta = theta / 180 * math.pi
    rotate_mat = torch.tensor([[math.cos(theta), -math.sin(theta)],
                               [math.sin(theta), math.cos(theta)]])
    new_data = copy.deepcopy(data)
    print(new_data.position.shape)
    new_data.position = torch.einsum('ij,tnj->tni', rotate_mat,
                                     new_data.position)
    new_data.velocity = torch.einsum('ij,tnj->tni', rotate_mat,
                                     new_data.velocity)
    new_data.acceleration = torch.einsum('ij,tnj->tni', rotate_mat,
                                         new_data.acceleration)
    new_data.destination = torch.einsum('ij,tnj->tni', rotate_mat,
                                        new_data.destination)
    new_data.waypoints = torch.einsum('ij,dnj->dni', rotate_mat,
                                      new_data.waypoints)
    new_data.obstacles = torch.einsum('ij,nj->ni', rotate_mat,
                                      new_data.obstacles) if new_data.obstacles.numel() else new_data.obstacles
    return new_data


def mirror_augmentation(data: RawData, theta: float) -> RawData:
    """
    Mirroring point in 'data' to augment data.
    Input:
        - data: Raw data to augment.
        - theta: Mirror angle, in degrees unit(0~180), anti-clockwise direction, e.g. 0 means up<->down mirroring and 90 means left<->right mirroring
    Return:
        - augmentation_data: Data after mirroring.
    """
    print('Data Augmentation - Mirror: Angle {}'.format(theta))

    theta = theta / 180 * math.pi
    mirror_mat = torch.tensor([[math.cos(2 * theta), math.sin(2 * theta)],
                               [math.sin(2 * theta), -math.cos(2 * theta)]])
    new_data = copy.deepcopy(data)
    new_data.position = torch.einsum('ij,tnj->tni', mirror_mat,
                                     new_data.position)
    new_data.velocity = torch.einsum('ij,tnj->tni', mirror_mat,
                                     new_data.velocity)
    new_data.acceleration = torch.einsum('ij,tnj->tni', mirror_mat,
                                         new_data.acceleration)
    new_data.destination = torch.einsum('ij,tnj->tni', mirror_mat,
                                        new_data.destination)
    new_data.waypoints = torch.einsum('ij,dnj->dni', mirror_mat,
                                      new_data.waypoints)
    new_data.obstacles = torch.einsum('ij,nj->ni', mirror_mat,
                                      new_data.obstacles) if new_data.obstacles.numel() else new_data.obstacles
    return new_data
