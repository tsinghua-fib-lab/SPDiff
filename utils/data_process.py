import torch

def trajectories_split(trajectories):
    split_trajectories = []
    for traj in trajectories:
        tensor_traj = torch.tensor(traj)
        if(torch.all(torch.diff(tensor_traj[:, 2]) == 1)):
            split_trajectories.append(traj)
        else:
            left = 0
            for right in range(1, tensor_traj.shape[0]):
                if(tensor_traj[right, 2] - tensor_traj[right - 1, 2] > 1):
                    split_trajectories.append(traj[left:right])
                    left = right
            split_trajectories.append(traj[left:right])

    return split_trajectories


