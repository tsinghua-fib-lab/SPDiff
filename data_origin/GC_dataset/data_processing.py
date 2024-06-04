# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import tqdm

# path = './Annotation/'
# files = os.listdir(path)
# ped_traj = []
# for file in tqdm(files):
#     traj = []
#     with open(path + '/' + file) as f:
#         for line in f:
#             if len(line) > 1:
#                 traj.append(int(line))
#     traj = np.array(traj)
#     traj = traj.reshape((-1, 3))
#     traj = traj.tolist()
#     ped_traj.append(traj)
    
# np.save('./gc_dataset.npy', ped_traj)


trajs = np.load('./gc_dataset.npy', allow_pickle=True)
t_min_res = 1e10
t_max_res = 0
for traj in trajs:
    tem = [u[2] for u in traj]
    t_min = np.min(tem) * 0.05
    t_max = np.max(tem) * 0.05
    if t_min < t_min_res:
        t_min_res = t_min
    if t_max > t_max_res:
        t_max_res = t_max
print(t_min_res)
print(t_max_res)