import sys
# from utils.visualization import *
from data import *
import matplotlib.pyplot as plt
path = 'PIML-main/data/GC_Dataset/GC_Dataset_ped1-12685_time760-820_interp9_xrange5-25_yrange15-35.npy'
space_range = [[5, 15], [25, 35]]
saved_data = RawData()
saved_data.load_trajectory_data(path)
fig = plt.figure(figsize=(5, 5))
ax = plt.subplot()
ax.grid(linestyle='dotted')
ax.set_aspect(1.0, 'datalim')
ax.set_axisbelow(True)
# ax.set_xlabel('$x_1$ [m]')
# ax.set_ylabel('$x_2$ [m]')
ax.set_xlim(space_range[0][0], space_range[1][0])
ax.set_ylim(space_range[0][1], space_range[1][1])
video = state_animation(ax, saved_data, show_speed=False, movie_file=savename+".gif")