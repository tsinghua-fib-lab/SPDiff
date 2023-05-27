"""Utility functions for plots and animations."""

import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from data.dataset import RawData
import torch

def init_animation(ax, data: RawData, circle: dict={}, number: dict={}) -> dict:
    """Init entities needed in animation and return as a dict.
        - "title": Title of the figure
        - {ped_id}: The entities corresponding to pedestrian {ped_id}
    """
    actors = {}
    for ped_id in range(data.num_pedestrians):
        actors[ped_id] = {
            "circle": plt.Circle((0, 0), **circle, visible=False),
            "number": ax.text(0, 0, str(ped_id), **number, size="xx-small", visible=False, verticalalignment="center", horizontalalignment="center", color=(0, 0, 0, 1)),
            "legend": ax.text(0.1, 0.9 - 0.08 * ped_id, '', transform=ax.transAxes, visible=False),
            "route": (plt.plot([], [], ls='-', marker='.', color=(.5, .5, .5, .1), visible=False))[0],
        }
        ax.add_patch(actors[ped_id]["circle"])

    actors["title"] = plt.title('')
    if(data.obstacles.numel()):
        plt.plot(data.obstacles[:, 0], data.obstacles[:, 1], "-k")
    
    return actors

def update_animation(frame_num: int, data: RawData, actors: dict, show_speed=False,
    color=None) -> list:
    frame = data.get_frame(frame_num)
    actors_list = []
    for ped in range(frame["num_pedestrians"]):
        if(frame["mask_p"][ped] == 0):
            actors[ped]["circle"].set_visible(False)
            actors[ped]["number"].set_visible(False)
            actors[ped]["route"].set_visible(False)
            if(show_speed):
                actors[ped]["legend"].set_visible(False)
            continue
        speed = np.linalg.norm(frame["velocity"][ped, :])
        acc = np.linalg.norm(frame["acceleration"][ped, :])
        radius = 0.38 / 2
        if(color):
            color_ = color(frame)
        else:
            color_ = (0, 1.34 / (1.34 + speed), speed / (1.34 + speed), 0.4)
        actors[ped]["number"].set(position=frame["position"][ped, :], visible=True)
        actors[ped]["circle"].set(center=frame["position"][ped, :], radius=radius, color=color_, visible=True)
        route = torch.concat((frame['position'][(ped,), :], frame['destinations'][frame['destination_flag'][ped]:, ped, :]), dim=0)
        actors[ped]["route"].set(data=(route[:, 0], route[:, 1]), visible=True)
        if(show_speed):
            actors[ped]["legend"].set(text=f'$v_{{{ped}}} = {speed:.2f}m/s, a_{{{ped}}} = {acc:.2f}m/s^2$', visible=True)

        actors_list.append(actors[ped]["circle"])
        actors_list.append(actors[ped]["number"])
        actors_list.append(actors[ped]["route"])
        if(show_speed):
            actors_list.append(actors[ped]["legend"])

    if("source" in data.meta_data and data.meta_data["source"] == "GC dataset"):
        begin_frame = data.meta_data["begin_frame"]
        interpolation = data.meta_data["interpolation"]
        title_text = f'[GC Dataset]: Frame {int(frame_num//interpolation) + begin_frame} / {frame_num*data.meta_data["time_unit"]:.2f}s'
    elif("source" in data.meta_data and data.meta_data["source"] == "basic unit"):
        title_text = f'[Basic Unit {data.meta_data["scene"]}]: Frame {frame_num} / {frame_num*data.meta_data["time_unit"]:.2f}s'
    else:
        title_text = f'Frame {frame_num} / {frame_num*data.meta_data["time_unit"]:.2f}s'
    actors["title"].set(text=title_text)
    actors_list.append(actors["title"])
    
    return actors_list

def state_animation(ax, data:RawData, *, movie_file=None, writer=None, show_speed=False):
    """Generate animation for {data}."""
    if(movie_file): print(f"Saving animation to '{movie_file}'...")
    actors = init_animation(ax, data)

    def update(i):
        progress = round(i / data.num_steps * 100)
        print("\r", end="")
        print("Animation progress: {}%: ".format(progress), end="")
        sys.stdout.flush()
        return update_animation(i, data, actors, show_speed)

    ani = animation.FuncAnimation(
        ax.get_figure(), update,
        frames=data.num_steps,
        interval=data.meta_data["time_unit"] * 1000.0, blit=True)
    if movie_file:
        ani.save(movie_file, writer=writer, dpi=200)
    return ani


def state_animation_compare(ax, data1:RawData, data2:RawData, *, movie_file=None, writer=None, show_speed=False):
    """Generate animation to compare {data1} and {data2}.
        - data1: Data to compare, draw in colorful disks.
        - data2: Data as base, draw in black and white circle.

        Note: data1 and data2 should have same time unit.
    """
    if(movie_file): print(f"Saving compare animation to '{movie_file}'...")
    actors1 = init_animation(ax, data1, circle={"zorder":9}, number={"zorder":10})
    actors2 = init_animation(ax, data2, circle={"zorder":7}, number={"zorder":8, "alpha":0.2})

    def update(i):
        progress = round(i / data2.num_steps * 100)
        print("\r", end="")
        print("Animation progress: {}%: ".format(progress), end="")
        sys.stdout.flush()
        return update_animation(i, data1, actors1, show_speed) \
        + update_animation(i, data2, actors2, show_speed, color=lambda x: (0.2, 0.2, 0.2, 0.2))

    ani = animation.FuncAnimation(
        ax.get_figure(), update,
        frames=data2.num_steps,
        interval=data2.meta_data["time_unit"] * 1000.0, blit=True)
    if movie_file:
        ani.save(movie_file, writer=writer, dpi=200)
    return ani


def plot_trajectory(p_pred, p_label, name):
    # 首先初始化图像
    p_pred = p_pred.cpu()
    p_label = p_label.cpu()
    plt.figure()
    # 对于 p_pred, p_label 进行遍历, 绘制轨迹
    for t in range(p_pred.shape[0]):
        for n in range(p_pred.shape[1]):
            if not np.isnan(p_pred[t, n, 0]):
                if t>=25 and not np.isnan(p_pred[t-25, n, 0]):
                    plt.scatter(p_pred[t, n, 0], p_pred[t, n, 1], c='r', marker='o')
                else:
                    plt.scatter(p_pred[t, n, 0], p_pred[t, n, 1], c='r', marker='x')
                if t>0 and not np.isnan(p_pred[t-1, n, 0]):
                    plt.plot(p_pred[t-1:t+1, n, 0], p_pred[t-1:t+1, n, 1], c='r', linestyle='--')
            if not np.isnan(p_label[t, n, 0]):
                if t>=25 and not np.isnan(p_label[t-25, n, 0]):
                    plt.scatter(p_label[t, n, 0], p_label[t, n, 1], c='b', marker='o')
                else:
                    plt.scatter(p_label[t, n, 0], p_label[t, n, 1], c='b', marker='x')
                if t>0 and not np.isnan(p_label[t-1, n, 0]):
                    plt.plot(p_label[t-1:t+1, n, 0], p_label[t-1:t+1, n, 1], c='b', linestyle='--')
    f = plt.gcf()  #获取当前图像
    f.savefig('visualize_data/'+name+'.jpg')
    f.clear()  #释放内存
