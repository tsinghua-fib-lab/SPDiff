import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../../')
from data.dataset import RawData
from utils.visualization import state_animation
from utils.data_process import trajectories_split
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='GC dataset processor')
    parser.add_argument('-i', '--input', type=str, default='./cache/Annotation/',
                        help='output file path')
    parser.add_argument('-o', '--output', type=str, default='./cache/',
                        help='output file path')
    parser.add_argument('-p', '--ped', type=int, default='12684',
                        help='number of pedestrians to handle')
    parser.add_argument('-d', '--duration', type=float, default='60',
                        help='length of time snippet to save')
    parser.add_argument('-t', '--time', type=float, default='760',
                        help='begining of time snippet to save. Recommendation parameter: 760, 1000, 1100, 1280, 1560')
    parser.add_argument('-r', '--range', action='store_true',
                        help='whether to limit range to [[5, 15], [25, 35]]')
    parser.add_argument('-v', '--visulization', action='store_true',
                        help='whether to generate visulization animation')
    args, unknown = parser.parse_known_args() 
    return args

if __name__ == '__main__':
    args = get_args()
    # Handle xxx.txt, ..., yyy.txt, range in [1, 12685]
    ped_range = (1, args.ped + 1) 
    # Remove data point not in [xxx, yyy) (second), range in [0, 4800]
    time_range = (int(args.time), int(args.time + args.duration)) 
    space_range = [[5, 15], [25, 35]] if args.range else [[0, 0], [30, 35]]
    interpolation = 9 # Linear interpolate to make video more smoothly.
    interpolation_mode = 'cubic' # 'linear', 'slinear', 'quadratic', 'cubic'
    savename = args.output + f"GC_Dataset_ped{ped_range[0]}-{ped_range[1]}_time{time_range[0]}-{time_range[1]}_interp{interpolation}_xrange{space_range[0][0]}-{space_range[1][0]}_yrange{space_range[0][1]}-{space_range[1][1]}"

    # Assume raw video is 25fps
    time_unit = 20 / 25 / (interpolation + 1)
    meta_data = {
        "time_unit": time_unit, 
        "version": "v2.2",
        "begin_frame": time_range[0] * 25,
        "interpolation": interpolation,
        "source": "GC dataset"
    }
    frame_range = (int(time_range[0] / time_unit), int(time_range[1] / time_unit))

    # Get perspective transform matrix, to transform picture coordinate to world coordinatie
    length = 39
    width = 30
    # post1 = np.float32([[456, 118], [1441, 120],[64, 919],[1914, 939] ])
    # post2 = np.float32([[0, length],[width, length], [0, 0], [width, 0]])
    # import cv2
    # M = cv2.getPerspectiveTransform(post1, post2)
    M = np.array([[3.54477751e-02,  1.73477252e-02, -1.82112170e+01],
        [6.03523702e-04, -5.58259424e-02,  5.12654156e+01],
        [1.00205219e-05,  1.25487966e-03,  1.00000000e+00]])

    # Uncomment to show the transformed scene as a picture
    # import cv2
    # image = cv2.imread(filepath+'../Frame/000000.jpg')
    # post1 = np.float32([[456, 118], [1441, 120],[64, 919],[1914, 939] ])
    # post2 = np.float32([[0, length],[width, length], [0, 0], [width, 0]])
    # M = cv2.getPerspectiveTransform(post1, post2 * 20)
    # cv2.imshow("image", cv2.warpPerspective(image, M, (width*20,length*20)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    trajectories = []
    print('Processing...')
    for i in tqdm(range(ped_range[0], ped_range[1])):
        with open(args.input + str(i).zfill(6) + ".txt", "r") as f:
            traj = [int(x) for x in f.read().split('\n') if x]
            traj = np.array(traj, dtype=float).reshape(-1, 3)
            traj[:, 2] = traj[:, 2] / 20 * (interpolation + 1)

            # Coordinate transformation
            image_coordination = np.concatenate((traj[:, 0:2], \
                np.ones((traj.shape[0], 1))), axis=1)
            world_coordination = np.einsum('ij,nj->ni', M, image_coordination)
            traj[:, 0] = world_coordination[:, 0] / world_coordination[:, 2]
            traj[:, 1] = world_coordination[:, 1] / world_coordination[:, 2]
            
            # Interpolate
            traj_ = np.zeros([int(traj[-1, 2] - traj[0, 2] + 1), 3])
            traj_[:, 2] = np.arange(traj[0, 2], traj[-1, 2] + 1)
            try:
                traj_[:, 0] = interp1d(traj[:, 2], traj[:, 0], kind=interpolation_mode)(traj_[:, 2])
                traj_[:, 1] = interp1d(traj[:, 2], traj[:, 1], kind=interpolation_mode)(traj_[:, 2])
            except (ValueError): # traj_.shape[0] is too less to do high order interpolate
                traj_[:, 0] = np.interp(traj_[:, 2], traj[:, 2], traj[:, 0])
                traj_[:, 1] = np.interp(traj_[:, 2], traj[:, 2], traj[:, 1])
            traj = traj_
            
            # Remove frames not in frame_range
            traj = traj[(traj[:, 2] >= frame_range[0]) * (traj[:, 2] < frame_range[1]), :]
            if(len(traj) == 0): continue

            # Remove data points not in space_range
            traj = traj[(traj[:, 0] >= space_range[0][0]) * (traj[:, 0] <= space_range[1][0]) * (traj[:, 1] >= space_range[0][1]) * (traj[:, 1] <= space_range[1][1]), :]
            if(len(traj) == 0): continue

            trajectories.append([(x,y,int(f) - frame_range[0]) for x,y,f in traj])
                
    trajectories = trajectories_split(trajectories)

    destination = []
    for traj in trajectories:
        destination.append([(traj[-1][0], traj[-1][1], traj[-1][2])])


    #** 2m********** y** 13** 15。
    R = 0.14667 * width / 2
    theta = np.linspace(0, 2*np.pi, 100)
    obstacles = np.stack((R * np.cos(theta) + 0.45333*width, R * np.sin(theta) + 0.28974*length), axis=1)

    data = np.array((meta_data, trajectories, destination, obstacles), \
        dtype=object)
    np.save(savename + ".npy", data)

    saved_data = RawData()
    saved_data.load_trajectory_data(savename + ".npy")

    if(args.visulization):
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