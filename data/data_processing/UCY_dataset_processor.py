import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../../')
from data.dataset import RawData
from utils.visualization import state_animation
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='UCY dataset processor')
    parser.add_argument('-i', '--input', type=str, default='./cache/UCY/',
                        help='input file path')
    parser.add_argument('-o', '--output', type=str, default='./cache/',
                        help='output file path')
    parser.add_argument('-d', '--duration', type=float, default='20',
                        help='length of time snippet to save')
    parser.add_argument('-t', '--time', type=float, default='0',
                        help='begining of time snippet to save. Recommendation parameter: 760, 1000, 1100, 1280, 1560')
    # parser.add_argument('-r', '--range', action='store_true',
    #                     help='whether to limit range to [[5, 15], [25, 35]]')
    parser.add_argument('-v', '--visulization', action='store_true',
                        help='whether to generate visulization animation')
    args, unknown = parser.parse_known_args() 
    return args

if __name__ == '__main__':
    args = get_args()
    time_range = (int(args.time), int(args.time + args.duration))
    frame_range = [time_range[0] * 25, time_range[1] * 25]
    time_unit = 1.0/12.5
    savename = args.output + f"UCY_Dataset_time{time_range[0]}-{time_range[1]}_timeunit{time_unit:.2f}"

    meta_data = {
        "time_unit": time_unit, 
        "version": "v2.2",
        "begin_time": time_range[0],
        "source": "UCY dataset"
    }

    # Get perspective transform matrix, to transform picture coordinate to world coordinatie
    # length = 13
    # width = 12.6
    # post1 = np.float32([[166, 115], [561, 130], [132, 440], [602, 445]]) - np.float32([[360, 288]])
    # post2 = np.float32([[0, 0],[width, 0], [0, length], [width, length]])
    # import cv2
    # M = cv2.getPerspectiveTransform(post1, post2)
    # print(M)
    M = np.array([[ 2.84217540e-02,  2.97335273e-03,  6.02821031e+00],
                [-1.67162992e-03,  4.40195878e-02,  7.29109248e+00],
                [-9.83343172e-05,  5.42377797e-04,  1.00000000e+00]])

    # Uncomment to show the transformed scene as a picture
    # import cv2
    # image = cv2.imread('./cache/UCY/students_003.jpg')
    # post1 = np.float32([[166, 115], [561, 130], [132, 440], [602, 445]])
    # post2 = np.float32([[0, length],[width, length], [0, 0], [width, 0]])
    # M = cv2.getPerspectiveTransform(post1, post2 * 30)
    # cv2.imshow("image", cv2.warpPerspective(image, M, (width*30,length*30)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    trajectories = []
    print('Processing...')
    with open(args.input + "data_university_students/students003.vsp") as f:
        num_pedestrians = int(f.readline().split(' ')[0])
        trajs = []
        for i in tqdm(range(num_pedestrians)):
            S = int(f.readline().split(' ')[0])
            traj = np.zeros([S, 3])
            for j in range(S):
                traj[j, :] = np.array(f.readline().split(' ')[0:3], dtype=float)

            # Coordinate transformation
            image_coordination = np.concatenate((traj[:, 0:2], np.ones((traj.shape[0], 1))), axis=1)
            world_coordination = np.einsum('ij,nj->ni', M, image_coordination)
            traj[:, 0] = world_coordination[:, 0] / world_coordination[:, 2]
            traj[:, 1] = world_coordination[:, 1] / world_coordination[:, 2]

            # Interpolate
            begin_frame, end_frame = int(traj[0, 2]), int(traj[-1, 2])
            sample_frame = np.arange(begin_frame, end_frame + 1, time_unit * 25)
            traj_ = np.zeros([len(sample_frame), 3])
            traj_[:, 2] = sample_frame
            try:
                traj_[:, 0] = interp1d(traj[:, 2], traj[:, 0], kind='cubic')(traj_[:, 2])
                traj_[:, 1] = interp1d(traj[:, 2], traj[:, 1], kind='cubic')(traj_[:, 2])
            except (ValueError): # traj_.shape[0] is too less to do high order interpolate
                traj_[:, 0] = np.interp(traj_[:, 2], traj[:, 2], traj[:, 0])
                traj_[:, 1] = np.interp(traj_[:, 2], traj[:, 2], traj[:, 1])    
            traj = [(x,y,int(f / time_unit / 25)) for x,y,f in traj_ if ((f >= frame_range[0]) and (f <= frame_range[1]))]
            if(traj):
                trajectories.append(traj)

    destination = []
    for traj in trajectories:
        destination.append([(traj[-1][0], traj[-1][1], traj[-1][2])])

    data = np.array((meta_data, trajectories, destination, []), dtype=object)
    np.save(savename + ".npy", data)
    print(f'Saved processed data to {savename + ".npy"}\n')

    saved_data = RawData()
    saved_data.load_trajectory_data(savename + ".npy")

    if(args.visulization):
        fig = plt.figure(figsize=(5, 5))
        ax = plt.subplot()
        ax.grid(linestyle='dotted')
        ax.set_aspect(1.0, 'datalim')
        ax.set_axisbelow(True)
        ax.set_xlim(-5, 20)
        ax.set_ylim(-5, 16.5)
        video = state_animation(ax, saved_data, show_speed=False, movie_file=savename+".gif")
