import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../../')
from data.dataset import RawData

if __name__ == '__main__':
    filepath = '/data/zhangguozhen/dataset/pedsim/processed_data/UCY_Dataset/'
    savepath = './models/sgan/datasets/UCY/'    
    filenames = (
        [ # train
            # 'GC_Dataset_ped1-12685_time760-820_interp9_xrange5-25_yrange15-35', 
            # 'GC_Dataset_ped1-12685_time1000-1060_interp9_xrange5-25_yrange15-35', 
            # 'GC_Dataset_ped1-12685_time1100-1160_interp9_xrange5-25_yrange15-35', 
            # 'GC_Dataset_ped1-12685_time2104-2164_interp9_xrange5-25_yrange15-35', 
            # 'GC_Dataset_ped1-12685_time2164-2224_interp9_xrange5-25_yrange15-35', 
            # 'GC_Dataset_ped1-12685_time2224-2284_interp9_xrange5-25_yrange15-35', 
            'UCY_Dataset_time0-54_timeunit0.08',
            'UCY_Dataset_time54-108_timeunit0.08',
            
        ], 
        [ # val
            # 'GC_Dataset_ped1-12685_time1280-1340_interp9_xrange5-25_yrange15-35', 
            # 'GC_Dataset_ped1-12685_time2284-2344_interp9_xrange5-25_yrange15-35', 
            'UCY_Dataset_time108-162_timeunit0.08',
        ], 
        [ # test
            # 'GC_Dataset_ped1-12685_time1560-1620_interp9_xrange5-25_yrange15-35', 
            # 'GC_Dataset_ped1-12685_time2344-2404_interp9_xrange5-25_yrange15-35', 
            'UCY_Dataset_time162-216_timeunit0.08',
        ]
    )

    for filename in filenames[0]:        
        rawdata = RawData()
        rawdata.load_trajectory_data(filepath + filename + '.npy')

        with open(savepath + 'train/' + filename + '.txt', 'w') as f:
            for frame in range(rawdata.num_steps):
                for ped in range(rawdata.num_pedestrians):
                    if(rawdata.mask_p[frame, ped] == 1):
                        f.write(f'{frame}\t{ped}\t{rawdata.position[frame, ped, 0]}\t{rawdata.position[frame, ped, 1]}\n')
    
    for filename in filenames[1]:        
        rawdata = RawData()
        rawdata.load_trajectory_data(filepath + filename + '.npy')

        with open(savepath + 'val/' + filename + '.txt', 'w') as f:
            for frame in range(rawdata.num_steps):
                for ped in range(rawdata.num_pedestrians):
                    if(rawdata.mask_p[frame, ped] == 1):
                        f.write(f'{frame}\t{ped}\t{rawdata.position[frame, ped, 0]}\t{rawdata.position[frame, ped, 1]}\n')
    
    for filename in filenames[2]:        
        rawdata = RawData()
        rawdata.load_trajectory_data(filepath + filename + '.npy')

        with open(savepath + 'test/' + filename + '.txt', 'w') as f:
            for frame in range(rawdata.num_steps):
                for ped in range(rawdata.num_pedestrians):
                    if(rawdata.mask_p[frame, ped] == 1):
                        f.write(f'{frame}\t{ped}\t{rawdata.position[frame, ped, 0]}\t{rawdata.position[frame, ped, 1]}\n')
