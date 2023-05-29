# -*- coding: utf-8 -*-
"""
***

"""
import torch
import torch.utils
import numpy as np
import copy
import yaml
from collections import defaultdict
import pdb
import sys

sys.path.append('..')

# import functions.noises as NOISE
# import utils.data_augmentation as AUG
from data.data import RawData, TimeIndexedPedData, PointwisePedData, ChanneledTimeIndexedPedData, TimeIndexedPedDataPolarCoor


class BaseDataset(object):
    """docstring for BaseDataset"""

    def __init__(self):
        super(BaseDataset, self).__init__()

    def build_dataset(self, args):
        raise NotImplementedError

    def load_single_data(self, data_list, add_noise_flag=False):
        data_paths = []
        with open(data_list, 'r') as file:
            for line in file:
                data_paths.append(line.strip())

        data = []
        for path in data_paths:
            tmp_data = RawData()
            tmp_data.load_trajectory_data(path)
            data.append(tmp_data)

        self.raw_data = data

    def load_data(self, data_path, add_noise_flag=False):
        with open(data_path, 'r') as stream:
            data_paths = yaml.load(stream, Loader=yaml.FullLoader)

        data = defaultdict(list)
        for key in data_paths.keys():
            for path in data_paths[key]:
                tmp_data = RawData()
                tmp_data.load_trajectory_data(path)
                data[key].append(tmp_data)

        if add_noise_flag:  # TODO: add noise to training data
            pass

        self.raw_data = data

    @staticmethod
    def get_augmented_data(data: list, thetas: list = None, mirrors: list = None):
        out = []
        out += data
        for d in data:
            if thetas:
                for theta in thetas:
                    out.append(AUG.rotate_augmentation(d, theta))
            if mirrors:
                for theta in mirrors:
                    out.append(AUG.mirror_augmentation(d, theta))
        return out

    def data_augmentation(self, keys, thetas: list = None, mirrors: list = None):
        assert self.raw_data, 'Error: Must load raw data before data augmentation.'

        for key in keys:
            self.raw_data[key] = self.get_augmented_data(self.raw_data[key], thetas, mirrors)

    @staticmethod
    def split_train_val_test(n, train_ratio, val_ratio, test_ratio, seed, shuffle=False):
        """
        """
        if train_ratio + val_ratio + test_ratio != 1:
            raise Exception('illegal train valid test split!')

        train_idx, valid_idx, test_idx = [], [], []
        if shuffle:
            rnd_state = np.random.RandomState(seed)
            idx_all = [i for i in range(n)]
            idx_rnd = rnd_state.permutation(int(n * val_ratio))
            idx_all[:int(n * val_ratio)] = idx_rnd
        else:
            idx_all = [i for i in range(n)]

        val_ratio += train_ratio
        train_idx = np.array(copy.deepcopy(idx_all[:int(n * train_ratio)]))
        valid_idx = np.array(copy.deepcopy(idx_all[int(n * train_ratio):int(n * val_ratio)]))
        test_idx = np.array(copy.deepcopy(idx_all[int(n * val_ratio):]))
        return train_idx, valid_idx, test_idx

    def merge_pointwise_data(self, data_list):
        if len(data_list) == 1:
            return data_list[0]
        else:
            dataset = data_list[0]
            for data in data_list[1:]:
                dataset.add(data)
            return dataset


class PointwisePedDataset(BaseDataset):
    """docstring for PointwisePedDataset"""

    def __init__(self):
        super(PointwisePedDataset, self).__init__()

    def build_dataset(self, args):
        """
        """
        assert (self.raw_data), 'Error: Must load raw data before build dataset.'

        self.args = args
        self.time_unit = self.raw_data['train'][0].time_unit
        args.time_unit = self.time_unit

        self.dataset = defaultdict(list)
        for key in self.raw_data.keys():
            for raw_data in self.raw_data[key]:
                tmp_data = TimeIndexedPedData()
                tmp_data.make_dataset(args, raw_data)
                # pdb.set_trace()
                tmp_data.set_dataset_info(tmp_data, raw_data, list(range(len(tmp_data))))
                if key in {'train', 'valid'}:
                    tmp_data = tmp_data.to_pointwise_data()
                self.dataset[key].append(tmp_data)

        self.train_data = self.merge_pointwise_data(self.dataset['train'])
        self.train_data.to(args.device)
        self.valid_data = self.merge_pointwise_data(self.dataset['valid'])
        self.valid_data.to(args.device)

        print('\ntrain {}, valid {}'.format(
            len(self.train_data), len(self.valid_data)))

        if 'test' in self.dataset.keys():
            self.test_data = self.dataset['test']
            test_len = []
            for d in self.test_data:
                d.to(args.device)
                test_len.append(len(d))
            print(' test {}'.format(test_len))

        args.ped_feature_dim = self.train_data.ped_feature_dim
        args.obs_feature_dim = self.train_data.obs_feature_dim
        args.self_feature_dim = self.train_data.self_feature_dim

        print('Load data successfully!')

        # release space
        # self.dataset = None
        # self.raw_data = None
        # torch.cuda.empty_cache()

    def build_dataset_with_list(self, args):
        """
        """
        assert (self.raw_data), 'Error: Must load raw data before build dataset.'

        self.args = args
        self.time_unit = self.raw_data[0].time_unit

        self.dataset = []
        for raw_data in self.raw_data:
            tmp_data = TimeIndexedPedData()
            tmp_data.make_dataset(args, raw_data)
            tmp_data.set_dataset_info(tmp_data, raw_data, list(range(len(tmp_data))))
            self.dataset.append(tmp_data)

        train_idx, valid_idx, test_idx = self.split_train_val_test(
            len(self.dataset), args.train_ratio, args.val_ratio,
            args.test_ratio, args.seed, shuffle=False)

        self.train_data = []
        for i in train_idx:
            train_data = PointwisePedData()
            train_data.load_from_time_indexed_peddata(self.dataset[i])
            train_data.set_time_unit(self.dataset[i].time_unit)
            self.train_data.append(train_data)
        self.train_data = self.merge_pointwise_data(self.train_data)

        self.valid_data = []
        for i in valid_idx:
            val_data = PointwisePedData()
            val_data.load_from_time_indexed_peddata(self.dataset[i])
            val_data.set_time_unit(self.dataset[i].time_unit)
            self.valid_data.append(val_data)
        self.valid_data = self.merge_pointwise_data(self.valid_data)

        self.test_data = [self.dataset[i] for i in test_idx]
        test_data_len = [len(u) for u in self.test_data]

        # todo: 把test_data改为可以接收多个文件; 注意：分的时候需要有overlap才能充分利用数据。
        self.test_data = self.test_data[0]

        self.train_data.to(args.device)
        self.valid_data.to(args.device)
        self.test_data.to(args.device)

        args.ped_feature_dim = self.train_data.ped_feature_dim
        args.obs_feature_dim = self.train_data.obs_feature_dim
        args.self_feature_dim = self.train_data.self_feature_dim

        print('\ntrain {}, valid {}, test {}'.format(
            len(self.train_data), len(self.valid_data), len(self.test_data)))
        print('Load data successfully!')

    def old_build_dataset(self, args, raw_data):
        self.args = args
        self.time_unit = raw_data.time_unit

        self.dataset = TimeIndexedPedData()
        self.dataset.make_dataset(args, raw_data)
        self.dataset.set_dataset_info(self.dataset, raw_data, list(range(len(self.dataset))))

        self.num_frames = self.dataset.num_frames
        self.num_pedestrians = self.dataset.num_pedestrians
        self.ped_feature_dim = self.dataset.ped_feature_dim
        self.obs_feature_dim = self.dataset.obs_feature_dim
        self.self_feature_dim = self.dataset.self_feature_dim

        dataset_with_noise = self.dataset
        if args.add_noise:
            rw_noise = NOISE.random_walk_noise(raw_data.velocity, raw_data.mask_v, args.noise_std)
            raw_data.velocity += rw_noise
            dataset_with_noise = TimeIndexedPedData()
            dataset_with_noise.make_dataset(args, raw_data)
            dataset_with_noise.set_dataset_info(dataset_with_noise, raw_data, list(range(len(self.dataset))))

        train_idx, valid_idx, test_idx = self.split_train_val_test(
            len(self.dataset), args.train_ratio, args.val_ratio,
            args.test_ratio, args.seed, shuffle=False)

        self.train_data = PointwisePedData()
        self.train_data.load_from_time_indexed_peddata(dataset_with_noise, train_idx)
        self.train_data.set_time_unit(dataset_with_noise.time_unit)
        self.valid_data = PointwisePedData()
        self.valid_data.load_from_time_indexed_peddata(dataset_with_noise, valid_idx)
        self.valid_data.set_time_unit(dataset_with_noise.time_unit)
        self.test_data = TimeIndexedPedData(*self.dataset[test_idx])
        self.test_data.set_dataset_info(self.dataset, raw_data, test_idx)

        self.train_data.to(args.device)
        self.valid_data.to(args.device)
        self.test_data.to(args.device)

        args.ped_feature_dim = self.ped_feature_dim
        args.obs_feature_dim = self.obs_feature_dim
        args.self_feature_dim = self.self_feature_dim

        print('\ntrain {}, valid {}, test {}'.format(
            len(train_idx), len(valid_idx), len(test_idx)))
        print('Load data successfully!')


class PointwisePedDatasetOnlyTraining(BaseDataset):
    """docstring for PointwisePedDatasetOnlyTraining"""

    def __init__(self):
        super(PointwisePedDatasetOnlyTraining, self).__init__()

    def build_dataset(self, args):
        """
        TODO: 把timeindexedpeddataset支持接收多个文件
        """
        assert self.raw_data, 'Error: Must load raw data before build dataset.'

        self.args = args
        self.time_unit = self.raw_data['train'][0].time_unit
        args.time_unit = self.time_unit

        self.dataset = defaultdict(list)
        pointwise_set = {'train'}
        if not args.finetune_flag:
            pointwise_set.union({'valid'})
        for key in self.raw_data.keys():
            for raw_data in self.raw_data[key]:
                tmp_data = TimeIndexedPedData()
                tmp_data.make_dataset(args, raw_data)
                tmp_data.set_dataset_info(tmp_data, raw_data, list(range(len(tmp_data))))
                if key in pointwise_set:
                    tmp_data = tmp_data.to_pointwise_data()
                self.dataset[key].append(tmp_data)

        self.train_data = self.merge_pointwise_data(self.dataset['train'])
        self.train_data.to(args.device)

        if args.finetune_flag:
            self.valid_data = [d.to_channeled_time_index_data(args.valid_steps, 'split') for d in self.dataset['valid']]
            # self.valid_data = self.dataset['valid']
        else:
            self.valid_data = self.merge_pointwise_data(self.dataset['valid'])
            self.valid_data.to(args.device)

        print('\ntrain {}, valid {}'.format(len(self.train_data), len(self.valid_data)))

        if 'test' in self.dataset.keys():
            self.test_data = self.dataset['test']
            test_len = []
            for d in self.test_data:
                d.to(args.device)
                test_len.append(len(d))
            print(' test {}'.format(test_len))

        args.ped_feature_dim = self.train_data.ped_feature_dim
        args.obs_feature_dim = self.train_data.obs_feature_dim
        args.self_feature_dim = self.train_data.self_feature_dim

        print('Load data successfully!')


class TimeIndexedPedDataset(BaseDataset):
    """docstring for TimeIndexedPedDataset"""

    def __init__(self):
        super(TimeIndexedPedDataset, self).__init__()

    def build_dataset(self, args):
        """
        TODO: 把timeindexedpeddataset支持接收多个文件
        """
        assert self.raw_data, 'Error: Must load raw data before build dataset.'

        self.args = args
        self.time_unit = self.raw_data['train'][0].time_unit
        args.time_unit = self.time_unit

        self.dataset = defaultdict(list)
        pointwise_set = set()
        if not args.finetune_flag:
            pointwise_set.union({'train', 'valid'})
        for key in self.raw_data.keys():
            for raw_data in self.raw_data[key]:
                tmp_data = TimeIndexedPedData()
                tmp_data.make_dataset(args, raw_data)
                tmp_data.set_dataset_info(tmp_data, raw_data, list(range(len(tmp_data))))
                if key in pointwise_set:
                    tmp_data = tmp_data.to_pointwise_data()
                self.dataset[key].append(tmp_data)

        if args.finetune_flag:
            # pdb.set_trace()
            self.train_data = [d.to_channeled_time_index_data(args.valid_steps, 'slice') for d in self.dataset['train']]
            self.valid_data = [d.to_channeled_time_index_data(args.valid_steps, 'split') for d in self.dataset['valid']]
            # self.valid_data = self.dataset['valid']
        else:
            self.train_data = self.merge_pointwise_data(self.dataset['train'])
            self.train_data.to(args.device)
            self.valid_data = self.merge_pointwise_data(self.dataset['valid'])
            self.valid_data.to(args.device)

        print('\ntrain {}, valid {}'.format(len(self.train_data), len(self.valid_data)))

        if 'test' in self.dataset.keys():
            self.test_data = self.dataset['test']
            test_len = []
            for d in self.test_data:
                d.to(args.device)
                test_len.append(len(d))
            print(' test {}'.format(test_len))

        args.ped_feature_dim = self.valid_data[0].ped_feature_dim
        args.obs_feature_dim = self.valid_data[0].obs_feature_dim
        args.self_feature_dim = self.valid_data[0].self_feature_dim

        print('Load data successfully!')

class TimeIndexedPedDataset2(BaseDataset):
    """docstring for TimeIndexedPedDataset2"""

    def __init__(self):
        super(TimeIndexedPedDataset2, self).__init__()

    def build_dataset(self, args, finetune_flag):
        """
        TODO: 把timeindexedpeddataset支持接收多个文件
        """
        assert self.raw_data, 'Error: Must load raw data before build dataset.'

        self.args = args
        self.time_unit = self.raw_data['train'][0].time_unit
        args.time_unit = self.time_unit

        self.dataset = defaultdict(list)

        for key in self.raw_data.keys():
            for raw_data in self.raw_data[key]:
                tmp_data = TimeIndexedPedData()
                tmp_data.make_dataset(args, raw_data)
                tmp_data.set_dataset_info(tmp_data, raw_data, list(range(len(tmp_data))))

                self.dataset[key].append(tmp_data)

        if finetune_flag:
            self.train_data = self.dataset['train']
            self.train_data = [d.to_channeled_time_index_data(args.valid_steps, 'slice') for d in self.dataset['train']]
            # self.valid_data = [d.to_channeled_time_index_data(args.valid_steps, 'split') for d in self.dataset['valid']]
            self.valid_data = self.dataset['valid']
        else:
            # self.train_data = self.merge_pointwise_data(self.dataset['train'])
            # self.train_data.to(args.device)
            # self.valid_data = self.merge_pointwise_data(self.dataset['valid'])
            # self.valid_data.to(args.device)
            self.train_data = self.dataset['train']
            # self.train_data = [d.to_channeled_time_index_data(args.valid_steps, 'slice') for d in self.dataset['train']]
            # self.valid_data = [d.to_channeled_time_index_data(args.valid_steps, 'split') for d in self.dataset['valid']]
            self.valid_data = self.dataset['valid']

        print('\ntrain {}, valid {}'.format(len(self.train_data), len(self.valid_data)))

        if 'test' in self.dataset.keys():
            self.test_data = self.dataset['test']
            test_len = []
            for d in self.test_data:
                d.to(args.device)
                test_len.append(len(d))
            print(' test {}'.format(test_len))

        args.ped_feature_dim = self.valid_data[0].ped_feature_dim
        args.obs_feature_dim = self.valid_data[0].obs_feature_dim
        args.self_feature_dim = self.valid_data[0].self_feature_dim

        print('Load data successfully!')


class TimeIndexedPedDatasetforVis(BaseDataset):
    """docstring for TimeIndexedPedDatasetforVis"""

    def __init__(self):
        super(TimeIndexedPedDatasetforVis, self).__init__()

    def build_dataset(self, args):
        """
        TODO: 把timeindexedpeddataset支持接收多个文件
        """
        assert self.raw_data, 'Error: Must load raw data before build dataset.'

        self.args = args
        self.dataset = defaultdict(list)
        keys = list(self.raw_data.keys())
        for key in keys:
            for raw_data in self.raw_data[key]:
                self.time_unit = raw_data.time_unit
                tmp_data = TimeIndexedPedData()
                tmp_data.make_dataset(args, raw_data)
                tmp_data.set_dataset_info(tmp_data, raw_data, list(range(len(tmp_data))))
                self.dataset[key].append(tmp_data)
        args.time_unit = self.time_unit

        args.ped_feature_dim = self.dataset[keys[0]][0].ped_feature_dim
        args.obs_feature_dim = self.dataset[keys[0]][0].obs_feature_dim
        args.self_feature_dim = self.dataset[keys[0]][0].self_feature_dim

        print('Load data successfully!')


class PointwisePedDatasetPolar(BaseDataset):
    """docstring for PointwisePedDatasetPolar"""

    def __init__(self):
        super(PointwisePedDatasetPolar, self).__init__()

    def build_dataset(self, args):
        """
        """
        assert (self.raw_data), 'Error: Must load raw data before build dataset.'

        self.args = args
        self.time_unit = self.raw_data['train'][0].time_unit
        args.time_unit = self.time_unit

        self.dataset = defaultdict(list)
        for key in self.raw_data.keys():
            for raw_data in self.raw_data[key]:
                tmp_data = TimeIndexedPedDataPolarCoor()
                tmp_data.make_dataset(args, raw_data)
                tmp_data.set_dataset_info(tmp_data, raw_data, list(range(len(tmp_data))))
                tmp_data.to_polar_system()
                if key in {'train', 'valid'}:
                    tmp_data = tmp_data.to_pointwise_data()
                self.dataset[key].append(tmp_data)

        self.train_data = self.merge_pointwise_data(self.dataset['train'])
        self.train_data.to(args.device)
        self.valid_data = self.merge_pointwise_data(self.dataset['valid'])
        self.valid_data.to(args.device)

        print('\ntrain {}, valid {}'.format(
            len(self.train_data), len(self.valid_data)))

        if 'test' in self.dataset.keys():
            self.test_data = self.dataset['test']
            test_len = []
            for d in self.test_data:
                d.to(args.device)
                test_len.append(len(d))
            print(' test {}'.format(test_len))

        args.ped_feature_dim = self.train_data.ped_feature_dim
        args.obs_feature_dim = self.train_data.obs_feature_dim
        args.self_feature_dim = self.train_data.self_feature_dim

        print('Load data successfully!')


class TimeIndexedPedDatasetPolar(BaseDataset):
    """docstring for TimeIndexedPedDatasetPolar"""

    def __init__(self):
        super(TimeIndexedPedDatasetPolar, self).__init__()

    def build_dataset(self, args):
        """
        """
        assert self.raw_data, 'Error: Must load raw data before build dataset.'

        self.args = args
        self.time_unit = self.raw_data['train'][0].time_unit
        args.time_unit = self.time_unit

        self.dataset = defaultdict(list)
        pointwise_set = set()
        if not args.finetune_flag:
            pointwise_set.union({'train', 'valid'})
        for key in self.raw_data.keys():
            for raw_data in self.raw_data[key]:
                tmp_data = TimeIndexedPedDataPolarCoor()
                tmp_data.make_dataset(args, raw_data)
                tmp_data.set_dataset_info(tmp_data, raw_data, list(range(len(tmp_data))))
                tmp_data.to_polar_system()
                if key in pointwise_set:
                    tmp_data = tmp_data.to_pointwise_data()
                self.dataset[key].append(tmp_data)

        if args.finetune_flag:
            self.train_data = [d.to_channeled_time_index_data(args.valid_steps, 'slice') for d in self.dataset['train']]
            # self.valid_data = [d.to_channeled_time_index_data(args.valid_steps, 'split') for d in self.dataset['valid']]
            self.valid_data = self.dataset['valid']
        else:
            self.train_data = self.merge_pointwise_data(self.dataset['train'])
            self.train_data.to(args.device)
            self.valid_data = self.merge_pointwise_data(self.dataset['valid'])
            self.valid_data.to(args.device)

        print('\ntrain {}, valid {}'.format(len(self.train_data), len(self.valid_data)))

        if 'test' in self.dataset.keys():
            self.test_data = self.dataset['test']
            test_len = []
            for d in self.test_data:
                d.to(args.device)
                test_len.append(len(d))
            print(' test {}'.format(test_len))

        args.ped_feature_dim = self.valid_data[0].ped_feature_dim
        args.obs_feature_dim = self.valid_data[0].obs_feature_dim
        args.self_feature_dim = self.valid_data[0].self_feature_dim

        print('Load data successfully!')


