#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import torch
import torch.nn.functional as F
import time
import numpy as np
import pickle
import os
import h5py
from functools import partial
from typing import Sequence


from utils.make_master_file import DATASET_DICT
from utils.normalizer import init_normalizer, UnitTransformer, PointWiseUnitTransformer, MinMaxTransformer, TorchQuantileTransformer, IdentityTransformer
from torch.utils.data import Dataset
from utils.make_master_file import DATASET_DICT
from utils.utilities import downsample, resize







class MixedTemporalDataset(Dataset):
    def __init__(self, data_names, n_list = None, res = 128,t_in = 10, t_ar = 1, n_channels = None, normalize=False,train=True,data_weights=None):
        '''
        Dataset class for training pretraining multiple datasets
        :param data_names: names of datasets, specified in make_master_file.py
        :param n_list: num of training samples per dataset, should corresponds to the order of data_names
        :param res: input resolution for the model, 64/128/256/512/1024
        :param t_in: input timesteps, 10 for default
        :param t_ar: steps for auto-regressive pretraining, 1 for default
        :param n_channels: number of channels for dataset, if None, it auto reads max number of channels from config file, should be specified for test dataset
        :param normalize: if normalize data,  reversible instance normalization is implemented in each model
        :param train: if it is train dataset or (in distribution) test dataset
        '''
        self.data_names = data_names if isinstance(data_names, list) else [data_names]
        self.data_weights = data_weights if data_weights is not None else [1] * len(self.data_names)
        self.num_datasets = len(data_names)
        self.t_in = t_in
        self.t_ar = t_ar
        self.train = train
        self.res = res
        self.n_sizes = n_list if n_list is not None else [DATASET_DICT[name]['train_size'] if train else DATASET_DICT[name]['test_size'] for name in self.data_names]
        self.weighted_sizes = [size * weight for size, weight in zip(self.n_sizes, self.data_weights)]
        self.cumulative_sizes = np.cumsum(self.weighted_sizes)

        self.t_tests = [DATASET_DICT[name]['t_test'] for name in self.data_names]
        self.downsamples = [DATASET_DICT[name]['downsample'] for name in self.data_names]
        self.n_channels = max([DATASET_DICT[name]['n_channels'] for name in self.data_names]) if n_channels is None else n_channels

        self.data_files = []
        for name in self.data_names:
            if DATASET_DICT[name]['scatter_storage']:
                def open_hdf5_file(path, idx):
                    return h5py.File(f'{path}/data_{idx}.hdf5', 'r')['data'][:]
                path = DATASET_DICT[name]['train_path'] if train else DATASET_DICT[name]['test_path']
                self.data_files.append(partial(open_hdf5_file, path))
            else:
                self.data_files.append(h5py.File(DATASET_DICT[name]['train_path'] if train else DATASET_DICT[name]['test_path'], 'r'))


        self.normalize = normalize
        self.normalizers = []
        if normalize:
            print('Using normalizer for inputs')
            for data in self.data_files:
                self.normalizers.append(UnitTransformer(torch.from_numpy(data['data'][:500]).float()))


    def pad_data(self, x):
        '''
        pad data to unified shape
        :param x: H, W, T, C
        :return:  H', W', T', C'
        '''
        H, W, T, C = x.shape
        x = x.view(H, W, -1).permute(2, 0, 1) # T * C, H, W
        x = F.interpolate(x.unsqueeze(0), size=(self.res, self.res),mode='bilinear').squeeze(0).permute(1, 2, 0)
        x = x.view(*x.shape[:2], T, C)
        x_new = torch.ones([*x.shape[:-1], self.n_channels])
        x_new[..., :x.shape[-1]] = x  # H, W, T, Cmax

        return x_new

    def get_target_mask(self, x, size_orig):
        '''
        :param x: single data, H, W, T, C
        :param size_orig: original size of x
        :return: masks for evaluation (by resolution)
        '''
        msk = torch.zeros(*x.shape[:2], 1, x.shape[-1])    ## target mask shape H,W,1,C
        kx, ky = x.shape[0] // size_orig[0], x.shape[1] // size_orig[1]
        if kx ==0 or ky == 0:
            kx = 1 if kx ==0 else kx
            ky = 1 if ky == 0 else ky
        msk[::kx, ::ky, :, :size_orig[-1]] = 1

        return msk

    def __len__(self):
        return self.cumulative_sizes[-1]




    def __getitem__(self, idx):
        '''
        Logic of getitem: first find which dataset idx is in, then reshape it to H,W,T,C,
            for training dataset, we random sample start timestep
            for test dataset, we return the whole trajectory
        :param idx: id in the whole dataset
        :return: data slice
        '''
        dataset_idx = int(np.searchsorted(self.cumulative_sizes, idx + 1))

        if dataset_idx == 0:
            data_idx = idx
        else:
            data_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        data_idx //= self.data_weights[dataset_idx]
        # Use torch.tensor instead of torch.from_numpy to avoid numpy version conflicts
        if callable(self.data_files[dataset_idx]):
            raw_data = self.data_files[dataset_idx](data_idx)[:]
        else:
            raw_data = self.data_files[dataset_idx]['data'][data_idx][:]
        sample = torch.tensor(raw_data, dtype=torch.float32)
        if sample.ndim == 3:
            sample = sample.unsqueeze(-1)

        orig_size = list(sample.shape)
        orig_size[-1] = DATASET_DICT[self.data_names[dataset_idx]]['pred_channels'] if 'pred_channels' in DATASET_DICT[self.data_names[dataset_idx]].keys() else orig_size[-1]
        sample = self.pad_data(sample)


        if self.train:
            start_idx = np.random.randint(max(sample.shape[-2] - (self.t_in + self.t_ar) + 1, 1))
            x, y = sample[..., start_idx: start_idx + self.t_in,:], sample[..., start_idx + self.t_in: min(start_idx + self.t_in + self.t_ar, sample.shape[-2]),:]
            msk = torch.ones([*x.shape[:2], 1, x.shape[-1]])
        else:
            start_idx = 0
            x, y = sample[..., start_idx:start_idx + self.t_in,:], sample[..., self.t_in:self.t_in + self.t_tests[dataset_idx],:]
            msk = self.get_target_mask(sample, orig_size)

        if self.normalize:
            x = (x.unsqueeze(0) - self.normalizers[int(dataset_idx)].mean[..., start_idx: start_idx + self.t_in,:]) / (self.normalizers[int(dataset_idx)].std[..., start_idx: start_idx + self.t_in,:] + 1e-6)
            x = x.squeeze()

        if self.downsamples[dataset_idx] != (1, 1):
            x, y = x[::self.downsamples[dataset_idx][0],::self.downsamples[dataset_idx][1]], y[::self.downsamples[dataset_idx][0],::self.downsamples[dataset_idx][1]]

        idx_cls = torch.LongTensor([dataset_idx])
        return x, y, msk, idx_cls







class MixedMaskedDataset(Dataset):
    def __init__(self, data_names, n_list = None, res = 128,t_in = 10, t_ar = 1, n_channels = None, normalize=False,train=True,data_weights=None):
        '''
        Dataset class for training pretraining multiple datasets
        :param data_names: names of datasets, specified in make_master_file.py
        :param n_list: num of training samples per dataset, should corresponds to the order of data_names
        :param res: input resolution for the model, 64/128/256/512/1024
        :param t_in: input timesteps, 10 for default
        :param t_ar: steps for auto-regressive pretraining, 1 for default
        :param n_channels: number of channels for dataset, if None, it auto reads max number of channels from config file, should be specified for test dataset
        :param normalize: if normalize data,  reversible instance normalization is implemented in each model
        :param train: if it is train dataset or (in distribution) test dataset
        '''
        self.data_names = data_names if isinstance(data_names, list) else [data_names]
        self.data_weights = data_weights if data_weights is not None else [1] * len(self.data_names)
        self.num_datasets = len(data_names)
        self.t_in = t_in
        self.t_ar = t_ar
        self.train = train
        self.res = res
        self.n_sizes = n_list if n_list is not None else [DATASET_DICT[name]['train_size'] if train else DATASET_DICT[name]['test_size'] for name in self.data_names]
        self.weighted_sizes = [size * weight for size, weight in zip(self.n_sizes, self.data_weights)]
        self.cumulative_sizes = np.cumsum(self.weighted_sizes)

        self.t_tests = [DATASET_DICT[name]['t_test'] for name in self.data_names]
        self.downsamples = [DATASET_DICT[name]['downsample'] for name in self.data_names]
        self.n_channels = max([DATASET_DICT[name]['n_channels'] for name in self.data_names]) if n_channels is None else n_channels

        self.data_files = []
        for name in self.data_names:
            if DATASET_DICT[name]['scatter_storage']:
                def open_hdf5_file(path, idx):
                    return h5py.File(f'{path}/data_{idx}.hdf5', 'r')['data'][:]
                path = DATASET_DICT[name]['train_path'] if train else DATASET_DICT[name]['test_path']
                self.data_files.append(partial(open_hdf5_file, path))
            else:
                self.data_files.append(h5py.File(DATASET_DICT[name]['train_path'] if train else DATASET_DICT[name]['test_path'], 'r'))


        self.normalize = normalize
        self.normalizers = []
        if normalize:
            print('Using normalizer for inputs')
            for data in self.data_files:
                self.normalizers.append(UnitTransformer(torch.from_numpy(data['data'][:500]).float()))


    def pad_data(self, x):
        '''
        pad data to unified shape
        :param x: H, W, T, C
        :return:  H', W', T', C'
        '''
        H, W, T, C = x.shape
        x = x.view(H, W, -1).permute(2, 0, 1) # T * C, H, W
        x = F.interpolate(x.unsqueeze(0), size=(self.res, self.res),mode='bilinear').squeeze(0).permute(1, 2, 0)
        x = x.view(*x.shape[:2], T, C)
        x_new = torch.ones([*x.shape[:-1], self.n_channels])    # use 1 for void padding
        x_new[..., :x.shape[-1]] = x  # H, W, T, Cmax

        return x_new

    def get_target_mask(self, x, size_orig):
        '''
        :param x: single data, H, W, T, C
        :param size_orig: original size of x
        :return: masks for evaluation (by resolution)
        '''
        msk = torch.zeros(*x.shape[:2], 1, x.shape[-1])    ## target mask shape H,W,1,C
        kx, ky = x.shape[0] // size_orig[0], x.shape[1] // size_orig[1]
        if kx ==0 or ky == 0:
            kx = 1 if kx ==0 else kx
            ky = 1 if ky == 0 else ky
        msk[::kx, ::ky, :, :size_orig[-1]] = 1

        return msk

    def get_masked_input(self, x):
        '''
        :param x:  single data, H, W, T, C
        :param size_orig:  original size of x
        :return: masked input, TODO: downsampling resolution
        '''
        x_new = x.clone()
        x_new[:,:,-1,:] = -1
        return x_new


    def __len__(self):
        return self.cumulative_sizes[-1]




    def __getitem__(self, idx):
        '''
        Logic of getitem: first find which dataset idx is in, then reshape it to H,W,T,C,
            for training dataset, we random sample start timestep
            for test dataset, we return the whole trajectory
        :param idx: id in the whole dataset
        :return: data slice
        '''
        dataset_idx = int(np.searchsorted(self.cumulative_sizes, idx + 1))

        if dataset_idx == 0:
            data_idx = idx
        else:
            data_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        data_idx //= self.data_weights[dataset_idx]
        # Use torch.tensor instead of torch.from_numpy to avoid numpy version conflicts
        if callable(self.data_files[dataset_idx]):
            raw_data = self.data_files[dataset_idx](data_idx)[:]
        else:
            raw_data = self.data_files[dataset_idx]['data'][data_idx][:]
        sample = torch.tensor(raw_data, dtype=torch.float32)
        if sample.ndim == 3:
            sample = sample.unsqueeze(-1)

        orig_size = list(sample.shape)
        sample = self.pad_data(sample)


        if self.train:
            start_idx = np.random.randint(max(sample.shape[-2] - self.t_in + 1, 1))
            x = sample[..., start_idx: start_idx + self.t_in,:]
            x_msk = self.get_masked_input(x)


            target_msk = torch.ones([*x.shape[:2], 1, x.shape[-1]])
        else:
            x_msk, x = sample[...,:self.t_in,:], sample[..., self.t_in-1:self.t_in + self.t_tests[dataset_idx],:]
            target_msk = self.get_target_mask(sample, orig_size)
            x_msk = self.get_masked_input(x_msk)
        if self.downsamples[dataset_idx] != (1, 1):
            x_msk, x = x_msk[::self.downsamples[dataset_idx][0],::self.downsamples[dataset_idx][1]], x[::self.downsamples[dataset_idx][0],::self.downsamples[dataset_idx][1]]

        idx_cls = torch.LongTensor([dataset_idx])
        return x_msk, x, target_msk, idx_cls



class SteadyDataset2D(Dataset):
    def __init__(self, data_name, n_train=None, res=128, n_channels = None, normalize=False, train=True):
        '''
        :param data_name:
        :param n_train:
        :param res:
        :param t_in:
        :param t_ar:
        :param n_channels:
        :param normalize:
        :param train:
        '''
        self.data_name = data_name
        self.n_size = n_train if n_train is not None else DATASET_DICT[data_name]['train_size'] if train else DATASET_DICT[data_name]['test_size']
        self.train = train
        self.res = res
        self.n_channels = DATASET_DICT[data_name]['n_channels'] if n_channels is None else n_channels
        self.downsample = DATASET_DICT[data_name]['downsample']



        if DATASET_DICT[self.data_name]['scatter_storage']:
            def open_hdf5_file(path, idx, name):
                return h5py.File(f'{path}/data_{idx}.hdf5', 'r')[name][:]

            path = DATASET_DICT[self.data_name]['train_path'] if train else DATASET_DICT[self.data_name]['test_path']
            self.data_files = partial(open_hdf5_file, path)
        else:
            self.data_files = h5py.File(DATASET_DICT[self.data_name]['train_path'] if train else DATASET_DICT[self.data_name]['test_path'], 'r')

    def pad_data(self, x):
        '''
        pad data to unified shape
        :param x: H, W, T, C
        :return:  H', W', T', C'
        '''
        H, W, C = x.shape
        x = x.view(H, W, -1).permute(2, 0, 1)  # C, H, W
        x = F.interpolate(x.unsqueeze(0), size=(self.res, self.res), mode='bilinear').squeeze(0).permute(1, 2, 0).unsqueeze(-2)
        x_new = torch.ones([*x.shape[:-1], self.n_channels])
        x_new[..., :x.shape[-1]] = x  # H, W, T, Cmax

        return x_new


    def shuffle_channels(self, x, y):
        idx1, idx2 = torch.randperm(x.shape[-1])[:2]
        x[..., [idx1, idx2]] = x[..., [idx2, idx1]]
        y[...,[idx1, idx2]] = y[..., [idx2, idx1]]
        return x, y


    def get_target_mask(self, x, size_orig):
        '''
        :param x: single data, H, W, T, C
        :param size_orig: original size of x
        :return: masks for evaluation (by resolution)
        '''
        msk = torch.zeros(*x.shape[:2], 1, x.shape[-1])  ## target mask shape H,W,1,C
        kx, ky = x.shape[0] // size_orig[0], x.shape[1] // size_orig[1]
        if kx == 0 or ky == 0:
            kx = 1 if kx == 0 else kx
            ky = 1 if ky == 0 else ky
        msk[::kx, ::ky, :, :size_orig[-1]] = 1

        return msk


    def __getitem__(self, idx):
        '''
        Logic of getitem:  reshape data to H,W,L,T,C,
            for training dataset, we random sample start timestep
            for test dataset, we return the whole trajectory
        :param idx: id in the whole dataset
        :return: data slice
        '''
        # Use torch.tensor instead of torch.from_numpy to avoid numpy version conflicts
        if callable(self.data_files):
            sample_x = torch.tensor(self.data_files(idx, name='x')[:], dtype=torch.float32)
            sample_y = torch.tensor(self.data_files(idx, name='y')[:], dtype=torch.float32)
        else:
            sample_x = torch.tensor(self.data_files['x'][idx][:], dtype=torch.float32)
            sample_y = torch.tensor(self.data_files['y'][idx][:], dtype=torch.float32)

        if sample_x.ndim == 2:
            sample_x = sample_x.unsqueeze(-1)
            sample_y = sample_y.unsqueeze(-1)



        orig_size = list(sample_x.shape)
        orig_size[-1] = DATASET_DICT[self.data_name]['pred_channels'] if 'pred_channels' in DATASET_DICT[self.data_name].keys() else orig_size[-1]
        x, y = self.pad_data(sample_x), self.pad_data(sample_y)


        if self.train:
            msk = torch.ones([*x.shape[:2], 1, x.shape[-1]])
        else:
            msk = self.get_target_mask(x, orig_size)


        if self.downsample != (1, 1, 1):
            x, y = x[::self.downsample[0],::self.downsample[1]], y[::self.downsample[0],::self.downsample[1]]

        return x, y, msk

    def __len__(self):
        return self.n_size



class TemporalDataset3D(Dataset):
    def __init__(self, data_name, n_train=None, res=128, t_in=10, t_ar = 1, n_channels = None, normalize=False, train=True):
        '''

        :param data_name:
        :param n_train:
        :param res:
        :param t_in:
        :param t_ar:
        :param n_channels:
        :param normalize:
        :param train:
        '''
        self.data_name = data_name
        self.n_size = n_train if n_train is not None else DATASET_DICT[data_name]['train_size'] if train else DATASET_DICT[data_name]['test_size']
        self.train = train
        self.res = res
        self.t_in = t_in
        self.t_ar = t_ar
        self.t_test = DATASET_DICT[data_name]['t_test']
        self.n_channels = DATASET_DICT[data_name]['n_channels'] if n_channels is None else n_channels
        self.downsample = DATASET_DICT[data_name]['downsample']



        if DATASET_DICT[self.data_name]['scatter_storage']:
            def open_hdf5_file(path, idx):
                return h5py.File(f'{path}/data_{idx}.hdf5', 'r')['data'][:]

            path = DATASET_DICT[self.data_name]['train_path'] if train else DATASET_DICT[self.data_name]['test_path']
            self.data_files = partial(open_hdf5_file, path)
        else:
            self.data_files = h5py.File(DATASET_DICT[self.data_name]['train_path'] if train else DATASET_DICT[self.data_name]['test_path'], 'r')

    def pad_data(self, x):
        '''
        pad data to unified shape
        :param x: H, W, T, C
        :return:  H', W', T', C'
        '''
        H, W, L, T, C = x.shape
        x = x.view(H, W, L, -1).permute(3, 0, 1, 2)  # T * C, H, W, L
        x = F.interpolate(x.unsqueeze(0), size=(self.res, self.res, self.res), mode='trilinear').squeeze(0).permute(1, 2, 3, 0)
        x = x.view(*x.shape[:3], T, C)
        x_new = torch.ones([*x.shape[:-1], self.n_channels])
        x_new[..., :x.shape[-1]] = x  # H, W, L, T, Cmax

        return x_new

    def get_target_mask(self, x, size_orig):
        '''
        :param x: single data, H, W, T, C
        :param size_orig: original size of x
        :return: masks for evaluation (by resolution)
        '''
        msk = torch.zeros(*x.shape[:3], 1, x.shape[-1])  # H, W, L, 1, C
        kx, ky, kz = x.shape[0] // size_orig[0], x.shape[1] // size_orig[1], x.shape[2] // size_orig[2]
        if kx == 0 or ky == 0 or kz == 0:
            kx = 1 if kx == 0 else kx
            ky = 1 if ky == 0 else ky
            kz = 1 if kz == 0 else kz
        msk[::kx, ::ky, ::kz, :, :size_orig[-1]] = 1

        return msk


    def __getitem__(self, idx):
        '''
        Logic of getitem:  reshape data to H,W,L,T,C,
            for training dataset, we random sample start timestep
            for test dataset, we return the whole trajectory
        :param idx: id in the whole dataset
        :return: data slice
        '''



        # Use torch.tensor instead of torch.from_numpy to avoid numpy version conflicts
        if callable(self.data_files):
            sample = torch.tensor(self.data_files(idx)[:], dtype=torch.float32)
        else:
            sample = torch.tensor(self.data_files['data'][idx][:], dtype=torch.float32)
        if sample.ndim == 4:
            sample = sample.unsqueeze(-1)

        orig_size = list(sample.shape)
        orig_size[-1] = DATASET_DICT[self.data_name]['pred_channels'] if 'pred_channels' in DATASET_DICT[self.data_name].keys() else orig_size[-1]
        sample = self.pad_data(sample)


        if self.train:
            start_idx = np.random.randint(max(sample.shape[-2] - (self.t_in + self.t_ar) + 1, 1))
            x, y = sample[..., start_idx: start_idx + self.t_in,:], sample[..., start_idx + self.t_in: min(start_idx + self.t_in + self.t_ar, sample.shape[-2]),:]
            msk = torch.ones([*x.shape[:3], 1, x.shape[-1]])
        else:
            start_idx = 0
            x, y = sample[..., start_idx:start_idx + self.t_in,:], sample[..., self.t_in:self.t_in + self.t_test,:]
            msk = self.get_target_mask(sample, orig_size)


        if self.downsample != (1, 1, 1):
            x, y = x[::self.downsample[0],::self.downsample[1],::self.downsample[2]], y[::self.downsample[0],::self.downsample[1],::self.downsample[2]]

        return x, y, msk

    def __len__(self):
        return self.n_size
