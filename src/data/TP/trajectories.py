import logging
import os
import math
from copy import deepcopy

import numpy as np
from sklearn.preprocessing import scale

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def seq_collate(data):
    (index_list, obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list, max_value, min_value) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    max_value = torch.Tensor(max_value[0])
    min_value = torch.Tensor(min_value[0])
    out = {"index": torch.IntTensor(index_list),
           "obs": obs_traj,
           "gt": pred_traj,
           "obs_rel": obs_traj_rel,
           "gt_rel": pred_traj_rel,
           "non_linear_ped": non_linear_ped,
           "loss_mask": loss_mask,
           "seq_start_end": seq_start_end,
           "max": max_value,
           "min": min_value}
    
    return out


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        
        self.max = np.max(seq_list, axis=(0,2))
        self.min = np.min(seq_list, axis=(0,2))
        self.mean = torch.from_numpy(np.mean(seq_list, axis=(0,2))).to(dtype=torch.float32)
        self.std = torch.from_numpy(np.std(seq_list, axis=(0,2))).to(dtype=torch.float32)
        
    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            index,
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.max, self.min
        ]
        
        return out


class SimulatedForkTrajectory(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, num_data, obs_len=8, pred_len=12, threshold=0.002,
        min_ped=1, delim='\t',
        stochastic=True
    ):
        """
        Args:
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        self.num_data = num_data
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.theta = np.pi / 3
        self.sigma = 0.5
        self.threshold = threshold
        self.stochastic = stochastic
        
        self.traj = []
        for sgn in [1, -1]:
            traj_flat_x = torch.arange(-self.obs_len-1., 0., 1., dtype=torch.float)
            traj_flat_y = torch.zeros_like(traj_flat_x, dtype=torch.float)
            traj_flat = torch.cat([traj_flat_x[None, :], traj_flat_y[None, :]], dim=0)
            
            dist = torch.arange(1., self.pred_len+1., 1., dtype=torch.float)
            traj_ramp_x = dist * np.cos(sgn * self.theta, dtype=np.float32)
            traj_ramp_y = dist * np.sin(sgn * self.theta, dtype=np.float32)
            traj_ramp = torch.cat([traj_ramp_x[None, :], traj_ramp_y[None, :]], dim=0)
            
            traj = torch.cat([traj_flat, traj_ramp], dim=1)
    
            self.traj.append(traj)
        
        traj_cat = torch.cat(self.traj, dim=1)
        self.max = torch.max(traj_cat, dim=1).values
        self.min = torch.min(traj_cat, dim=1).values
        self.mean = torch.mean(traj_cat, dim=1)
        self.std = torch.std(traj_cat, dim=1)
        
    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        if self.stochastic:
            i = np.random.randint(2)
        else:
            i = 0
        
        traj = deepcopy(self.traj[i])
        traj += torch.normal(mean=torch.zeros_like(traj), std=torch.ones_like(traj) * self.sigma)
        
        traj_rel = traj[:, 1:] - traj[:, :-1]
        traj = traj[:, 1:]
        
        #non_linear = poly_fit(traj, self.pred_len, self.threshold)
        non_linear = torch.zeros([1, 1])
        loss_mask = torch.ones([1, self.obs_len + self.pred_len])
        
        out = [
            index,
            traj[:, :self.obs_len][None, :, :], traj[:, self.obs_len:][None, :, :],
            traj_rel[:, :self.obs_len][None, :, :], traj_rel[:, self.obs_len:][None, :, :],
            non_linear, loss_mask, self.max, self.min
        ]
        return out


class SimulatedCrossTrajectory(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, num_data, obs_len=8, pred_len=12, threshold=0.002,
        min_ped=1, delim='\t',
        stochastic=True
    ):
        """
        Args:
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        self.num_data = num_data
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.theta = np.pi / 3
        self.sigma = 0.5
        self.threshold = threshold
        self.stochastic = stochastic
        
        self.traj = []
        for sgn in [1, -1]:
            traj_flat_x = torch.arange(-self.obs_len-1., 0., 1., dtype=torch.float)
            traj_flat_y = torch.zeros_like(traj_flat_x, dtype=torch.float)
            traj_flat = torch.cat([traj_flat_x[None, :], traj_flat_y[None, :]], dim=0)
            
            dist = torch.arange(1., self.pred_len+1., 1., dtype=torch.float)
            traj_ramp_x = dist * np.cos(sgn * self.theta, dtype=np.float32)
            dist = torch.cat([torch.arange(1., 3., 1., dtype=torch.float),
                              torch.arange(2., -self.pred_len+4., -1., dtype=torch.float)])
            traj_ramp_y = dist * np.sin(sgn * self.theta, dtype=np.float32)
            traj_ramp = torch.cat([traj_ramp_x[None, :], traj_ramp_y[None, :]], dim=0)
            
            traj = torch.cat([traj_flat, traj_ramp], dim=1)
    
            #traj = torch.nn.functional.pad(traj, (0,1), mode='replicate')[:, 1:]
            self.traj.append(traj)
        
        traj_cat = torch.cat(self.traj, dim=1)
        self.max = torch.max(traj_cat, dim=1).values
        self.min = torch.min(traj_cat, dim=1).values
        self.mean = torch.mean(traj_cat, dim=1)
        self.std = torch.std(traj_cat, dim=1)
        
    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        if self.stochastic:
            i = np.random.randint(2)
        else:
            i = 0
        
        traj = deepcopy(self.traj[i])
        traj += torch.normal(mean=torch.zeros_like(traj), std=torch.ones_like(traj) * self.sigma)
        
        traj_rel = traj[:, 1:] - traj[:, :-1]
        traj = traj[:, 1:]
        
        #non_linear = poly_fit(traj, self.pred_len, self.threshold)
        non_linear = torch.zeros([1, 1])
        loss_mask = torch.ones([1, self.obs_len + self.pred_len])
        
        out = [
            index,
            traj[:, :self.obs_len][None, :, :], traj[:, self.obs_len:][None, :, :],
            traj_rel[:, :self.obs_len][None, :, :], traj_rel[:, self.obs_len:][None, :, :],
            non_linear, loss_mask, self.max, self.min
        ]
        return out

    