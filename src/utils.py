from abc import ABC
import os
import argparse
from yacs.config import CfgNode
from typing import Optional
import shutil
import math
import torch
from torch import Tensor
from pathlib import Path


def load_config(args: argparse.Namespace) -> CfgNode:
    from default_params import _C as cfg
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg_ = cfg.clone()
    if os.path.isfile(args.config_file):
        conf = args.config_file
        print(f"Configuration file loaded from {conf}.")
        cfg_.merge_from_file(conf)
        cfg_.OUTPUT_DIR = os.path.join(cfg_.OUTPUT_DIR,
                                       os.path.splitext(conf)[0])

    else:
        raise FileNotFoundError

    if cfg_.LOAD_TUNED and args.mode != "tune":
        cfg_ = load_tuned(args, cfg_)
    cfg_.freeze()

    print(f"output dirname: {cfg_.OUTPUT_DIR}")
    os.makedirs(cfg_.OUTPUT_DIR, exist_ok=True)
    if os.path.isfile(args.config_file):
        shutil.copy2(args.config_file, os.path.join(
            cfg_.OUTPUT_DIR, 'config.yaml'))

    return cfg_


def load_tuned(args: argparse.Namespace, cfg: CfgNode) -> CfgNode:
    import optuna
    study_path = os.path.join(cfg.OUTPUT_DIR, "optuna.db")
    if not os.path.exists(study_path):
        return cfg

    study_path = os.path.join("sqlite:///", study_path)
    print("load params from optuna database")
    study = optuna.load_study(storage=study_path, study_name="my_opt")
    trial_dict = study.best_trial.params

    for key in list(trial_dict.keys()):
        if type(trial_dict[key]) == str:
            exec(f"cfg.{key} = '{trial_dict[key]}'")
        else:
            exec(f"cfg.{key} = {trial_dict[key]}")

    return cfg


def optimizer_to_cuda(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()


class DynamicBufferModule(ABC, torch.nn.Module):
    """Torch module that allows loading variables from the state dict even in the case of shape mismatch."""

    def get_tensor_attribute(self, attribute_name: str) -> Tensor:
        """Get attribute of the tensor given the name.
        Args:
            attribute_name (str): Name of the tensor
        Raises:
            ValueError: `attribute_name` is not a torch Tensor
        Returns:
            Tensor: Tensor attribute
        """
        attribute = getattr(self, attribute_name)
        if isinstance(attribute, Tensor):
            return attribute

        raise ValueError(
            f"Attribute with name '{attribute_name}' is not a torch Tensor")

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args):
        """Resizes the local buffers to match those stored in the state dict.
        Overrides method from parent class.
        Args:
          state_dict (dict): State dictionary containing weights
          prefix (str): Prefix of the weight file.
          *args:
        """
        persistent_buffers = {k: v for k, v in self._buffers.items(
        ) if k not in self._non_persistent_buffers_set}
        local_buffers = {k: v for k,
                         v in persistent_buffers.items() if v is not None}

        for param in local_buffers.keys():
            for key in state_dict.keys():
                if key.startswith(prefix) and key[len(prefix):].split(".")[0] == param:
                    if not local_buffers[param].shape == state_dict[key].shape:
                        attribute = self.get_tensor_attribute(param)
                        attribute.resize_(state_dict[key].shape)

        super()._load_from_state_dict(state_dict, prefix, *args)


class GaussianKDE(DynamicBufferModule):
    """Gaussian Kernel Density Estimation.
    Args:
        dataset (Optional[Tensor], optional): Dataset on which to fit the KDE model. Defaults to None.
    """

    def __init__(self, dataset: Optional[Tensor] = None):
        super().__init__()

        self.register_buffer("bw_transform", Tensor())
        self.register_buffer("dataset", Tensor())
        self.register_buffer("norm", Tensor())

        #self.bw_transform = Tensor()
        #self.dataset = Tensor()
        #self.norm = Tensor()

        if dataset is not None:
            self.fit(dataset)

    def forward(self, features: Tensor) -> Tensor:
        """Get the KDE estimates from the feature map.
        Args:
          features (Tensor): Feature map extracted from the CNN
        Returns: KDE Estimates
        """
        features = torch.matmul(features, self.bw_transform)

        estimate = torch.zeros(features.shape[0]).to(features.device)
        for i in range(features.shape[0]):
            embedding = ((self.dataset - features[i]) ** 2).sum(dim=1)
            embedding = torch.exp(-embedding / 2) * self.norm
            estimate[i] = torch.mean(embedding)

        return estimate

    def fit(self, dataset: Tensor) -> None:
        """Fit a KDE model to the input dataset.
        Args:
          dataset (Tensor): Input dataset.
        Returns:
            None
        """

        num_samples, dimension = dataset.shape

        # compute scott's bandwidth factor
        factor = num_samples ** (-1 / (dimension + 4))

        cov_mat = self.cov(dataset.T)
        inv_cov_mat = torch.linalg.inv(cov_mat)
        inv_cov = inv_cov_mat / factor**2

        # transform data to account for bandwidth
        bw_transform = torch.linalg.cholesky(inv_cov)
        dataset = torch.matmul(dataset, bw_transform)

        #
        norm = torch.prod(torch.diag(bw_transform))
        norm *= math.pow((2 * math.pi), (-dimension / 2))

        self.bw_transform = bw_transform
        self.dataset = dataset
        self.norm = norm

    @staticmethod
    def cov(tensor: Tensor) -> Tensor:
        """Calculate the unbiased covariance matrix.
        Args:
            tensor (Tensor): Input tensor from which covariance matrix is computed.
        Returns:
            Output covariance matrix.
        """
        mean = torch.mean(tensor, dim=1, keepdim=True)
        cov = torch.matmul(tensor - mean, (tensor - mean).T) / \
            (tensor.size(1) - 1)
        return cov
