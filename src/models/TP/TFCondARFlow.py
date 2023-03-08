from yacs.config import CfgNode
import copy
import math
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from utils import optimizer_to_cuda


class ARFlow(nn.Module):
    def __init__(self, cfg: CfgNode) -> None:
        super(ARFlow, self).__init__()
        
        self.output_path = Path(cfg.OUTPUT_DIR)

        self.obs_len = cfg.DATA.OBSERVE_LENGTH
        self.pred_len = cfg.DATA.PREDICT_LENGTH
        
        self.feature_dim = 2
        conditioning_length = 16
        self.pe_dim = 16
        self.input_size = self.feature_dim + self.pe_dim
        self.d_model = 16
        num_heads = 4
        num_encoder_layers = 3
        num_decoder_layers = 3
        dim_feedforward_scale = 4
        dropout_rate = 0.1
        act_type = "gelu"
        flow_type = "RealNVP"
        target_dim = 2
        n_blocks = 3
        n_hidden = 2
        hidden_size = 64
        dequantize = False
        self.scaling = True
        prediction_length = self.pred_len
        
        self.encoder_input = nn.Linear(self.input_size, self.d_model)
        self.decoder_input = nn.Linear(self.input_size, self.d_model)

        # [B, T, d_model] where d_model / num_heads is int
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward_scale * self.d_model,
            dropout=dropout_rate,
            activation=act_type,
        )

        flow_cls = {
            "RealNVP": RealNVP,
            "MAF": MAF,
        }[flow_type]
        self.flow = flow_cls(
            input_size=target_dim,
            n_blocks=n_blocks,
            n_hidden=n_hidden,
            hidden_size=hidden_size,
            cond_label_size=conditioning_length,
        )
        self.dequantize = dequantize
        
        self.dist_args_proj = nn.Linear(self.d_model, conditioning_length)
        
        position = torch.arange(self.obs_len + self.pred_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.pe_dim, 2) * (-math.log(10000.0) / self.pe_dim))
        self.pe = torch.zeros(self.obs_len + self.pred_len, 1, self.pe_dim)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.pe = self.pe.cuda()

        if self.scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        # mask
        self.register_buffer(
            "tgt_mask",
            self.transformer.generate_square_subsequent_mask(prediction_length),
        )
        
        self.optimizer = optim.Adam(self.parameters(), 5e-4)
        self.optimizers = [self.optimizer]

    def predict(self, data_dict: Dict) -> Dict:
        inputs = torch.cat([data_dict['obs'], data_dict['gt']])

        enc_inputs = inputs[: self.obs_len, ...]
        dec_inputs = enc_inputs[-1][None]
        
        _, scale = self.scaler(enc_inputs)
        if self.scaling:
            self.flow.scale = scale
        
        enc_pe = self.pe[: self.obs_len, ...]
        dec_pe = self.pe[self.obs_len-1 : -1, ...]
        
        enc_inputs = torch.cat([enc_inputs, enc_pe.tile(1, enc_inputs.shape[1], 1)], dim=-1)
        
        enc_out = self.transformer.encoder(self.encoder_input(enc_inputs))
        
        sample = dec_inputs
        prob_map = []
        preds = []
        current_dec_inputs = []
        
        for k in range(self.pred_len):
            current_dec_inputs.append(torch.cat([sample, dec_pe[k][None]], dim=-1))
            dec_output = self.transformer.decoder(
                self.decoder_input(torch.cat(current_dec_inputs, dim=0)),
                enc_out)[-1][None]
            sample_num = 10000
            dist_args = self.dist_args_proj(dec_output).expand(sample_num, -1, -1)
        
            pos, log_prob = self.flow.sample_with_log_prob(cond=dist_args)
            pos_log_prob = torch.cat([pos, torch.exp(log_prob)[..., None]], dim=-1)
            prob_map.append(pos_log_prob[None, ...])
            sample = pos[-1:]
            preds.append(sample)
            
        preds = torch.cat(preds, dim=0)
        prob_map = torch.cat(prob_map, dim=0)
        data_dict[("pred", 0)] = preds
        data_dict[("prob", 0)] = prob_map
        return data_dict

    def update(self, data_dict: Dict) -> Dict:
        inputs = torch.cat([data_dict['obs'], data_dict['gt']])

        enc_inputs = inputs[: self.obs_len, ...]
        dec_inputs = inputs[self.obs_len-1 : -1, ...]
        
        _, scale = self.scaler(enc_inputs)
        if self.scaling:
            self.flow.scale = scale
        
        enc_pe = self.pe[: self.obs_len, ...]
        dec_pe = self.pe[self.obs_len-1 : -1, ...]
        
        enc_inputs = torch.cat([enc_inputs, enc_pe.tile(1, enc_inputs.shape[1], 1)], dim=-1)
        dec_inputs = torch.cat([dec_inputs, dec_pe.tile(1, dec_inputs.shape[1], 1)], dim=-1)

        enc_out = self.transformer.encoder(
            self.encoder_input(enc_inputs)
        )
        
        dec_output = self.transformer.decoder(
            self.decoder_input(dec_inputs),
            enc_out,
            tgt_mask=self.tgt_mask,
        )
        
        dist_args = self.dist_args_proj(dec_output)
        
        gt = data_dict['gt']
        if self.dequantize:
            gt += torch.rand_like(data_dict['gt'])
        
        loss = -self.flow.log_prob(gt, dist_args)
        loss = loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.mean().item()}
    
    def save(self, epoch: int = 0, path: Path=None) -> None:
        if path is None:
            path = self.output_path / "ckpt.pt"
            
        ckpt = {
            'epoch': epoch,
            'state': self.state_dict(),
            'optim_state': self.optimizer.state_dict(),
        }

        torch.save(ckpt, path)
        
    def check_saved_path(self, path: Path = None) -> bool:
        if path is None:
            path = self.output_path / "ckpt.pt"        
        
        return path.exists()

    def load(self, path: Path=None) -> int:
        if path is None:
            path = self.output_path / "ckpt.pt"
        
        ckpt = torch.load(path)
        self.load_state_dict(ckpt['state'])

        self.optimizer.load_state_dict(ckpt['optim_state'])
        optimizer_to_cuda(self.optimizer)
    
        return ckpt['epoch']
    

def create_masks(
    input_size, hidden_size, n_hidden, input_order="sequential", input_degrees=None
):
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_order == "sequential":
        degrees += (
            [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        )
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += (
            [torch.arange(input_size) % input_size - 1]
            if input_degrees is None
            else [input_degrees % input_size - 1]
        )

    elif input_order == "random":
        degrees += (
            [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
        )
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += (
            [torch.randint(min_prev_degree, input_size, (input_size,)) - 1]
            if input_degrees is None
            else [input_degrees - 1]
        )

    # construct masks
    masks = []
    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]


class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """

    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians += log_abs_det_jacobian
        return x, torch.sum(sum_log_abs_det_jacobians, dim=-1)

    def inverse(self, u, y):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians += log_abs_det_jacobian
        return u, torch.sum(sum_log_abs_det_jacobians, dim=-1)


class BatchNorm(nn.Module):
    """ RealNVP BatchNorm layer """

    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer("running_mean", torch.zeros(input_size))
        self.register_buffer("running_var", torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            
            self.batch_mean = x.reshape(-1, x.shape[-1]).mean(0)
            # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)
            self.batch_var = x.reshape(-1, x.shape[-1]).var(0)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(
                self.batch_mean.data * (1 - self.momentum)
            )
            self.running_var.mul_(self.momentum).add_(
                self.batch_var.data * (1 - self.momentum)
            )

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta
        
        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        #        print('in sum log var {:6.3f} ; out sum log var {:6.3f}; sum log det {:8.3f}; mean log_gamma {:5.3f}; mean beta {:5.3f}'.format(
        #            (var + self.eps).log().sum().data.numpy(), y.var(0).log().sum().data.numpy(), log_abs_det_jacobian.mean(0).item(), self.log_gamma.mean(), self.beta.mean()))
        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_abs_det_jacobian.expand_as(x)


class LinearMaskedCoupling(nn.Module):
    """ Modified RealNVP Coupling Layers per the MAF paper """

    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None):
        super().__init__()

        self.register_buffer("mask", mask)

        # scale function
        s_net = [
            nn.Linear(
                input_size + (cond_label_size if cond_label_size is not None else 0),
                hidden_size,
            )
        ]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        # translation function
        self.t_net = copy.deepcopy(self.s_net)
        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear):
                self.t_net[i] = nn.ReLU()

    def forward(self, x, y=None):
        # apply mask
        mx = x * self.mask
        
        # run through model
        s = self.s_net(mx if y is None else torch.cat([y, mx], dim=-1))
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=-1)) * (
            1 - self.mask
        )
        
        # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)
        log_s = torch.tanh(s) * (1 - self.mask)
        u = x * torch.exp(log_s) + t
        # u = (x - t) * torch.exp(log_s)
        # u = mx + (1 - self.mask) * (x - t) * torch.exp(-s)

        # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size done at model log_prob
        # log_abs_det_jacobian = -(1 - self.mask) * s
        # log_abs_det_jacobian = -log_s #.sum(-1, keepdim=True)
        log_abs_det_jacobian = log_s

        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        # apply mask
        mu = u * self.mask

        # run through model
        s = self.s_net(mu if y is None else torch.cat([y, mu], dim=-1))
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=-1)) * (
            1 - self.mask
        )
        
        log_s = torch.tanh(s) * (1 - self.mask)
        x = (u - t) * torch.exp(-log_s)
        # x = u * torch.exp(log_s) + t
        # x = mu + (1 - self.mask) * (u * s.exp() + t)  # cf RealNVP eq 7

        # log_abs_det_jacobian = (1 - self.mask) * s  # log det dx/du
        # log_abs_det_jacobian = log_s #.sum(-1, keepdim=True)
        log_abs_det_jacobian = -log_s

        return x, log_abs_det_jacobian


class MaskedLinear(nn.Linear):
    """ MADE building block layer """

    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        super().__init__(input_size, n_outputs)

        self.register_buffer("mask", mask)

        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(
                torch.rand(n_outputs, cond_label_size) / math.sqrt(cond_label_size)
            )

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out


class MADE(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        activation="ReLU",
        input_order="sequential",
        input_degrees=None,
    ):
        """
        Args:
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of MADEs
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(
            input_size, hidden_size, n_hidden, input_order, input_degrees
        )

        # setup activation
        if activation == "ReLU":
            activation_fn = nn.ReLU()
        elif activation == "Tanh":
            activation_fn = nn.Tanh()
        else:
            raise ValueError("Check activation function.")

        # construct model
        self.net_input = MaskedLinear(
            input_size, hidden_size, masks[0], cond_label_size
        )
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [
            activation_fn,
            MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2, 1)),
        ]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        # MAF eq 4 -- return mean and log std
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=-1)
        u = (x - m) * torch.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = -loga
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        # MAF eq 3
        # D = u.shape[-1]
        x = torch.zeros_like(u)
        # run through reverse model
        for i in self.input_degrees:
            m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=-1)
            x[..., i] = u[..., i] * torch.exp(loga[..., i]) + m[..., i]
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=-1)


class Flow(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.__scale = None
        self.net = None

        # base distribution for calculation of log prob under the model
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))

    @property
    def base_dist(self):
        return Normal(self.base_dist_mean, self.base_dist_var)

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale

    def forward(self, x, cond):
        if self.scale is not None:
            x /= self.scale
        u, log_abs_det_jacobian = self.net(x, cond)
        return u, log_abs_det_jacobian

    def inverse(self, u, cond):
        x, log_abs_det_jacobian = self.net.inverse(u, cond)
        if self.scale is not None:
            x *= self.scale
            log_abs_det_jacobian += torch.log(torch.abs(self.scale))
        return x, log_abs_det_jacobian

    def log_prob(self, x, cond):
        u, sum_log_abs_det_jacobians = self.forward(x, cond)
        return torch.sum(self.base_dist.log_prob(u), dim=-1) + sum_log_abs_det_jacobians

    def sample(self, sample_shape=torch.Size(), cond=None):
        if cond is not None:
            shape = cond.shape[:-1]
        else:
            shape = sample_shape

        u = self.base_dist.sample(shape)
        sample, _ = self.inverse(u, cond)
        return sample
    
    def sample_with_log_prob(self, sample_shape=torch.Size(), cond=None):
        if cond is not None:
            shape = cond.shape[:-1]
        else:
            shape = sample_shape

        u = self.base_dist.sample(shape)
        sample, sum_log_abs_det_jacobians = self.inverse(u, cond)
        return sample, torch.sum(self.base_dist.log_prob(u), dim=-1) + sum_log_abs_det_jacobians


class RealNVP(Flow):
    def __init__(
        self,
        n_blocks,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        batch_norm=True,
    ):
        super().__init__(input_size)

        # construct model
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [
                LinearMaskedCoupling(
                    input_size, hidden_size, n_hidden, mask, cond_label_size
                )
            ]
            mask = 1 - mask
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)


class MAF(Flow):
    def __init__(
        self,
        n_blocks,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        activation="ReLU",
        input_order="sequential",
        batch_norm=True,
    ):
        super().__init__(input_size)

        # construct model
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [
                MADE(
                    input_size,
                    hidden_size,
                    n_hidden,
                    cond_label_size,
                    activation,
                    input_order,
                    self.input_degrees,
                )
            ]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)


class Scaler(ABC, nn.Module):
    def __init__(self, keepdim: bool = False):
        super().__init__()
        self.keepdim = keepdim

    @abstractmethod
    def compute_scale(
        self, data: torch.Tensor
    ) -> torch.Tensor:
        pass

    def forward(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        scale = self.compute_scale(data)

        dim = 0
        if self.keepdim:
            scale = scale.unsqueeze(dim=dim)
            return data / scale, scale
        else:
            return data / scale.unsqueeze(dim=dim), scale


class MeanScaler(Scaler):
    """
    The ``MeanScaler`` computes a per-item scale according to the average
    absolute value over time of each item. The average is computed only among
    the observed values in the data tensor, as indicated by the second
    argument. Items with no observed data are assigned a scale based on the
    global average.
    Parameters
    ----------
    minimum_scale
        default scale that is used if the time series has only zeros.
    """

    def __init__(self, minimum_scale: float = 1e-10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("minimum_scale", torch.tensor(minimum_scale))

    def compute_scale(
        self, data: torch.Tensor
    ) -> torch.Tensor:

        dim = 0

        # these will have shape (N, C)
        observed_indicator = torch.ones_like(data)
        num_observed = observed_indicator.sum(dim=dim)
        sum_observed = (data.abs() * observed_indicator).sum(dim=dim)

        # first compute a global scale per-dimension
        total_observed = num_observed.sum(dim=0)
        denominator = torch.max(total_observed, torch.ones_like(total_observed))
        default_scale = sum_observed.sum(dim=0) / denominator

        # then compute a per-item, per-dimension scale
        denominator = torch.max(num_observed, torch.ones_like(num_observed))
        scale = sum_observed / denominator

        # use per-batch scale when no element is observed
        # or when the sequence contains only zeros
        scale = torch.where(
            sum_observed > torch.zeros_like(sum_observed),
            scale,
            default_scale * torch.ones_like(num_observed),
        )

        return torch.max(scale, self.minimum_scale).detach()


class NOPScaler(Scaler):
    """
    The ``NOPScaler`` assigns a scale equals to 1 to each input item, i.e.,
    no scaling is applied upon calling the ``NOPScaler``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_scale(
        self, data: torch.Tensor
    ) -> torch.Tensor:
        dim = 0
        return torch.ones_like(data).mean(dim=dim)
    

if __name__ == "__main__":
    pass
