from yacs.config import CfgNode
from typing import Dict, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from utils import optimizer_to_cuda


from models.TP.socialLSTM import (
    init_weights,
    TrajectoryGenerator,
    TrajectoryDiscriminator,
    gan_g_loss,
    gan_d_loss,
    l2_loss
)


class socialGAN(nn.Module):
    def __init__(self, cfg: CfgNode) -> None:
        super(socialGAN, self).__init__()

        self.output_path = Path(cfg.OUTPUT_DIR)

        self.obs_len = cfg.DATA.OBSERVE_LENGTH
        self.pred_len=cfg.DATA.PREDICT_LENGTH
        self.best_k = 20
        self.d_step = 2
        
        self.g = TrajectoryGenerator(
            obs_len=cfg.DATA.OBSERVE_LENGTH,
            pred_len=cfg.DATA.PREDICT_LENGTH,
            embedding_dim=16,
            encoder_h_dim=32,
            decoder_h_dim=32,
            mlp_dim=64,
            num_layers=1,
            noise_dim=(8,),
            noise_type='gaussian',
            noise_mix_type='global',
            pooling_type='none',
            #pooling_type='pool_net',
            pool_every_timestep=False,
            dropout=0,
            bottleneck_dim=0,
            #bottleneck_dim=8,
            neighborhood_size=2.0,
            grid_size=8,
            batch_norm=False
        )

        self.d = TrajectoryDiscriminator(
            obs_len=cfg.DATA.OBSERVE_LENGTH,
            pred_len=cfg.DATA.PREDICT_LENGTH,
            embedding_dim=16,
            #h_dim=48,
            h_dim=64,
            mlp_dim=64,
            num_layers=1,
            dropout=0.0,
            batch_norm=False,
            d_type='global')

        self.g.apply(init_weights).type(torch.FloatTensor).train()
        self.d.apply(init_weights).type(torch.FloatTensor).train()

        self.optimizer_g = optim.Adam(self.g.parameters(), 5e-4)
        self.optimizer_d = optim.Adam(self.d.parameters(), 5e-4)

        self.optimizers = [self.optimizer_g, self.optimizer_d]

        self.g_loss_fn = gan_g_loss
        self.d_loss_fn = gan_d_loss

        self.clipping_threshold_g = 1.5
        self.clipping_threshold_d = 0
        self.l2_loss_weight = 1.0

    def forward(self, data_dict: Dict) -> Tuple[torch.Tensor, Dict]:
        return super().forward(data_dict)

    def predict(self, data_dict: Dict, return_prob=False) -> Dict:
        if return_prob:
            data_dict = self.g.sample(data_dict)
        else:
            data_dict = self.g.predict(data_dict)
        return data_dict

    def predict_from_new_obs(self, data_dict: Dict, time_step: int) -> Dict:
        # TODO: need to implement the density estimation & update
        return data_dict

    def update(self, data_dict: Dict) -> Dict:
        for _ in range(self.d_step):
            data_dict = self.predict(data_dict)
            traj_real = torch.cat([data_dict["obs_st_slstm"].permute(1, 0, 2), data_dict["gt_st_slstm"].permute(1, 0, 2)], dim=0)
            traj_real_rel = torch.cat([data_dict["obs_st_slstm_rel"].permute(1, 0, 2), data_dict["gt_st_slstm_rel"].permute(1, 0, 2)], dim=0)
            traj_fake = torch.cat([data_dict["obs_st_slstm"].permute(1, 0, 2), data_dict[("pred_st_slstm", 0)].permute(1, 0, 2)], dim=0)
            traj_fake_rel = torch.cat([data_dict["obs_st_slstm_rel"].permute(1, 0, 2), data_dict[("pred_st_slstm_rel", 0)].permute(1, 0, 2)], dim=0)

            scores_fake = self.d(traj_fake, traj_fake_rel, data_dict["seq_start_end_slstm"])
            scores_real = self.d(traj_real, traj_real_rel, data_dict["seq_start_end_slstm"])

            # Compute loss with optional gradient penalty
            d_loss = self.d_loss_fn(scores_real, scores_fake)

            self.optimizer_d.zero_grad()
            d_loss.backward()
            if self.clipping_threshold_d > 0:
                nn.utils.clip_grad_norm_(self.d.parameters(), self.clipping_threshold_d)
            self.optimizer_d.step()

        g_loss = 0
        #loss_mask = data_dict["loss_mask"][:, self.obs_len:]
        bs, t, _ = data_dict["gt_st_slstm"].shape
        loss_mask = torch.ones(bs, t).cuda()
        g_l2_loss_rel = []
        for _ in range(self.best_k):
            data_dict = self.predict(data_dict)
            if self.l2_loss_weight > 0:
                g_l2_loss_rel.append(self.l2_loss_weight * l2_loss(
                    data_dict[("pred_st_slstm_rel", 0)],
                    data_dict["gt_st_slstm_rel"],
                    loss_mask,
                    mode='raw'))
        
        if self.l2_loss_weight > 0:
            g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
            g_l2_loss_rel = torch.min(g_l2_loss_rel, dim=1)[0]
            g_loss += g_l2_loss_rel.mean()

        traj_fake = torch.cat([data_dict["obs_st_slstm"].permute(1, 0, 2), data_dict[("pred_st_slstm", 0)].permute(1, 0, 2)], dim=0)
        traj_fake_rel = torch.cat([data_dict["obs_st_slstm_rel"].permute(1, 0, 2), data_dict[("pred_st_slstm_rel", 0)].permute(1, 0, 2)], dim=0)

        scores_fake = self.d(traj_fake, traj_fake_rel, data_dict["seq_start_end_slstm"])
        discriminator_loss = self.g_loss_fn(scores_fake)

        g_loss += discriminator_loss

        self.optimizer_g.zero_grad()
        g_loss.backward()
        if self.clipping_threshold_g > 0:
            nn.utils.clip_grad_norm_(
                self.g.parameters(), self.clipping_threshold_g
            )
        self.optimizer_g.step()

        return {"d_loss": d_loss.item(), "g_loss": g_loss.item()}

    def save(self, epoch: int = 0, path: Path=None) -> None:
        if path is None:
            path = self.output_path / "ckpt.pt"
            
        ckpt = {
            'epoch': epoch,
            'g_state': self.g.state_dict(),
            'd_state': self.d.state_dict(),
            'g_optim_state': self.optimizer_g.state_dict(),
            'd_optim_state': self.optimizer_d.state_dict()
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
        self.g.load_state_dict(ckpt['g_state'])
        try:
            self.d.load_state_dict(ckpt['d_state'])

            self.optimizer_g.load_state_dict(ckpt['g_optim_state'])
            self.optimizer_d.load_state_dict(ckpt['d_optim_state'])

            optimizer_to_cuda(self.optimizer_g)
            optimizer_to_cuda(self.optimizer_d)
            return ckpt["epoch"]
        except KeyError:
            print("skip loading discriminator model weights")

        