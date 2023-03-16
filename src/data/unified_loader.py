# I created this data loader by refering following great reseaches & github repos.
# VP: stochastic video generation https://github.com/edenton/svg
# MP: On human motion prediction using recurrent neural network https://github.com/wei-mao-2019/LearnTrajDep
#     Trajectron++ https://github.com/StanfordASL/Trajectron-plus-plus
#     Motion Indeterminacy Diffusion https://github.com/gutianpei/mid
# TP: Social GAN https://github.com/agrimgupta92/sgan
from pathlib import Path
import dill
from yacs.config import CfgNode
import torch
from torch.utils.data import DataLoader


def unified_loader(cfg: CfgNode, rand=True, split="train", batch_size=None) -> DataLoader:
    # train, val, test
    if cfg.DATA.TASK == "TP":
        from .TP.trajectron_dataset import EnvironmentDataset, hypers
        
        if 'longer' in cfg.DATA.DATASET_NAME and split != "train":
            i = int(cfg.DATA.DATASET_NAME[-1])
            cfg.defrost()
            cfg.DATA.OBSERVE_LENGTH -= i
            cfg.DATA.DATASET_NAME = cfg.DATA.DATASET_NAME[:-8]
            cfg.freeze()
            
        if cfg.DATA.DATASET_NAME == 'sdd' and split != 'train':
            i = cfg.DATA.PREDICT_LENGTH - 12
            cfg.defrost()
            cfg.DATA.OBSERVE_LENGTH -= i
            cfg.freeze()
            
        
        if cfg.DATA.DATASET_NAME == "sdd" and split == "val":
            # previous methods use the test split for validation
            env_path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / 'processed_data' / f"{cfg.DATA.DATASET_NAME}_test.pkl"
        else:
            env_path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / 'processed_data' / f"{cfg.DATA.DATASET_NAME}_{split}.pkl"
            
        with open(env_path, 'rb') as f:
            env = dill.load(f, encoding='latin1')

        dataset = EnvironmentDataset(env,
                                    state=hypers[cfg.DATA.TP.STATE],
                                    pred_state=hypers[cfg.DATA.TP.PRED_STATE],
                                    node_freq_mult=hypers['scene_freq_mult_train'],
                                    scene_freq_mult=hypers['node_freq_mult_train'],
                                    hyperparams=hypers,
                                    min_history_timesteps=1 if cfg.DATA.TP.ACCEPT_NAN and split == 'train' else cfg.DATA.OBSERVE_LENGTH - 1,
                                    min_future_timesteps=cfg.DATA.PREDICT_LENGTH,
                                    #augment=hypers['augment'] and split == 'train'
                                    )
        
        # assume we have only 'PEDESTRIAN' node
        for node_type_dataset in dataset:
            if node_type_dataset.node_type == 'PEDESTRIAN':
                dataset = node_type_dataset
                break                                         
        
    # train, val, test
    elif cfg.DATA.TASK == "MP":
        if cfg.DATA.DATASET_NAME == "h36motion":
            import os
            from data.MP.h36motion import H36motion
            dataset_train = H36motion(
                path_to_data=os.path.join(cfg.DATA.PATH, cfg.DATA.TASK, "h3.6m", "dataset"),
                actions="all",
                input_n=cfg.DATA.OBSERVE_LENGTH,
                output_n=cfg.DATA.PREDICT_LENGTH,
                split=0,
                load_3d=False)

            if split == "train":
                dataset = dataset_train
            elif split == "val":
                dataset_val = H36motion(
                    path_to_data=os.path.join(cfg.DATA.PATH, cfg.DATA.TASK, "h3.6m", "dataset"),
                    actions="smoking",
                    input_n=cfg.DATA.OBSERVE_LENGTH,
                    output_n=cfg.DATA.PREDICT_LENGTH,
                    split=2,
                    data_mean=dataset_train.data_mean,
                    data_std=dataset_train.data_std,
                    onehotencoder=dataset_train.onehotencoder,
                    load_3d=False)

                dataset = dataset_val
            elif split == "test":
                dataset_test = H36motion(
                    path_to_data=os.path.join(cfg.DATA.PATH, cfg.DATA.TASK, "h3.6m", "dataset"),
                    actions="smoking",
                    input_n=cfg.DATA.OBSERVE_LENGTH,
                    output_n=cfg.DATA.PREDICT_LENGTH,
                    split=1,
                    data_mean=dataset_train.data_mean,
                    data_std=dataset_train.data_std,
                    onehotencoder=dataset_train.onehotencoder,
                    load_3d=False)

                dataset = dataset_test
            
    # only train, test
    elif cfg.DATA.TASK == "VP":
        from data.VP.datasets_factory import VP_dataset
        if cfg.DATA.DATASET_NAME == "bair":
            path = Path(cfg.DATA.PATH) / cfg.DATA.TASK
            img_width = 64
        elif cfg.DATA.DATASET_NAME == "kth":
            path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / "kth_action"
            img_width = 128
        elif cfg.DATA.DATASET_NAME == "mnist":
            path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / "moving-mnist-example" / f"moving-mnist-{split}.npz"
            img_width = 64
        dataset = VP_dataset(dataset_name=cfg.DATA.DATASET_NAME,
                            data_path=path,
                            split=split,
                            img_width=img_width,
                            input_n=cfg.DATA.OBSERVE_LENGTH,
                            output_n=cfg.DATA.PREDICT_LENGTH,
                            injection_action="concat")

    if cfg.DATA.TASK == "TP":
        #from data.TP.trajectories import seq_collate
        from .TP.preprocessing import dict_collate as seq_collate
    elif cfg.DATA.TASK == "VP":
        from .VP.mnist import seq_collate
    elif cfg.DATA.TASK == "MP":
        from .MP.h36motion import seq_collate
        
    if batch_size is None:
        batch_size = cfg.DATA.BATCH_SIZE
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=rand,
        num_workers=cfg.DATA.NUM_WORKERS,
        collate_fn=seq_collate,
        drop_last=True if split == 'train' else False,
        pin_memory=True)
    
    return loader


