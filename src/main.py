import os
import argparse
from typing import List, Dict
from yacs.config import CfgNode
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from collections import OrderedDict

from utils import load_config
from data.unified_loader import unified_loader
from models.build_model import Build_Model
from metrics.build_metrics import Build_Metrics
from visualization.build_visualizer import Build_Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pytorch training & testing code for task-agnostic time-series prediction")
    parser.add_argument("--config_file", type=str, default='',
                        metavar="FILE", help='path to config file')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "tune"], default="train")
    parser.add_argument(
        "--visualize", action="store_true", help="flag for whether visualize the results in mode:test")

    return parser.parse_args()


def train(cfg: CfgNode, save_model=True) -> None:
    validation = cfg.SOLVER.VALIDATION and cfg.DATA.TASK != "VP"

    data_loader = unified_loader(cfg, rand=True, split="train")
    if validation:
        val_data_loader = unified_loader(cfg, rand=False, split="val")
        val_loss = np.inf

    start_epoch = 0
    model = Build_Model(cfg)

    if model.check_saved_path():
        # model saved at the end of each epoch. resume training from next epoch
        start_epoch = model.load() + 1
        print('loaded pretrained model')

    if cfg.SOLVER.USE_SCHEDULER:
        schedulers = [torch.optim.lr_scheduler.StepLR(optimizer,
                                                      step_size=int(
                                                          cfg.SOLVER.ITER/10),
                                                      last_epoch=start_epoch-1,
                                                      gamma=0.7) for optimizer in model.optimizers]

    with tqdm(range(start_epoch, cfg.SOLVER.ITER)) as pbar:
        for i in pbar:
            loss_list = []
            for data_dict in data_loader:
                data_dict = {k: data_dict[k].cuda()
                             if isinstance(data_dict[k], torch.Tensor)
                             else data_dict[k]
                             for k in data_dict}

                loss_list.append(model.update(data_dict))

            loss_info = aggregate(loss_list)
            pbar.set_postfix(OrderedDict(loss_info))

            # validation
            if (i+1) % cfg.SOLVER.SAVE_EVERY == 0:
                if validation:
                    curr_val_loss = evaluate_model(
                        cfg, model, val_data_loader)["score"]
                    if curr_val_loss < val_loss:
                        val_loss = curr_val_loss
                        if save_model:
                            model.save(epoch=i)
                else:
                    if save_model:
                        model.save(epoch=i)

        if cfg.SOLVER.USE_SCHEDULER:
            [scheduler.step() for scheduler in schedulers]
    return curr_val_loss


def evaluate_model(cfg: CfgNode, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, visualize=False):
    model.eval()
    metrics = Build_Metrics(cfg)
    visualizer = Build_Visualizer(cfg)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    update_timesteps = [1]

    run_times = {0: []}
    run_times.update({t: [] for t in update_timesteps})

    result_info = {}

    if visualize:
        with torch.no_grad():
            result_list = []
            print("timing the computation, evaluating probability map, and visualizing... ")
            data_loader_one_each = unified_loader(
                cfg, rand=False, split="test", batch_size=1)
            for i, data_dict in enumerate(tqdm(data_loader_one_each, leave=False, total=10)):
                data_dict = {k: data_dict[k].cuda()
                             if isinstance(data_dict[k], torch.Tensor)
                             else data_dict[k]
                             for k in data_dict}
                dict_list = []

                result_dict = model.predict(
                    deepcopy(data_dict), return_prob=True)  # warm-up
                torch.cuda.synchronize()
                starter.record()
                result_dict = model.predict(
                    deepcopy(data_dict), return_prob=True)
                ender.record()
                torch.cuda.synchronize()
                curr_run_time = starter.elapsed_time(ender)
                run_times[0].append(curr_run_time)

                for t in update_timesteps:
                    starter.record()
                    result_dict = model.predict_from_new_obs(result_dict, t)
                    ender.record()
                    torch.cuda.synchronize()
                    curr_run_time = starter.elapsed_time(ender)
                    run_times[t].append(curr_run_time)

                dict_list.append(deepcopy(result_dict))
                dict_list = metrics.denormalize(
                    dict_list)  # denormalize the output
                if cfg.TEST.KDE:
                    torch.cuda.synchronize()
                    starter.record()
                    dict_list = kde(dict_list)
                    ender.record()
                    torch.cuda.synchronize()
                    run_times[0][-1] += starter.elapsed_time(ender)
                dict_list = visualizer.prob_to_grid(dict_list)
                result_list.append(metrics(deepcopy(dict_list)))

                if visualize:
                    visualizer(dict_list)
                if i == 9:
                    break

            result_info.update(aggregate(result_list))
            print(result_info)

        print(f"execution time: {np.mean(run_times[0]):.2f} " +
              u"\u00B1" + f"{np.std(run_times[0]):.2f} [ms]")
        print(f"execution time: {np.mean(run_times[1]):.2f} " +
              u"\u00B1" + f"{np.std(run_times[1]):.2f} [ms]")
        result_info.update({"execution time": np.mean(
            run_times[0]), "time std": np.std(run_times[0])})

    print("evaluating ADE/FDE metrics ...")
    with torch.no_grad():
        result_list = []
        for i, data_dict in enumerate(tqdm(data_loader, leave=False)):
            data_dict = {k: data_dict[k].cuda()
                         if isinstance(data_dict[k], torch.Tensor)
                         else data_dict[k]
                         for k in data_dict}

            dict_list = []
            for _ in range(cfg.TEST.N_TRIAL):
                result_dict = model.predict(
                    deepcopy(data_dict), return_prob=False)
                dict_list.append(deepcopy(result_dict))

            dict_list = metrics.denormalize(dict_list)
            result_list.append(deepcopy(metrics(dict_list)))
        d = aggregate(result_list)
        result_info.update({k: d[k] for k in d.keys() if d[k] != 0.0})

    np.set_printoptions(precision=4)
    print(result_info)

    model.train()

    return result_info


def test(cfg: CfgNode, visualize) -> None:
    data_loader = unified_loader(cfg, rand=False, split="test")
    model = Build_Model(cfg)
    try:
        model.load()
    except FileNotFoundError:
        print("no model saved")
    result_info = evaluate_model(cfg, model, data_loader, visualize)
    import json
    with open(os.path.join(cfg.OUTPUT_DIR, "metrics.json"), "w") as fp:
        json.dump(result_info, fp)


def aggregate(dict_list: List[Dict]) -> Dict:
    if "nsample" in dict_list[0]:
        ret_dict = {k: np.sum([d[k] for d in dict_list], axis=0) / np.sum(
            [d["nsample"] for d in dict_list]) for k in dict_list[0].keys()}
    else:
        ret_dict = {k: np.mean([d[k] for d in dict_list], axis=0)
                    for k in dict_list[0].keys()}

    return ret_dict


def tune(cfg: CfgNode) -> None:
    import optuna

    def objective_with_arg(cfg):
        _cfg = cfg.clone()
        _cfg.defrost()

        def objective(trial):
            _cfg.MODEL.FLOW.N_BLOCKS = trial.suggest_int(
                "MODEL.FLOW.N_BLOCKS", 1, 3)
            _cfg.MODEL.FLOW.N_HIDDEN = trial.suggest_int(
                "MODEL.FLOW.N_HIDDEN", 1, 3)
            _cfg.MODEL.FLOW.HIDDEN_SIZE = trial.suggest_int(
                "MODEL.FLOW.HIDDEN_SIZE", 32, 128, step=16)
            _cfg.MODEL.FLOW.CONDITIONING_LENGTH = trial.suggest_int(
                "MODEL.FLOW.CONDITIONING_LENGTH", 8, 64, step=8)
            _cfg.SOLVER.LR = trial.suggest_float(
                "SOLVER.LR", 1e-6, 1e-3, log=True)
            _cfg.SOLVER.WEIGHT_DECAY = trial.suggest_float(
                "SOLVER.WEIGHT_DECAY", 1e-12, 1e-5, log=True)

            return train(_cfg, save_model=False)

        return objective

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.HyperbandPruner()

    study = optuna.create_study(sampler=sampler, pruner=pruner,
                                direction='minimize',
                                storage=os.path.join(
                                    "sqlite:///", cfg.OUTPUT_DIR, "optuna.db"),
                                study_name='my_opt',
                                load_if_exists=True)
    study.optimize(objective_with_arg(cfg), n_jobs=4,
                   n_trials=200, gc_after_trial=True)

    trial = study.best_trial

    print(trial.value, trial.params)


def kde(dict_list: List):
    from utils import GaussianKDE
    for data_dict in dict_list:
        for k in list(data_dict.keys()):
            if k[0] == "prob":
                prob = data_dict[k]
                batch_size, _, timesteps, _ = prob.shape
                prob_, gt_traj_log_prob = [], []
                for b in range(batch_size):
                    prob__, gt_traj_prob__ = [], []
                    for i in range(timesteps):
                        kernel = GaussianKDE(prob[b, :, i, :-1])
                        # estimate the prob of predicted future positions for fair comparison of inference time
                        kernel(prob[b, :, i, :-1])
                        prob__.append(deepcopy(kernel))
                        gt_traj_prob__.append(
                            kernel(data_dict["gt"][b, None, i].float()))
                    prob_.append(deepcopy(prob__))
                    gt_traj_log_prob.append(
                        torch.cat(gt_traj_prob__, dim=-1).log())
                gt_traj_log_prob = torch.stack(gt_traj_log_prob, dim=0)
                gt_traj_log_prob = torch.nan_to_num(
                    gt_traj_log_prob, neginf=-10000)
                data_dict[k] = prob_
                data_dict[("gt_traj_log_prob", k[1])] = gt_traj_log_prob

    return dict_list


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    if args.mode == "train":
        train(cfg)
    elif args.mode == "test":
        test(cfg, args.visualize)
    elif args.mode == "tune":
        tune(cfg)


if __name__ == "__main__":
    main()
