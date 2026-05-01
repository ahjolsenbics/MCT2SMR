import os
from pathlib import Path
import torch


def to_one_hot_3d(tensor, out_channels, device):  # shape = [batch, s, h, w]
    # n, d, h, w = tensor.size()
    # one_hot = torch.zeros(n, out_channels, d, h, w, device=device).scatter_(1, tensor.view(n, 1, d, h, w), 1)

    b, n, d, h, w = tensor.size()
    one_hot = torch.zeros(b, out_channels, d, h, w, device=device).scatter_(1, tensor.view(b, 1, d, h, w), 1)

    return one_hot


def create_paths(result_base_path):
    result_path = create_experiment_name(result_base_path=result_base_path, mode="number")
    for p in ["log", "token", "val", "checkpoints", "test", "visual_extractor", "meta_models"]:
        sub_path = os.path.join(result_path, p)
        Path(sub_path).mkdir(parents=True, exist_ok=True)
    return result_path


def create_experiment_name(result_base_path, mode="number"):
    for i in range(1, 999):
        result_path = os.path.join(result_base_path, f"exp_{i}")
        if not os.path.exists(result_path):
            Path(result_path).mkdir(parents=True, exist_ok=True)
            return result_path


def create_run_name(result_base_path, mode="number"):
    for i in range(1, 999):
        result_path = os.path.join(result_base_path, f"run_{i}")
        if not os.path.exists(result_path):
            Path(result_path).mkdir(parents=True, exist_ok=True)
            return result_path


def show_args(opt, logger):
    dict = vars(opt)
    for key, value in dict.items():
        logger.info(f"{key}: {value}")