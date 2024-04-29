import torch
import pandas as pd
import numpy as np
import os, sys


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def init_torch(device="cuda"):
    """
    Initializes torch device

    Args:
    device (str): device to use

    Returns:
    torch.device: torch device
    """
    if device == "cpu":
        return device
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def plot_res(
    train_stats: dict, title: str, smooth_kernel: int = 20, render_as: str = "notebook"
):
    """
    Plots the training stats

    Args:
    train_stats (dict): training stats
    title (str): title for the plot
    smooth_kernel (int): kernel size for smoothing
    render_as (str): render mode
    """
    kernel = np.ones(smooth_kernel) / smooth_kernel
    r_vec = train_stats["rewards"]
    eps = train_stats["exploration_eps"] * 100
    r_vec = np.convolve(r_vec, kernel, mode="valid")
    eps = np.convolve(eps, kernel, mode="valid")
    x = range(len(r_vec))
    fig = pd.DataFrame(index=x, data={"rewards": r_vec, "exploration_eps": eps}).plot(
        backend="plotly",
        title=title,
        labels={"index": "episode_num", "exploration_eps": "exploration_eps %"},
    )
    fig.show(render_as, render_mode="webgl")
