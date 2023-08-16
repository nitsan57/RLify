import torch
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def init_torch(device='cuda'):
    if device == 'cpu':
        return device
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def plot_res(vec, title, smooth_kernel=20, render_as="notebook"):
    fig = px.line(x=range(len(np.convolve(vec, np.ones(smooth_kernel)/smooth_kernel,mode='valid'))), y=np.convolve(vec, np.ones(smooth_kernel)/smooth_kernel, mode='valid'), title=title)
    # fig.show(render_mode='webgl')
    fig.show(render_as, render_mode='webgl')


def plot_log_loss(path, smooth_kernel=(7)):
    with open(path) as f:
        lines = f.readlines()
    data = {'y_train':[], 'y_val': []}
    for l in lines:
        if "axe" not in l.lower():
            if 'train' in l.lower():
                number = float(l.split(":")[-1])
                data['y_train'].append(number)
            if 'val' in l.lower():
                number = float(l.split(":")[-1])
                data['y_val'].append(number)
    data['y_val'] =  y=np.convolve(data['y_val'], np.ones(smooth_kernel)/smooth_kernel, mode='valid')
    data['y_train'] =  y=np.convolve(data['y_train'], np.ones(smooth_kernel)/smooth_kernel, mode='valid')
    # data['x'] = len(data['y_val'])
    pd.DataFrame(data).plot().show()