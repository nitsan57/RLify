import torch
import torch.nn as nn
import numpy as np
from .base_model import BaseModel


class ReccurentLayer(BaseModel):
    """
    Base class for RNNs
    """

    is_rnn = True

    def __init__(self, input_shape, out_shape):
        super().__init__(input_shape, out_shape)

    def forward(self, x):
        """
        Forward pass of the model

        Args:
            x: PackedSequence: the input data
        """
        super().forward(x)

    def reset(self):
        raise NotImplementedError()


class GRU(ReccurentLayer):
    """
    GRU model
    """

    def __init__(self, hidden_dim=64, num_grus=2, *args, **kwargs):
        """
        Args:
            hidden_dim: int: the hidden dimension
            num_grus: int: the number of GRUs
            *args: args: args to pass to the base class
            **kwargs: kwargs: kwargs to pass to the base
        """
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_grus = num_grus
        self.input_size_dict = {
            k: np.prod(self.input_shape[k]) for k in self.input_shape
        }
        self.num_inputs = len(self.input_size_dict)
        self.hidden_dim = hidden_dim
        self.reset()  # init hidden state
        self.num_grus = num_grus

        self.l1 = torch.nn.ModuleDict(
            {
                k: nn.GRU(
                    int(input_size), hidden_dim, num_layers=num_grus, batch_first=True
                )
                for k, input_size in self.input_size_dict.items()
            }
        )
        self.concat_layer = nn.Sequential(
            nn.Linear(int(hidden_dim * self.num_inputs), hidden_dim), torch.nn.ReLU()
        )
        self.out_layer = nn.Linear(hidden_dim, np.prod(self.out_shape))

    def forward(self, x: torch.tensor):
        concat_tensor = []

        for k in self.l1.keys():
            self.l1[k].flatten_parameters()
            out, h = self.l1[k](x[k], self.hidden_state[k])
            self.hidden_state[k] = h.detach()
            concat_tensor.append(out)
        concat_tensor = torch.cat(concat_tensor, -1)
        out = self.concat_layer(concat_tensor)
        out = self.out_layer(out)
        return out.reshape(-1, *self.out_shape)

    def reset(self):
        self.hidden_state = {k: None for k in self.input_shape}
