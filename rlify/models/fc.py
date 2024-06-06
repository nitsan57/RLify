import torch
import numpy as np
import torch.nn as nn
from .base_model import BaseModel


class FC(BaseModel):
    """
    A Basic fully connected model
    """

    def __init__(self, embed_dim=64, depth=2, activation=nn.ReLU(), *args, **kwargs):
        """

        Args:
            embed_dim: int: the embedding dimension
            depth: int: the depth of the model
            activation: torch.nn.Module: the activation function
            *args: args: args to pass to the base class
            **kwargs: kwargs: kwargs to pass to the base class


        """
        super().__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.activation = activation

        self.l1 = torch.nn.ModuleDict(
            {
                k: nn.Sequential(nn.Linear(input_size, embed_dim), self.activation)
                for k, input_size in self.input_size_dict.items()
            }
        )
        self.embed_layer = torch.nn.ModuleDict(
            {
                k: nn.Sequential(
                    *[
                        nn.Sequential(nn.Linear(embed_dim, embed_dim), self.activation)
                        for i in range(depth)
                    ]
                )
                for k in self.input_size_dict
            }
        )
        self.concat_layer = nn.Sequential(
            nn.Linear(embed_dim * self.num_inputs, embed_dim), self.activation
        )
        self.out_layer = nn.Linear(embed_dim, np.prod(self.out_shape))

    def forward(self, x, d=None):  # d is for rnn api compability

        res_dict = dict()
        for k in x:
            layer_in = torch.flatten(x[k], start_dim=1)
            out = self.l1[k](layer_in)
            out = self.embed_layer[k](out)
            res_dict[k] = out

        res = torch.cat(list(res_dict.values()), 1)
        res = self.concat_layer(res)
        res = self.out_layer(res)
        return res

    def reset(self):
        pass
