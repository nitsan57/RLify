import torch
import numpy as np
import torch.nn as nn
from .base_model import AbstractModel

class FC(AbstractModel):
    
    def __init__(self, input_shape, out_shape, embed_dim = 128, repeat=3):
        super().__init__(input_shape, out_shape)

        self.input_size_dict = {k: np.prod(self.input_shape[k]) for k in self.input_shape}
        self.num_inputs = len(self.input_size_dict)

        self.mean_in_params = torch.nn.ParameterDict({k:torch.nn.Parameter(torch.zeros(self.input_shape[k])+1e-2) for k in self.input_shape})
        self.std_in_params = torch.nn.ParameterDict({k:torch.nn.Parameter(torch.ones(self.input_shape[k])) for k in self.input_shape})

        self.l1 = torch.nn.ModuleDict({k:nn.Sequential(nn.Linear(input_size, embed_dim), nn.LeakyReLU()) for k,input_size in self.input_size_dict.items()})
        self.embed_layer = torch.nn.ModuleDict({k:nn.Sequential(*[nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.LeakyReLU()) for i in range(repeat)])  for k in self.input_size_dict})
        self.concat_layer = nn.Sequential(nn.Linear(embed_dim * self.num_inputs, embed_dim), torch.nn.LeakyReLU())
        self.l2 = nn.Linear(embed_dim, out_shape)
    

    def forward(self, x, d=None): # d is for rnn api compability

        res_dict = dict()
        
        for k in x:
            layer = self.l1[k]
            normed = (x[k] - self.mean_in_params[k])/(torch.abs((self.std_in_params[k])) + 1e-3)
            layer_in = torch.flatten(normed, start_dim=1)
            out = layer(layer_in)
            res_dict[k] = out

        for k in x:
            layer = self.embed_layer[k]
            out = layer(res_dict[k])
            res_dict[k] = out


        res = torch.cat(list(res_dict.values()),1)
        res = self.concat_layer(res)
        
        out = self.l2(res)
        return out


    def reset(self):
        pass
