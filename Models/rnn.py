import torch
import torch.nn as nn
import numpy as np
from .base_model import AbstractModel

class GRU(AbstractModel):
    is_rnn = True
    def __init__(self, input_shape, out_shape, hidden_dim = 64, num_grus=2):
        super().__init__(input_shape, out_shape)

        self.input_size_dict = {k: np.prod(self.input_shape[k]) for k in self.input_shape}
        self.num_inputs = len(self.input_size_dict)
        
        self.hidden_dim = hidden_dim
        self.reset() # init hidden state
        self.num_grus = num_grus

        # self.mean_in_params = torch.nn.ParameterDict({k:torch.nn.Parameter(torch.zeros(self.input_shape[k])+1e-2) for k in self.input_shape})
        # self.std_in_params = torch.nn.ParameterDict({k:torch.nn.Parameter(torch.ones(self.input_shape[k])) for k in self.input_shape})

        self.l1 = torch.nn.ModuleDict({k:nn.GRU(int(input_size), hidden_dim, num_layers=num_grus, batch_first=True) for k,input_size in self.input_size_dict.items()})
        self.concat_layer = nn.Sequential(nn.Linear(int(hidden_dim * self.num_inputs), hidden_dim), torch.nn.ReLU())
        self.l2 = nn.Linear(hidden_dim, out_shape)


    def forward(self, x):
        """x is padded pack"""
        temp_k = list(x.keys())[0]
        device = x[temp_k].data.device

        concat_tensor = []

        for k in x:
            layer = self.l1[k]
            layer.flatten_parameters()           
            out,h = layer(x[k], self.hidden_state[k])
            self.hidden_state[k] = h.detach()
            padded_output, output_lens = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            all_outs_len= torch.sum(output_lens)
            relevant_flatten_out = torch.zeros((all_outs_len, *padded_output.shape[2:]), device=device)
            last_idx = 0
            for i, out_len in enumerate(output_lens):
                curr_idx = last_idx+out_len
                relevant_flatten_out[last_idx:curr_idx] = padded_output[i][:out_len]
                last_idx = curr_idx

            concat_tensor.append(relevant_flatten_out)
        concat_tensor = torch.cat(concat_tensor, 1)
        out = self.concat_layer(concat_tensor)
        out = self.l2(out)
        return out


    def reset(self):
        self.hidden_state = {k: None for k in self.input_shape}

