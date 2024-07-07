import torch
import torch.nn as nn
import numpy as np

from rlify.agents.agent_utils import ObsWrapper
from .base_model import BaseModel


def pack_from_done_indices(data, dones):
    """
    Packs the data from the done indices to torch.nn.utils.rnn.PackedSequence

    """
    done_indices = torch.where(dones == True)[0].cpu().numpy().astype(np.int32)
    sorted_seq_lens, seq_indices = get_seqs_indices_for_pack(
        done_indices
    )  # sorted_data_sub_indices
    rev_indices = seq_indices.argsort()
    assert np.all(np.sort(sorted_seq_lens, kind="stable")[::-1] == sorted_seq_lens)
    b = done_indices
    max_colected_len = np.max(sorted_seq_lens)

    packed_obs = ObsWrapper(tensors=True)
    for k in data:
        obs_shape = data[k][-1].shape
        temp = []
        curr_idx = 0
        for i, d_i in enumerate(done_indices):
            temp.append(data[k][curr_idx : d_i + 1])
            curr_idx = d_i + 1  #

        temp = [temp[i] for i in seq_indices]
        max_new_seq_len = max_colected_len
        new_lens = sorted_seq_lens

        padded_seq_batch = torch.nn.utils.rnn.pad_sequence(temp, batch_first=True)
        b = padded_seq_batch.shape[0]
        padded_seq_batch = padded_seq_batch.reshape(
            (b, max_new_seq_len, np.prod(obs_shape))
        )
        pakced_states = torch.nn.utils.rnn.pack_padded_sequence(
            padded_seq_batch, lengths=np.array(new_lens), batch_first=True
        )
        packed_obs[k] = pakced_states
    return packed_obs, rev_indices


def get_seqs_indices_for_pack(done_indices):
    """
    calculates the sequence indices for packing the data
    returns seq_lens, sorted_data_sub_indices
    """
    env_indices = np.zeros_like(done_indices, dtype=np.int32)
    env_indices[0] = -1
    env_indices[1:] = done_indices[:-1]
    all_lens = done_indices - env_indices
    seq_indices = np.argsort(all_lens, kind="stable")[::-1]
    seq_lens = all_lens[seq_indices]
    return seq_lens, seq_indices  # , sorted_data_sub_indices


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

    def forward(self, x: torch.nn.utils.rnn.PackedSequence):
        concat_tensor = []

        for k in self.l1.keys():
            self.l1[k].flatten_parameters()
            out, h = self.l1[k](x[k], self.hidden_state[k])
            self.hidden_state[k] = h.detach()
            # padded_output, output_lens = torch.nn.utils.rnn.pad_packed_sequence(
            #     out, batch_first=True
            # )
            # padded_output = padded_output[rev_indices]
            # output_lens = output_lens[rev_indices]
            # all_outs_len = torch.sum(output_lens)
            # relevant_flatten_out = torch.zeros(
            #     (all_outs_len, *padded_output.shape[2:]), device=device
            # )
            # last_idx = 0
            # for i, out_len in enumerate(output_lens):
            #     curr_idx = last_idx + out_len
            #     relevant_flatten_out[last_idx:curr_idx] = padded_output[i][:out_len]
            #     last_idx = curr_idx
            concat_tensor.append(out)
        concat_tensor = torch.cat(concat_tensor, -1)
        out = self.concat_layer(concat_tensor)
        out = self.out_layer(out)
        return out

    def reset(self):
        self.hidden_state = {k: None for k in self.input_shape}
