import torch
from abc import ABC, abstractmethod
import numpy as np
from rlify.agents.agent_utils import ObsShapeWraper


class BaseModel(torch.nn.Module, ABC):
    """
    Base class for all NN models
    """

    is_rnn = False

    def __init__(self, input_shape, out_shape: tuple):
        """
        Args:
            input_shape (tuple): input shape of the model
            out_shape (tuple): output shape of the model
        """
        super().__init__()
        self.input_shape = ObsShapeWraper(input_shape)

        self.input_size_dict = {
            k: np.prod(self.input_shape[k]) for k in self.input_shape
        }
        self.num_inputs = len(self.input_size_dict)
        self.out_shape = out_shape

    def get_total_params(self):
        """
        Returns the total number of parameters in the model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @abstractmethod
    def forward(self, x, dones=None):
        """
        Forward pass of the model
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """resets hidden state, if not rnn - can be just: pass"""
        raise NotImplementedError

    @property
    def device(self):
        """
        Returns the device of the model
        """
        return next(self.parameters()).device
