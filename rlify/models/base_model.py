import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np
from rlify.agents.agent_utils import ObsShapeWraper

from .model_factory import models_db

class AbstractModel(torch.nn.Module, ABC):
    is_rnn=False

    def __init__(self, input_shape, out_shape):
        super().__init__()
    
        self.input_shape = ObsShapeWraper(input_shape)


        self.out_shape = np.array(out_shape)
        
        if len(self.out_shape.shape) == 0:
            self.out_shape = self.out_shape.reshape((1,))


    def get_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

    # def get_output_dim(self):
    #     return self.out_shape
    
    def __init_subclass__(cls):
        models_db.register(cls)

        
    @abstractmethod
    def forward(self, x, dones = None):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """if not rnn - impl can be just: pass, otherwise resets hidden state"""
        raise NotImplementedError

