from torch.distributions import Normal
import torch
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F


class MCAW():
    def __init__(self, low, high, locs_scales) -> None:
        self.device = locs_scales.device
        num_models = len(low)
        self.models = []
        
        self.out_shape = locs_scales.shape[0], num_models # sample or get locs

        mu_dims = np.arange(num_models)
        sigma_dims = np.arange(num_models, 2*num_models)

        for i in range(num_models):
            model = CAW(low[i], high[i], locs_scales[:, mu_dims[i]], locs_scales[:, sigma_dims[i]])
            self.models.append(model)
            

    def sample(self, sample_shape=torch.Size()):
        res = []
        for i,m in enumerate(self.models):
            res.append(m.sample(sample_shape))
        return torch.stack(res, -1)
    
    
    def log_prob(self, actions):
        res = []
        for i,m in enumerate(self.models):
            res.append(m.log_prob(actions[:, i]))
        return torch.stack(res, -1)
    
    @property
    def loc(self):
        res = []
        for i,m in enumerate(self.models):
            res.append(m.loc)
        return torch.stack(res, -1)


    @property
    def scale(self):
        res = []
        for i,m in enumerate(self.models):
            res.append(m.scale)
            self.res[i] = m.scale
        return torch.cat(res).reshape(self.out_shape)
    


    def entropy(self):
        # res = torch.zeros(self.out_shape)
        m_e = 0 
        for i,m in enumerate(self.models):
            m_e = m.entropy() + m_e
        return m_e / len(self.models)

        
class CAW(Normal):
    def __init__(self, low, high, loc, scale) -> None:
        """

        """
        low, high = float(low), float(high)
        self.low = low
        self.high = high
        coeff = (high - low) / 2
        bias = low + coeff

        loc = torch.tanh(loc)*coeff+bias
        scale = torch.nn.functional.sigmoid(scale)
        # scale = torch.clamp(scale, 1e-4, 1)
        # if scale.shape[0] > 10:
        #     print(np.round(scale[0].item(), 3), np.round(loc[0].item(), 3))
        
        super().__init__(loc, scale)
        # self.sample_activation = torch.nn.Identity() #torch.nn.Tanh() #torch.nn.Sigmoid()


    def sample(self, sample_shape=torch.Size()):
        sample = super().sample(sample_shape)
        # self.sample_activation(sample)
        # print(sample[0], torch.clamp(sample, self.low, self.high)[0])
        return torch.clamp(sample, self.low, self.high)
    
    # @property
    # def loc(self):
    #     return self.sample_activation(self.loc)
    # def get_loc(self):
    #     return self.sample_activation(self.loc)


class MDA():
    """
    Multivariate Discrete Action Space
    """
    def __init__(self, start: np.array, possible_actions: int, n_actions: np.array, x : torch.tensor) -> None:
        """
        Args:
            start (np.array): an offset for start of each action
            possible_actions (int): number of possible actions
            n_actions (np.array): number of actions for each action
            x (torch.tensor): the logits for each action
        """

        self.device = x.device
        self.start = start
        self.possible_actions = possible_actions
        num_models = n_actions
        self.models = []
        self.out_shape = x.shape[0], num_models # sample or get locs

        f_i = 0
        for i in range(num_models):
            model = Categorical(logits=F.log_softmax(x[:, f_i: i+self.possible_actions], dim=1)) #(low[i], high[i], x[:, mu_dims[i]], x[:, sigma_dims[i]])
            f_i = i + self.possible_actions
            self.models.append(model)
            

    def sample(self, sample_shape=torch.Size()):
        res = []
        for i,m in enumerate(self.models):
            res.append(m.sample(sample_shape))
        return torch.stack(res, -1)
    
    
    def log_prob(self, actions):
        res = []
        for i,m in enumerate(self.models):
            res.append(m.log_prob(actions[:, i]))
        return torch.stack(res, -1)
    
    @property
    def probs(self):
        res = []
        for i,m in enumerate(self.models):
            res.append(m.probs)
        return torch.stack(res, -1)
    
    
    def entropy(self):
        m_e = 0 
        for i,m in enumerate(self.models):
            m_e = m.entropy() + m_e
        return m_e / len(self.models)