import numpy as np
from abc import ABC, abstractclassmethod

class Explorer(ABC):
    
    """Class that acts a linear exploration method"""
    def __init__(self) -> None:
        super().__init__()

    @abstractclassmethod
    def explore(self):
        raise NotImplementedError

    @abstractclassmethod
    def update(self):
        raise NotImplementedError

    @abstractclassmethod
    def act(self, action_space, obs, num_obs):
        raise NotImplementedError


class RandomExplorer(Explorer):
    """Class that acts a linear exploration method"""
    def __init__(self, exploration_epsilon=1, eps_end=0.05, eps_dec=1e-2) -> None:
        self.exploration_epsilon = exploration_epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec


    def explore(self):
        if np.random.random() < self.exploration_epsilon:
            return True
        return False

    def update(self):
        self.exploration_epsilon = self.exploration_epsilon * (1-self.eps_dec) if self.exploration_epsilon > self.eps_end else self.eps_end

    def act(self, action_space, obs, num_obs):
        if np.issubdtype(action_space.dtype, np.integer):
            return self.act_discrete(action_space, obs, num_obs)
        else:
            return self.act_cont(action_space, obs, num_obs)

    def act_discrete(self, action_space, obs, num_obs):
        return np.random.choice(action_space.n, num_obs)
    
    def act_cont(self, action_space, obs, num_obs):
        res = []
        for i,low in enumerate(action_space.low):
            high = action_space.high[i]
            res.append(np.random.uniform(low, high, num_obs))

        return np.stack(res, -1)
    


class HeuristicExplorer(RandomExplorer):
    """
    A class for custom exploration methods- defined by user in init heuristic_function(obs) - > action
    """
    def __init__(self,heuristic_function, exploration_epsilon=1, eps_end=0.05, eps_dec=1e-2) -> None:
        super().__init__(exploration_epsilon, eps_end, eps_dec)
        self.heuristic_function = heuristic_function

    def explore(self):
        """
        Check if it as exploration action or not
        """
        if np.random.random() < self.exploration_epsilon:
            return True
        return False

    def update(self):
        """
        Update the exploration epsilon
        """
        self.exploration_epsilon = self.exploration_epsilon * (1-self.eps_dec) if self.exploration_epsilon > self.eps_end else self.eps_end

    def act(self, action_space, obs, num_obs):
        """
        Call the heuristic function to get the action
        """
        return self.heuristic_function(obs)