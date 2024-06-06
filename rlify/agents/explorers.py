import numpy as np
from abc import ABC, abstractmethod


class Explorer(ABC):
    """Abstrcat Exploration Class"""

    def __init__(self) -> None:
        super().__init__()
        self.inner_state = None

    @abstractmethod
    def explore(self):
        """
        Returns True if it is an exploration action time step
        """
        raise NotImplementedError

    @abstractmethod
    def update(self):
        """
        updates the exploration epsilon
        """
        raise NotImplementedError

    @abstractmethod
    def act(self, action_space, obs, num_obs):
        """
        Responsible for storing an inner state if needed(in self.inner_state attr)
        Returns the action to be taken
        """
        raise NotImplementedError


class RandomExplorer(Explorer):
    """Class that acts a linear exploration method"""

    def __init__(
        self, exploration_epsilon: int = 1, eps_end: float = 0.05, eps_dec: float = 1e-2
    ) -> None:
        """
        Args:
            exploration_epsilon: The initial exploration epsilon
            eps_end: The final exploration epsilon
            eps_dec: The decay rate of the exploration epsilon
        """
        self.exploration_epsilon = exploration_epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec

    def explore(self):
        """
        Returns True if it is an exploration action time step (randomness based on the exploration epsilon)
        """
        if np.random.random() < self.exploration_epsilon:
            return True
        return False

    def update(self):
        """
        updates the exploration epsilon in linear mode: exploration_epsilon * (1-self.eps_dec)
        """
        self.exploration_epsilon = (
            self.exploration_epsilon * (1 - self.eps_dec)
            if self.exploration_epsilon > self.eps_end
            else self.eps_end
        )

    def act(self, action_space, obs, num_obs):
        """
        Args:
            action_space: The action space of the env
            obs: The observation of the env
            num_obs: The number of observations to act on
        Reutns a random action from the action space
        """
        if np.issubdtype(action_space.dtype, np.integer):
            return self._act_discrete(action_space, obs, num_obs)
        else:
            return self._act_cont(action_space, obs, num_obs)

    def _act_discrete(self, action_space, obs, num_obs):
        return np.random.choice(action_space.n, num_obs)[..., np.newaxis]

    def _act_cont(self, action_space, obs, num_obs):
        res = []
        for i, low in enumerate(action_space.low):
            high = action_space.high[i]
            res.append(np.random.uniform(low, high, num_obs))
        return np.stack(res, -1)


class HeuristicExplorer(RandomExplorer):
    """
    A class for custom exploration methods- defined by user in init heuristic_function(inner_state, obs) - > action
    """

    def __init__(
        self, heuristic_function, exploration_epsilon=1, eps_end=0.05, eps_dec=1e-2
    ) -> None:
        """
        Args:
            heuristic_function: A function that takes in the inner_state, observation (ObsWrapper) and returns a tuple: (inner_state, action) the inner state (could be None) and the action to be taken,
                            please notice that the actions shape is b,n_actions,action_dim
        """
        super().__init__(exploration_epsilon, eps_end, eps_dec)
        self.inner_state = None
        self.heuristic_function = heuristic_function

    def explore(self):
        if np.random.random() < self.exploration_epsilon:
            return True
        return False

    def update(self):
        self.exploration_epsilon = (
            self.exploration_epsilon * (1 - self.eps_dec)
            if self.exploration_epsilon > self.eps_end
            else self.eps_end
        )

    def act(self, action_space, obs, num_obs):
        """
        Call the heuristic function to get the action, also updates the inner state
        """
        res = self.heuristic_function(self.inner_state, obs)
        self.inner_state = res[0]
        return res[1]
