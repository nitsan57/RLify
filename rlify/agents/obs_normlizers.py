from abc import ABC

import torch

from rlify.agents.agent_utils import ObsWrapper


class ObsNormlizer(ABC, torch.nn.Module):

    def forward(self, obs: ObsWrapper):
        """
        Normalizes the observations.
        Args:
            obs: The observations to normalize.
        """
        raise NotImplementedError

    def denormalize(self, obs: ObsWrapper):
        """
        Denormalizes the observations.
        Args:
            obs: The observations to denormalize.
        """
        raise NotImplementedError


class ZObsNormlizer(ObsNormlizer):
    """
    This class is used to normalize and denormalize observations using the z-score.
    """

    def __init__(self, mean: ObsWrapper | dict, std: ObsWrapper | dict):
        """
        Args:
            mean: The mean of the observations.
            std: The standard deviation of the observations.
        """
        self.mean = ObsWrapper(mean)
        self.std = ObsWrapper(std)

    def forward(self, obs: ObsWrapper):
        """
        Normalizes the observations.
        Args:
            obs: The observations to normalize.
        """
        return (obs - self.mean) / (self.std + 1e-6)

    def denormalize(self, obs: ObsWrapper):
        """
        Denormalizes the observations.
        Args:
            obs: The observations to denormalize.
        """
        return obs * (self.std + 1e-6) + self.mean


class AutoNormlizer(ObsNormlizer):
    """
    This class is used to normalize and denormalize observations using the z-score.
    """

    def __init__(
        self,
        input_shape: dict[str:int],
        normilize_type: str = "z-score",
        rnn_obs: bool = False,
    ):
        """
        Args:
            input_shape: The shape of the input observations.
            normilize_type: The type of normalization to use. Support [z-score, min-max]
        """
        super().__init__()
        assert normilize_type in [
            "z-score",
            "min-max",
        ], "normilize_type should be in ['z-score', 'min-max']"
        self.normilize_type = normilize_type
        self.rnn_obs = rnn_obs
        self.param1 = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(torch.zeros(input_shape[k])) for k in input_shape}
        )
        self.param2 = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(torch.ones(input_shape[k])) for k in input_shape}
        )
        self.eps = 1e-6
        self.inited = False
        self.i = 0

    def forward(self, obs: ObsWrapper):
        """
        Normalizes the observations.
        Args:
            obs: The observations to normalize.
        """

        if self.inited:
            # if self.i % 100 == 0:
            #     print("AutoNormlizer params", self.param1["data"], self.param2["data"])
            #     print("obs", obs["data"][0])
            #     print("norm", ((obs - self.param1) * (self.param2))["data"][0])
            #     self.i = 0
            # self.i += 1
            return (obs - self.param1) / (self.param2)
        else:
            for k in self.param1.keys():
                if self.normilize_type == "z-score":
                    if self.rnn_obs:
                        self.param1[k].data = obs[k].flatten(0, 1).mean(0)
                        self.param2[k].data = 1 / (obs[k].flatten(0, 1).std(0) + 1e-1)
                    else:
                        self.param1[k].data = obs[k].mean(0)
                        self.param2[k].data = 1 / (obs[k].std(0) + 1e-1)
                elif self.normilize_type == "min-max":
                    if self.rnn_obs:
                        self.param2[k].data = 1 / (
                            obs[k].flatten(0, 1).max((0))[0].abs() + 1e-1
                        )
                        self.param1[k].data = torch.zeros_like(self.param2[k].data)
                    else:
                        self.param2[k].data = 1 / (obs[k].max(0)[0].abs() + 1e-1)
                        self.param1[k].data = torch.zeros_like(self.param2[k].data)

            self.inited = True
            return (obs - self.param1) / (self.param2)

    def denormalize(self, obs: ObsWrapper):
        """
        Denormalizes the observations.
        Args:
            obs: The observations to denormalize.
        """
        return obs * (torch.abs(self.param2)) + self.param1
