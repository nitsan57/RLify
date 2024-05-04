import numpy as np
from rlify.agents.agent_utils import ObsWraper
from rlify.agents.heuristic_agent import Heuristic_Agent
import torch
import gymnasium as gym
import pytest
from rlify.agents.tests.test_cases import generate_test_cases


# make pytest config for single_test
@pytest.mark.parametrize(
    "env_name,num_parallel_envs,is_rnn",
    [("CartPole-v1", 4, False)],
)
def test_single(env_name, num_parallel_envs, is_rnn):
    """
    Test DQN agent on a single environment

    Args:
    env_name: str: name of the environment
    num_parallel_envs: int: number of parallel environments
    is_rnn: bool: whether to use RNN or not

    """

    env = gym.make(env_name, render_mode=None)

    def heuristic_func(inner_state, obs: ObsWraper):
        # an function that does not keep inner state
        b_shape = len(obs)
        actions = np.zeros((b_shape, 1))  # single discrete action
        # just a dummy heuristic for a gym env with np.array observations (for more details about the obs object check ObsWraper)
        # the heuristic check whether the first number of each observation is positive, if so, it returns action=1, else 0
        actions[torch.where(obs["data"][:, 0] > 0)[0].cpu()] = 1
        return None, actions

    agent = Heuristic_Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        heuristic_func=heuristic_func,
        tensorboard_dir="/tmp/heuristic_test_tensorboard",
        num_parallel_envs=num_parallel_envs,
    )
    reward = agent.run_env(env, best_act=True)
    agent.save_agent("/tmp/heuristic/heuristic_test.pt")
    agent.load_agent("/tmp/heuristic/heuristic_test.pt")
    assert True


def main():
    test_single("Taxi-v3", 4, False)


if __name__ == "__main__":
    main()
