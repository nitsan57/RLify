from rlify.agents.explorers import RandomExplorer
from rlify.models import fc, rnn
from rlify.agents.vdqn_agent import VDQN_Agent
import torch
import gymnasium as gym
import pytest
from rlify.agents.tests.test_cases import generate_test_cases


# make pytest config for single_test
@pytest.mark.parametrize(
    "env_name,num_parallel_envs,is_rnn",
    generate_test_cases(continuous_envs=False),
)
def test_single(env_name, num_parallel_envs, is_rnn):
    """
    Test VDQN agent on a single environment

    Args:
    env_name: str: name of the environment
    num_parallel_envs: int: number of parallel environments
    is_rnn: bool: whether to use RNN or not

    """
    env = gym.make(env_name)
    models_shapes = VDQN_Agent.get_models_input_output_shape(
        env.observation_space, env.action_space
    )
    Q_input_shape = models_shapes["Q_model"]["input_shape"]
    Q_out_shape = models_shapes["Q_model"]["out_shape"]

    if is_rnn:
        Q_model = rnn.GRU(
            input_shape=Q_input_shape,
            hidden_dim=64,
            num_grus=2,
            out_shape=Q_out_shape,
        )
    else:
        Q_model = fc.FC(
            input_shape=Q_input_shape,
            embed_dim=64,
            depth=2,
            activation=torch.nn.ReLU(),
            out_shape=Q_out_shape,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = VDQN_Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        batch_size=1024,
        max_mem_size=10**6,
        num_parallel_envs=num_parallel_envs,
        lr=3e-4,
        Q_model=Q_model,
        discount_factor=0.99,
        explorer=RandomExplorer(1, 0.05, 0.01),
        num_epochs_per_update=20,
        tensorboard_dir="/tmp/vdqn_test_tensorboard/",
    )
    train_stats = agent.train_n_steps(env=env, n_steps=800)
    reward = agent.run_env(env, best_act=True)
    agent.save_agent("/tmp/vdqn/vdqn_test.pt")
    agent.load_agent("/tmp/vdqn/vdqn_test.pt")
    assert True


def main():
    test_single("Taxi-v3", 4, False)


if __name__ == "__main__":
    main()
