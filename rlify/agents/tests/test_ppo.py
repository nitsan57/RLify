from rlify.models import fc, rnn
from rlify.agents.ppo_agent import PPO_Agent
import torch
import gymnasium as gym
import pytest
from rlify.agents.tests.test_cases import generate_test_cases


# make pytest config for single_test
@pytest.mark.parametrize(
    "env_name,num_parallel_envs,is_rnn",
    generate_test_cases(),
)
def test_single(env_name, num_parallel_envs, is_rnn):
    """
    Test PPO agent on a single environment

    Args:
    env_name: str: name of the environment
    num_parallel_envs: int: number of parallel environments
    is_rnn: bool: whether to use RNN or not

    """
    env = gym.make(env_name)
    models_shapes = PPO_Agent.get_models_input_output_shape(
        env.observation_space, env.action_space
    )
    policy_input_shape = models_shapes["policy_nn"]["input_shape"]
    policy_out_shape = models_shapes["policy_nn"]["out_shape"]
    critic_input_shape = models_shapes["critic_nn"]["input_shape"]
    critic_out_shape = models_shapes["critic_nn"]["out_shape"]

    if is_rnn:
        policy_nn = rnn.GRU(
            input_shape=policy_input_shape,
            hidden_dim=64,
            num_grus=2,
            out_shape=policy_out_shape,
        )
        critic_nn = rnn.GRU(
            input_shape=critic_input_shape,
            hidden_dim=64,
            num_grus=2,
            out_shape=critic_out_shape,
        )
    else:
        policy_nn = fc.FC(
            input_shape=policy_input_shape,
            embed_dim=64,
            depth=2,
            activation=torch.nn.ReLU(),
            out_shape=policy_out_shape,
        )
        critic_nn = fc.FC(
            input_shape=critic_input_shape,
            embed_dim=64,
            depth=2,
            activation=torch.nn.ReLU(),
            out_shape=critic_out_shape,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = PPO_Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        batch_size=1024,
        max_mem_size=10**5,
        num_parallel_envs=num_parallel_envs,
        lr=3e-4,
        entropy_coeff=0.05,
        policy_nn=policy_nn,
        critic_nn=critic_nn,
        discount_factor=0.99,
        tensorboard_dir="/tmp/ppo/ppo_test_tensorboard/",
    )
    train_stats = agent.train_n_steps(env=env, n_steps=400)
    reward = agent.run_env(env, best_act=True)
    agent.save_agent("/tmp/ppo/ppo_test.pt")
    agent.load_agent("/tmp//ppo/ppo_test.pt")
    assert True


def main():
    test_single("LunarLanderContinuous-v2", 1, True)


if __name__ == "__main__":
    main()
