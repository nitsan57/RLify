from rlify.agents.agent_utils import ObsWrapper
from rlify.agents.explorers import RandomExplorer
from rlify.agents.vdqn_agent import VDQN_Agent
from rlify.models import fc, rnn, dtree
from rlify.agents.dqn_agent import DQN_Agent
from rlify.agents.ppo_agent import PPO_Agent
from rlify.agents.ddpg_agent import DDPG_Agent
from rlify.agents.heuristic_agent import Heuristic_Agent

# import os
# os.environ["LD_LIBRARY_PATH"] =  os.environ.get("LD_LIBRARY_PATH") + "/home/nitsan57/.mujoco/mujoco210/bin:"
# os.environ["LD_LIBRARY_PATH"] =  os.environ.get("LD_LIBRARY_PATH") + "/usr/lib/nvidia:"
# os.environ["PATH"] =  os.environ.get("LD_LIBRARY_PATH") + os.environ["PATH"]
# os.environ["LD_PRELOAD"] =  os.environ.get("LD_PRELOAD", "") + "/usr/lib/x86_64-linux-gnu/libGLEW.so"

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from rlify import utils
import gymnasium as gym
import plotly.express as px

import gymnasium as gym


def norm_obs(x):
    return (x / 255).astype(np.float32)


def main():

    device = utils.init_torch()
    # # env_name = "Hopper-v4" #"CarRacing-v2" #"LunarLanderContinuous-v2" #"Pendulum-v1"  "CartPole-v1" #"LunarLander-v2" #"Acrobot-v1"
    env_name = "Taxi-v3"
    env = gym.make(env_name, render_mode=None)
    # this time we will use a simple GRU as our reccurent NN:
    # models_shapes = DDPG_Agent.get_models_input_output_shape(
    #     env.observation_space, env.action_space
    # )
    # q_model_input_shape = models_shapes["Q_model"]["input_shape"]
    # q_model_out_shape = models_shapes["Q_model"]["out_shape"]
    # q_mle_model_input_shape = models_shapes["Q_mle_model"]["input_shape"]
    # q_mle_model_out_shape = models_shapes["Q_mle_model"]["out_shape"]
    # Q_model = rnn.GRU(
    #     input_shape=q_model_input_shape,
    #     hidden_dim=64,
    #     num_grus=2,
    #     out_shape=q_model_out_shape,
    # )
    # Q_mle_model = rnn.GRU(
    #     input_shape=q_mle_model_input_shape,
    #     hidden_dim=64,
    #     num_grus=2,
    #     out_shape=q_mle_model_out_shape,
    # )
    # Q_model = fc.FC(
    #     input_shape=q_model_input_shape,
    #     embed_dim=64,
    #     depth=2,
    #     activation=torch.nn.ReLU(),
    #     out_shape=q_model_out_shape,
    # )
    # Q_mle_model = fc.FC(
    #     input_shape=q_mle_model_input_shape,
    #     embed_dim=64,
    #     depth=2,
    #     activation=torch.nn.ReLU(),
    #     out_shape=q_mle_model_out_shape,
    # )
    # agent = DDPG_Agent(
    #     obs_space=env.observation_space,
    #     action_space=env.action_space,
    #     device=device,
    #     batch_size=2048,
    #     max_mem_size=10**6,
    #     num_parallel_envs=16,
    #     lr=3e-4,
    #     Q_model=Q_model,
    #     Q_mle_model=Q_mle_model,
    #     discount_factor=0.99,
    #     tensorboard_dir="./tensorboard/test_rnn_ddpg",
    # )
    models_shapes = VDQN_Agent.get_models_input_output_shape(
        env.observation_space, env.action_space
    )
    q_model_input_shape = models_shapes["Q_model"]["input_shape"]
    q_model_out_shape = models_shapes["Q_model"]["out_shape"]
    # Q_model = rnn.GRU(
    #     input_shape=q_model_input_shape,
    #     hidden_dim=64,
    #     num_grus=2,
    #     out_shape=q_model_out_shape,
    # )
    Q_model = fc.FC(
        input_shape=q_model_input_shape,
        embed_dim=128,
        depth=4,
        activation=torch.nn.ReLU(),
        out_shape=q_model_out_shape,
    )

    agent = VDQN_Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        batch_size=64,
        max_mem_size=10**6,
        num_parallel_envs=1,
        lr=3e-4,
        Q_model=Q_model,
        discount_factor=0.99,
        tensorboard_dir="./tensorboard/test_rnn_vdqn",
        explorer=RandomExplorer(1, 0.05, 0.03),
    )
    train_stats = agent.train_n_steps(env=env, n_steps=200000)
    utils.plot_res(
        train_stats, "train_rewared_" + env_name, smooth_kernel=10, render_as="browser"
    )  # supports notebook, browser as well
    breakpoint()
    env = gym.make(env_name, render_mode="human")
    agent.run_env(env, best_act=True, num_runs=1)

    # return
    # ###############################################
    # env_name = "Acrobot-v1"
    # # env = gym.make(env_name, render_mode=None)
    # # # this time we will use a simple GRU as our reccurent NN:
    # # models_shapes = DDPG_Agent.get_models_input_output_shape(env.observation_space, env.action_space)
    # # q_model_input_shape = models_shapes["Q_model"]["input_shape"]
    # # q_model_out_shape = models_shapes["Q_model"]["out_shape"]
    # # q_mle_model_input_shape = models_shapes["Q_mle_model"]["input_shape"]
    # # q_mle_model_out_shape = models_shapes["Q_mle_model"]["out_shape"]
    # # Q_model = rnn.GRU(input_shape=q_model_input_shape, hidden_dim=64, num_grus=2, out_shape=q_model_out_shape)
    # # Q_mle_model = rnn.GRU(input_shape=q_mle_model_input_shape, hidden_dim=64, num_grus=2, out_shape=q_mle_model_out_shape)
    # # agent = DDPG_Agent(obs_space=env.observation_space, action_space=env.action_space, device=device, batch_size=1024, max_mem_size=10**6, num_parallel_envs=16,
    # #                 lr=3e-4, Q_model=Q_model, Q_mle_model=Q_mle_model, discount_factor=0.99, tensorboard_dir = None)
    # # train_stats = agent.train_n_steps(env=env,n_steps=100000)

    # models_shapes = DDPG_Agent.get_models_input_output_shape(
    #     env.observation_space, env.action_space
    # )
    # Q_model = fc.FC(
    #     input_shape=models_shapes["Q_model"]["input_shape"],
    #     embed_dim=128,
    #     depth=2,
    #     activation=torch.nn.ReLU(),
    #     out_shape=models_shapes["Q_model"]["out_shape"],
    # )
    # Q_mle_model = fc.FC(
    #     input_shape=models_shapes["Q_mle_model"]["input_shape"],
    #     embed_dim=128,
    #     depth=2,
    #     activation=torch.nn.ReLU(),
    #     out_shape=models_shapes["Q_mle_model"]["out_shape"],
    # )

    # # Q_model = rnn.GRU(
    # #     input_shape=models_shapes["Q_model"]["input_shape"],
    # #     hidden_dim=128,
    # #     num_grus=3,
    # #     out_shape=models_shapes["Q_model"]["out_shape"],
    # # )
    # # Q_mle_model = rnn.GRU(
    # #     input_shape=models_shapes["Q_mle_model"]["input_shape"],
    # #     hidden_dim=128,
    # #     num_grus=3,
    # #     out_shape=models_shapes["Q_mle_model"]["out_shape"],
    # # )
    # agent_c = DDPG_Agent(
    #     obs_space=env.observation_space,
    #     action_space=env.action_space,
    #     device=device,
    #     batch_size=64,
    #     max_mem_size=10**6,
    #     num_parallel_envs=16,
    #     lr=3e-4,
    #     Q_model=Q_model,
    #     Q_mle_model=Q_mle_model,
    #     discount_factor=0.99,
    #     explorer=RandomExplorer(1, 0.05, 0.03),
    #     tensorboard_dir="./tensorboard",
    #     num_epochs_per_update=10,
    #     target_update="soft[tau=0.005]",
    # )
    # # models_shapes = PPO_Agent.get_models_input_output_shape(
    # #     env.observation_space, env.action_space
    # # )
    # # policy_nn = fc.FC(
    # #     input_shape=models_shapes["policy_nn"]["input_shape"],
    # #     embed_dim=64,
    # #     depth=2,
    # #     activation=torch.nn.ReLU(),
    # #     out_shape=models_shapes["policy_nn"]["out_shape"],
    # # )
    # # critic_nn = fc.FC(
    # #     input_shape=models_shapes["critic_nn"]["input_shape"],
    # #     embed_dim=64,
    # #     depth=2,
    # #     activation=torch.nn.ReLU(),
    # #     out_shape=models_shapes["critic_nn"]["out_shape"],
    # # )

    # # agent_c = PPO_Agent(
    # #     obs_space=env.observation_space,
    # #     action_space=env.action_space,
    # #     device=device,
    # #     batch_size=1024,
    # #     max_mem_size=10**5,
    # #     num_parallel_envs=4,
    # #     lr=3e-4,
    # #     entropy_coeff=0.05,
    # #     policy_nn=policy_nn,
    # #     critic_nn=critic_nn,
    # #     discount_factor=0.99,
    # #     tensorboard_dir=None,
    # # )
    # train_stats_c = agent_c.train_n_steps(env=env, n_steps=400000)
    # metric_df = agent_c.get_train_metrics()
    # metric_df.plot(backend="plotly").show()
    # utils.plot_res(
    #     train_stats_c, title="dqo ReLU", smooth_kernel=100, render_as="browser"
    # )
    # print("run_env 10 times:")
    # print(agent_c.run_env(env, best_act=True, num_runs=10))
    # print("----------------------------------")
    # breakpoint()

    # Saving the agent ckpt is easy
    # _ = agent.save_agent("lunar_ppo.pt")

    env_name = "LunarLander-v2"
    env = gym.make(env_name, render_mode=None)
    models_shapes = PPO_Agent.get_models_input_output_shape(
        env.observation_space, env.action_space
    )
    policy_input_shape = models_shapes["policy_nn"]["input_shape"]
    policy_out_shape = models_shapes["policy_nn"]["out_shape"]
    critic_input_shape = models_shapes["critic_nn"]["input_shape"]
    critic_out_shape = models_shapes["critic_nn"]["out_shape"]
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

    agent = PPO_Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        batch_size=1024,
        max_mem_size=10**5,
        num_parallel_envs=4,
        lr=3e-4,
        entropy_coeff=0.05,
        policy_nn=policy_nn,
        critic_nn=critic_nn,
        discount_factor=0.99,
        kl_div_thresh=0.05,
        clip_param=0.2,
        tensorboard_dir="./tensorboard/",
    )
    train_stats = agent.train_n_steps(env=env, n_steps=2000)
    # Saving the agent ckpt is easy
    _ = agent.save_agent("lunar_ppo.pt")


if __name__ == "__main__":
    main()
