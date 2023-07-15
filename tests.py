
from Agents.agent_utils import LinearRandomExplorer, ObsWraper
from Models import fc, rnn
from Agents.dqn_agent import DQN_Agent
from Agents.ppo_agent import PPO_Agent

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import utils
import gymnasium as gym
import plotly.express as px

def unittest():
    """Not implemnted yet"""
    ObsWraper(np.zeros((3,3))).shape
    obs = ObsWraper(
        {'data': 3}
    )
    obs2 = ObsWraper(
        {'data': 4}
    )
    obs3 = ObsWraper(
        {'data': 5}
    )
    obs_list = [obs, obs2, obs3]
    ObsWraper(obs_list).shape
    import torch

    obs = ObsWraper(
        {'data': torch.tensor(3)}
    )
    obs2 = ObsWraper(
        {'data': torch.tensor(4)}
    )
    obs3 = ObsWraper(
        {'data': torch.tensor(5)}
    )
    obs_list = [obs, obs2, obs3]
    ObsWraper(obs_list)


def run_for_d(a, e_name):
    env_c2 = gym.make(e_name, render_mode="rgb_array")
    reward = a.run_env(env_c2, render=True, best_act=True)
    print("Run Reward:", reward)
    import pygame
    pygame.display.quit()


def test_2e2():

    device = utils.init_torch()
   

    env_name = "LunarLander-v2"
    env = gym.make(env_name, render_mode="rgb_array", continuous=True)
    model_class_c = rnn.GRU
    model_kwargs_c = {} #{'embed_dim': 64, 'repeat':2}
    agent_c = PPO_Agent(obs_space=env.observation_space, action_space=env.action_space, device=device, batch_size=1024, max_mem_size=10**5, num_parallel_envs=4, lr=3e-4, entropy_coeff=0.05, model_class=model_class_c, model_kwargs=model_kwargs_c, discount_factor=0.99,kl_div_thresh=0.05, clip_param=0.3)
    train_stats_C = agent_c.train_n_steps(env=env,n_steps=250)


    env = gym.make(env_name, render_mode="rgb_array", continuous=True)
    reward = agent_cont.run_env(env, render=False, best_act=True)
    print("Run Reward:", reward)
    import pygame
    pygame.display.quit()


    env_name = "LunarLander-v2"
    env = gym.make(env_name, render_mode="rgb_array")
    model_class = fc.FC
    model_kwargs = {'embed_dim': 64, 'repeat':2}
    agent = PPO_Agent(obs_space=env.observation_space, action_space=env.action_space, device=device, batch_size=1024, max_mem_size=10**5, num_parallel_envs=4, lr=3e-4, entropy_coeff=0.05, model_class=model_class, model_kwargs=model_kwargs, discount_factor=0.99)
    train_stats = agent.train_n_steps(env=env,n_steps=25)


    # TEST CONTINOUS 
    car_env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    agent_car = PPO_Agent(obs_space=car_env.observation_space, action_space=car_env.action_space, device=device, batch_size=4096, max_mem_size=10**5,num_parallel_envs=4, lr=3e-4, model_class=fc.FC, random_explorer = LinearRandomExplorer(0.3,0,0))
    train_stats_c = agent_car.train_n_steps(env=car_env,n_steps=100)

    # test single obs int
    env_c = gym.make("Taxi-v3", render_mode="rgb_array")
    model_class_c = fc.FC
    model_kwargs_c = {'embed_dim': 64, 'repeat':2}
    agent_c = PPO_Agent(obs_space=env_c.observation_space, action_space=env_c.action_space, device='cpu', batch_size=4096, max_mem_size=10**5,num_parallel_envs=1, lr=3e-4, entropy_coeff=0.1, model_class=model_class_c, model_kwargs=model_kwargs_c, horizon=-1, discount_factor=0.99, kl_div_thresh=10)
    train_stats_c = agent_c.train_n_steps(env=env_c,n_steps=250)



    car_env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array", )
    action_space = car_env.action_space
    obs_space = car_env.observation_space
    agent_car = PPO_Agent(obs_space=obs_space, action_space=action_space, device=device, batch_size=4096, max_mem_size=10**5,num_parallel_envs=8, lr=0.0001, model_class=rnn.GRU)
    train_stats_c = agent_car.train_n_steps(env=car_env,n_steps=2500)


    env_name = "LunarLander-v2"
    env = gym.make(env_name, render_mode="rgb_array", continuous=True) #continuous=True
    model_class = rnn.GRU
    model_kwargs = {'hidden_dim': 64, 'num_grus':2}
    agent_cont = PPO_Agent(obs_space=env.observation_space, action_space=env.action_space, device=device, batch_size=1024, max_mem_size=10**5, num_parallel_envs=4, lr=3e-4, entropy_coeff=0.1, model_class=model_class, model_kwargs=model_kwargs, horizon=-1, discount_factor=0.99)
    train_stats_cont = agent_cont.train_n_steps(env=env,n_steps=350)

    env = gym.make(env_name, render_mode="human", continuous=True)
    reward = train_stats_cont.run_env(env, render=True, best_act=True)
    print("Run Reward:", reward)
    import pygame
    pygame.display.quit()


    env_name = "CartPole-v1"
    env_c = gym.make(env_name, render_mode="rgb_array")
    model_class_c = fc.FC
    model_kwargs_c = {'embed_dim': 64, 'repeat':2}
    agent_c = DQN_Agent(obs_space=env_c.observation_space, action_space=env_c.action_space, batch_size=16, max_mem_size=10**5,num_parallel_envs=16, lr=1e-3,  model_class=model_class_c, model_kwargs=model_kwargs_c, horizon=-1, discount_factor=0.99, random_explorer = LinearRandomExplorer(1,0.05,0.01), target_update_time=100)
    train_stats_c = agent_c.train_episodial(env=env_c,n_episodes=16)

    # test single obs int
    env_c = gym.make("Taxi-v3", render_mode="rgb_array")
    model_class_c = fc.FC
    model_kwargs_c = {'embed_dim': 64, 'repeat':2}
    agent_c = PPO_Agent(obs_space=env_c.observation_space, action_space=env_c.action_space, device='cpu', batch_size=4096, max_mem_size=10**5,num_parallel_envs=1, lr=3e-4, entropy_coeff=0.1, model_class=model_class_c, model_kwargs=model_kwargs_c, horizon=-1, discount_factor=0.99, kl_div_thresh=10)
    train_stats_c = agent_c.train_n_steps(env=env_c,n_steps=2500)

    # parallel envs test
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    agent = PPO_Agent(obs_space=env.observation_space, action_space=env.action_space, device=device, batch_size=64, max_mem_size=10**5,num_parallel_envs=2, lr=0.0001, model_class=fc.FC)
    agent.train_episodial(env=env,n_episodes=2)
    _ = agent.save_agent("/tmp/lunar_ppo.pt")
    agent.load_agent("/tmp/lunar_ppo.pt")


    # single env test
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    agent = PPO_Agent(obs_space=env.observation_space, action_space=env.action_space, device=device, batch_size=64, max_mem_size=10**5,num_parallel_envs=1, lr=0.0001, model_class=rnn.GRU)
    agent.train_episodial(env=env,n_episodes=2)



    # Continious envs test
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array", )
    agent = PPO_Agent(obs_space=env.observation_space, action_space=env.action_space, device=device, batch_size=64, max_mem_size=10**5,num_parallel_envs=2, lr=0.0001, model_class=rnn.GRU)
    agent.train_episodial(env=env,n_episodes=2)



    car_env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array", )
    agent_car = PPO_Agent(obs_space=car_env.observation_space, action_space=car_env.action_space, device=device, batch_size=4096, max_mem_size=10**5,num_parallel_envs=8, lr=0.0001, model_class=rnn.GRU)
    train_stats_c = agent_car.train_n_steps(env=car_env,n_steps=2500)
    

    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    model_class = fc.FC
    model_kwargs = {'embed_dim': 64, 'repeat':2}
    agent_car_dqn = DQN_Agent(obs_space=env.observation_space, action_space=env.action_space, device=device, batch_size=4096, max_mem_size=10**5,num_parallel_envs=16, lr=3e-4, model_class=model_class, model_kwargs=model_kwargs, random_explorer = LinearRandomExplorer(0.99, 0.03,0.001))
    train_stats_car = agent_car_dqn.train_n_steps(env=env,n_steps=2500)

    # TEST DQN
    car_env = gym.make("MountainCar-v0", render_mode="rgb_array", )
    model_class_car =  fc.FC #rnn.GRU  #
    model_kwargs_car =  {'embed_dim': 92, 'repeat':3} # {} #
    agent_car = DQN_Agent(obs_space=car_env.observation_space, action_space=car_env.action_space, device=device, batch_size=4096, max_mem_size=10**5,num_parallel_envs=16, lr=3e-4, model_class=model_class_car, model_kwargs=model_kwargs_car, random_explorer = LinearRandomExplorer(0.99,0,0.001))
    train_stats_car = agent_car.train_n_steps(env=car_env,n_steps=2500)

    # TEST CONTINOUS 
    car_env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    agent_car = PPO_Agent(obs_space=car_env.observation_space, action_space=car_env.action_space, device=device, batch_size=4096, max_mem_size=10**5,num_parallel_envs=4, lr=3e-4, model_class=fc.FC, random_explorer = LinearRandomExplorer(0.3,0,0))
    train_stats_c = agent_car.train_n_steps(env=car_env,n_steps=1000)

def main():
    test_2e2()




if __name__ == "__main__":
    main()