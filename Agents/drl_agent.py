from tqdm import tqdm
from abc import ABC, abstractmethod
import gymnasium as gym
import utils
from .agent_utils import ExperienceReplay
from .explorers import RandomExplorer
import numpy as np
import functools
import operator
from .agent_utils import ParallelEnv
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from .agent_utils import ExperienceReplay, ObsShapeWraper, ObsWraper
import uuid
import pygame
import logging
import pandas as pd
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
import datetime

from Models import model_factory
import copy

class RL_Agent(ABC):
    """
    RL_Agent is an abstract class that defines the basic structure of an RL agent.
    It is used as a base class for all RL agents.
    """
    TRAIN = 0
    EVAL = 1

    def __init__(self, obs_space: gym.spaces, action_space : gym.spaces, max_mem_size=10e6, batch_size=256, explorer = RandomExplorer() , num_parallel_envs=4, model_class=None, model_kwargs=dict(), lr=0.0001, device=None, norm_params={}, experience_class=ExperienceReplay, discount_factor=0.99, tensorboard_dir = './tensorboard') -> None:
        """
        Args:
            obs_space (gym.spaces): observation space of the environment
            action_space (gym.spaces): action space of the environment
            max_mem_size (int, optional): maximum size of the experience replay buffer. Defaults to 10e6.
            batch_size (int, optional): batch size for training. Defaults to 256.
            explorer (Explorer, optional): exploration method. Defaults to RandomExplorer().
            num_parallel_envs (int, optional): number of parallel environments. Defaults to 4.
            model_class (AbstractModel, optional): model class. Defaults to None.
            model_kwargs (dict, optional): model kwargs. Defaults to dict().
            lr (float, optional): learning rate. Defaults to 0.0001.
            device (torch.device, optional): device to run on. Defaults to None.
            norm_params (dict, optional): normalization parameters. Defaults to {}.
        """
        super(RL_Agent, self).__init__()
        self.id = uuid.uuid4()
        self.writer = SummaryWriter(f'{tensorboard_dir}/{self.__class__.__name__}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
        self.env = None
        self.r_func = lambda s,a,r: r
        self.explorer = explorer

        self.norm_params = copy.copy(norm_params)
        self.model_kwargs = model_kwargs
        self.obs_space = obs_space #obs_shape
        self.obs_shape = ObsShapeWraper(obs_space.shape) #obs_shape
        self.discount_factor = discount_factor
        if norm_params is None:
            norm_params = {'mean': {}, 'std': {}}
            for k in self.obs_shape.keys():
                norm_params['mean'] = ObsWraper({k: 0})
                norm_params['std'] = ObsWraper({k: 1})
        self.norm_params = norm_params
        self.model_class = model_class
        self.batch_size = batch_size
        self.num_parallel_envs = batch_size if num_parallel_envs is None else num_parallel_envs
        if model_class.is_rnn:
            logger.info(f"RNN model detected, using batch_size = num_parallel_env = {self.num_parallel_envs}")
            self.batch_size = self.num_parallel_envs

        self.eval_mode = self.EVAL
        
        self.experience = experience_class(max_mem_size, self.obs_shape)
        self.mem_size = max_mem_size
        
        self.device = device if device is not None else utils.init_torch() #default goes to cuda -> cpu' or enter manualy
        
        self.lr = lr
        self.action_space = action_space
        self.define_action_space()
        self.init_models()
        self.metrics = defaultdict(list)


    @abstractmethod
    def init_models(self):
        raise NotImplementedError
    #     self.rnn = self.model_class.is_rnn
    #     if self.rnn:
    #         self.update_policy = self.update_policy_rnn


    @abstractmethod
    def update_policy(self, *exp):
        raise NotImplementedError
    

    def get_train_metrics(self):
        return pd.DataFrame(self.metrics)
        


    def define_action_space(self):
        self.action_dtype = self.action_space.dtype
        
        if np.issubdtype(self.action_dtype, np.integer):           
            try:
                self.n_actions = self.action_space.shape[0]
            except:
                self.n_actions = 1

            self.possible_actions = self.action_space.n
            self.best_act = self.best_act_discrete
        else:
            self.n_actions = self.action_space.shape[0]
            self.possible_actions = 'continuous'
            self.best_act = self.best_act_cont
            

    def __del__(self):
        self.close_env_procs()
        self.writer.close()


    @abstractmethod
    def save_agent(self,f_name: str)-> dict:
        save_dict = {
        'agent_type' : self.__class__.__name__,
        'obs_space' : self.obs_space,
        'obs_shape' : self.obs_shape,
        'model_class' : self.model_class.__name__,
        'action_space' : self.action_space,
        'model_kwargs' : self.model_kwargs,
        'norm_params' : self.norm_params,
        
        }
        return save_dict


    @abstractmethod
    def load_agent(self, f_name: str):
        checkpoint = torch.load(f_name,map_location=self.device)
        self.action_space = checkpoint['action_space']

        if 'obs_space' in checkpoint:
            self.obs_space = checkpoint['obs_space']

        if 'obs_shape' in checkpoint:
            self.obs_shape = checkpoint['obs_shape']

        if 'norm_params' in checkpoint:
            self.norm_params = checkpoint['norm_params']        
        
        if 'model_kwargs' in checkpoint:
            self.model_kwargs = checkpoint['model_kwargs']
        
        if 'model_class' in checkpoint:
            model_class = checkpoint['model_class']
            self.model_class = model_factory.get_model_class(model_class)
        self.init_models()
            

        return checkpoint


    def set_train_mode(self):
        self.reset_rnn_hidden()
        self.eval_mode = self.TRAIN


    def set_eval_mode(self):
        self.reset_rnn_hidden()
        self.eval_mode = self.EVAL


    def train_episodial(self, env, n_episodes, max_episode_len=None, disable_tqdm=False):
        if n_episodes < self.num_parallel_envs:
            self.set_num_parallel_env(n_episodes)
        train_r = self._train_n_iters(env, n_episodes, True, max_episode_len=max_episode_len, disable_tqdm=disable_tqdm)
        return train_r


    def train_n_steps(self, env, n_steps, max_episode_len=None, disable_tqdm=False):
        return self._train_n_iters(env, n_steps, episodes=False, max_episode_len=max_episode_len, disable_tqdm=disable_tqdm)

    
    def _train_n_iters(self, env, n_iters, episodes=False, max_episode_len=None, disable_tqdm=False):
        """General train function, if episodes is true- each iter is episode, otherwise train steps"""
        self.set_train_mode()
        
        pbar = tqdm(total=n_iters, leave=None, disable=disable_tqdm)
        curr_training_steps = 0
        train_rewards = []

        if episodes:
            to_update_idx = 0
        else:
            to_update_idx = 1

        i = 0
        ep_number = 0
        best_agent_score = None
        while i < n_iters:
            rewards_vector = self.collect_episode_obs(env, max_episode_len, num_to_collect_in_parallel=self.num_parallel_envs)
            num_steps_collected = 0
            for r in rewards_vector:
                train_rewards.append(np.sum(r))
                num_steps_collected += len(r)
            mean_rewrad = np.mean(train_rewards[-len(rewards_vector):])

            if best_agent_score is None or mean_rewrad > best_agent_score:
                best_agent_score = mean_rewrad
                self.save_agent('/tmp/best_{}.pt'.format(self.id))

            collect_info = [self.num_parallel_envs, num_steps_collected]
            curr_training_steps += num_steps_collected
            desciption = f"episode {ep_number}, R:{np.round(mean_rewrad, 2):08}, total_steps:{curr_training_steps}"
            pbar.set_description(desciption)

            pbar.update(collect_info[to_update_idx])
            i += collect_info[to_update_idx]
            ep_number += self.num_parallel_envs
            self.update_policy()
            for k in self.metrics:
                self.writer.add_scalar(k, self.metrics[k][-1], ep_number)
            self.reset_rnn_hidden()

        pbar.close()
        self.close_env_procs()
        return train_rewards


    def set_num_parallel_env(self, num_parallel_envs):
        assert self.num_parallel_envs <= self.batch_size, f"please provide batch_size>= num_parallel_envs current: {self.batch_size}, {num_parallel_envs},"
        self.num_parallel_envs = num_parallel_envs


    @abstractmethod
    def act(self, observations, num_obs=1, extra_info=False):
        raise NotImplementedError
    
  
    def load_highest_score_agent(self):
        self.load_agent('/tmp/best_{}.pt'.format(self.id))


    def get_highest_score_agent_ckpt_path(self):
        return '/tmp/best_{}.pt'.format(self.id)
        


    @abstractmethod
    def best_act_discrete(self, observations, num_obs=1, extra_info=False):
        raise NotImplementedError
    

    @abstractmethod
    def best_act_cont(self, observations, num_obs=1, extra_info=False):
        raise NotImplementedError

    def get_seqs_indices_for_pack(self, done_indices):
        """returns seq_lens, sorted_data_sub_indices"""
        env_indices = np.zeros_like(done_indices, dtype=np.int32)
        env_indices[0] = -1
        env_indices[1:] = done_indices[:-1]
        all_lens = done_indices - env_indices
        data_sub_indices = np.array([list(range(env_indices[i]+1, done_indices[i]+1, 1)) for i in range(len(all_lens-1))], dtype=object)

        seq_indices = np.argsort(all_lens, kind='stable')[::-1]
        sorted_data_sub_indices = data_sub_indices[seq_indices]
        sorted_data_sub_indices = np.concatenate(sorted_data_sub_indices).astype(np.int32)
        seq_lens = all_lens[seq_indices]
        return seq_lens, seq_indices, sorted_data_sub_indices


    def pack_from_done_indices(self, data, seq_indices, sorted_seq_lens, done_indices):
        """returns pakced obs"""
        assert np.all(np.sort(sorted_seq_lens, kind='stable')[::-1] == sorted_seq_lens)
        max_colected_len = np.max(sorted_seq_lens)

        packed_obs = ObsWraper()
        for k in data:
            obs_shape = data[k][-1].shape
            temp = []
            curr_idx = 0
            for i,d_i in enumerate(done_indices):
                temp.append(data[k][curr_idx:d_i+1])
                curr_idx = d_i+1# 
            
            temp = [temp[i] for i in seq_indices]
            max_new_seq_len = max_colected_len
            new_lens = sorted_seq_lens

            padded_seq_batch = torch.nn.utils.rnn.pad_sequence(temp, batch_first=True)
            padded_seq_batch = padded_seq_batch.reshape((self.num_parallel_envs, max_new_seq_len, np.prod(obs_shape)))
            pakced_states = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=np.array(new_lens), batch_first=True)
            packed_obs[k] = pakced_states

        return packed_obs


    def pack_sorted_data(self, sorted_data, sorted_seq_lens):
        states = ObsWraper()
        for k in sorted_data:
            tmp = [torch.from_numpy(x).float() for x in sorted_data[k]]
            states[k] = self.pack_sorted_data_h(tmp, sorted_seq_lens).to(self.device)
        return states


    def pack_sorted_data_h(self, data, seq_lens):
        batch_size = len(data)
        max_seq_len = np.max(seq_lens).astype(np.int32)
        padded_seq_batch = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
        padded_seq_batch = padded_seq_batch.reshape((batch_size, max_seq_len, np.prod(data[-1].shape)))
        p_data = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=seq_lens, batch_first=True)
        return p_data


    def norm_obs(self, observations):
        # ObsWraper(observations)
        return observations
        if type(observations) == list:
            import pdb;pdb.set_trace()
        return  (observations - self.norm_params['mean']) / self.norm_params['std']


    def pre_process_obs_for_act(self, observations, num_obs):
        observations = ObsWraper(observations)
        
        observations = self.norm_obs(observations)
        len_obs = len(observations)

        
        if num_obs == 1 and self.obs_shape == observations.shape:
            # BATCH=1
            len_obs = 1
            observations = observations[np.newaxis, :]
            if self.rnn:
                observations = observations[np.newaxis, :]
        elif num_obs != len_obs and num_obs != 1:
            raise Exception(f"number of observations do not match real observation num obs: {num_obs}, vs real len: {len(observations)}")
        # return observations
        # if self.rnn:
        #     seq_lens = np.ones(len_obs)
        #     states = self.pack_sorted_data(observations, seq_lens)
        #     # states = torch.from_numpy(observations).to(self.device)
        # else:
        states = observations.get_as_tensors(self.device)
        return states


    def return_correct_actions_dim(self, selected_actions, num_obs):
        selected_actions = selected_actions.reshape(num_obs,*self.action_space.shape)
        return selected_actions.astype(self.action_dtype)


    def close_env_procs(self):
        if self.env is not None:
            self.env.close_procs()
            self.env = None


    def set_intrisic_reward_func(self, func):
        self.r_func = func


    def intrisic_reward_func(self, state, action, reward):
        return self.r_func(state, action, reward)

    
    def collect_episode_obs(self, env, max_episode_len=None, num_to_collect_in_parallel=None, env_funcs={"step": "step", "reset": "reset"}):
        # supports run on different env api
        self.reset_rnn_hidden()
        if num_to_collect_in_parallel is None:
            num_to_collect_in_parallel = self.num_parallel_envs
        if env.__class__.__name__ != "ParallelEnv":
            if self.env is None:
                self.env = ParallelEnv(env, num_to_collect_in_parallel)
            else:
                self.env.change_env(env)
        else:
            self.env = env

        step_function = getattr(self.env, env_funcs["step"])

        reset_function = getattr(self.env, env_funcs["reset"])

        if max_episode_len:
            def episode_len_exceeded(x): return x > max_episode_len
        else:
            def episode_len_exceeded(x): return False

        observations = []

        for item in reset_function():
            observations.append([item[0]])
        
        env_dones = np.array([False for i in range(num_to_collect_in_parallel)])

        latest_observations = [observations[i][-1] for i in range(num_to_collect_in_parallel)]        

        rewards = [[] for i in range(num_to_collect_in_parallel)]
        actions = [[] for i in range(num_to_collect_in_parallel)]
        dones = [[] for i in range(num_to_collect_in_parallel)]
        truncated = [[] for i in range(num_to_collect_in_parallel)]

        max_episode_steps = 0

        while not all(env_dones):
            relevant_indices = np.where(env_dones == False)[0].astype(np.int32)         
            
            if self.explorer.explore():
                explore_action = self.explorer.act(self.action_space, latest_observations, num_to_collect_in_parallel)
                current_actions = self.return_correct_actions_dim(explore_action, num_to_collect_in_parallel)
            else:
                current_actions = self.act(latest_observations, num_to_collect_in_parallel)
            # TODO DEBUG
            # allways use all envs to step, even some envs are done already
            next_obs, reward, terminated, trunc, info = step_function(current_actions)

            for i in relevant_indices:

                actions[i].append(current_actions[i])
                intrisic_reward = self.intrisic_reward_func(latest_observations[i], current_actions[i], reward[i])
                rewards[i].append(intrisic_reward)
                truncated[i].append(trunc[i])
                done = terminated[i] or trunc[i]
                dones[i].append(done)
                env_dones[i] = done

                max_episode_steps += 1
                if done:
                    # import pdb;pdb.set_trace()
                    continue

                if episode_len_exceeded(max_episode_steps):
                    dones[i][-1] = True
                    truncated[i][-1] = True
                    env_dones[i] = True
                    break
            
                observations[i].append(next_obs[i])

            latest_observations = [observations[i][-1] for i in range(num_to_collect_in_parallel)]
            

        observations = functools.reduce(operator.iconcat, observations, [])
        observations = self.norm_obs(observations) # it is normalized in act, so here we save it normlized to replay buffer aswell
        actions = functools.reduce(operator.iconcat, actions, [])
        rewards_x = functools.reduce(operator.iconcat, rewards, [])
        dones = functools.reduce(operator.iconcat, dones, [])
        truncated = functools.reduce(operator.iconcat, truncated, [])
        
        # next_observations = functools.reduce(operator.iconcat, next_observations, [])
        # next_observations = self.norm_obs(next_observations) # it is normalized in act, so here we save it normlized to replay buffer aswell

        self.experience.append(observations, actions, rewards_x, dones, truncated)
        self.reset_rnn_hidden()
        self.explorer.update()
        return rewards


    @abstractmethod
    def reset_rnn_hidden(self,):
        """if agent uses rnn, this callback is called in many places so please impliment it"""
        raise NotImplementedError


    @abstractmethod
    def get_last_collected_experiences(self, number_of_episodes):
        # Mainly for paired alg
        raise NotImplementedError

    @abstractmethod
    def clear_exp(self):
        self.experience.clear()
    

    def run_env(self, env, best_act=True, num_runs=1):
        "runs env in eval"
        self.set_eval_mode()
        env = ParallelEnv(env, 1)
        act_func = self.act
        if best_act:
            act_func = self.best_act

        mean_r = 0

        for i in range(num_runs):
            obs, _ = env.reset()[0]
            # break
            R = 0
            t = 0
            while True:
                action = act_func(obs)
                obs, r, terminated, truncated, _ = env.step(action)
                R += r
                t += 1
                done = terminated or truncated 
                if done:
                    break
            mean_r +=R
        pygame.display.quit()
        return mean_r / num_runs
