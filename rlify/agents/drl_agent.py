from attr import dataclass
from tqdm import tqdm
from abc import ABC, abstractmethod
import gymnasium as gym
from rlify import utils
from rlify.agents.explorers import Explorer, RandomExplorer
import numpy as np
import functools
import operator
from .agent_utils import ParallelEnv, TrainMetrics
import torch
from .agent_utils import ObsShapeWraper, ObsWrapper
from rlify.agents.experience_replay import ExperienceReplay
import uuid
import logging
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from munch import Munch
import os
import pygame

logger = logging.getLogger(__name__)
import datetime

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def pad_from_done_indices(data, dones):
    """
    Packs the data from the done indices to torch.nn.utils.rnn.PackedSequence

    """
    if isinstance(data, ObsWrapper):
        return pad_states_from_done_indices(data, dones)
    elif isinstance(data, torch.Tensor):
        return pad_tensors_from_done_indices(data, dones)


def pad_states_from_done_indices(data, dones):
    """
    Packs the data from the done indices to torch.nn.utils.rnn.PackedSequence

    """
    done_indices = torch.where(dones == True)[0].cpu().numpy().astype(np.int32)
    padded_obs = ObsWrapper(tensors=True)
    for k in data:
        temp = []
        curr_idx = 0
        for i, d_i in enumerate(done_indices):
            temp.append(data[k][curr_idx : d_i + 1])
            curr_idx = d_i + 1  #
        padded_seq_batch = torch.nn.utils.rnn.pad_sequence(temp, batch_first=True)
        padded_obs[k] = padded_seq_batch
    lengths = done_indices - np.roll(done_indices, 1)
    lengths[0] = done_indices[0]
    return padded_obs, lengths


def pad_tensors_from_done_indices(data, dones):
    """
    Packs the data from the done indices to torch.nn.utils.rnn.PackedSequence

    """
    done_indices = torch.where(dones == True)[0].cpu().numpy().astype(np.int32)
    temp = []
    curr_idx = 0
    for i, d_i in enumerate(done_indices):
        temp.append(data[curr_idx : d_i + 1])
        curr_idx = d_i + 1  #
    padded_seq_batch = torch.nn.utils.rnn.pad_sequence(temp, batch_first=True)
    padded_obs = padded_seq_batch
    lengths = done_indices - np.roll(done_indices, 1)
    lengths[0] = done_indices[0]
    return padded_obs, lengths


class RLDataset(Dataset):
    def __init__(
        self, states, actions, rewards, dones, truncated, next_states, prepare_for_rnn
    ):
        states, actions, rewards, dones, truncated, next_states = self._prepare_data(
            states, actions, rewards, dones, truncated, next_states
        )
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.truncated = truncated
        self.next_states = next_states
        self.terminated = dones * (1 - truncated)
        self.loss_flag = torch.ones_like(rewards)
        self.prepare_for_rnn = prepare_for_rnn
        self.max_len = len(states)
        if self.prepare_for_rnn:
            (
                self.states,
                self.actions,
                self.rewards,
                self.dones,
                self.truncated,
                self.next_states,
                self.loss_flag,
                lengths,
            ) = self._pad_experiecne(
                states, actions, rewards, dones, truncated, next_states
            )
            self.max_len = lengths.max()

    def __len__(self):
        return self.max_len

    def _prepare_data(self, states, actions, rewards, dones, truncated, next_states):
        """
        Prepares the data for training
        Args:
            states: The states
            actions: The actions
            rewards: The rewards
            dones: The dones
            truncated: The truncateds

        Returns:
            The prepared data
        """
        actions = torch.from_numpy(actions)
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(-1)
        rewards = torch.from_numpy(rewards)
        dones = torch.from_numpy(dones)
        truncated = torch.from_numpy(truncated)
        states = states.get_as_tensors("cpu")
        next_states = next_states.get_as_tensors("cpu")
        return states, actions, rewards, dones, truncated, next_states

    def _pad_experiecne(self, states, actions, rewards, dones, truncateds, next_states):
        """
        Creates a padded version of the data
        Args:
            states: The states
            actions: The actions
            rewards: The rewards
            dones: The dones
            truncateds: The truncateds

        Returns:
            The padded states, actions, rewards, dones, truncateds, loss_flag, and lengths
        """
        padded_states, lengths = pad_states_from_done_indices(states, dones)
        padded_actions, lengths = pad_tensors_from_done_indices(actions, dones)
        padded_rewards, lengths = pad_tensors_from_done_indices(rewards, dones)
        padded_dones, lengths = pad_tensors_from_done_indices(dones, dones)
        padded_truncateds, lengths = pad_tensors_from_done_indices(truncateds, dones)
        padded_next_states, lengths = pad_states_from_done_indices(next_states, dones)
        loss_flag, lengths = pad_tensors_from_done_indices(
            torch.ones_like(rewards), dones
        )
        return (
            padded_states,
            padded_actions,
            padded_rewards,
            padded_dones,
            padded_truncateds,
            padded_next_states,
            loss_flag,
            lengths,
        )

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            self.truncated[idx],
            self.next_states[idx],
            self.loss_flag[idx],
        )

    def collate_fn(self, batch):
        states, actions, rewards, dones, truncated, next_states, loss_flag = zip(*batch)
        states = ObsWrapper.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)
        truncated = torch.stack(truncated)
        next_states = ObsWrapper.stack(next_states)
        loss_flag = torch.stack(loss_flag)
        return states, actions, rewards, dones, truncated, next_states, loss_flag


class IData(ABC):
    @abstractmethod
    def get_data_loader(self, batch_size: int, shuffle: bool):
        raise NotImplementedError


class RLData(IData):
    def __init__(
        self,
        states,
        actions,
        rewards,
        dones,
        truncated,
        next_states,
        prepare_for_rnn,
        num_workers: int = 2,
    ):
        self.num_workers = num_workers
        self.prepare_for_rnn = prepare_for_rnn
        self.dataset = RLDataset(
            states, actions, rewards, dones, truncated, next_states, prepare_for_rnn
        )
        self.can_shuffle = False if self.prepare_for_rnn else True

    def get_data_loader(self, batch_size, shuffle):
        if not (shuffle == self.can_shuffle or shuffle == False):
            logging.warning(
                "Shuffle is not allowed when preparing data for RNN, changing shuffle to False"
            )
            shuffle = False

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
        )


class RL_Agent(ABC):
    """
    RL_Agent is an abstract class that defines the basic structure of an RL agent.
    It is used as a base class for all RL agents.
    """

    TRAIN = 0
    EVAL = 1

    def __init__(
        self,
        obs_space: gym.spaces,
        action_space: gym.spaces,
        max_mem_size: int = int(10e6),
        batch_size: int = 256,
        explorer: Explorer = RandomExplorer(),
        num_parallel_envs: int = 4,
        num_epochs_per_update: int = 10,
        lr: float = 3e-4,
        device: str = None,
        experience_class: object = ExperienceReplay,
        discount_factor: float = 0.99,
        reward_normalization=True,
        tensorboard_dir: str = "./tensorboard",
    ) -> None:
        """

        Args:
            obs_space (gym.spaces): observation space of the environment
            action_space (gym.spaces): action space of the environment
            max_mem_size (int, optional): maximum size of the experience replay buffer. Defaults to 10e6.
            batch_size (int, optional): batch size for training. Defaults to 256.
            explorer (Explorer, optional): exploration method. Defaults to RandomExplorer().
            num_parallel_envs (int): number of parallel environments. Defaults to 4.
            num_epochs_per_update (int): Training epochs per update. Defaults to 10.
            lr (float, optional): learning rate. Defaults to 0.0001.
            device (torch.device, optional): device to run on. Defaults to None.
            experience_class (object, optional): experience replay class. Defaults to ExperienceReplay.\
            discount_factor (float, optional): discount factor. Defaults to 0.99.
            reward_normalization (bool, optional): whether to normalize the rewards by maximum absolut value. Defaults to True.
            tensorboard_dir (str, optional): tensorboard directory. Defaults to './tensorboard'.
            
        """
        super(RL_Agent, self).__init__()

        self.id = uuid.uuid4()
        self.init_tb_writer(tensorboard_dir)
        self.num_epochs_per_update = num_epochs_per_update

        self.env = None
        self.r_func = lambda s, a, r: r
        self.explorer = explorer
        self.stat_agg_stage = True
        self.obs_mean = None
        self.obs_std = None
        self.reward_normalization = reward_normalization
        self.max_reward = 0

        self.obs_space = obs_space
        self.obs_shape = ObsShapeWraper(obs_space)  # obs_shape
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.set_num_parallel_env(num_parallel_envs)
        self.device = (
            device if device is not None else utils.init_torch()
        )  # default goes to cuda -> cpu' or enter manualy

        self.lr = lr
        self.define_action_space(action_space)
        models = self.setup_models()
        self.validate_models(models)
        self.experience = experience_class(max_mem_size, self.obs_shape, self.n_actions)

        self.metrics = TrainMetrics()

    def contains_reccurent_nn(self):
        return self.rnn_models

    def validate_models(self, models):
        self.rnn_models = False
        assert (
            models is not None
        ), "setup_models should return a list of models, or empty list"
        is_rnn_list = []
        for m in models:
            is_rnn_list.append(m.is_rnn)
        if len(is_rnn_list) > 0:
            assert (
                np.array(is_rnn_list) == is_rnn_list[0]
            ).all(), "all models should have the same rnn status - either all reccurent, or either all not reccurent"
            self.rnn_models = is_rnn_list[0]

    def init_tb_writer(self, tensorboard_dir: str = None):
        """
        Initializes tensorboard writer

        Args:
            tensorboard_dir (str): tensorboard directory
        """
        if tensorboard_dir is not None:
            self.writer = SummaryWriter(
                f'{tensorboard_dir}/{self.__class__.__name__}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
            )
        else:  # just emulate a dummy writer
            self.writer = Munch()
            self.writer.add_scalar = lambda *args, **kwargs: None
            self.writer.close = lambda *args, **kwargs: None

    @staticmethod
    @abstractmethod
    def get_models_input_output_shape(obs_space, action_space):
        """
        Calculates the input and output shapes of the models
        Args:
            obs_space: observation space
            action_space: action space

        Returns:
            dictionary: dictionary containing the input and output shapes of the models
        """
        raise NotImplementedError

    @abstractmethod
    def setup_models(self) -> list[torch.nn.Module]:
        """
        Initializes the NN models
        """
        raise NotImplementedError

    @abstractmethod
    def update_policy(self, trajectories_dataset):
        """
        Updates the models and according to the agnets logic
        """
        raise NotImplementedError

    def get_train_metrics(self):
        """
        Returns the training metrics
        """
        return self.metrics.get_metrcis_df()

    def read_obs_space_properties(obs_space):
        """
        Returns the observation space properties
        """
        obs_shape = ObsShapeWraper(obs_space)
        return obs_shape

    def read_action_space_properties(action_space):
        """
        Returns the action space properties
        """
        if np.issubdtype(action_space.dtype, np.integer):
            try:
                n_actions = action_space.shape[0]
            except:
                n_actions = 1
            possible_actions = action_space.n
        else:
            n_actions = action_space.shape[0]
            possible_actions = "continuous"

        return n_actions, possible_actions

    def define_action_space(self, action_space):
        """
        Defines the action space
        """
        self.action_space = action_space
        self.action_dtype = self.action_space.dtype
        self.n_actions, self.possible_actions = RL_Agent.read_action_space_properties(
            action_space
        )

    def __del__(self):
        """
        Destructor
        """
        if hasattr(self, "env"):
            self.close_env_procs()
        if hasattr(self, "writer"):
            self.writer.close()

    @staticmethod
    def read_nn_properties(ckpt_fname):
        checkpoint = torch.load(ckpt_fname, map_location="cpu")
        relevant_keys = []
        for k in checkpoint:
            if isinstance(checkpoint[k], dict) and "approximated_args" in checkpoint[k]:
                relevant_keys.append(k)
        return pd.DataFrame(checkpoint)["critic_nn"][
            ["approximated_args", "class_type"]
        ].to_dict()

    def _generate_nn_save_key(self, model: torch.nn.Module):
        """

        Generates a key for saving the model
        the key includes the approximated args, class type - for reproducibility and the state dict of the model
        Args:
            model: the model to save

        Returns:
            dictionary: dictionary containing the model's state

        """

        default_module_args = (
            torch.nn.Module().__dict__.keys() | torch.nn.Module.__dict__.keys()
        )
        class_type = str(model.__class__)
        return {
            "approximated_args": {
                k: v for k, v in model.__dict__.items() if k not in default_module_args
            },
            "class_type": class_type,
            "state_dict": model.state_dict(),
        }

    @abstractmethod
    def save_agent(self, f_name: str) -> dict:
        """
        Saves the agent to a file.

        Args:
            f_name (str): file name
        Returns: a dictionary containing the agent's state.
        """
        save_dict = {
            "agent_type": self.__class__.__name__,
            "obs_space": self.obs_space,
            "obs_shape": self.obs_shape,
            "action_space": self.action_space,
            "discount_factor": self.discount_factor,
        }
        try:
            os.makedirs(os.path.dirname(f_name), exist_ok=True)
        except:
            pass
        torch.save(save_dict, f_name)
        return save_dict

    @abstractmethod
    def load_agent(self, f_name: str):
        """
        Loads the agent from a file.
        Returns: a dictionary containing the agent's state.
        """
        checkpoint = torch.load(f_name, map_location=self.device)
        self.action_space = checkpoint["action_space"]
        self.obs_space = checkpoint["obs_space"]
        self.obs_shape = checkpoint["obs_shape"]
        self.discount_factor = checkpoint["discount_factor"]
        self.define_action_space(self.action_space)
        self.setup_models()
        return checkpoint

    @abstractmethod
    def set_train_mode(self):
        """
        sets the agent to train mode - all models are set to train mode
        """
        self.reset_rnn_hidden()

    @abstractmethod
    def set_eval_mode(self):
        """
        sets the agent to train mode - all models are set to eval mode
        """
        self.reset_rnn_hidden()

    def train_episodial(
        self,
        env: gym.Env,
        n_episodes: int,
        max_episode_len: int = None,
        disable_tqdm: bool = False,
    ):
        """
        Trains the agent for a given number of episodes

        Args:
            env (gym.Env): the environment to train on
            n_episodes (int): number of episodes to train
            max_episode_len (int, optional): maximum episode length - truncates after that. Defaults to None.
            disable_tqdm (bool, optional): disable tqdm. Defaults to False.
        Returns:
            train rewards
        """
        if n_episodes < self.num_parallel_envs:
            self.set_num_parallel_env(n_episodes)
        train_r = self._train_n_iters(
            env,
            n_episodes,
            True,
            max_episode_len=max_episode_len,
            disable_tqdm=disable_tqdm,
        )
        return train_r

    def train_n_steps(
        self,
        env: gym.Env,
        n_steps: int,
        max_episode_len: int = None,
        disable_tqdm: bool = False,
    ):
        """
        Trains the agent for a given number of steps

        Args:
            env (gym.Env): the environment to train on
            n_steps (int): number of steps to train
            max_episode_len (int, optional): maximum episode length - truncates after that. Defaults to None.
            disable_tqdm (bool, optional): disable tqdm. Defaults to False.
        Returns:
            train rewards
        """
        return self._train_n_iters(
            env,
            n_steps,
            episodes=False,
            max_episode_len=max_episode_len,
            disable_tqdm=disable_tqdm,
        )

    def _train_n_iters(
        self,
        env: gym.Env,
        n_iters: int,
        episodes: bool = False,
        max_episode_len: bool = None,
        disable_tqdm: bool = False,
    ):
        """
        Trains the agent for a given number of steps

        Args:
            env (gym.Env): the environment to train on
            n_iters (int): number of steps/episodes to train
            episodes (bool, optional): whether to train for episodes or steps. Defaults to False.
            max_episode_len (int, optional): maximum episode length - truncates after that. Defaults to None.
            disable_tqdm (bool, optional): disable tqdm. Defaults to False.
        Returns:
            train rewards
        """
        self.set_train_mode()

        pbar = tqdm(total=n_iters, leave=None, disable=disable_tqdm)
        curr_training_steps = 0
        train_stats = {"rewards": [], "exploration_eps": []}

        if episodes:
            to_update_idx = 0
        else:
            to_update_idx = 1

        i = 0
        ep_number = 0
        best_agent_score = None
        while i < n_iters:
            self.set_eval_mode()
            rewards_vector = self.collect_episode_obs(
                env, max_episode_len, num_to_collect_in_parallel=self.num_parallel_envs
            )
            num_steps_collected = 0
            for r in rewards_vector:
                train_stats["rewards"].append(np.sum(r))
                train_stats["exploration_eps"].append(self.explorer.exploration_epsilon)
                num_steps_collected += len(r)
            mean_rewrad = np.mean(train_stats["rewards"][-len(rewards_vector) :])

            if best_agent_score is None or mean_rewrad > best_agent_score:
                best_agent_score = mean_rewrad
                self.save_agent("/tmp/best_{}.pt".format(self.id))

            collect_info = [self.num_parallel_envs, num_steps_collected]
            curr_training_steps += num_steps_collected
            desciption = f"episode {ep_number}, curr_mean_R:{np.round(mean_rewrad, 2):08}, best_mean_R:{np.round(best_agent_score,2)}, total_steps:{curr_training_steps}"
            pbar.set_description(desciption)

            pbar.update(collect_info[to_update_idx])
            i += collect_info[to_update_idx]
            ep_number += self.num_parallel_envs
            self.set_train_mode()
            trajectory_data = self.get_trajectories_data()
            self.update_policy(trajectory_data)
            self.metrics.on_epoch_end()

            for k in self.metrics:
                self.writer.add_scalar(k, self.metrics[k][-1], ep_number)
            self.writer.add_scalar("mean_rewrad", mean_rewrad, ep_number)
            self.writer.add_scalar(
                "exploration_eps", self.explorer.exploration_epsilon, ep_number
            )
            self.reset_rnn_hidden()

        pbar.close()
        self.close_env_procs()
        return pd.DataFrame(train_stats, index=range(len(train_stats["rewards"])))

    @abstractmethod
    def get_trajectories_data(self):
        """
        Returns the trajectories data
        """
        raise NotImplementedError

    def set_num_parallel_env(self, num_parallel_envs):
        """
        Sets the number of parallel environments

        Args:
            num_parallel_envs (int): number of parallel environments
        """
        self.num_parallel_envs = num_parallel_envs
        assert (
            self.num_parallel_envs <= self.batch_size
        ), f"please provide batch_size>= num_parallel_envs current: {self.batch_size}, {num_parallel_envs},"
        self.num_parallel_envs = num_parallel_envs

    @abstractmethod
    def act(self, observations: np.array, num_obs: int = 1) -> np.array:
        """

        Args:
            observations: The observations to act on
            num_obs: The number of observations to act on

        Returns:
            The selected actions (np.ndarray)
        """
        raise NotImplementedError

    def load_highest_score_agent(self):
        """
        Loads the highest score agent from training
        """
        self.load_agent("/tmp/best_{}.pt".format(self.id))

    def get_highest_score_agent_ckpt_path(self):
        """
        Returns the path of the highest score agent from training
        """
        return "/tmp/best_{}.pt".format(self.id)

    @abstractmethod
    def best_act(self, observations, num_obs=1):
        """
        The highest probabilities actions in a detrminstic way

        Args:
            observations: The observations to act on
            num_obs: The number of observations to act on
        Returns:
            The highest probabilty action to be taken in a detrministic way
        """
        raise NotImplementedError

    def norm_obs(self, observations):
        """
        Normalizes the observations according to the pre given normalization parameters [future api - currently not availble]
        """
        return observations
        return (observations - self.norm_params["mean"]) / self.norm_params["std"]

    def pre_process_obs_for_act(
        self, observations: [np.array, ObsWrapper, dict], num_obs: int
    ):
        """
        Pre processes the observations for act

        Args:
            observations: The observations to act on
            num_obs: The number of observations to act on
        Returns:
            The pre processed observations an ObsWrapper object with the right dims
        """
        if type(observations) != ObsWrapper:
            observations = ObsWrapper(observations)

        len_obs = len(observations)
        if num_obs != len_obs:
            raise Exception(
                f"number of observations do not match real observation num obs: {num_obs}, vs real len: {len_obs}"
            )
        states = observations.get_as_tensors(self.device)
        return states

    def return_correct_actions_dim(self, actions: np.array, num_obs: int):
        """
        Returns the correct actions dimention

        Args:
            actions: The selected actions
            num_obs: The number of observations to act on
        """
        actions = actions.reshape(num_obs, *self.action_space.shape)
        return actions.astype(self.action_dtype)

    def close_env_procs(self):
        """
        Closes the environment processes
        """

        if self.env is not None:
            self.env.close_procs()
            self.env = None

    def set_intrisic_reward_func(self, func):
        """
        sets the agents inner reward function to a custom function that takes state, action, reward and returns reward for the algorithm::

            # Create some agent
            agent = PPO_Agent(obs_space=env.observation_space, action_space=env.action_space tensorboard_dir = None)
            def dummy_reward_func(state, action, reward):
                if state[0] > 0:
                    return reward + 1
            agent.set_intrisic_reward_func(dummy_reward_func)
            # now train normaly

        Args:
            func (function): a function that takes state, action, reward and returns reward for the algorithm
        """
        self.r_func = func

    def intrisic_reward_func(self, state: np.array, action: np.array, reward: np.array):
        """
        Calculates the agents inner reward
        """
        return self.r_func(state, action, reward)

    def collect_episode_obs(
        self,
        env,
        max_episode_len=None,
        num_to_collect_in_parallel=None,
    ) -> float:
        """
        Collects observations from the environment

        Args:
            env (gym.env): gym environment
            max_episode_len (int, optional): maximum episode length. Defaults to None.
            num_to_collect_in_parallel (int, optional): number of parallel environments. Defaults to None.
            env_funcs (dict, optional): dictionary of env functions mapping to call on the environment. Defaults to {"step": "step", "reset": "reset"}.
        Returns:
            float: total reward collected
        """
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

        if max_episode_len:

            def episode_len_exceeded(x):
                return x > max_episode_len

        else:

            def episode_len_exceeded(x):
                return False

        observations = []

        for item in self.env.reset():
            observations.append([item[0]])

        env_dones = np.array([False for i in range(num_to_collect_in_parallel)])

        latest_observations = [
            observations[i][-1] for i in range(num_to_collect_in_parallel)
        ]

        rewards = [[] for i in range(num_to_collect_in_parallel)]
        actions = [[] for i in range(num_to_collect_in_parallel)]
        dones = [[] for i in range(num_to_collect_in_parallel)]
        truncated = [[] for i in range(num_to_collect_in_parallel)]

        max_episode_steps = 0
        time_to_act = 0
        while not all(env_dones):
            relevant_indices = np.where(env_dones == False)[0].astype(np.int32)

            if self.explorer.explore():
                explore_action = self.explorer.act(
                    self.action_space, latest_observations, num_to_collect_in_parallel
                )
                assert (
                    explore_action.shape[0] == num_to_collect_in_parallel
                ), f"The explorer heuristic functions does not returns the correct number of actions (batch dim) expected: {num_to_collect_in_parallel}, got: {explore_action.shape[0]}"
                current_actions = self.return_correct_actions_dim(
                    explore_action, num_to_collect_in_parallel
                )
            else:
                current_actions = self.act(
                    latest_observations, num_to_collect_in_parallel
                )
            # TODO DEBUG
            # allways use all envs to step, even some envs are done already
            next_obs, reward, terminated, trunc, info = self.env.step(current_actions)
            for i in relevant_indices:
                actions[i].append(current_actions[i])
                intrisic_reward = self.intrisic_reward_func(
                    latest_observations[i], current_actions[i], reward[i]
                )
                rewards[i].append(intrisic_reward)
                truncated[i].append(trunc[i])
                done = terminated[i] or trunc[i]
                dones[i].append(done)
                env_dones[i] = done

                max_episode_steps += 1
                if done:
                    continue

                if episode_len_exceeded(max_episode_steps):
                    dones[i][-1] = True
                    truncated[i][-1] = True
                    env_dones[i] = True
                    break

                observations[i].append(next_obs[i])

            latest_observations = [
                observations[i][-1] for i in range(num_to_collect_in_parallel)
            ]
        observations = functools.reduce(operator.iconcat, observations, [])
        actions = functools.reduce(operator.iconcat, actions, [])
        rewards_x = functools.reduce(operator.iconcat, rewards, [])
        dones = functools.reduce(operator.iconcat, dones, [])
        truncated = functools.reduce(operator.iconcat, truncated, [])

        rewards_x = np.array(rewards_x).astype(np.float32)

        self.max_reward = max(np.abs(rewards_x).max(), self.max_reward)
        if self.reward_normalization:
            rewards_x /= self.max_reward + 1e-4
        self.experience.append(observations, actions, rewards_x, dones, truncated)
        self.reset_rnn_hidden()
        self.explorer.update()
        return rewards

    @abstractmethod
    def reset_rnn_hidden(
        self,
    ):
        """if agent uses rnn, when the hidden states are reset.
        this callback is called in many places so please impliment it in you agent"""
        raise NotImplementedError

    def get_last_collected_experiences(self, number_of_episodes: int):
        """
        returns the last collected experiences

        Args:
            number_of_episodes (int): number of episodes to return
        """
        # Mainly for paired alg
        self.experience.get_last_episodes(number_of_episodes)

    def clear_exp(self):
        """
        clears the experience replay buffer
        """
        self.experience.clear()

    def run_env(
        self, env: gym.Env, best_act: bool = True, num_runs: int = 1
    ) -> np.array:
        """
        Runs the environment with the agent in eval mode
        Args:

            env (gym.Env): the environment to run
            best_act (bool, optional): whether to use the best action or the agent's action. Defaults to True.
            num_runs (int, optional): number of runs. Defaults to 1.

        """

        self.set_eval_mode()
        num_envs = 1  # maybe later supp for parralel exceution
        env = ParallelEnv(env, num_envs)
        act_func = self.act
        if best_act:
            act_func = self.best_act

        all_rewards = []

        for i in range(num_runs):
            obs, _ = env.reset()[0]
            obs = [obs]  # change this for supp for parralel exceution
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
            self.reset_rnn_hidden()
            all_rewards.append(R)
        all_rewards = np.array(all_rewards)
        pygame.quit()
        return {
            "mean": all_rewards.mean(),
            "std": all_rewards.std(),
            "all_runs": all_rewards,
        }
