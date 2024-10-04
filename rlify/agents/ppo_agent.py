import logging
import torch
import torch.nn as nn
import numpy as np
from .agent_utils import ObsWrapper, calc_gaes, LambdaDataset, IData, LambdaData
from rlify.agents.experience_replay import ForgettingExperienceReplay
from .action_spaces_utils import MCAW, MDA
from .explorers import Explorer, RandomExplorer
from .drl_agent import RL_Agent
import adabelief_pytorch
from rlify.utils import HiddenPrints
import gc
from torch.utils.data import Dataset
import gymnasium as gym


class PPODataset(Dataset):
    """
    Dataset for PPO.
    """

    def __init__(
        self,
        states,
        actions,
        dones,
        returns,
        advantages,
        logits,
        prepare_for_rnn,
    ):
        """

        Args:
            states (np.ndarray): The states.
            actions (np.ndarray): The actions.
            dones (np.ndarray): The dones.
            returns (np.ndarray): The returns.
            advantages (np.ndarray): The advantages.
            logits (np.ndarray): The logits.
            prepare_for_rnn (bool): Whether to prepare for RNN.

        """
        obs_collection = (states,)
        tensor_collection = actions, returns, advantages, logits
        self.x_dataset = LambdaDataset(
            obs_collection,
            tensor_collection=tensor_collection,
            dones=dones,
            prepare_for_rnn=prepare_for_rnn,
        )
        self.prepare_for_rnn = prepare_for_rnn

    def __len__(self):
        return len(self.x_dataset)

    def __getitems__(self, idx):
        obs_collection, tensor_collection, dones, loss_flag = (
            self.x_dataset.__getitems__(idx)
        )
        states = obs_collection[0]
        actions, returns, advantages, logits = tensor_collection
        return (
            states,
            actions,
            dones,
            returns,
            advantages,
            logits,
            loss_flag,
        )

    def __getitem__(self, idx):
        return self.__getitems__(idx)

    def collate_fn(self, batch):
        return batch


class PPOData(IData):
    """
    A class for PPO data.
    """

    def __init__(
        self,
        states,
        actions,
        dones,
        returns,
        advantages,
        logits,
        prepare_for_rnn,
    ):
        """

        Args:
            states (np.ndarray): The states.
            actions (np.ndarray): The actions.
            dones (np.ndarray): The dones.
            returns (np.ndarray): The returns.
            advantages (np.ndarray): The advantages.
            logits (np.ndarray): The logits.
            prepare_for_rnn (bool): Whether to prepare for RNN.

        """
        dataset = PPODataset(
            states, actions, dones, returns, advantages, logits, prepare_for_rnn
        )
        super().__init__(dataset, prepare_for_rnn)


class PPO_Agent(RL_Agent):
    """Proximal Policy Optimization (PPO) reinforcement learning agent.
    Inherits from RL_Agent.
    """

    def __init__(
        self,
        obs_space: gym.spaces,
        action_space: gym.spaces,
        policy_nn,
        critic_nn,
        batch_size: int = 1024,
        entropy_coeff: float = 0.1,
        kl_div_thresh: float = 0.03,
        clip_param: float = 0.1,
        explorer: Explorer = RandomExplorer(0, 0, 0),
        num_parallel_envs: int = 4,
        num_epochs_per_update: int = 10,
        lr: float = 3e-4,
        device: str = None,
        experience_class: object = ForgettingExperienceReplay,
        max_mem_size: int = int(10e5),
        discount_factor: float = 0.99,
        normlize_obs: str = "auto",
        reward_normalization=True,
        tensorboard_dir: str = "./tensorboard",
        dataloader_workers: int = 0,
        accumulate_gradients_per_epoch: bool = None,
    ):
        """
        Example::

            env_name = 'Pendulum-v1'
            env = gym.make(env_name, render_mode=None)
            models_shapes = PPO_Agent.get_models_input_output_shape(env.observation_space, env.action_space)
            policy_input_shape = models_shapes["policy_nn"]["input_shape"]
            policy_out_shape = models_shapes["policy_nn"]["out_shape"]
            critic_input_shape = models_shapes["critic_nn"]["input_shape"]
            critic_out_shape = models_shapes["critic_nn"]["out_shape"]
            policy_nn = fc.FC(input_shape=policy_input_shape, embed_dim=128, depth=3, activation=torch.nn.ReLU(), out_shape=policy_out_shape)
            critic_nn = fc.FC(input_shape=critic_input_shape, embed_dim=128, depth=3, activation=torch.nn.ReLU(), out_shape=critic_out_shape)
            agent = PPO_Agent(obs_space=env.observation_space, action_space=env.action_space, device=device, batch_size=1024, max_mem_size=10**5,
                            num_parallel_envs=4, lr=3e-4, entropy_coeff=0.05, policy_nn=policy_nn, critic_nn=critic_nn, discount_factor=0.99, tensorboard_dir = None)
            train_stats = agent.train_n_steps(env=env,n_steps=250000)

        Args:
            obs_space (gym.spaces): The observation space of the environment.
            action_space (gym.spaces): The action space of the environment.
            policy_nn (nn.Module): The policy neural network.
            critic_nn (nn.Module): The critic neural network.
            batch_size (int): The batch size for training.
            entropy_coeff (float): The coefficient for the entropy regularization term.
            kl_div_thresh (float): The threshold for the KL divergence between old and new policy.
            clip_param (float): The clipping parameter for the PPO loss.
            explorer (Explorer): The exploration strategy.
            num_parallel_envs (int): The number of parallel environments.
            num_epochs_per_update (int): The number of epochs per update.
            lr (float): The learning rate.
            device (str): The device to use for training.
            experience_class (object): The experience replay class.
            max_mem_size (int): The maximum memory size for experience replay.
            discount_factor (float): The discount factor for future rewards.
            reward_normalization (bool): Whether to normalize rewards.
            tensorboard_dir (str): The directory to save tensorboard logs.
            dataloader_workers (int): The number of workers for the dataloader.

        """
        self.policy_nn = policy_nn
        self.critic_nn = critic_nn
        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            batch_size=batch_size,
            explorer=explorer,
            num_parallel_envs=num_parallel_envs,
            num_epochs_per_update=num_epochs_per_update,
            lr=lr,
            device=device,
            experience_class=experience_class,
            max_mem_size=max_mem_size,
            discount_factor=discount_factor,
            normlize_obs=normlize_obs,
            reward_normalization=reward_normalization,
            tensorboard_dir=tensorboard_dir,
            dataloader_workers=dataloader_workers,
        )

        self.kl_div_thresh = kl_div_thresh
        self.clip_param = clip_param
        self.optimization_step = 0
        self.losses = []
        self.entropy_coeff = entropy_coeff
        self.criterion = nn.MSELoss().to(self.device)
        if self.possible_actions == "continuous":
            self.best_act_func = self.best_act_cont
        else:
            self.best_act_func = self.best_act_discrete

    def set_train_mode(self):
        super().set_train_mode()
        self.policy_nn.train()
        self.critic_nn.train()

    def set_eval_mode(self):
        super().set_eval_mode()
        self.policy_nn.eval()
        self.critic_nn.eval()

    @staticmethod
    def get_models_input_output_shape(obs_space, action_space) -> dict:
        """
        Returns the input and output shapes of the Q model.
        """
        n_actions, possible_actions = RL_Agent.read_action_space_properties(
            action_space
        )
        if np.issubdtype(action_space.dtype, np.integer):
            out_shape = possible_actions * n_actions
        else:
            out_shape = 2 * n_actions  # for param trick

        obs_space = RL_Agent.read_obs_space_properties(obs_space)

        return {
            "critic_nn": {
                "input_shape": obs_space,
                "out_shape": (1,),
            },
            "policy_nn": {
                "input_shape": obs_space,
                "out_shape": out_shape,
            },
        }

    def setup_models(self):
        # self.exp_sigma = self.GSDE()
        # self.exp_sigma.to(self.device)
        self.policy_nn = self.policy_nn.to(self.device)
        self.critic_nn = self.critic_nn.to(self.device)
        if np.issubdtype(self.action_dtype, np.integer):
            self.actor_model = lambda x: MDA(
                self.action_space.start,
                self.possible_actions,
                self.n_actions,
                self.policy_nn(x).reshape(-1, self.possible_actions),
            )
        else:
            self.actor_model = lambda x: MCAW(
                self.action_space.low,
                self.action_space.high,
                self.policy_nn(x).reshape(-1, 2 * self.n_actions),
            )

        # weight = list(self.policy_nn.children())[-1].weight.data
        # bias = list(self.policy_nn.children())[-1].bias.data
        # list(self.policy_nn.children())[-1].weight.data = (weight) * 0.01
        # list(self.policy_nn.children())[-1].bias.data = (bias) * 0.01

        with HiddenPrints():
            self.actor_optimizer = adabelief_pytorch.AdaBelief(
                self.policy_nn.parameters(),
                self.lr,
                amsgrad=False,
            )
            self.critic_optimizer = adabelief_pytorch.AdaBelief(
                self.critic_nn.parameters(),
                self.lr,
                amsgrad=False,
            )
            return [self.policy_nn, self.critic_nn]

    def save_agent(self, f_name) -> dict:
        save_dict = super().save_agent(f_name)
        save_dict["actor_optimizer"] = self.actor_optimizer.state_dict()
        save_dict["policy_nn"] = self._generate_nn_save_key(self.policy_nn)
        save_dict["critic_optimizer"] = self.critic_optimizer.state_dict()
        save_dict["critic_nn"] = self._generate_nn_save_key(self.critic_nn)
        save_dict["entropy_coeff"] = self.entropy_coeff
        save_dict["discount_factor"] = self.discount_factor
        torch.save(save_dict, f_name)
        return save_dict

    def load_agent(self, f_name):
        checkpoint = super().load_agent(f_name)
        self.policy_nn.load_state_dict(checkpoint["policy_nn"]["state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_nn.load_state_dict(checkpoint["critic_nn"]["state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    def reset_rnn_hidden(
        self,
    ):
        self.policy_nn.reset()
        self.critic_nn.reset()

    def set_num_parallel_env(self, num_parallel_envs):
        super().set_num_parallel_env(num_parallel_envs)

    def best_act(self, observations, num_obs=1):
        return self.best_act_func(observations, num_obs)

    def best_act_discrete(self, observations, num_obs=1):
        states = self.pre_process_obs_for_act(observations, num_obs)

        with torch.no_grad():
            actions_dist = self.actor_model(states)
        selected_actions = torch.argmax(actions_dist.probs, 1).detach().cpu().numpy()
        return self.return_correct_actions_dim(selected_actions, num_obs)

    def best_act_cont(self, observations, num_obs=1):
        states = self.pre_process_obs_for_act(observations, num_obs)

        with torch.no_grad():
            actions_dist = self.actor_model(states)

        selected_actions = actions_dist.loc.detach().cpu().numpy()
        return self.return_correct_actions_dim(selected_actions, num_obs)

    def act(self, observations, num_obs=1):
        states = self.pre_process_obs_for_act(observations, num_obs)

        with torch.no_grad():
            actions_dist = self.actor_model(states)
            action = actions_dist.sample()

        selected_actions = action.detach().cpu().numpy()
        return self.return_correct_actions_dim(selected_actions, num_obs)

    def get_trajectories_data(self):
        return self._get_ppo_experiences()

    def calc_logits_values(self, states, actions, dones):
        data = LambdaData(
            obs_collection=(states,),
            tensor_collection=(actions,),
            dones=dones,
            prepare_for_rnn=self.contains_reccurent_nn(),
        )
        dl = data.get_dataloader(
            self.get_train_batch_size(),
            shuffle=False,
            num_workers=self.dataloader_workers,
        )

        values = []
        logits = []
        batched_loss_flags_list = []
        training = self.training

        self.set_eval_mode()
        with torch.no_grad():

            for mb in dl:
                (
                    obs_collection,
                    tensor_collection,
                    batched_dones,
                    batched_loss_flags,
                ) = mb
                batched_loss_flags_list.append(batched_loss_flags)
                b = batched_dones.shape[0]
                batched_states = obs_collection[0]
                batched_actions = tensor_collection[0]
                batched_actions = batched_actions.to(self.device, non_blocking=True)
                batched_states = batched_states.to(self.device)
                dist = self.actor_model(batched_states)
                logit = dist.log_prob(batched_actions.reshape(-1, self.n_actions))
                logit = logit.reshape(((b, -1, self.n_actions)))
                logits.append(logit.to("cpu"))
                values.append(self.critic_nn(batched_states).to("cpu"))

        if training:
            self.set_train_mode()
        try:
            batched_loss_flags = torch.cat(batched_loss_flags_list).flatten()
        except:
            breakpoint()
        if self.contains_reccurent_nn():
            return (
                torch.cat(logits, 1)
                .reshape(-1, self.n_actions)
                .detach()
                .cpu()
                .numpy()[batched_loss_flags],
                torch.cat(values, 1)
                .flatten()
                .detach()
                .cpu()
                .numpy()[batched_loss_flags],
            )
        else:
            return (
                torch.cat(logits)
                .reshape(-1, self.n_actions)
                .detach()
                .cpu()
                .numpy()[batched_loss_flags],
                torch.cat(values).flatten().detach().cpu().numpy()[batched_loss_flags],
            )

    def _get_ppo_experiences(self, num_episodes=None):
        """
        Get the experiences for PPO
        Args:
            num_episodes (int): Number of episodes to get.

        Returns:
            tuple: (states, actions, rewards, dones, truncated, next_states)

        """
        if num_episodes is None:
            num_episodes = self.num_parallel_envs
            states, actions, rewards, dones, truncated, next_states = (
                self.experience.get_last_episodes(num_episodes)
            )
        logits, values = self.calc_logits_values(states, actions, dones)
        advantages, returns = calc_gaes(
            rewards,
            values,
            terminated=dones * (1 - truncated),
            discount_factor=self.discount_factor,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        returns = np.expand_dims(returns, -1)
        trajectory_data = PPOData(
            states,
            actions,
            dones,
            returns,
            advantages,
            logits,
            self.contains_reccurent_nn(),
        )
        return trajectory_data

    def update_policy(self, trajectory_data: PPOData):
        """
        Update the policy network.
        Args: exp (tuple): Experience tuple.
        """

        shuffle = False if (self.policy_nn.is_rnn) else True
        ppo_dataloader = trajectory_data.get_dataloader(
            self.get_train_batch_size(),
            shuffle=shuffle,
            num_workers=self.dataloader_workers,
        )
        for e in range(self.num_epochs_per_update):
            kl_div_bool = False
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            for b, mb in enumerate(ppo_dataloader):
                (
                    batched_states,
                    batched_actions,
                    batched_dones,
                    batched_returns,
                    batched_advantages,
                    batched_logits,
                    batched_loss_flags,
                ) = mb
                batched_returns = batched_returns.to(self.device, non_blocking=True)
                batched_advantages = batched_advantages.to(
                    self.device, non_blocking=True
                )
                old_log_probs = batched_logits.to(self.device, non_blocking=True)
                batched_actions = batched_actions.to(self.device, non_blocking=True)
                batched_states = batched_states.to(self.device)
                batched_dones = batched_dones.to(self.device)
                dist = self.actor_model(batched_states)
                critic_values = self.critic_nn(batched_states)
                new_log_probs = dist.log_prob(
                    batched_actions.reshape(-1, self.n_actions)
                )
                old_log_probs = old_log_probs.reshape_as(new_log_probs)
                log_ratio = torch.clamp(new_log_probs, np.log(1e-3), 0.0) - torch.clamp(
                    old_log_probs, np.log(1e-3), 0.0
                )
                ratio = log_ratio.exp().mean(-1)
                log_ratio = ratio.log()
                batched_advantages = batched_advantages.reshape_as(ratio)
                surr1 = ratio * batched_advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 / (1.0 + self.clip_param), 1.0 + self.clip_param
                    )
                    * batched_advantages
                )
                actor_criterion = lambda x, y: -torch.min(x, y).mean()
                actor_loss = self.criterion_using_loss_flag(
                    actor_criterion, surr1, surr2, batched_loss_flags
                )
                entropy = dist.entropy()
                actor_loss = actor_loss - self.apply_regularization(
                    self.entropy_coeff, entropy, batched_loss_flags
                )

                kl_div = (
                    -(ratio * log_ratio - (ratio - 1))[batched_loss_flags.flatten()]
                    .mean()
                    .item()
                )
                if np.abs(kl_div) > self.kl_div_thresh:
                    kl_div_bool = True
                    if e == 0 and b == 0:
                        raise Exception(
                            "Something unexpected happend please report it - kl div exceeds before first grad call",
                            e,
                            kl_div,
                        )
                    break

                critic_loss = self.criterion_using_loss_flag(
                    self.criterion,
                    critic_values,
                    batched_returns,
                    batched_loss_flags,
                )
                if not self.accumulate_gradients_per_epoch:
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    critic_loss = self.criterion_using_loss_flag(
                        self.criterion,
                        critic_values,
                        batched_returns,
                        batched_loss_flags,
                    )
                    critic_loss.backward()
                    self.critic_optimizer.step()
                else:
                    actor_loss = actor_loss / len(ppo_dataloader)
                    actor_loss.backward()
                    critic_loss = self.criterion_using_loss_flag(
                        self.criterion,
                        critic_values,
                        batched_returns,
                        batched_loss_flags,
                    )
                    critic_loss = critic_loss / len(ppo_dataloader)
                    critic_loss.backward()

                self.metrics.add("kl_div", kl_div)
                self.metrics.add("critic_loss", critic_loss.item())
                self.metrics.add("actor_loss", actor_loss.item())

            if kl_div_bool:
                break
            if self.accumulate_gradients_per_epoch:
                self.actor_optimizer.step()
                self.critic_optimizer.step()
