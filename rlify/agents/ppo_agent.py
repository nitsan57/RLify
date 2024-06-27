import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .agent_utils import ObsWrapper, calc_gaes
from rlify.agents.experience_replay import ForgettingExperienceReplay
from .action_spaces_utils import MCAW, MDA
from .explorers import Explorer, RandomExplorer
from .drl_agent import RL_Agent, IData, RLData, RLDataset, pad_tensors_from_done_indices
import adabelief_pytorch
from rlify.utils import HiddenPrints
import gc
from torch.utils.data import Dataset, DataLoader


class PPODataset(Dataset):
    def __init__(
        self,
        rl_dataset: RLDataset,
        values,
        returns,
        advantages,
        logits,
    ):
        self.rl_dataset = rl_dataset
        values, returns, advantages, logits = self._prepare_values_logits(
            values, returns, advantages, logits
        )
        self.values = values
        self.returns = returns
        self.advantages = advantages
        self.logits = logits
        if self.rl_dataset.prepare_for_rnn:
            (
                self.values,
                self.returns,
                self.logits,
                self.advantages,
                lengths,
            ) = self._pad_values_logits(
                values, returns, advantages, logits, self.rl_dataset.dones
            )

    def __len__(self):
        return self.rl_dataset.__len__()

    def _prepare_values_logits(self, values, returns, advantages, logits):
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
        values = torch.from_numpy(values)
        returns = torch.from_numpy(returns)
        advantages = torch.from_numpy(advantages)
        logits = torch.from_numpy(logits)
        return values, returns, advantages, logits

    def _pad_values_logits(self, values, returns, advantages, logits, dones):
        """
        Creates a padded version of the data
        Args:
            values: The values
            logits: The logits

        Returns:
            The padded values and logits, and lengths
        """
        padded_values, lengths = pad_tensors_from_done_indices(values, dones)
        padded_returns, lengths = pad_tensors_from_done_indices(returns, dones)
        padded_advantages = pad_tensors_from_done_indices(advantages, dones)
        padded_logits, lengths = pad_tensors_from_done_indices(logits, dones)
        return (
            padded_values,
            padded_advantages,
            padded_returns,
            padded_logits,
            lengths,
        )

    def __getitem__(self, idx):
        """
        Gets the item at the index
        Args:
            idx: item idx

        Returns:
            batched version of the data at idx
            states, actions, rewards, dones, truncated, next_states, loss_flag

        """
        states, actions, rewards, dones, truncated, next_states, loss_flag = (
            self.rl_dataset.__getitem__(idx)
        )
        returns = self.returns[idx]
        values = self.values[idx]
        advantages = self.advantages[idx]
        logits = self.logits[idx]
        return (
            states,
            actions,
            rewards,
            dones,
            truncated,
            next_states,
            loss_flag,
            values,
            returns,
            advantages,
            logits,
        )

    def collate_fn(self, batch):
        (
            states,
            actions,
            rewards,
            dones,
            truncated,
            next_states,
            loss_flag,
            values,
            returns,
            advantages,
            logits,
        ) = zip(*batch)
        (
            states,
            actions,
            rewards,
            dones,
            truncated,
            next_states,
            loss_flag,
        ) = self.rl_dataset.collate_fn(
            list(
                zip(states, actions, rewards, dones, truncated, next_states, loss_flag)
            )
        )
        values = torch.stack(values)
        returns = torch.stack(returns)
        advantages = torch.stack(advantages)
        logits = torch.stack(logits)
        return (states, actions, dones, returns, advantages, logits, loss_flag)
        return (
            states,
            actions,
            rewards,
            dones,
            truncated,
            values,
            returns,
            advantages,
            logits,
            loss_flag,
        )


class PPOData(IData):
    def __init__(
        self,
        rl_data: RLData,
        values,
        returns,
        advantages,
        logits,
        num_workers: int = 2,
    ):
        self.num_workers = num_workers
        self.rl_data = rl_data
        self.dataset = PPODataset(
            rl_data.dataset,
            values,
            returns,
            advantages,
            logits,
        )
        self.can_shuffle = False if self.rl_data.prepare_for_rnn else True

    def get_data_loader(self, batch_size, shuffle=True):
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


class PPO_Agent(RL_Agent):
    """Proximal Policy Optimization (PPO) reinforcement learning agent.
    Inherits from RL_Agent.
    """

    def __init__(
        self,
        policy_nn,
        critic_nn,
        batch_size: int = 1024,
        entropy_coeff: float = 0.1,
        num_epochs_per_update: int = 10,
        kl_div_thresh: float = 0.03,
        clip_param: float = 0.1,
        experience_class: ForgettingExperienceReplay = ForgettingExperienceReplay,
        explorer: Explorer = RandomExplorer(0, 0, 0),
        *args,
        **kwargs,
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
            batch_size (int): Batch size for sampling from replay buffer.
            entropy_coeff (float): Entropy regularization coefficient.
            num_epochs_per_update (int): Training epochs per update.
            kl_div_thresh (float): KL divergence threshold.
            clip_param (float): Clipping parameter.
            experience_class (ForgettingExperienceReplay): Experience replay class to use.
            explorer (Explorer): Class for random exploration.
            kwArgs: Additional RL_Agent arguments.
        """
        self.policy_nn = policy_nn
        self.critic_nn = critic_nn
        super().__init__(
            *args,
            **kwargs,
            num_epochs_per_update=num_epochs_per_update,
            batch_size=batch_size,
            experience_class=experience_class,
            explorer=explorer,
        )  # inits

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
            self.actor_model = lambda x, d=torch.ones((1, 1)): MDA(
                self.action_space.start,
                self.possible_actions,
                self.n_actions,
                self.policy_nn(x, d),
            )
        else:
            self.actor_model = lambda x, d=torch.ones((1, 1)): MCAW(
                self.action_space.low, self.action_space.high, self.policy_nn(x, d)
            )

        weight = list(self.policy_nn.children())[-1].weight.data
        bias = list(self.policy_nn.children())[-1].bias.data
        list(self.policy_nn.children())[-1].weight.data = (weight) * 0.01
        list(self.policy_nn.children())[-1].bias.data = (bias) * 0.01

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
            actions_dist = self.actor_model(states, torch.ones((num_obs, 1)))
        selected_actions = torch.argmax(actions_dist.probs, 1).detach().cpu().numpy()
        return self.return_correct_actions_dim(selected_actions, num_obs)

    def best_act_cont(self, observations, num_obs=1):
        states = self.pre_process_obs_for_act(observations, num_obs)

        with torch.no_grad():
            actions_dist = self.actor_model(states, torch.ones((num_obs, 1)))

        selected_actions = actions_dist.loc.detach().cpu().numpy()
        return self.return_correct_actions_dim(selected_actions, num_obs)

    def act(self, observations, num_obs=1):
        states = self.pre_process_obs_for_act(observations, num_obs)

        with torch.no_grad():
            actions_dist = self.actor_model(
                states, torch.ones((num_obs, 1), device=self.device)
            )
            action = actions_dist.sample()

        selected_actions = action.detach().cpu().numpy()

        return self.return_correct_actions_dim(selected_actions, num_obs)

    def get_trajectories_data(self):
        return self._get_ppo_experiences()

    # def reccurent_minibatch_generator(self, padded_data, max_len):
    #     # write a code piece
    #     # run a for loop that goes through the padded states and calculates the values and logits
    #     num_sequences_per_batch = self.batch_size // self.num_parallel_envs
    #     # Arrange a list that determines the sequence count for each mini batch
    #     # for i in range(0, max_len, num_sequences_per_batch):

    #     for i in range(0, max_len, num_sequences_per_batch):
    #         temp = [
    #             d[:, i : i + num_sequences_per_batch].to(self.device)
    #             for d in padded_data
    #         ]
    #         yield temp

    # def minibatch_generator(self, data):
    #     # write a code piece
    #     # run a for loop that goes through the padded states and calculates the values and logits
    #     # Arrange a list that determines the sequence count for each mini batch

    #     for i in range(0, len(data), self.batch_size):
    #         temp = [d[:, i : i + self.batch_size].to(self.device) for d in data]
    #         yield temp

    def calc_logits_values(self, trajectory_data: RLData):
        mb_gen = trajectory_data.get_data_loader(self.batch_size, shuffle=False)
        values = []
        logits = []
        self.set_eval_mode()
        with torch.no_grad():
            for mb in mb_gen:
                (
                    batched_states,
                    batched_actions,
                    batched_rewards,
                    batched_dones,
                    batched_truncated,
                    batched_next_states,
                    batched_loss_flags,
                ) = mb
                batched_actions = batched_actions.to(self.device, non_blocking=True)
                batched_states = batched_states.to(self.device)
                values.append(
                    self.critic_nn(batched_states, batched_dones.to(self.device))
                    .squeeze()
                    .to("cpu")
                )
                logit = self.actor_model(
                    batched_states, batched_dones.to(self.device)
                ).log_prob(batched_actions)
                logits.append(logit.to("cpu"))
        return torch.cat(logits), torch.cat(values)

    # def ppo_minibatch_generator(
    #     self, states, actions, rewards, dones, truncated, next_states, logits, values
    # ):

    #     terminated = dones * (1 - truncated)
    #     advantages, returns = calc_gaes(
    #         rewards, values, terminated, self.discount_factor
    #     )
    #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
    #     returns = returns.unsqueeze(-1)
    #     if False:
    #         padded_states, trajectories_lengths = self.pad_from_done_indices(
    #             states, dones
    #         )
    #         # write a code piece
    #         # run a for loop that goes through the padded states and calculates the values and logits
    #         num_sequences_per_batch = self.batch_size // len(padded_states)
    #         # )  # Arrange a list that determines the sequence count for each mini batch
    #         max_len = trajectories_lengths.max()
    #         for i in range(0, max_len, num_sequences_per_batch):
    #             batch_states = padded_states[:, i : i + num_sequences_per_batch]
    #             breakpoint()
    #             values.append(
    #                 self.critic_nn(batch_states.to(self.device), dones).squeeze()
    #             )
    #             dist = self.actor_model(batch_states, dones).log_prob(
    #                 actions[i : i + num_sequences_per_batch]
    #             )
    #             logits.append(dist.log_prob(actions[i : i + num_sequences_per_batch]))
    #     else:
    #         mb_gen = self.minibatch_generator(
    #             [states, actions, rewards, dones, truncated, next_states],
    #             trajectories_lengths.shape[1],
    #         )
    #         # for mb in mb_gen:
    #         #     batch_states, batched_actions, batched_dones, batched_rewards, batched_dones, batched_truncated, batch_next_states = mb
    #         #     batched_logits, batched_values = self.calc_logits_values(batch_states, batched_actions, batched_dones)
    #         #     yield batch_states, batched_actions, batched_dones, value, logit

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
        rl_data = RLData(
            states,
            actions,
            rewards,
            dones,
            truncated,
            next_states,
            self.contains_reccurent_nn(),
        )
        logits, values = self.calc_logits_values(rl_data)
        values = values.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        advantages, returns = calc_gaes(
            rewards,
            values,
            terminated=dones * (1 - truncated),
            discount_factor=self.discount_factor,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        returns = np.expand_dims(returns, -1)
        trajectory_data = PPOData(
            rl_data,
            values,
            returns,
            advantages,
            logits,
        )
        return trajectory_data

        # (
        #     states,
        #     actions,
        #     rewards,
        #     dones,
        #     truncated,
        #     next_states,
        #     loss_flag,
        #     trajectories_lengths,
        # ) = self.experience.get_last_episodes(num_episodes, padded=True)
        # if self.contains_reccurent_nn():
        #     (
        #         states,
        #         actions,
        #         rewards,
        #         dones,
        #         truncated,
        #         next_states,
        #         loss_flag,
        #         trajectories_lengths,
        #     ) = self.experience.get_last_episodes(num_episodes, padded=True)

        # else:
        #     (
        #         states,
        #         actions,
        #         rewards,
        #         dones,
        #         truncated,
        #         next_states,
        #     ) = self.experience.get_last_episodes(num_episodes, padded=False)

        # can go inside expericnce buffer
        #############
        # gen = self.minibatch_generator(
        #     [states, actions, rewards, dones, truncated, next_states],
        #     trajectories_lengths.shape[1],
        # # )
        # breakpoint()

        # breakpoint()
        # #     terminated = dones * (1 - truncated)
        # #     advantages, returns = calc_gaes(
        # #         rewards, values, terminated, self.discount_factor
        # #     )
        # #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        # #     returns = returns.unsqueeze(-1)
        # # logits, values = self.calc_logits_values(states, actions, dones)

        # # mini_batch_generator = self.mini_batch_generator([states, actions, rewards, dones, truncated])
        # # for mini_batch in mini_batch_generator:
        # #     states, actions, rewards, dones, truncated = mini_batch

        # prev_i = 0
        # values = []
        # logits = []
        # # self.set_eval_mode()
        # # with torch.no_grad():
        # if False:
        #     padded_states, trajectories_lengths = self.pad_from_done_indices(
        #         states, dones
        #     )
        #     # write a code piece
        #     # run a for loop that goes through the padded states and calculates the values and logits
        #     num_sequences_per_batch = self.batch_size // len(padded_states)
        #     # )  # Arrange a list that determines the sequence count for each mini batch
        #     max_len = trajectories_lengths.max()
        #     for i in range(0, max_len, num_sequences_per_batch):
        #         batch_states = padded_states[:, i : i + num_sequences_per_batch]
        #         breakpoint()
        #         values.append(
        #             self.critic_nn(batch_states.to(self.device), dones).squeeze()
        #         )
        #         dist = self.actor_model(batch_states, dones).log_prob(
        #             actions[i : i + num_sequences_per_batch]
        #         )
        #         logits.append(dist.log_prob(actions[i : i + num_sequences_per_batch]))
        # else:

        #     def gen():
        #         self.set_eval_mode()
        #         with torch.no_grad():
        #             for b in range(0, len(states), self.batch_size):
        #                 batch_states = states[b : b + self.batch_size].to(self.device)
        #                 batched_actions = actions[b : b + self.batch_size].to(
        #                     self.device
        #                 )
        #                 batched_dones = dones[b : b + self.batch_size].to(self.device)
        #                 # values.append(
        #                 #     self.critic_nn(batch_states, batched_dones).squeeze().to("cpu")
        #                 # )
        #                 logit = self.actor_model(batch_states, batched_dones).log_prob(
        #                     batched_actions
        #                 )
        #                 # logits.append(logit.flatten().to("cpu"))
        #                 yield batch_states, batched_actions, batched_dones, self.critic_nn(
        #                     batch_states, batched_dones
        #                 ).squeeze(), logit.flatten()

        # return gen
        # breakpoint()

        # # self.set_eval_mode()
        # # with torch.no_grad():
        # #     values = self.critic_nn(states, dones).squeeze()
        # #     dist = self.actor_model(states, dones)
        # # self.set_train_mode()
        # # logits = dist.log_prob(actions)
        # return (
        #     states,
        #     actions,
        #     rewards,
        #     dones,
        #     truncated,
        #     torch.cat(values),
        #     torch.cat(logits),
        # )

    def update_policy(self, trajectory_data: PPOData):
        """
        Update the policy network.
        Args: exp (tuple): Experience tuple.
        """

        # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

        # all_samples_len = len(states)
        # b_size = self.batch_size if not self.policy_nn.is_rnn else all_samples_len
        # rand_perm = False if (self.policy_nn.is_rnn) else True
        shuffle = False if (self.policy_nn.is_rnn) else True
        ppo_dataloader = trajectory_data.get_data_loader(
            self.batch_size, shuffle=shuffle
        )
        for e in range(self.num_epochs_per_update):
            # mini_batch_generator = self.rollout.mini_batch_generator()
            # for mini_batch in mini_batch_generator:
            # for b in range(0, all_samples_len, b_size):
            # indices_perm = (
            #     torch.randperm(len(returns))
            #     if rand_perm
            #     else torch.arange(len(returns))
            # )
            # states = states[indices_perm]
            # actions = actions[indices_perm]
            # returns = returns[indices_perm]
            # advantages = advantages[indices_perm]
            # logits = logits[indices_perm]
            kl_div_bool = False
            # for b in range(0, all_samples_len, b_size):
            for b, mb in enumerate(ppo_dataloader):
                (
                    batch_states,
                    batched_actions,
                    # batched_rewards,
                    batched_dones,
                    # batched_truncated,
                    # batched_values,
                    batched_returns,
                    batched_advantages,
                    batched_logits,
                    batched_loss_flags,
                ) = mb
                # batch_states = states[b : b + b_size]
                # batched_actions = actions[b : b + b_size]
                # batched_returns = returns[b : b + b_size]
                # batched_advantage = advantages[b : b + b_size]
                # batched_logits = logits[b : b + b_size]
                # batched_dones = dones[b : b + b_size]
                batched_returns = batched_returns.to(self.device, non_blocking=True)
                batched_advantages = batched_advantages.to(
                    self.device, non_blocking=True
                )
                old_log_probs = batched_logits.to(self.device, non_blocking=True)
                batched_actions = batched_actions.to(self.device, non_blocking=True)
                batch_states = batch_states.to(self.device)
                batched_dones = batched_dones.to(self.device)
                dist = self.actor_model(batch_states, batched_dones)
                critic_values = self.critic_nn(batch_states, batched_dones)
                new_log_probs = dist.log_prob(batched_actions)

                log_ratio = torch.clamp(new_log_probs, np.log(1e-3), 0.0) - torch.clamp(
                    old_log_probs, np.log(1e-3), 0.0
                )
                ratio = log_ratio.exp().mean(-1)
                log_ratio = torch.log(ratio)

                surr1 = ratio * batched_advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 / (1.0 + self.clip_param), 1.0 + self.clip_param
                    )
                    * batched_advantages
                )
                entropy = dist.entropy().mean()
                actor_loss = (
                    -(torch.min(surr1, surr2).mean()) - self.entropy_coeff * entropy
                )
                kl_div = (
                    -(ratio * log_ratio - (ratio - 1)).mean().item()
                )  # kl_div = (old_log_probs - new_log_probs).mean().item() #

                if np.abs(kl_div) > self.kl_div_thresh:
                    kl_div_bool = True
                    if e == 0 and b == 0:
                        raise Exception(
                            "Something unexpected happend please report it - kl div exceeds before first grad call",
                            e,
                            kl_div,
                        )
                    # print("kl div exceeded", e, kl_div)
                    break
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                # nn.utils.clip_grad_norm_(self.policy_nn.parameters(), 0.5)
                self.actor_optimizer.step()
                critic_loss = self.criterion(critic_values, batched_returns)
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                self.critic_optimizer.step()
                self.metrics.add("kl_div", kl_div)
                self.metrics.add("critic_loss", critic_loss.item())
                self.metrics.add("actor_loss", actor_loss.item())

            if kl_div_bool:
                break
