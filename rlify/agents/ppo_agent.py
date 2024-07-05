import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data.dataset import ConcatDataset
from .agent_utils import ObsWrapper, calc_gaes, LambdaDataset, IData, LambdaData
from rlify.agents.experience_replay import ForgettingExperienceReplay
from .action_spaces_utils import MCAW, MDA
from .explorers import Explorer, RandomExplorer
from .drl_agent import RL_Agent
import adabelief_pytorch
from rlify.utils import HiddenPrints
import gc
from torch.utils.data import Dataset


class PPODataset(Dataset):
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
        obsWrapper_obs_collection, tensor_collection, dones, loss_flag = (
            self.x_dataset.__getitems__(idx)
        )
        states = obsWrapper_obs_collection[0]
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

    def calc_logits_values(self, states, actions, dones):
        data = LambdaData(
            obsWrapper_obs_collection=(states,),
            tensor_collection=(actions,),
            dones=dones,
            prepare_for_rnn=self.contains_reccurent_nn(),
        )
        dl = data.get_dataloader(
            self.batch_size, shuffle=False, num_workers=self.dataloader_workers
        )

        values = []
        logits = []
        self.set_eval_mode()
        with torch.no_grad():
            for mb in dl:
                (
                    obsWrapper_obs_collection,
                    tensor_collection,
                    batched_dones,
                    batched_loss_flags,
                ) = mb
                batched_states = obsWrapper_obs_collection[0]
                batched_actions = tensor_collection[0]
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
        return (
            torch.cat(logits).detach().cpu().numpy(),
            torch.cat(values).detach().cpu().numpy(),
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
            self.batch_size, shuffle=shuffle, num_workers=self.dataloader_workers
        )
        for e in range(self.num_epochs_per_update):
            kl_div_bool = False
            for b, mb in enumerate(ppo_dataloader):
                (
                    batch_states,
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
                batch_states = batch_states.to(self.device)
                batched_dones = batched_dones.to(self.device)
                breakpoint()
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
                kl_div = -(ratio * log_ratio - (ratio - 1)).mean().item()

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
