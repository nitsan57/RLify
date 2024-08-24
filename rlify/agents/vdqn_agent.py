import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rlify.agents.experience_replay import ExperienceReplay
from rlify.agents.explorers import Explorer, RandomExplorer
from rlify.agents.agent_utils import IData, LambdaDataset, calc_returns
from rlify.agents.drl_agent import RL_Agent
from rlify.models.base_model import BaseModel
import gymnasium as gym
from torch.utils.data import Dataset


class DQNDataset(Dataset):
    """
    Dataset for DQN
    """

    def __init__(
        self,
        states,
        actions,
        rewards,
        returns,
        dones,
        truncated,
        next_states,
        prepare_for_rnn,
    ):
        """

        Args:
            states: np.array: the states
            actions: np.array: the actions
            rewards: np.array: the rewards
            returns: np.array: the returns
            dones: np.array: the dones
            truncated: np.array: the truncated
            next_states: np.array: the next states
            prepare_for_rnn: bool: whether to prepare for RNN or not
        """
        obs_collection = states, next_states
        tensor_collection = actions, rewards, returns, truncated

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
        states, next_states = obs_collection
        actions, rewards, returns, truncated = tensor_collection
        return (
            states,
            actions,
            rewards,
            returns,
            dones,
            truncated,
            next_states,
            loss_flag,
        )

    def __getitem__(self, idx):
        return self.__getitems__(idx)

    def collate_fn(self, batch):
        return batch


class DQNData(IData):
    """
    DQN Data
    """

    def __init__(
        self,
        states,
        actions,
        rewards,
        returns,
        dones,
        truncated,
        next_states,
        prepare_for_rnn,
    ):
        """

        Args:
            states: np.array: the states
            actions: np.array: the actions
            rewards: np.array: the rewards
            returns: np.array: the returns
            dones: np.array: the dones
            truncated: np.array: the truncated
            next_states: np.array: the next states
            prepare_for_rnn: bool: whether to prepare for RNN or not

        """
        dataset = DQNDataset(
            states,
            actions,
            rewards,
            returns,
            dones,
            truncated,
            next_states,
            prepare_for_rnn,
        )
        super().__init__(dataset, prepare_for_rnn)


class VDQN_Agent(RL_Agent):
    """
    DQN Agent
    """

    def __init__(
        self,
        obs_space: gym.spaces,
        action_space: gym.spaces,
        Q_model: BaseModel,
        dqn_reg: float = 0.0,
        batch_size: int = 64,
        soft_exploit: bool = True,
        explorer: Explorer = RandomExplorer(),
        num_parallel_envs: int = 4,
        num_epochs_per_update: int = 10,
        lr: float = 3e-4,
        device: str = None,
        experience_class: object = ExperienceReplay,
        max_mem_size: int = int(10e6),
        discount_factor: float = 0.99,
        reward_normalization=True,
        tensorboard_dir: str = "./tensorboard",
        dataloader_workers: int = 0,
        accumulate_gradients_per_epoch: bool = None,
    ):
        """
        Example::

            env_name = "CartPole-v1"
            env = gym.make(env_name, render_mode=None)
            models_shapes = VDQN_Agent.get_models_input_output_shape(env.observation_space, env.action_space)
            Q_input_shape = models_shapes["Q_model"]["input_shape"]
            Q_out_shape = models_shapes["Q_model"]["out_shape"]
            Q_model = fc.FC(input_shape=Q_input_shape, out_shape=Q_out_shape)
            agent = VDQN_Agent(obs_space=env.observation_space, action_space=env.action_space, batch_size=64, max_mem_size=10**5, num_parallel_envs=16,
                                lr=3e-4, Q_model=Q_model, discount_factor=0.99, target_update='hard[update_freq=10]', tensorboard_dir = None, num_epochs_per_update=2)
            train_stats = agent.train_n_steps(env=env,n_steps=40000)

        Args:
            obs_space (gym.spaces): The observation space.
            action_space (gym.spaces): The action space.
            Q_model (BaseModel): The Q model.
            dqn_reg (float, optional): The DQN regularization. Defaults to 0.0.
            batch_size (int, optional): The batch size. Defaults to 64.
            soft_exploit (bool, optional): Whether to use soft exploitation. Defaults to True.
            explorer (Explorer, optional): The explorer. Defaults to RandomExplorer().
            num_parallel_envs (int, optional): The number of parallel environments. Defaults to 4.
            num_epochs_per_update (int, optional): The number of epochs per update. Defaults to 10.
            lr (float, optional): The learning rate. Defaults to 3e-4.
            device (str, optional): The device. Defaults to None.
            experience_class (object, optional): The experience class. Defaults to ExperienceReplay.
            max_mem_size (int, optional): The maximum memory size. Defaults to int(10e6).
            discount_factor (float, optional): The discount factor. Defaults to 0.99.
            reward_normalization (bool, optional): Whether to normalize the rewards. Defaults to True.
            tensorboard_dir (str, optional): The tensorboard directory. Defaults to "./tensorboard".
            dataloader_workers (int, optional): The number of dataloader workers. Defaults to 0.
            accumulate_gradients_per_epoch (bool, optional): Whether to accumulate gradients per epoch. Defaults to None.
        """
        self.Q_model = Q_model
        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            batch_size=batch_size,
            num_parallel_envs=num_parallel_envs,
            num_epochs_per_update=num_epochs_per_update,
            lr=lr,
            device=device,
            experience_class=experience_class,
            max_mem_size=max_mem_size,
            discount_factor=discount_factor,
            reward_normalization=reward_normalization,
            tensorboard_dir=tensorboard_dir,
            dataloader_workers=dataloader_workers,
            explorer=explorer,
        )  # inits
        self.soft_exploit = soft_exploit
        self.dqn_reg = dqn_reg
        self.criterion = nn.MSELoss().to(self.device)
        is_valid, msg = self.check_action_space()
        assert is_valid, msg

    def check_action_space(self):
        is_valid = self.action_dtype is not None
        msg = None
        if not np.issubdtype(self.action_dtype, np.integer):
            is_valid = False
            msg = "currently is not supported continuous space in vdqn"
        return is_valid, msg

    def setup_models(self):
        """
        Initializes the Q Model and optimizer.
        """
        self.Q_model = self.Q_model.to(self.device)
        self.optimizer = optim.Adam(self.Q_model.parameters(), lr=self.lr)
        return [self.Q_model]

    @staticmethod
    def get_models_input_output_shape(obs_space, action_space) -> dict:
        n_actions, possible_actions = RL_Agent.read_action_space_properties(
            action_space
        )
        assert (
            possible_actions != "continuous"
        ), f"currently not supported continuous space in dqn"
        obs_space = RL_Agent.read_obs_space_properties(obs_space)

        return {
            "Q_model": {
                "input_shape": obs_space,
                "out_shape": possible_actions,
            }
        }

    def set_train_mode(self):
        super().set_train_mode()
        self.Q_model.train()

    def set_eval_mode(self):
        super().set_eval_mode()
        self.Q_model.eval()

    def best_act(self, observations, num_obs=1):
        all_actions = self.act_base(observations, num_obs=num_obs)
        selected_actions = (
            torch.argmax(all_actions, -1).detach().cpu().numpy().astype(np.int32)
        )
        return self.return_correct_actions_dim(selected_actions, num_obs)

    def save_agent(self, f_name: str) -> dict:
        save_dict = super().save_agent(f_name)
        save_dict["optimizer"] = self.optimizer.state_dict()
        save_dict["Q_model"] = self._generate_nn_save_key(self.Q_model)
        torch.save(save_dict, f_name)
        return save_dict

    def load_agent(self, f_name):
        checkpoint = super().load_agent(f_name)
        self.Q_model.load_state_dict(checkpoint["Q_model"]["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint

    def contains_reccurent_nn(self):
        return self.Q_model.is_rnn

    def act_base(self, observations: np.array, num_obs: int = 1) -> torch.Tensor:
        """
        Returns the Q values for the given observations.

        Args:
            observations (np.array): The observations.
            num_obs (int, optional): The number of observations. Defaults to 1.

        Returns:
            The Q values (torch.tensor)
        """
        states = self.pre_process_obs_for_act(observations, num_obs)
        with torch.no_grad():
            all_actions_values = self.Q_model(states)
            all_actions_values = torch.squeeze(all_actions_values, 1)

        return all_actions_values

    def act(self, observations: np.array, num_obs: int = 1) -> np.ndarray:
        if not self.soft_exploit:
            return self.best_act(observations, num_obs=num_obs)

        all_actions_values = self.act_base(observations, num_obs=num_obs)
        selected_actions = (
            torch.multinomial(torch.softmax(all_actions_values, 1), 1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.int32)
        )
        return self.return_correct_actions_dim(selected_actions, num_obs)

    def reset_rnn_hidden(
        self,
    ):
        self.Q_model.reset()

    def get_trajectories_data(self):
        return self._get_dqn_experiences()

    def _get_dqn_experiences(self) -> tuple[torch.Tensor]:
        """
        loads experiences from the replay buffer and returns them as tensors.

        Returns:
            tuple: (states, actions, rewards, dones, truncated, next_states, returns)

        """
        first_experience_batch = self.experience.sample_random_episodes(
            self.num_parallel_envs
        )
        observations, actions, rewards, dones, truncated, next_observations = (
            first_experience_batch
        )
        returns = calc_returns(rewards, (dones * (1 - truncated)), self.discount_factor)
        rl_data = DQNData(
            observations,
            actions,
            rewards,
            returns,
            dones,
            truncated,
            next_observations,
            self.contains_reccurent_nn(),
        )
        return rl_data

    def update_policy(self, trajectory_data: DQNData):
        shuffle = False if (self.Q_model.is_rnn) else True
        dataloader = trajectory_data.get_dataloader(
            self.get_train_batch_size(),
            shuffle=shuffle,
            num_workers=self.dataloader_workers,
        )
        for e in range(self.num_epochs_per_update):
            self.optimizer.zero_grad(set_to_none=True)
            for b, mb in enumerate(dataloader):
                (
                    batched_states,
                    batched_actions,
                    batched_rewards,
                    batched_returns,
                    batched_dones,
                    batched_truncated,
                    batched_next_states,
                    batched_loss_flags,
                ) = mb
                batched_next_states = batched_next_states.to(
                    self.device, non_blocking=True
                )
                batched_not_terminated = 1 - (batched_dones * (1 - batched_truncated))
                batched_not_terminated = batched_not_terminated.to(
                    self.device, non_blocking=True
                )
                batched_returns = batched_returns.to(self.device, non_blocking=True)
                batched_rewards = batched_rewards.to(self.device, non_blocking=True)
                batched_actions = batched_actions.to(
                    self.device, non_blocking=True
                ).squeeze()
                batched_dones = batched_dones.to(self.device, non_blocking=True)
                batched_states = batched_states.to(self.device)
                v_table = self.Q_model(batched_states)
                # flat in case of RNN (b*seq_len, ...)
                v_table = v_table.reshape(-1, self.possible_actions)
                q_values = v_table[
                    np.arange(len(v_table)), batched_actions.long().flatten()
                ]
                with torch.no_grad():
                    q_next = self.Q_model(batched_next_states)  # .detach().max(-1)[0]
                    q_next = q_next.detach().max(-1)[0]
                    q_next = q_next.reshape_as(batched_rewards)

                expected_next_values = (
                    batched_rewards
                    + batched_not_terminated * self.discount_factor * q_next.detach()
                )
                expected_next_values = torch.max(expected_next_values, batched_returns)
                loss = self.criterion_using_loss_flag(
                    self.criterion,
                    q_values,
                    expected_next_values,
                    batched_loss_flags,
                ) + self.apply_regularization(
                    self.dqn_reg, q_values.pow(2), batched_loss_flags
                )

                if not self.accumulate_gradients_per_epoch:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                else:
                    loss = loss / len(dataloader)
                    loss.backward()

                self.metrics.add("q_loss", loss.item())
                self.metrics.add("q_magnitude", q_values.mean().item())

            if self.accumulate_gradients_per_epoch:
                self.optimizer.step()
            self.reset_rnn_hidden()
