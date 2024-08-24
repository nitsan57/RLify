import torch
import numpy as np
from rlify.agents.experience_replay import ExperienceReplay
from rlify.agents.explorers import Explorer, RandomExplorer
from rlify.agents.vdqn_agent import DQNData, VDQN_Agent
import copy
import gymnasium as gym

from rlify.models.base_model import BaseModel


class DQN_Agent(VDQN_Agent):
    """
    DQN Agent
    """

    def __init__(
        self,
        obs_space: gym.spaces,
        action_space: gym.spaces,
        Q_model: BaseModel,
        target_update: str = "hard[update_freq=10]",
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
    ):
        """
        Example::

            env_name = "CartPole-v1"
            env = gym.make(env_name, render_mode=None)
            models_shapes = DQN_Agent.get_models_input_output_shape(env.observation_space, env.action_space)
            Q_input_shape = models_shapes["Q_model"]["input_shape"]
            Q_out_shape = models_shapes["Q_model"]["out_shape"]
            Q_model = fc.FC(input_shape=Q_input_shape, out_shape=Q_out_shape)
            agent = DQN_Agent(
            obs_space=env.observation_space,
            action_space=env.action_space,
            Q_model=Q_model,
            batch_size=64,
            max_mem_size=int(10e6),
            num_parallel_envs=4,
            num_epochs_per_update=10,
            lr=3e-4,
            discount_factor=0.99,
            target_update="hard[update_freq=10]",
            )
            train_stats = agent.train_n_steps(env=env, n_steps=40000)

        Args:
            obs_space (gym.spaces): The observation space of the environment.
            action_space (gym.spaces): The action space of the environment.
            Q_model (BaseModel): The Q-network model.
            dqn_reg (float, optional): The L2 regularization coefficient for the Q-network. Defaults to 0.0.
            target_update (str, optional): The target update rule. Defaults to "hard[update_freq=10]".
            batch_size (int, optional): The batch size for training. Defaults to 64.
            soft_exploit (bool, optional): Whether to use soft exploitation during action selection. Defaults to True.
            explorer (Explorer, optional): The exploration strategy. Defaults to RandomExplorer().
            num_parallel_envs (int, optional): The number of parallel environments. Defaults to 4.
            num_epochs_per_update (int, optional): The number of epochs per update. Defaults to 10.
            lr (float, optional): The learning rate. Defaults to 3e-4.
            device (str, optional): The device to use for training. Defaults to None.
            experience_class (object, optional): The experience replay class. Defaults to ExperienceReplay.
            max_mem_size (int, optional): The maximum size of the experience replay memory. Defaults to int(10e6).
            discount_factor (float, optional): The discount factor for future rewards. Defaults to 0.99.
            reward_normalization (bool, optional): Whether to normalize rewards. Defaults to True.
            tensorboard_dir (str, optional): The directory to save TensorBoard logs. Defaults to "./tensorboard".
            dataloader_workers (int, optional): The number of workers for the data loader. Defaults to 0.

        """
        self.init_target_update_rule(target_update)
        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            Q_model=Q_model,
            dqn_reg=dqn_reg,
            batch_size=batch_size,
            soft_exploit=soft_exploit,
            explorer=explorer,
            max_mem_size=max_mem_size,
            num_parallel_envs=num_parallel_envs,
            num_epochs_per_update=num_epochs_per_update,
            lr=lr,
            device=device,
            experience_class=experience_class,
            discount_factor=discount_factor,
            reward_normalization=reward_normalization,
            tensorboard_dir=tensorboard_dir,
            dataloader_workers=dataloader_workers,
        )

    def setup_models(self):
        """
        Initializes the Q and target Q networks.
        """
        super().setup_models()
        self.target_Q_model = copy.deepcopy(self.Q_model).to(self.device)
        DQN_Agent.hard_target_update(self, manual_update=True)
        return [self.Q_model, self.target_Q_model]

    @staticmethod
    def get_models_input_output_shape(obs_space, action_space):
        return VDQN_Agent.get_models_input_output_shape(obs_space, action_space)

    def init_target_update_rule(self, target_update):
        """
        Initializes the target update rule.

        Args:
            target_update (str): 'soft[tau=0.01]' or 'hard[update_freq=10]' target update
        """
        self.target_update_counter = 0
        self.target_update_time = 1
        target_update, target_update_param = target_update.split("[")
        try:
            target_update_param = float(target_update_param.split("=")[-1][:-1])

        except:
            target_update_param = float(target_update_param[:-1])

        if target_update.lower() == "soft":
            self.update_target = self.soft_target_update
            self.tau = target_update_param
            assert self.tau < 1 and self.tau > 0, "tau must be between 0 and 1"
        elif target_update.lower() == "hard":
            self.target_update_time = target_update_param
            self.update_target = self.hard_target_update
        else:
            raise ValueError(
                f"target_update_type must be 'soft[update_each=]' or 'hard[tau=]', got {target_update}"
            )

    def set_train_mode(self):
        super().set_train_mode()
        self.target_Q_model.train()

    def set_eval_mode(self):
        super().set_eval_mode()

    def hard_target_update(self, manual_update: bool = False):
        """
        Hard update model parameters.

        Args:
            manual_update (bool, optional): Whether to force an update. Defaults to False - in case of force update target_update_counter is not updated.

        """
        self.target_update_counter += 1 * (
            1 - manual_update
        )  # add 1 only if not manual_update
        if self.target_update_counter > self.target_update_time or manual_update:

            self.target_Q_model.load_state_dict(self.Q_model.state_dict())
            for p in self.target_Q_model.parameters():
                p.requires_grad = False

            self.target_update_counter = (
                0 if manual_update == False else self.target_update_counter
            )
        self.target_Q_model.eval()

    def soft_target_update(self):
        """
        Soft update model parameters.
        """
        for target_param, local_param in zip(
            self.target_Q_model.parameters(), self.Q_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
        self.target_Q_model.eval()

    def save_agent(self, f_name) -> dict:
        save_dict = super().save_agent(f_name)
        torch.save(save_dict, f_name)
        return save_dict

    def load_agent(self, f_name):
        checkpoint = super().load_agent(f_name)
        DQN_Agent.hard_target_update(self, manual_update=True)
        return checkpoint

    def reset_rnn_hidden(
        self,
    ):
        super().reset_rnn_hidden()
        self.target_Q_model.reset()

    def update_policy(self, trajectory_data: DQNData):
        """
        Updates the policy.
        Using the DQN algorithm.
        """
        shuffle = False if (self.Q_model.is_rnn) else True
        dataloader = trajectory_data.get_dataloader(
            self.get_train_batch_size(),
            shuffle=shuffle,
            num_workers=self.dataloader_workers,
        )
        for e in range(self.num_epochs_per_update):
            self.optimizer.zero_grad(set_to_none=True)
            self.reset_rnn_hidden()
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
                batched_states = batched_states.to(self.device, non_blocking=True)
                batched_not_terminated = 1 - batched_dones * (1 - batched_truncated)
                batched_not_terminated = batched_not_terminated.to(
                    self.device, non_blocking=True
                )
                batched_returns = batched_returns.to(self.device, non_blocking=True)
                batched_rewards = batched_rewards.to(self.device, non_blocking=True)
                batched_actions = batched_actions.to(
                    self.device, non_blocking=True
                ).squeeze()
                batched_dones = batched_dones.to(self.device)
                batched_terminated = 1 - batched_not_terminated
                v_table = self.Q_model(batched_states).reshape(
                    -1, self.possible_actions
                )
                # print("v_table", v_table)
                q_values = v_table[
                    np.arange(len(v_table)), batched_actions.long().flatten()
                ]
                q_values = q_values.reshape_as(batched_actions)
                with torch.no_grad():
                    q_next = (
                        self.target_Q_model(batched_next_states).detach().max(-1)[0]
                    )
                    q_next = q_next.reshape_as(batched_rewards)  # * 100
                expected_next_values = (
                    batched_rewards
                    + (1 - batched_terminated) * self.discount_factor * q_next
                )
                expected_next_values = torch.max(expected_next_values, batched_returns)
                expected_next_values = expected_next_values.reshape_as(q_values)
                expected_next_values = expected_next_values
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
                    self.update_target()
                else:
                    loss = loss / len(dataloader)
                    loss.backward()

                self.metrics.add("q_loss", loss.item())
                self.metrics.add("q_magnitude", q_values.mean().item())
            if self.accumulate_gradients_per_epoch:
                self.optimizer.step()
                self.update_target()
