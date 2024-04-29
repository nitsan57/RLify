import torch
import torch.nn as nn
import numpy as np
from rlify.agents.vdqn_agent import VDQN_Agent
import copy


class DQN_Agent(VDQN_Agent):
    """
    DQN Agent
    """

    def __init__(self, target_update: str = "hard[update_freq=10]", *args, **kwargs):
        """
        Example::

            env_name = "CartPole-v1"
            env = gym.make(env_name, render_mode=None)
            models_shapes = DQN_Agent.get_models_input_output_shape(env.observation_space, env.action_space)
            Q_input_shape = models_shapes["Q_model"]["input_shape"]
            Q_out_shape = models_shapes["Q_model"]["out_shape"]
            Q_model = fc.FC(input_shape=Q_input_shape, out_shape=Q_out_shape)
            agent = DQN_Agent(obs_space=env.observation_space, action_space=env.action_space, batch_size=64, max_mem_size=10**5, num_parallel_envs=16,
                                lr=3e-4, Q_model=Q_model, discount_factor=0.99, target_update='hard[update_freq=10]', tensorboard_dir = None, num_epochs_per_update=2)
            train_stats = agent.train_n_steps(env=env,n_steps=80000)

        Args:
            target_update (str): 'soft[tau=0.01]' or 'hard[update_freq=10]' target update
            args: Additional VDQN_Agent arguments.
            kwargs: Additional VDQN_Agent arguments.

        """
        self.init_target_update_rule(target_update)
        super().__init__(*args, **kwargs)

    def setup_models(self):
        """
        Initializes the Q and target Q networks.
        """
        super().setup_models()
        self.target_Q_model = copy.deepcopy(self.Q_model).to(self.device)
        DQN_Agent.hard_target_update(self, manual_update=True)

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

    def best_act_cont(self, observations, num_obs=1):
        return self.act(observations, num_obs)

    def best_act_discrete(self, observations, num_obs=1):
        all_actions = self.act_base(observations, num_obs=num_obs)
        selected_actions = (
            torch.argmax(all_actions, -1).detach().cpu().numpy().astype(np.int32)
        )
        return self.return_correct_actions_dim(selected_actions, num_obs)

    def save_agent(self, f_name) -> dict:
        save_dict = super().save_agent(f_name)
        torch.save(save_dict, f_name)
        return save_dict

    def load_agent(self, f_name):
        checkpoint = super().load_agent(f_name)
        DQN_Agent.hard_target_update(self, manual_update=True)
        return checkpoint

    def act_base(self, observations, num_obs=1):
        states = self.pre_process_obs_for_act(observations, num_obs)
        with torch.no_grad():
            self.Q_model.eval()
            all_actions = self.Q_model(states, torch.ones((num_obs, 1)))
            all_actions = torch.squeeze(all_actions, 1)
        self.Q_model.train()

        return all_actions

    def act(self, observations, num_obs=1) -> np.ndarray:
        if not self.soft_exploit:
            return self.best_act(observations, num_obs=num_obs)
        all_actions = self.act_base(observations, num_obs=num_obs)
        selected_actions = (
            torch.multinomial(torch.softmax(all_actions, 1), 1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.int32)
        )
        return self.return_correct_actions_dim(selected_actions, num_obs)

    def reset_rnn_hidden(
        self,
    ):
        super().reset_rnn_hidden()
        self.target_Q_model.reset()

    def update_policy(self, *exp):
        """
        Updates the policy.
        Using the DQN algorithm.
        """
        if len(exp) == 0:
            states, actions, rewards, dones, truncated, next_states, returns = (
                self._get_dqn_experiences()
            )
        else:
            states, actions, rewards, dones, truncated, next_states, returns = exp
        for g in range(self.num_epochs_per_update):
            terminated = dones * (1 - truncated)
            all_samples_len = len(states)

            b_size = all_samples_len if self.Q_model.is_rnn else self.batch_size

            for b in range(0, all_samples_len, b_size):
                batched_states = states[b : b + b_size]
                batched_actions = actions[b : b + b_size].squeeze()
                batched_next_states = next_states[b : b + b_size]
                batched_rewards = rewards[b : b + b_size]
                batched_dones = dones[b : b + b_size]
                batched_terminated = terminated[b : b + b_size]
                batched_returns = returns[b : b + b_size]

                v_table = self.Q_model(batched_states, batched_dones)

                # only because last batch is smaller
                real_batch_size = batched_states.len
                q_values = v_table[np.arange(real_batch_size), batched_actions.long()]

                with torch.no_grad():
                    q_next = (
                        self.target_Q_model(batched_next_states, batched_dones)
                        .detach()
                        .max(1)[0]
                    )

                expected_next_values = (
                    batched_rewards
                    + (1 - batched_terminated) * self.discount_factor * q_next
                )
                expected_next_values = torch.max(expected_next_values, batched_returns)
                loss = (
                    self.criterion(q_values, expected_next_values)
                    + self.dqn_reg * q_values.pow(2).mean()
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                self.update_target()
                self.metrics.add("q_loss", loss.item())
                self.metrics.add("q_magnitude", q_values.mean().item())
