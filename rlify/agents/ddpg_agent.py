import torch
import torch.optim as optim
import numpy as np
from rlify.agents.agent_utils import ObsShapeWraper
from rlify.agents.dqn_agent import DQN_Agent
import adabelief_pytorch
from rlify.agents.vdqn_agent import DQNData
from rlify.models.base_model import BaseModel
from rlify.utils import HiddenPrints
import copy


class DDPG_Agent(DQN_Agent):
    """
    DQN Agent
    """

    def __init__(
        self,
        Q_mle_model: BaseModel,
        target_update: str = "soft[tau=0.005]",
        *args,
        **kwargs,
    ):
        """

        Example::

            env_name = "CartPole-v1"
            env = gym.make(env_name, render_mode=None)
            models_shapes = DDPG_Agent.get_models_input_output_shape(env.observation_space, env.action_space)
            Q_input_shape = models_shapes["Q_model"]["input_shape"]
            Q_out_shape = models_shapes["Q_model"]["out_shape"]
            Q_mle_input_shape = models_shapes["Q_mle_model"]["input_shape"]
            Q_mle_out_shape = models_shapes["Q_mle_model"]["out_shape"]
            Q_model = fc.FC(
                input_shape=Q_input_shape,
                out_shape=Q_out_shape,
            )
            Q_mle_model = fc.FC(
                input_shape=Q_mle_input_shape,
                out_shape=Q_mle_out_shape,
            )
            agent = DDPG_Agent(obs_space=env.observation_space, action_space=env.action_space, Q_model=Q_model, Q_mle_model=Q_mle_model)
            train_stats = agent.train_n_steps(env=env_c,n_steps=40000)

        Args:
            Q_mle_model (BaseModel): The MLE model to use.
            args: Additional DQN_Agent arguments.
            kwargs: Additional DQN_Agent arguments.

        """
        self.Q_mle_model = Q_mle_model
        super().__init__(target_update=target_update, *args, **kwargs)

    def setup_models(self):
        """
        Initializes the Q, target Q and MLE networks.
        """
        super().setup_models()

        if np.issubdtype(self.action_dtype, np.integer):
            low = 0
            high = self.possible_actions - 1
            coeff = (high - low) / 2
            bias = low + coeff
            self.cont_transform_bias = torch.zeros(1).to(self.device) + bias
            self.cont_transform_coeff = torch.ones(1).to(self.device) * coeff
        else:
            low, high = np.array(self.action_space.low), np.array(
                self.action_space.high
            )
            coeff = (high - low) / 2
            bias = low + coeff
            self.cont_transform_bias = torch.from_numpy(bias).to(self.device)
            self.cont_transform_coeff = torch.from_numpy(coeff).to(self.device)

        self.Q_mle_model = self.Q_mle_model.to(self.device)
        self.target_Q_mle_model = copy.deepcopy(self.Q_mle_model).to(self.device)
        for p in self.target_Q_mle_model.parameters():
            p.requires_grad = False

        self.q_mle_optimizer = optim.Adam(self.Q_mle_model.parameters(), lr=self.lr)
        return [
            self.Q_model,
            self.target_Q_model,
            self.Q_mle_model,
            self.target_Q_mle_model,
        ]

    @staticmethod
    def get_models_input_output_shape(obs_space, action_space) -> dict:
        """
        Returns the input and output shapes of the Q model.
        """
        q_mle_input = ObsShapeWraper(obs_space)
        q_input = copy.deepcopy(ObsShapeWraper(obs_space))
        q_input["action"] = action_space.shape
        return {
            "Q_model": {
                "input_shape": q_input,
                "out_shape": (1,),
            },
            "Q_mle_model": {
                "input_shape": q_mle_input,
                "out_shape": (int(np.prod(action_space.shape)),),
            },
        }

    def check_action_space(self):
        return True, None

    def set_train_mode(self):
        super().set_train_mode()
        self.Q_mle_model.train()
        self.target_Q_mle_model.train()

    def set_eval_mode(self):
        super().set_eval_mode()
        self.Q_mle_model.eval()

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

            self.target_Q_mle_model.load_state_dict(self.Q_mle_model.state_dict())
            for p in self.target_Q_mle_model.parameters():
                p.requires_grad = False

            self.target_update_counter = (
                0 if manual_update == False else self.target_update_counter
            )
        self.target_Q_model.eval()
        self.target_Q_mle_model.eval()

    def soft_target_update(self):
        for target_param, local_param in zip(
            self.target_Q_model.parameters(), self.Q_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

        for target_param, local_param in zip(
            self.target_Q_mle_model.parameters(), self.Q_mle_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def best_act(self, observations, num_obs=1):
        return self.act(observations, num_obs)

    def reset_rnn_hidden(
        self,
    ):
        super().reset_rnn_hidden()
        self.Q_mle_model.reset()
        self.target_Q_mle_model.reset()

    def save_agent(self, f_name: str) -> dict:
        save_dict = super().save_agent(f_name)
        save_dict["q_mle_optimizer"] = self.q_mle_optimizer.state_dict()
        save_dict["Q_mle_model"] = self._generate_nn_save_key(self.Q_mle_model)
        torch.save(save_dict, f_name)
        return save_dict

    def load_agent(self, f_name: str):
        checkpoint = super().load_agent(f_name)
        self.Q_mle_model.load_state_dict(checkpoint["Q_mle_model"]["state_dict"])
        self.q_mle_optimizer.load_state_dict(checkpoint["q_mle_optimizer"])
        DDPG_Agent.hard_target_update(self, manual_update=True)
        return checkpoint

    def actor_action(
        self,
        observations: torch.tensor,
        num_obs: int = 1,
        use_target: bool = False,
    ):
        """
        Returns the actor action for a batch of observations.

        Args:
            observations (np.ndarray, torch.tensor): The observations to act on
            num_obs (int, optional): The number of observations to act on. Defaults to 1.

        Returns:
            torch.tensor: The actions

        """
        states = self.pre_process_obs_for_act(observations, num_obs)

        if use_target:
            action_pred = self.target_Q_mle_model(states)
        else:
            action_pred = self.Q_mle_model(states)
        action_pred_shape = action_pred.shape
        action_pred = torch.tanh(action_pred).flatten(
            0,
            -2,
        )
        cont_transform_coeff = self.cont_transform_coeff.expand(
            len(action_pred), self.n_actions
        )
        cont_transform_bias = self.cont_transform_bias.expand(
            len(action_pred), self.n_actions
        )
        all_actions = action_pred * cont_transform_coeff + cont_transform_bias
        return all_actions.reshape(action_pred_shape)

    def get_actor_action_value(
        self,
        states: torch.tensor,
        actions: torch.tensor,
        use_target: bool = False,
    ):
        """
        Returns the actor action value for a batch of observations.

        Args:
            states (torch.tensor): The observations to act on
            dones (torch.tensor): The dones of the observations
            actions (torch.tensor): The actions to act on

        Returns:
            torch.tensor: The actions values
        """
        states["action"] = actions
        if use_target:
            actions_values = self.target_Q_model(states).squeeze(-1)
        else:
            actions_values = self.Q_model(states).squeeze(-1)
        return actions_values

    def act(self, observations: np.array, num_obs: int = 1):
        with torch.no_grad():
            actor_acts = self.actor_action(observations, num_obs)
            if self.possible_actions != "continuous":
                actor_acts = actor_acts.round()
        actions = self.return_correct_actions_dim(
            actor_acts.unsqueeze(-1).detach().cpu().numpy(), num_obs
        )
        return actions

    def update_policy(self, trajectory_data: DQNData):
        """
        Updates the policy, using the DDPG algorithm.
        """
        shuffle = False if (self.Q_model.is_rnn) else True
        dataloader = trajectory_data.get_dataloader(
            self.get_train_batch_size(),
            shuffle=shuffle,
            num_workers=self.dataloader_workers,
        )
        for g in range(self.num_epochs_per_update):
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
                batched_not_terminated = (
                    1 - (batched_dones * (1 - batched_truncated))
                ).to(self.device, non_blocking=True)
                batched_returns = batched_returns.to(self.device, non_blocking=True)
                batched_rewards = batched_rewards.to(self.device, non_blocking=True)
                batched_actions = batched_actions.to(self.device, non_blocking=True)
                batched_dones = batched_dones.to(self.device)
                real_batch_size = batched_states.len
                q_values = self.get_actor_action_value(
                    batched_states, batched_actions, use_target=False
                )
                with torch.no_grad():
                    actor_next_action = self.actor_action(
                        batched_next_states,
                        real_batch_size,
                        use_target=True,
                    )
                    q_next = self.get_actor_action_value(
                        batched_next_states,
                        actor_next_action,
                        use_target=True,
                    )
                expected_next_values = (
                    batched_rewards
                    + (batched_not_terminated * self.discount_factor) * q_next.detach()
                )
                expected_next_values = torch.max(expected_next_values, batched_returns)
                loss = self.apply_function_with_loss_flag(
                    self.criterion,
                    q_values,
                    expected_next_values,
                    batched_loss_flags,
                ) + self.apply_regularization(
                    self.dqn_reg, q_values.pow(2), batched_loss_flags
                )
                actor_action = self.actor_action(
                    batched_states, real_batch_size, use_target=False
                )
                actor_values = self.get_actor_action_value(
                    batched_states, actor_action, use_target=False
                )
                # simple maximisze of actor reward (without changing Q function)
                actor_loss = self.apply_function_with_loss_flag(
                    lambda x, y: -x.mean(),
                    actor_values,
                    torch.zeros_like(actor_values),
                    batched_loss_flags,
                )

                self.q_mle_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()  # simple maximisze of actor reward (without changing Q function)
                self.q_mle_optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                self.update_target()
                self.metrics.add("q_loss", loss.item())
                self.metrics.add("actor_loss", actor_loss.item())
                self.metrics.add("q_magnitude", q_values.mean().item())
