import torch
from rlify.agents.explorers import Explorer, HeuristicExplorer, RandomExplorer

# from .action_spaces_utils import CAW
from .drl_agent import RL_Agent


class Heuristic_Agent(RL_Agent):
    """
    A Heuristic Agent that uses a heuristic function to act.
    """

    def __init__(self, heuristic_func, **kwargs):
        """

        Args:
            heuristic_function: A function that takes in the inner_state, observation (ObsWraper) and returns a tuple: (inner_state, action) the inner state (could be None) and the action to be taken,
                                please notice that the actions shape is b,n_actions,action_dim
                                Please check more ObsWraper for  more info on the observation input object
            kwargs: Arguments for the RL_Agent base class

        Example::

            env_name = "CartPole-v1"
            env_c = gym.make(env_name, render_mode=None)
            def heuristic_func(inner_state, obs: ObsWraper):
                # an function that does not keep inner state
                b_shape = len(obs)
                actions = np.zeros((b_shape, 1)) # single discrete action
                # just a dummy heuristic for a gym env with np.array observations (for more details about the obs object check ObsWraper)
                # the heuristic check whether the first number of each observation is positive, if so, it returns action=1, else 0
                actions[torch.where(obs['data'][:,0] > 0)[0].cpu()] = 1
                return None, actions

            agent_c = Heuristic_Agent(obs_space=env_c.observation_space, action_space=env_c.action_space, heuristic_func=heuristic_func)
            reward = agent_c.run_env(env_c, best_act=True)
            print("Run Reward:", reward)


        """
        explorer = HeuristicExplorer(heuristic_func, 1, 1, 1)
        super(Heuristic_Agent, self).__init__(explorer=explorer, **kwargs)  # inits

    def setup_models(self):
        """
        Does nothing in this agent.
        """
        pass

    def get_models_input_output_shape(obs_space, action_space):
        """
        Does nothing in this agent.
        """
        return {}

    def set_train_mode(self):
        return super().set_train_mode()

    def set_eval_mode(self):
        return super().set_eval_mode()

    def save_agent(self, f_name) -> dict:
        save_dict = super().save_agent(f_name)
        save_dict["explorer_inner"] = self.explorer.inner_state
        torch.save(save_dict, f_name)
        return save_dict

    def load_agent(self, f_name):
        checkpoint = super().load_agent(f_name)
        self.explorer.inner_state = checkpoint["explorer_inner"]
        return checkpoint

    def train(self, env, n_episodes: int):
        train_episode_rewards = super().train(env, n_episodes)
        self.experience.clear()
        return train_episode_rewards

    def act(self, observations, num_obs=1):
        observations = self.pre_process_obs_for_act(observations, num_obs)
        explore_action = self.explorer.act(self.action_space, observations, num_obs)

        assert (
            explore_action.shape[0] == num_obs
        ), f"The explorer heuristic functions does not returns the correct number of actions (batch dim) expected: {num_obs}, got: {explore_action.shape[0]}"
        return self.return_correct_actions_dim(explore_action, num_obs)

    def best_act(self, observations, num_obs=1):
        return self.act(observations, num_obs=num_obs)

    def reset_rnn_hidden(
        self,
    ):
        """reset nn hidden_state - does nothing in this agent"""
        pass

    def update_policy(self, *exp):
        """
        does nothing in this agent.
        """
        pass

    def get_last_collected_experiences(self, num_episodes):
        """Mainly for Paired Algorithm support"""
        exp = self.experience.get_last_episodes(num_episodes)
        return exp

    def clear_exp(self):
        return super().clear_exp()
