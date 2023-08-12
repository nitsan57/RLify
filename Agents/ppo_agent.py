from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .agent_utils import ExperienceReplay, ExperienceReplayBeta, ForgettingExperienceReplayBeta, calc_gaes
from .action_spaces_utils import MCAW, MDA
from .explorers import RandomExplorer
from .drl_agent import RL_Agent
import adabelief_pytorch
from utils import HiddenPrints



class PPO_Agent(RL_Agent):
    """Proximal Policy Optimization (PPO) reinforcement learning agent.
    Inherits from RL_Agent.
    """
    
    def __init__(self, batch_size=1024, entropy_coeff=0.1, num_epochs_per_update=10, 
                 kl_div_thresh=0.03, clip_param=0.1, experience_class = ForgettingExperienceReplayBeta,
                 explorer = RandomExplorer(0,0,0), **kwargs):
        """
        Initialize the PPO agent.
        
        Args:
            batch_size (int): Batch size for sampling from replay buffer.
            entropy_coeff (float): Entropy regularization coefficient. 
            num_epochs_per_update (int): Training epochs per update.
            kl_div_thresh (float): KL divergence threshold.
            clip_param (float): Clipping parameter.
            experience_class (class): Experience replay class to use.
            explorer (class): Class for random exploration.
            **kwArgs: Additional RL_Agent arguments.
        """
        
        super().__init__(**kwargs, batch_size=batch_size, experience_class=experience_class, explorer=explorer) # inits
        self.kl_div_thresh = kl_div_thresh
        self.clip_param =  clip_param
        self.optimization_step = 0
        self.losses = []
        self.entropy_coeff = entropy_coeff
        self.criterion = nn.MSELoss().to(self.device)
        self.num_epochs_per_update = num_epochs_per_update


    def set_train_mode(self):
        """Set agent in train mode."""
        super().set_train_mode()
        self.policy_nn.train()
        self.critic_nn.train()
        

    def set_eval_mode(self):
        super().set_eval_mode()
        self.policy_nn.eval()
        self.critic_nn.eval()


    def init_models(self):
        """
        Initialize the neural network models.
        """
        # self.exp_sigma = self.GSDE()
        # self.exp_sigma.to(self.device)

        if np.issubdtype(self.action_dtype, np.integer):
            out_shape = self.possible_actions * self.n_actions
            self.policy_nn = self.model_class(input_shape = self.obs_shape, out_shape=out_shape, **self.model_kwargs).to(self.device)
            self.actor_model = lambda x,d=torch.ones((1,1)) : MDA(self.action_space.start, self.possible_actions, self.n_actions, self.policy_nn(x,d))
     
        else:
            out_shape = 2 * self.n_actions # for param trick
            self.policy_nn = self.model_class(input_shape = self.obs_shape, out_shape=out_shape, **self.model_kwargs).to(self.device)
            self.actor_model = lambda x,d=torch.ones((1,1)) : MCAW(self.action_space.low, self.action_space.high, self.policy_nn(x,d))
        
        weight = list(self.policy_nn.children())[-1].weight.data
        bias = list(self.policy_nn.children())[-1].bias.data
        list(self.policy_nn.children())[-1].weight.data = ((weight)*0.01)
        list(self.policy_nn.children())[-1].bias.data = ((bias)*0.01)

        # self.policy_nn = torch.compile(self.policy_nn, mode="max-autotune")
        
        with HiddenPrints():
            # self.actor_optimizer = optim.RMSprop(self.policy_nn.parameters(), self.lr, centered=True)
            self.actor_optimizer =adabelief_pytorch.AdaBelief(self.policy_nn.parameters(), self.lr, print_change_log=False, amsgrad=False)

        
        self.critic_nn = self.model_class(input_shape=self.obs_shape, out_shape=1, **self.model_kwargs).to(self.device) #output single value - V(s) and not Q(s,a) as before\

        with HiddenPrints():
            # self.critic_optimizer = optim.RMSprop(self.critic_nn.parameters(), self.lr, centered=True)
            self.critic_optimizer = adabelief_pytorch.AdaBelief(self.critic_nn.parameters(), self.lr, print_change_log=False, amsgrad=False)


    def save_agent(self,f_name) -> dict:
        """
        Save the agent.
        Args: f_name (str): File name.
        returns: save_dict (dict): Dictionary of agent parameters.
        """
        save_dict = super().save_agent(f_name)
        save_dict['actor_optimizer'] = self.actor_optimizer.state_dict()
        save_dict['policy_nn'] = self.policy_nn.state_dict()
        save_dict['critic_optimizer'] = self.critic_optimizer.state_dict()
        save_dict['critic_model'] = self.critic_nn.state_dict()
        save_dict['entropy_coeff'] = self.entropy_coeff
        save_dict['discount_factor'] = self.discount_factor
        torch.save(save_dict, f_name)
        return save_dict


    def load_agent(self,f_name):
        """
        Loads the agent weights.
        Args: f_name (str): File name.
        """
        checkpoint = super().load_agent(f_name)
        self.define_action_space()
        
        self.init_models()
        self.policy_nn.load_state_dict(checkpoint['policy_nn'])
        try:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        except:pass
        self.critic_nn.load_state_dict(checkpoint['critic_model'])
        try:
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        except:pass


    def reset_rnn_hidden(self,):
        """reset nn hidden_state"""
        self.policy_nn.reset()
        self.critic_nn.reset()


    def set_num_parallel_env(self, num_parallel_envs):
        """
        Set the number of parallel environments.
        Args: num_parallel_envs (int): Number of parallel environments.
        """
        super().set_num_parallel_env(num_parallel_envs)


    def get_last_collected_experiences(self, number_of_episodes):
        return self._get_ppo_experiences(number_of_episodes, safe_check=False)


    def train(self, env, n_episodes):
        train_episode_rewards = super().train(env, n_episodes)
        self.experience.clear()
        return train_episode_rewards


    def best_act_discrete(self, observations, num_obs=1):
        """batched observation only!"""
        
        states = self.pre_process_obs_for_act(observations, num_obs)

        self.policy_nn.eval()
        with torch.no_grad():
            actions_dist = self.actor_model(states, torch.ones((num_obs,1)))
        self.policy_nn.train()
        selected_actions = torch.argmax(actions_dist.probs, 1).detach().cpu().numpy().astype(np.int32)
        return self.return_correct_actions_dim(selected_actions, num_obs)


    def best_act_cont(self, observations, num_obs=1):
        """batched observation only!"""

        states = self.pre_process_obs_for_act(observations, num_obs)

        self.policy_nn.eval()   
        with torch.no_grad():
            actions_dist = self.actor_model(states, torch.ones((num_obs,1)))
        self.policy_nn.train()

        selected_actions = actions_dist.loc.detach().cpu().numpy().astype(np.float32)
        return self.return_correct_actions_dim(selected_actions, num_obs)


    def collect_episode_obs(self, env, max_episode_len=None, num_to_collect_in_parallel=None, env_funcs={"step": "step", "reset": "reset"}):
        # self.exploration_noise = torch.normal(torch.zeros(np.prod(self.obs_shape)), 1).to(self.device)
        return super().collect_episode_obs(env, max_episode_len, num_to_collect_in_parallel, env_funcs)

    
    def act(self, observations, num_obs=1, extra_info=False):
        """batched observation only!"""
        if self.eval_mode:
            return self.best_act(observations, num_obs=num_obs)
        states = self.pre_process_obs_for_act(observations, num_obs)
        
        with torch.no_grad():
            self.policy_nn.eval()
            actions_dist = self.actor_model(states, torch.ones((num_obs,1), device=self.device))
            # gsde support: [wip]
            # if self.gdse:
                # mu, sigma = actions_dist.loc, actions_dist.scale
            #     # import pdb
            #     # pdb.set_trace()p
            # *torch.nn.functional.normalize(
                # action = mu + (self.exploration_noise*states['data'].data).sum(1)
                # action = mu + self.exp_sigma(states)
            #     action = actions_dist.sample()
            # else:
            action = actions_dist.sample()

        selected_actions = action.detach().cpu().numpy().astype(np.float32)

        return self.return_correct_actions_dim(selected_actions, num_obs)


    def _get_ppo_experiences(self, num_episodes= None, safe_check=True):
        """Current PPO only suports random_Samples = False!!"""

        if num_episodes is None:
            num_episodes = self.num_parallel_envs

        if safe_check:
            assert num_episodes <= self.num_parallel_envs
            
        states, actions, rewards, dones, truncated, next_states = self.experience.get_last_episodes(num_episodes)

        actions = torch.from_numpy(actions).to(self.device)
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(-1)
        rewards = torch.from_numpy(rewards).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        truncated = torch.from_numpy(truncated).to(self.device)        
        states = states.get_as_tensors(self.device) # OBS WRAPER

        self.set_eval_mode()
        with torch.no_grad():
            values = self.critic_nn(states, dones).squeeze()
            dist = self.actor_model(states, dones)
        self.set_train_mode()
        logits = dist.log_prob(actions)
        return states, actions, rewards, dones, truncated, values, logits
        

    def update_policy(self, *exp):
        """
        Update the policy network.
        Args: exp (tuple): Experience tuple.
        """
        if len(exp) == 0:
            states, actions, rewards, dones, truncated, values, logits = self._get_ppo_experiences()
        else:
            states, actions, rewards, dones, truncated, values, logits = exp
        
        terminated = dones*(1-truncated)
        advantages, returns = calc_gaes(rewards, values, terminated, self.discount_factor)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        # advantages = (advantages / advantages.abs().max())
        returns = returns.unsqueeze(-1)
        
        all_samples_len = len(states)
        b_size = self.batch_size if not self.model_class.is_rnn else all_samples_len

        for e in range(self.num_epochs_per_update):
            indices_perm = torch.randperm(len(returns)) if not self.model_class.is_rnn else torch.arange(len(returns))
            states = states[indices_perm]
            actions = actions[indices_perm]
            returns = returns[indices_perm]
            advantages = advantages[indices_perm]
            logits = logits[indices_perm]
            kl_div_bool = False
            for b in range(0, all_samples_len, b_size):
                batch_states = states[b:b+b_size]
                batched_actions = actions[b:b+b_size]
                batched_returns = returns[b:b+b_size]
                batched_advantage = advantages[b:b+b_size]
                batched_logits = logits[b:b+b_size]
                batched_dones = dones[b:b+b_size]

                dist = self.actor_model(batch_states, batched_dones)
                critic_values = self.critic_nn(batch_states, batched_dones)
                new_log_probs = dist.log_prob(batched_actions)

                old_log_probs = batched_logits # from acted policy
                log_ratio = (torch.clamp(new_log_probs, np.log(1e-3), 0.0) - torch.clamp(old_log_probs, np.log(1e-3), 0.0))
                ratio = log_ratio.exp().mean(-1)
                log_ratio = torch.log(ratio)

                surr1 = ratio * batched_advantage
                surr2 = torch.clamp(ratio, 1.0/(1.0 + self.clip_param), 1.0 + self.clip_param) * batched_advantage
                entropy = dist.entropy().mean()
                actor_loss  = -(torch.min(surr1, surr2).mean()) - self.entropy_coeff * entropy
                kl_div = -(ratio*log_ratio - (ratio-1)).mean().item()  # kl_div = (old_log_probs - new_log_probs).mean().item() #
                
                if np.abs(kl_div) > self.kl_div_thresh:
                    kl_div_bool = True
                    if e == 0 and b == 0:
                        print("Something unexpected happend please report it - kl div exceeds before first grad call", e, kl_div)
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
            if kl_div_bool:
                break

    
    def clear_exp(self):
        super().clear_exp()