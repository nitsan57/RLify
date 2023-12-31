import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from rlify.agents.explorers import Explorer, RandomExplorer
from .agent_utils import ExperienceReplay
# from .action_spaces_utils import CAW
from .drl_agent import RL_Agent
import adabelief_pytorch
from utils import HiddenPrints
from collections import defaultdict

class DQN_Agent(RL_Agent):
    """
    DQN Agent
    """
    def __init__(self, dqn_reg: float=0.0, batch_size: int=64, target_update: str='hard[update_freq=10]' ,soft_exploit: bool=True, explorer: Explorer = RandomExplorer(), **kwargs):
        """
        Example::

            env_name = "CartPole-v1"
            env_c = gym.make(env_name, render_mode=None)
            agent_c = DQN_Agent(obs_space=env_c.observation_space, action_space=env_c.action_space)
            train_stats_c = agent_c.train_n_steps(env=env_c,n_steps=100000)

        Args:
            dqn_reg (float, optional): L2 regularization for the Q network. Defaults to 0.0.
            batch_size (int, optional): Batch size for training. Defaults to 64.
            soft_exploit (bool, optional): Whether to use soft exploit. Defaults to True.
            explorer (Explorer, optional): The explorer to use. Defaults to RandomExplorer().
            target_update (str): 'soft[tau=0.01]' or 'hard[update_freq=10]' target update
            kwargs: Additional RL_Agent arguments.

        """
        super(DQN_Agent, self).__init__(explorer=explorer, **kwargs, batch_size=batch_size)  # inits
        self.init_target_update_rule(target_update)
    
        self.soft_exploit = soft_exploit
        self.dqn_reg = dqn_reg
        self.criterion = nn.MSELoss().to(self.device)


    def init_models(self):
        """
        Initializes the Q and target Q networks.
        """
        if not np.issubdtype(self.action_dtype, np.integer):
            assert False, "currently is not supported continuous space in dqn"
            self.continous_transform = lambda x: CAW(self.action_space.low, self.action_space.high, x[:,0], torch.nn.Softplus()(x[:,1])) 

        if np.issubdtype(self.action_dtype, np.integer):
            self.Q_model = self.model_class(input_shape=self.obs_shape, out_shape=self.possible_actions, **self.model_kwargs).to(self.device)
            self.target_Q_model = self.model_class(input_shape=self.obs_shape, out_shape=self.possible_actions, **self.model_kwargs).to(self.device)
        else:
            self.Q_model = self.model_class(input_shape=self.obs_shape, out_shape=2  or 1, **self.model_kwargs).to(self.device)
            self.target_Q_model = self.model_class(input_shape=self.obs_shape, out_shape=2  or 1, **self.model_kwargs).to(self.device)
        
        with HiddenPrints():
            self.optimizer = adabelief_pytorch.AdaBelief(self.Q_model.parameters(), self.lr, print_change_log=False, amsgrad=False)
        self.target_update_counter = 0
        self.target_update_time = 1
        self.hard_target_update(force_update=True)


    def init_target_update_rule(self, target_update):
        target_update, target_update_param = target_update.split('[')
        try:
            target_update_param = float(target_update_param.split('=')[-1][:-1])
            
        except:
            target_update_param = float(target_update_param[:-1])
            
        self.target_update_counter = 0
        if target_update.lower() == 'soft':
            self.update_target = self.soft_target_update
            self.tau = target_update_param
            assert self.tau < 1 and self.tau > 0, "tau must be between 0 and 1"
        elif target_update.lower() == 'hard':
            self.target_update_time = target_update_param
            self.update_target = self.hard_target_update
        else:
            raise ValueError(f"target_update_type must be 'soft[update_each=]' or 'hard[tau=]', got {target_update}")

    def set_train_mode(self):
        """
        Sets the Q network to train mode.
        """
        super().set_train_mode()
        self.Q_model.train()
        


    def set_eval_mode(self):
        """
        Sets the Q network to eval mode.
        """
        super().set_eval_mode()
        self.Q_model.eval()


    def hard_target_update(self, force_update: bool = False):
        """

        Hard update model parameters.

        Args:
            force_update (bool, optional): Whether to force an update. Defaults to False - in case of force update target_update_counter is not updated.

        """
        self.target_update_counter += 1*(1 - force_update) # add 1 only if not force_update
        if self.target_update_counter > self.target_update_time or force_update:
            
            self.target_Q_model.load_state_dict(self.Q_model.state_dict())
            for p in self.target_Q_model.parameters():
                p.requires_grad = False

            self.target_update_counter = 0 if force_update == False else self.target_update_counter
        self.target_Q_model.eval()
        


    def soft_target_update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(self.target_Q_model.parameters(), self.Q_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        self.target_Q_model.eval()


    def best_act_cont(self, observations, num_obs=1, extra_info=False):
        """
        Returns the best action for a continuous action space.
        """
        return self.act(observations, num_obs)
    


    def best_act_discrete(self, observations, num_obs=1, extra_info=False):
        """
        Returns the best action for a discrete action space.
        """
        all_actions = self.act_base(observations, num_obs=num_obs, extra_info=extra_info)            
        selected_actions = torch.argmax(all_actions, -1).detach().cpu().numpy().astype(np.int32)
        return self.return_correct_actions_dim(selected_actions, num_obs)


    def save_agent(self,f_name) -> dict:
        """
        Saves the agent to a file.
        Returns: a dictionary containing the agent's state.
        """
        save_dict = super().save_agent(self)
        save_dict['optimizer'] = self.optimizer.state_dict()
        save_dict['model'] = self.Q_model.state_dict()
        save_dict['discount_factor'] = self.discount_factor        
        torch.save(save_dict, f_name)
        return save_dict


    def load_agent(self,f_name):
        """
        Loads the agent from a file.
        """
        checkpoint = super().load_agent(f_name)
        self.Q_model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.hard_target_update(force_update=True)
        if 'discount_factor' in checkpoint:
            self.discount_factor = checkpoint['discount_factor']
        return checkpoint


    def train(self, env, n_episodes):
        """
        Trains the agent.
        """
        train_episode_rewards = super().train(env, n_episodes)
        self.experience.clear()
        return train_episode_rewards
    

    def act_base(self, observations, num_obs=1,extra_info=False):
        states = self.pre_process_obs_for_act(observations, num_obs)
        with torch.no_grad():
            self.Q_model.eval()
            all_actions = self.Q_model(states, torch.ones((num_obs,1)))
            all_actions = torch.squeeze(all_actions,1)
        self.Q_model.train()
    
        return all_actions


    def act(self, observations, num_obs=1,extra_info=False):
        """batched observation only!"""
        if not self.soft_exploit:
            return self.best_act(observations, num_obs=num_obs, extra_info=extra_info)
        all_actions = self.act_base(observations, num_obs=num_obs, extra_info=extra_info)
        selected_actions = torch.multinomial(torch.softmax(all_actions, 1), 1).detach().cpu().numpy().astype(np.int32)            
        return self.return_correct_actions_dim(selected_actions, num_obs)
    

    def reset_rnn_hidden(self,):
        """reset nn hidden_state"""
        self.Q_model.reset()
        self.target_Q_model.reset()


    def _get_dqn_experiences(self):
        """Get a mix of samples, including all last episode- makes sure we dont miss any seen states"""
        random_samples=(not self.model_class.is_rnn)
        # normal exp replay
        if random_samples:
            latest_experience_batch = self.experience.get_last_episodes(self.num_parallel_envs)

            last_episode_len = len(latest_experience_batch[0])
            #50% last episodes, 50% random
            random_experience_batch = self.experience.sample_random_batch(last_episode_len)
            observations, actions, rewards, dones, truncated, next_observations = random_experience_batch

            latest_observations, latest_actions, latest_rewards, latest_dones, latest_truncated, latest_next_observations = latest_experience_batch 
            rand_perm = torch.randperm(2*len(observations))
            observations = observations.cat(latest_observations)[rand_perm]  #np.concatenate([observations, latest_observations])[rand_perm]
            actions = np.concatenate([actions, latest_actions])[rand_perm]
            rewards = np.concatenate([rewards, latest_rewards])[rand_perm]
            dones = np.concatenate([dones, latest_dones])[rand_perm]
            truncated = np.concatenate([truncated, latest_truncated])[rand_perm]
            next_observations = (next_observations.cat(latest_next_observations))[rand_perm]

        else:
            observations, actions, rewards, dones, truncated, next_observations = self.experience.get_last_episodes(self.num_parallel_envs)
        
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        truncated = torch.from_numpy(truncated).to(self.device)
        observations = observations.get_as_tensors(self.device)
        next_observations = next_observations.get_as_tensors(self.device)

        return observations, actions, rewards, dones, truncated, next_observations


    def update_policy(self, *exp):
        """
        Updates the policy.
        Using the DQN algorithm.
        """
        if len(exp) == 0:
            # states, actions, rewards, dones, truncated, next_states = self._get_dqn_experiences(random_samples=(False if self.model_class.is_rnn else True)) #self._get_dqn_experiences(random_samples=(not self.rnn))
            states, actions, rewards, dones, truncated, next_states = self._get_dqn_experiences()
        else:
            states, actions, rewards, dones, truncated, next_states = exp

        terminated = dones * (1 - truncated)
        epoch_metrics = defaultdict(float)
        all_samples_len = len(states)

        b_size = all_samples_len if self.model_class.is_rnn else self.batch_size
        num_grad_updates = int(all_samples_len/self.batch_size) if self.model_class.is_rnn else 1
        
        for g in range(num_grad_updates):
            for b in range(0,all_samples_len, b_size):
                batched_states = states[b:b+b_size]
                batched_actions = actions[b:b+b_size]
                batched_next_states = next_states[b:b+b_size]
                batched_rewards = rewards[b:b+b_size]
                batched_dones = dones[b:b+b_size]
                batched_terminated = terminated[b:b+b_size]

                v_table = self.Q_model(batched_states, batched_dones)
                
                # only because last batch is smaller
                real_batch_size = batched_states.len
                q_values = v_table[np.arange(real_batch_size), batched_actions.long()]

                with torch.no_grad():
                    q_next = self.target_Q_model(batched_next_states, batched_dones).detach().max(1)[0]


                expected_next_values = batched_rewards + (1-batched_terminated) * self.discount_factor * q_next

                loss = self.criterion(q_values, expected_next_values) + self.dqn_reg*q_values.pow(2).mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                epoch_metrics['q_loss'] += loss.item()
                epoch_metrics['q_magnitude'] += q_values.mean().item()
                epoch_metrics['updates'] +=1
                

                self.update_target()
        self.metrics['q_loss'].append(epoch_metrics['q_loss']/epoch_metrics['updates'])
        self.metrics['q_magnitude'].append(epoch_metrics['q_magnitude']/epoch_metrics['updates'])


    def get_last_collected_experiences(self, num_episodes):
        """Mainly for Paired Algorithm support"""
        exp  = self.experience.get_last_episodes(num_episodes)
        return exp
   

    def clear_exp(self):
        self.experience.clear()