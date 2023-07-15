from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .agent_utils import MCAW, ExperienceReplay, ExperienceReplayBeta, ForgettingExperienceReplayBeta, LinearRandomExplorer, calc_returns, calc_gaes, calc_returns_fixed_horizon, fixed_horizon_gaes, CAW
from .drl_agent import RL_Agent
from torch.distributions import Categorical, Normal
import adabelief_pytorch
from utils import HiddenPrints



class PPO_Agent(RL_Agent):
    """
    Init ppo agent with entropy coeff and number epochs to update
    - kwargs are all RL Agent args
    """
    #TODO SUPPORT DIFFERNET OPTIMIZERS
    def __init__(self, batch_size=1024, entropy_coeff=0.05, num_epochs_per_update=10, kl_div_thresh=0.03, clip_param=0.1, experience_class = ForgettingExperienceReplayBeta,  random_explorer = LinearRandomExplorer(0,0,0),**kwargs):
        """ppo recommended setting is 1 parallel env (with non RNN envs, bacuase of too many gradients updates)"""
        super().__init__(**kwargs, batch_size=batch_size, experience_class=experience_class, random_explorer=random_explorer) # inits 
        if self.num_parallel_envs > 4 and not self.model_class.is_rnn:
            print("Warning: PPO is online algorithm and do not benefit from multiple envirnoments, please set num_parallel_envs <=4, convargence issue might preset if not")

        self.kl_div_thresh = kl_div_thresh
        self.clip_param =  clip_param #0.05 #0.03

        if self.horizon == -1:
            self.calc_gaes_func = calc_gaes
            self.calc_returns_func = calc_returns
        else:
            self.calc_gaes_func = fixed_horizon_gaes #calc_gaes
            self.calc_returns_func = calc_returns_fixed_horizon #calc_returns

        self.losses = []
        self.kl_div = []


        self.entropy_coeff = entropy_coeff
        
        self.criterion = nn.SmoothL1Loss(beta=0.01).to(self.device) #nn.MSELoss().to(device)
        self.init_models()
        self.num_epochs_per_update = num_epochs_per_update
        #FOR PPO UPDATE:
        self.init_ppo_buffers()


    def _warp_model_continuous(self, x):
        res = self.policy_nn(x)
        return self.continous_transform(res)
        

    def set_train_mode(self):
        super().set_train_mode()
        self.policy_nn.train()
        self.critic_nn.train()
        

    def set_eval_mode(self):
        super().set_eval_mode()
        self.policy_nn.eval()
        self.critic_nn.eval()

    def init_models(self):
        super().init_models()
        # self.exp_sigma = self.GSDE()
        # self.exp_sigma.to(self.device)
        
        if np.issubdtype(self.action_dtype, np.integer):
            self.policy_nn = self.model_class(input_shape = self.obs_shape, out_shape=self.possible_actions  or 1, **self.model_kwargs).to(self.device)
            # self.policy_nn = torch.compile(self.policy_nn)
            self.actor_model = lambda x : Categorical(logits=F.log_softmax(self.policy_nn(x), dim=1))
            # self.actor_model = torch.compile(self.actor_model, mode="max-autotune")
     
        else:
            out_shape = 2 * self.n_actions
            self.policy_nn = self.model_class(input_shape = self.obs_shape, out_shape=out_shape  or 1, **self.model_kwargs).to(self.device)
            # self.policy_nn = torch.compile(self.policy_nn)
           
            
            self.actor_model = lambda x : self._warp_model_continuous(x)

        list(self.policy_nn.children())[-1].weight.data.fill_(0.01)
        list(self.policy_nn.children())[-1].bias.data.fill_(0.01)
        with HiddenPrints():
            self.actor_optimizer = adabelief_pytorch.AdaBelief(self.policy_nn.parameters(), self.lr, print_change_log=False, amsgrad=False)
        
        self.critic_nn = self.model_class(input_shape=self.obs_shape, out_shape=1,**self.model_kwargs).to(self.device) #output single value - V(s) and not Q(s,a) as before

        list(self.critic_nn.children())[-1].weight.data.fill_(1)
        list(self.critic_nn.children())[-1].bias.data.fill_(1)
        with HiddenPrints():
            self.critic_optimizer = adabelief_pytorch.AdaBelief(self.critic_nn.parameters(), self.lr, print_change_log=False, amsgrad=False)

    def continous_transform(self, x):

        # CAW(self.action_space.low, self.action_space.high, x[:,0], x[:,1])

        return MCAW(self.action_space.low, self.action_space.high, x)#CAW(self.action_space.low, self.action_space.high, x[:,0], torch.nn.Softplus()(x[:,1]))


    def save_agent(self,f_name) -> dict:
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
        super().set_num_parallel_env(num_parallel_envs)
        self.init_ppo_buffers()


    def get_last_collected_experiences(self, number_of_episodes):
        return self._get_ppo_experiences(number_of_episodes, safe_check=False)


    def train(self, env, n_episodes):
        train_episode_rewards = super().train(env, n_episodes)
        self.experience.clear()
        self.init_ppo_buffers()
        return train_episode_rewards


    def best_act_discrete(self, observations, num_obs=1):
        """batched observation only!"""
        if not self.eval_mode:
            print("warning using this function will not accumulate train data, if this is the intention use eval mode to avoid the message")
        
        states = self.pre_process_obs_for_act(observations, num_obs)

        self.policy_nn.eval()
        with torch.no_grad():
            actions_dist = self.actor_model(states)
        self.policy_nn.train()

        selected_actions = torch.argmax(actions_dist.probs, 1).detach().cpu().numpy().astype(np.int32)
        return self.return_correct_actions_dim(selected_actions, num_obs)


    def best_act_cont(self, observations, num_obs=1):
        """batched observation only!"""
        if not self.eval_mode:
            print("warning using this function will not accumulate train data, if this is the intention use eval mode to avoid the message")

        states = self.pre_process_obs_for_act(observations, num_obs)

        self.policy_nn.eval()   
        with torch.no_grad():
            actions_dist = self.actor_model(states)
        self.policy_nn.train()

            # selected_actions = torch.argmax(actions_dist.probs, 1).detach().cpu().numpy().astype(np.int32)
        selected_actions = actions_dist.get_locs().detach().cpu().numpy().astype(np.float32)
        return self.return_correct_actions_dim(selected_actions, num_obs)



    def collect_episode_obs(self, env, max_episode_len=None, num_to_collect_in_parallel=None, env_funcs={"step": "step", "reset": "reset"}):
        # self.exploration_noise = torch.normal(torch.zeros(np.prod(self.obs_shape)), 0.015).to(self.device)
        return super().collect_episode_obs(env, max_episode_len, num_to_collect_in_parallel, env_funcs)


    def random_act_discrete(self,obs,  num_obs):
        actions = super().random_act_discrete(obs, num_obs)
        states = self.pre_process_obs_for_act(obs, num_obs)
        self.save_train_info(states, torch.from_numpy(actions.squeeze()).to(self.device), num_obs)
        return actions
    

    def random_act_cont(self,obs, num_obs):
        actions = self.return_correct_actions_dim(np.random.uniform(float(self.action_space.low), float(self.action_space.high), num_obs), num_obs)
        states = self.pre_process_obs_for_act(obs, num_obs)
        self.save_train_info(states, torch.from_numpy(actions).to(self.device), num_obs)
        return actions


    def save_train_info(self, states, actions, num_obs):
        self.policy_nn.eval()
        self.critic_nn.eval()
        with torch.no_grad():
            actions_dist = self.actor_model(states)
            log_probs = actions_dist.log_prob(actions).detach().float()
            values = self.critic_nn(states).detach().flatten().float()
        self.policy_nn.train()
        self.critic_nn.train()
        for i in range(num_obs):
            self.logits[i].append(log_probs[i])
            self.values[i].append(values[i])
            if self.policy_nn.is_rnn:
                self.states[i].append(states['data'].data[i])
            # else:
            #     self.states[i].append(states['data'][i])


        if self.store_entropy:
            all_ent = actions_dist.entropy().detach().cpu().numpy()
            for i in range(self.num_parallel_envs):
                self.stored_entropy[i].append(all_ent[i])

        return actions

    
    def act(self, observations, num_obs=1, extra_info=False):
        """batched observation only!"""
        if self.eval_mode:
            return self.best_act(observations, num_obs=num_obs)
        states = self.pre_process_obs_for_act(observations, num_obs)
        
        with torch.no_grad():
            self.policy_nn.eval()
            actions_dist = self.actor_model(states)
            
                

            # if self.gdse:
            #     # mu, sigma = actions_dist.get_locs(), actions_dist.get_scales()
            #     # import pdb
            #     # pdb.set_trace()
            #     # action = mu + (self.exploration_noise*states.data).sum(1)
            #     # action = mu + self.exp_sigma(states)
            #     action = actions_dist.sample()
            # else:
            action = actions_dist.sample()


        selected_actions = action.detach().cpu().numpy().astype(np.float32)

        self.save_train_info(states, action, num_obs)
        return self.return_correct_actions_dim(selected_actions, num_obs)


    def init_ppo_buffers(self):
        self.logits = [[] for i in range(self.num_parallel_envs)]
        self.values = [[] for i in range(self.num_parallel_envs)]
        self.states = [[] for i in range(self.num_parallel_envs)]


    def _get_ppo_experiences(self, num_episodes= None, safe_check=True):
        """Current PPO only suports random_Samples = False!!"""

        if num_episodes is None:
            num_episodes = self.num_parallel_envs

        if safe_check:
            assert num_episodes <= self.num_parallel_envs
            
        states, actions, rewards, dones, truncated, next_states = self.experience.get_last_episodes(num_episodes)


        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        truncated = torch.from_numpy(truncated).to(self.device)
        states = states.get_as_tensors(self.device)
        next_states = next_states.get_as_tensors(self.device)
        # ONLY POSSIBLE SINCE RANDOM SAMPLES ARE FALSE!!!
        done_indices = np.where(dones.cpu().numpy() == True)[0]


        values = torch.zeros_like(rewards, dtype=torch.float32, device=self.device)
        logits = torch.zeros((rewards.shape[0], self.n_actions), dtype=torch.float32, device=self.device)
        first_indice = -1
        tensor_creation_func = (lambda x: torch.tensor(x).reshape(len(x), self.n_actions)) if self.n_actions == 1 else torch.cat

        for i in range(num_episodes):
            current_done_indice = done_indices[i]
            curr_episode_len = current_done_indice - first_indice
            values[first_indice+1:current_done_indice+1] = torch.tensor(self.values[i][:curr_episode_len], device=self.device)
            logits[first_indice+1:current_done_indice+1] = tensor_creation_func(self.logits[i][:curr_episode_len]).reshape(logits[first_indice+1:current_done_indice+1].shape)
            first_indice = current_done_indice


        return [states, actions, rewards, dones, truncated, next_states, values, logits]



    def update_policy_reg(self):
        
        states, actions, rewards, dones, truncated, next_states, values, logits = self._get_ppo_experiences()

        # advantages = self.calc_gaes_func(rewards, values, dones, self.discount_factor, horizon=self.horizon)
        terminated = dones
        
        terminated = dones*(1-truncated)
        advantages = self.calc_gaes_func(rewards, values, terminated, self.discount_factor, horizon=self.horizon)
        
        # returns = advantages + values
        
        # returns = self.calc_returns_func(rewards, dones, self.discount_factor, horizon=self.horizon)
        returns = self.calc_returns_func(rewards, terminated, self.discount_factor, horizon=self.horizon)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        self.init_ppo_buffers()

        all_samples_len = len(states)
        for e in range(self.num_epochs_per_update):
            indices_perm = torch.randperm(len(returns))
            states = states[indices_perm]
            actions = actions[indices_perm]
            returns = returns[indices_perm]
            advantages = advantages[indices_perm]
            logits = logits[indices_perm]
            kl_div_bool = False
            for b in range(0, all_samples_len, self.batch_size):
                batch_states = states[b:b+self.batch_size]
                batched_actions = actions[b:b+self.batch_size]
                batched_returns = returns[b:b+self.batch_size]
                batched_advantage = advantages[b:b+self.batch_size]
                batched_logits = logits[b:b+self.batch_size]
                dist = self.actor_model(batch_states)

                critic_values = self.critic_nn(batch_states)
                

                entropy = dist.entropy().mean()
                
                new_log_probs = dist.log_prob(batched_actions)

                old_log_probs = batched_logits # from acted policy
                
                new_log_probs = new_log_probs.reshape(old_log_probs.shape)
                
                log_ratio = (torch.clamp(new_log_probs, np.log(1e-3), 0.0) - torch.clamp(old_log_probs, np.log(1e-3), 0.0)).mean(1)
                ratio = log_ratio.exp()
               
                surr1 = ratio
                surr2 = torch.clamp(ratio, 1.0/(1.0 + self.clip_param), 1.0 + self.clip_param)
                actor_loss  = - ((torch.min(surr1* batched_advantage, surr2* batched_advantage)).mean()) - self.entropy_coeff * entropy
                critic_loss = self.criterion(critic_values, torch.unsqueeze(batched_returns, 1))
                # kl_div = (old_log_probs - new_log_probs).mean().item()
                kl_div = (ratio*log_ratio - (ratio-1)).mean().item()
                self.kl_div.append(kl_div)
                if np.abs(kl_div) > self.kl_div_thresh:
                    kl_div_bool = True
                    if e == 0 and b == 0:
                        print("Something unexpected happend please report it - kl div before first grad call", e, kl_div)
                        # import pdb
                        # pdb.set_trace()
                    # print("kl div exceeded", e, kl_div)
                    break
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_nn.parameters(), 0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                self.critic_optimizer.step()
            if kl_div_bool:
                break


    def update_policy_rnn(self, *exp):
        self.reset_rnn_hidden()
        if len(exp) == 0:
            states, actions, rewards, dones, truncated, next_states, values, logits = self._get_ppo_experiences()
        else:
            states, actions, rewards, dones, truncated, next_states, values, logits = exp
        del next_states

        # terminated = dones
        
        terminated = dones*(1-truncated)
# 
        advantages = self.calc_gaes_func(rewards, values, terminated, self.discount_factor, horizon=self.horizon)

        # returns = advantages + values
        
        returns = self.calc_returns_func(rewards, terminated, self.discount_factor, horizon=self.horizon)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)


        done_indices = torch.where(dones == True)[0].cpu().numpy().astype(np.int32)
        assert len(done_indices) <= self.batch_size ,f"batch size < number of env to train, {self.batch_size}, {len(done_indices)}"
        seq_lens, seq_indices, sorted_data_sub_indices = self.get_seqs_indices_for_pack(done_indices)
        # self.init_ppo_buffers()

        pakced_states = self.pack_from_done_indices(states, seq_indices, seq_lens, done_indices)

        # pakced_states = states[sorted_data_sub_indices] # do not do it, it is already sorted
        sorted_actions = actions[sorted_data_sub_indices]
        sorted_returns = returns[sorted_data_sub_indices]
        sorted_advantage = advantages[sorted_data_sub_indices]
        sorted_logits = logits[sorted_data_sub_indices]

        num_grad_updates = self.num_epochs_per_update

        # p_bool = 0
        for e in range(num_grad_updates):
            
            dist = self.actor_model(pakced_states)
            entropy = dist.entropy().mean()

            new_log_probs = dist.log_prob(sorted_actions)

            old_log_probs = sorted_logits # from acted policy
            new_log_probs = new_log_probs.reshape(old_log_probs.shape)

            log_ratio = (torch.clamp(new_log_probs, np.log(1e-3), 0.0) - torch.clamp(old_log_probs, np.log(1e-3), 0.0)).mean(1)
            ratio = log_ratio.exp()
            
        
            surr1 = ratio
            surr2 = torch.clamp(ratio, 1.0/(1.0 + self.clip_param), 1.0 + self.clip_param)

            actor_loss  = -((torch.min(surr1 * sorted_advantage, surr2 * sorted_advantage)).mean()) - self.entropy_coeff * entropy ##- ((torch.min(surr1, surr2))*sorted_advantage).mean() - self.entropy_coeff * entropy
            
            
            # kl_div = (old_log_probs - new_log_probs).mean().item()
            kl_div = (ratio*log_ratio - (ratio-1)).mean().item() ## APPROX IN https://towardsdatascience.com/approximating-kl-divergence-4151c8c85ddd
            
            self.kl_div.append(kl_div)
            if np.abs(kl_div) > self.kl_div_thresh:
                if e == 0:
                    print("Something unexpected happend please report it - kl div before first grad call", e, kl_div)
                    pass
                break

            #NORMAL
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_nn.parameters(), 0.5)
            self.actor_optimizer.step()

            critic_values = self.critic_nn(pakced_states)
            critic_loss = self.criterion(critic_values, torch.unsqueeze(sorted_returns, 1))
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_optimizer.step()

            self.reset_rnn_hidden()
        self.init_ppo_buffers()
        # self.clear_exp()





    def clear_exp(self):
        self.experience.clear()
        self.init_ppo_buffers()

