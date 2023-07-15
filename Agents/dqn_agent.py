import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .agent_utils import ExperienceReplayBeta, ExperienceReplay, CAW
from .drl_agent import RL_Agent
import adabelief_pytorch
from utils import HiddenPrints

class DQN_Agent(RL_Agent):
    #TODO SUPPORT DIFFERNET OPTIMIZERS
    def __init__(self, dqn_reg=0.0, target_update_time = 100, batch_size=64, soft_exploit=True, **kwargs):
        super(DQN_Agent, self).__init__(**kwargs, batch_size=batch_size)  # inits
        self.soft_exploit = soft_exploit
        self.dqn_reg = dqn_reg
        ### curr unused ###
        self.tau = 0.001
        self.alpha = 0.1
        ###################
        l1 = nn.SmoothL1Loss(beta=0.01).to(self.device)
        l2 = nn.MSELoss().to(self.device)
        self.criterion = nn.MSELoss().to(self.device) #lambda x,y: torch.sqrt(l2(x,y))
        # self.criterion = nn.MSELoss().to(self.device)
        if not np.issubdtype(self.action_dtype, np.integer):
            assert False, "currently is not supported continuous space in dqn"
            self.continous_transform = lambda x: CAW(self.action_space.low, self.action_space.high, x[:,0], torch.nn.Softplus()(x[:,1])) 
        self.init_models()


        self.target_update_time = target_update_time # update target every X learning epochs
        self.target_update_counter = 0
        self.losses =[]


    def init_models(self):
        super().init_models()
        if np.issubdtype(self.action_dtype, np.integer):
            self.Q_model = self.model_class(input_shape=self.obs_shape, out_shape=self.possible_actions, **self.model_kwargs).to(self.device)
            self.target_Q_model = self.model_class(input_shape=self.obs_shape, out_shape=self.possible_actions, **self.model_kwargs).to(self.device)
        else:
            self.Q_model = self.model_class(input_shape=self.obs_shape, out_shape=2  or 1, **self.model_kwargs).to(self.device)
            self.target_Q_model = self.model_class(input_shape=self.obs_shape, out_shape=2  or 1, **self.model_kwargs).to(self.device)
            # self.policy_nn = torch.compile(self.policy_nn)
            # self.actor_model = lambda x : self._warp_model_continuous(x)
        
        list(self.Q_model.children())[-1].weight.data.fill_(0.001)
        list(self.Q_model.children())[-1].bias.data.fill_(0.001)
        with HiddenPrints():
            self.optimizer = adabelief_pytorch.AdaBelief(self.Q_model.parameters(), self.lr, print_change_log=False, amsgrad=False)
        self.hard_target_update()


    def set_train_mode(self):
        super().set_train_mode()
        self.Q_model.train()
        


    def set_eval_mode(self):
        super().set_eval_mode()
        self.Q_model.eval()


    def hard_target_update(self):
        self.target_Q_model.load_state_dict(self.Q_model.state_dict())
        for p in self.target_Q_model.parameters():
            p.requires_grad = False
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
        return self.act(observations, num_obs)
    


    def best_act_discrete(self, observations, num_obs=1, extra_info=False):
        all_actions = self.act_base(observations, num_obs=num_obs, extra_info=extra_info)            
        selected_actions = torch.argmax(all_actions, -1).detach().cpu().numpy().astype(np.int32)
        return self.return_correct_actions_dim(selected_actions, num_obs)



    def get_entropy(self, obs, batch_size, seq_lens=None):
        if self.rnn:
            return self.get_entropy_rnn(obs, batch_size, seq_lens=None)
        return self.get_entropy_reg(obs, batch_size, seq_lens=None)


    def save_agent(self,f_name) -> dict:
        save_dict = super().save_agent(self)
        save_dict['optimizer'] = self.optimizer.state_dict()
        save_dict['model'] = self.Q_model.state_dict()
        save_dict['discount_factor'] = self.discount_factor
        torch.save(save_dict, f_name)
        return save_dict


    def load_agent(self,f_name):
        checkpoint = super().load_agent(f_name)
        self.Q_model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.hard_target_update()
        if 'discount_factor' in checkpoint:
            self.discount_factor = checkpoint['discount_factor']
        return checkpoint


    def train(self, env, n_episodes):
        train_episode_rewards = super().train(env, n_episodes)
        self.experience.clear()
        return train_episode_rewards
    

    def act_base(self, observations, num_obs=1,extra_info=False):
        """batched observation only!"""
        

        states = self.pre_process_obs_for_act(observations, num_obs)
        with torch.no_grad():
            self.Q_model.eval()
            all_actions = self.Q_model(states)
            all_actions = torch.squeeze(all_actions,1)
        self.Q_model.train()
        

        if self.store_entropy:
            print("Fix me entropy stroage")
            all_ent = self.calc_entropy_from_vec(all_actions)
            for i in range(self.num_parallel_envs):
                self.stored_entropy[i].append(all_ent[i])
    
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


    def _get_dqn_experiences(self, random_samples):
        """Get a mix of samples, including all last episode- makes sure we dont miss any seen states"""

        if type(self.experience) in [ExperienceReplayBeta]:
            # try to get about self.num_parallel_envs game lens
            observations, actions, rewards, dones, truncated, next_observations = self.experience.sample_random_batch(self.num_parallel_envs*400)
        else:
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


    def update_policy_reg(self, *exp):
        if len(exp) == 0:
            states, actions, rewards, dones, truncated, next_states = self._get_dqn_experiences(random_samples=True) #self._get_dqn_experiences(random_samples=(not self.rnn))
        else:
            states, actions, rewards, dones, truncated, next_states = exp


        all_samples_len = len(states)

        for b in range(0,all_samples_len, self.batch_size):
            batched_states = states[b:b+self.batch_size]
            batched_actions = actions[b:b+self.batch_size]
            batched_next_states = next_states[b:b+self.batch_size]
            batched_rewards = rewards[b:b+self.batch_size]
            batched_dones = dones[b:b+self.batch_size]
            batched_truncated= truncated[b:b+self.batch_size]
            v_table = self.Q_model(batched_states)
            # only because last batch is smaller
            real_batch_size = batched_states.len
            q_values = v_table[np.arange(real_batch_size), batched_actions.long()]

            with torch.no_grad():
                q_next = self.target_Q_model(batched_next_states).detach().max(1)[0]
            # with torch.no_grad():
            #     q_next = self.Q_model(batched_next_states).detach().max(1)[0]
            
            expected_next_values = batched_rewards + (1-(batched_dones*(1-batched_truncated))) * self.discount_factor * q_next          

            loss = self.criterion(q_values, expected_next_values) + self.dqn_reg*q_values.pow(2).mean()
            self.optimizer.zero_grad(set_to_none=True)
            self.losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            

            self.target_update_counter += 1
            if self.target_update_counter > self.target_update_time:
                # print("updated")
                self.hard_target_update()
                self.target_update_counter = 0


    def update_policy_rnn(self, *exp):
        
        if len(exp) == 0:
            states, actions, rewards, dones, truncated, next_states = self._get_dqn_experiences(random_samples=(False if self.rnn else True))
            # states, actions, rewards, dones, next_states = self.get_last_collected_experiences(self.num_parallel_envs)
        else:
            states, actions, rewards, dones, truncated, next_states = exp
        num_samples = len(states)

        done_indices = torch.where(dones == True)[0].cpu().numpy().astype(np.int32)
        seq_lens, seq_indices, sorted_data_sub_indices = self.get_seqs_indices_for_pack(done_indices)
        sorted_actions = actions[sorted_data_sub_indices]
        sorted_rewards = rewards[sorted_data_sub_indices]
        sorted_dones = dones[sorted_data_sub_indices]
        sorted_truncated = truncated[sorted_data_sub_indices]
        pakced_states = self.pack_from_done_indices(states, seq_indices, seq_lens, done_indices)
        pakced_next_states = self.pack_from_done_indices(next_states, seq_indices, seq_lens, done_indices)

        num_grad_updates = num_samples // self.batch_size
        for i in range(num_grad_updates):
            v_table = self.Q_model(pakced_states)
            # v_table = v_table #.reshape(normal_b_size, len(self.action_space))
            
            q_values = v_table[np.arange(v_table.shape[0]), sorted_actions.long()]
            with torch.no_grad():
                q_next = self.target_Q_model(pakced_next_states).detach().max(1)[0] #.reshape(normal_b_size, len(self.action_space))
            expected_next_values = sorted_rewards + (1-(sorted_dones*(1-sorted_truncated))) * self.discount_factor * q_next
            loss = self.criterion(q_values, expected_next_values)

            # Optimize the model
            self.optimizer.zero_grad(set_to_none=True)
            self.losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.target_update_counter += 1*num_grad_updates
            if self.target_update_counter > self.target_update_time:
                self.hard_target_update()
                self.target_update_counter = 0
            


    def get_last_collected_experiences(self, num_episodes):
        """Mainly for Paired Algorithm support"""
        # return [torch.from_numpy(x)).to(self.device) for x in self.experience.get_last_episodes(num_episodes)]

        exp  = self.experience.get_last_episodes(num_episodes)
        res = []
        for i,x in enumerate(exp):
            if i == self.experience.states_index or i == self.experience.next_states_index:
                res.append(x.to(self.device))
            else:
                res.append(torch.from_numpy(x).to(self.device))
        return res

    
    def clear_exp(self):
        self.experience.clear()