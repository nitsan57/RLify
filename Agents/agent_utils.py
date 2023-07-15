from collections import namedtuple
from multiprocessing import Process, Pipe
import numpy as np
import torch
import copy
import gc
import gymnasium as gym
from torch.distributions import Normal


class LinearRandomExplorer():
    """Class that acts a linear exploration method"""
    def __init__(self, exploration_epsilon=1, eps_end=0.05, eps_dec=1e-2) -> None:
        self.exploration_epsilon = exploration_epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec


    def explore(self):
        if np.random.random() < self.exploration_epsilon:
            return True
        return False

    def update(self):
        self.exploration_epsilon = self.exploration_epsilon * (1-self.eps_dec) if self.exploration_epsilon > self.eps_end else self.eps_end


class MCAW():
    def __init__(self, low, high, locs_scales_tesnor) -> None:
        num_models = len(low)
        self.continuous_models = []
        self.out_shape = locs_scales_tesnor.shape[0], num_models # sample or get locs
        self.device = locs_scales_tesnor.device
        mu_dims = np.arange(num_models)
        sigma_dims = np.arange(num_models, 2*num_models) 
        for i in range(num_models):
            model = CAW(low[i], high[i], locs_scales_tesnor[:, mu_dims[i]],locs_scales_tesnor[:,sigma_dims[i]])
            self.continuous_models.append(model)
            

    def sample(self, sample_shape=torch.Size()):
        res = torch.zeros(self.out_shape, device=self.device)
        for i,m in enumerate(self.continuous_models):
            res[:,i] = m.sample(sample_shape)
        return res
    
    
    def log_prob(self, actions):
        res = torch.zeros(self.out_shape, device=self.device)

        for i,m in enumerate(self.continuous_models):
            res[:,i] = m.log_prob(actions[:, i])
            
        return res
    

    def get_scales(self):
        res = torch.zeros(self.out_shape, device=self.device)
        for i,m in enumerate(self.continuous_models):
            res[:,i] = m.loc
        return res

    def get_locs(self):
        res = torch.zeros(self.out_shape)
        for i,m in enumerate(self.continuous_models):
            res[:,i] = m.scale
        return res
    
    def entropy(self):
        res = torch.zeros(self.out_shape)
        for i,m in enumerate(self.continuous_models):
            res[:,i] = m.entropy()
        return res.mean(1)

        
class CAW(Normal):
    def __init__(self, low, high, loc, scale) -> None:
        low, high = float(low), float(high)
        self.low = low
        self.high = high
        coeff = (high - low) / 2
        bias = low + coeff
        loc = torch.tanh(4*loc)*coeff+bias
        scale = torch.nn.Softplus()(scale)
        super().__init__(loc, scale)
        self.activation = torch.nn.Identity() #torch.nn.Sigmoid()


    def sample(self, sample_shape=torch.Size()):
        return torch.clamp(self.activation(super().sample(sample_shape)), self.low, self.high)
    
    def get_scales(self):
        return self.scale

    def get_locs(self):
        return self.loc


def calc_returns(rewards, dones, discount_factor, horizon=10):
    """works with rewards vector which consitst of many epidsodes"""
    curr = 0
    device = rewards.device
    returns = torch.zeros_like(rewards, device=device)
    curr= rewards[-1]
    returns[-1] = rewards[-1]

    curr_discount_value = discount_factor
    norm_param = 1
    for reveres_idx in reversed(range(len(rewards)-1)):
        r = rewards[reveres_idx]
        if dones[reveres_idx]:
            norm_param = 1
            curr_discount_value = discount_factor
        

        curr = r + discount_factor * curr*(1-dones[reveres_idx])
        norm_param += curr_discount_value
        curr_discount_value *=discount_factor

        returns[reveres_idx] = curr #/norm_param  # append in revers e
    
    return returns


def calc_returns_fixed_horizon(rewards, dones, discount_factor, horizon=20):
    """works with rewards vector which consitst of many epidsodes"""
    norm_param = discount_factor * horizon
    curr = 0
    device = rewards.device
    returns = torch.zeros_like(rewards, device=device)
    horizoned_returns = torch.zeros_like(returns, device=device)
    

    curr= rewards[-1]
    curr_done_idx = len(dones)-1
    returns[-1] = rewards[-1]
    horizoned_returns[-1] = rewards[-1]

    for reveres_idx in reversed(range(len(rewards)-1)):
        r = rewards[reveres_idx]
        curr_done_idx = reveres_idx if dones[reveres_idx] else curr_done_idx
        curr = r + discount_factor * curr*(1-dones[reveres_idx])


        returns[reveres_idx] = curr  # append in reverse
        horizoned_returns[reveres_idx] = returns[reveres_idx]
        if (reveres_idx + horizon) <= curr_done_idx: #len(returns):
            horizoned_returns[reveres_idx]-=returns[reveres_idx + horizon]
            horizoned_returns[reveres_idx] /=norm_param
        else:
            portion = (reveres_idx + horizon-1 - curr_done_idx)  / horizon
            if portion ==1:
                portion = 1-1/(norm_param)
            horizoned_returns[reveres_idx] = horizoned_returns[reveres_idx]/(norm_param*(1-portion))
        
            # horizoned_returns[reveres_idx] = horizoned_returns[reveres_idx]/norm_param
        # else:
            # horizoned_returns[reveres_idx] = horizoned_returns[reveres_idx]/norm_param
    
    return horizoned_returns


def fixed_horizon_gaes(rewards, values, dones, discount_factor=1, decay=0.9, horizon=20):
    """
    works with rewards vector which consitst of many epidsodes
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    
    gamma = discount_factor
    curr_done_idx = len(dones)-1
    device = rewards.device

    next_values = torch.cat([values[1:], torch.zeros(1, device=device)])
    # deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]
    deltas = rewards + gamma * next_values * (1 - dones) - values
    gaes = torch.zeros_like(deltas, device=device)
    horizoned_gaes = torch.zeros_like(deltas, device=device)
    gaes[-1] = deltas[-1]
    horizoned_gaes[-1] = deltas[-1]

    for i in reversed(range(len(deltas)-1)):        
        curr_done_idx = i if dones[i] else curr_done_idx
        gaes[i] = deltas[i] + gamma * decay * gaes[i+1] * (1 - dones[i])
        horizoned_gaes[i] = gaes[i] 
        if i + horizon < curr_done_idx:
            horizoned_gaes[i]-=gaes[i + horizon]


    return horizoned_gaes


def calc_gaes(rewards, values, dones, discount_factor=0.99, decay=0.95, horizon=-1):
    """
    works with rewards vector which consitst of many epidsodes
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    gamma = discount_factor
    device = rewards.device

    next_values = torch.cat([values[1:], torch.zeros(1, device=device)])
    # deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]
    deltas = rewards - values + gamma * next_values * (1 - dones) 

    gaes = torch.zeros_like(deltas, device=device)
    gaes[-1] = deltas[-1]

    for i in reversed(range(len(deltas)-1)):
        gaes[i] = deltas[i] + gamma * decay * gaes[i+1] * (1 - dones[i])
    return gaes


class ObsShapeWraper(dict):
    dict_types = [dict, gym.spaces.Dict]
    
    def __init__(self, obs_shape):
            res = {}
            self.dict_types.append(type(self))
            if type(obs_shape) in self.dict_types:
                try:
                    for x in obs_shape:
                        if len(obs_shape[x].shape) == 0:
                            res[x] = (1,)
                        else:
                            res[x] = obs_shape[x].shape
                except AttributeError:
                    for x in obs_shape:
                        if len(obs_shape[x]) == 0:
                            res[x] = (1,)
                        else:
                            res[x] = obs_shape[x]
                super(ObsShapeWraper, self).__init__(res)
            else:
                try:
                    res = obs_shape.shape
                except AttributeError:
                    res = obs_shape
                super(ObsShapeWraper, self).__init__({'data': tuple([*res])})


class ObsWraper:
    def __init__(self, data=None, keep_dims=False, tensors=False):
        self.obj_constructor = None
        self.len = 0
        self.data = {}
        self.shape = {}
        if type(data) == ObsWraper:
            self.data = copy.deepcopy(data.data)
            self.obj_constructor = data.obj_constructor
            self.len = data.len
            self.shape = copy.deepcopy(data.shape)
            return

        if np.issubdtype(type(data), np.integer) or np.issubdtype(type(data), np.float):
            data = np.array(data,ndmin=1).astype(np.float32)

        if data is None:
            return self._init_from_none_()
        
        if type(data) == list or type(data) == tuple:
            if type(data[0]) == ObsWraper:
                return self.init_from_list_obsWrapper_obs(data)
            else:
                return self.init_from_list_generic_data(data)
        else:
            try:
                if type(data) == dict:
                    to_add = data
                else:
                    raise TypeError
                self.data = {}
                
                for k, v in to_add.items():
                    if tensors or torch.is_tensor(v):
                        self.obj_constructor = torch.tensor
                        v = torch.atleast_1d(v)
                    else:
                        self.obj_constructor = np.array
                        v = np.array(v,ndmin=1)

                    self.data[k] = v
                    len_v = len(self.data[k])

                    if self.len == 0:
                        self.len = len_v
                    assert self.len == len_v, "cant init a dict with a value with different len"

            except TypeError:
                if keep_dims:
                    to_add = np.array(data,ndmin=1)
                else:
                    to_add = np.expand_dims(data, axis=0)
                self.data = {'data': to_add}
                self.len = len(to_add)

        self.update_shape()


    def update_shape(self):
        for k, v in self.items():
            try:
                self.shape[k] = v.shape
            except AttributeError as e:
                try:
                    self.shape[k] = v.data.shape
                except:
                    self.shape[k] = None


    def init_from_list_obsWrapper_obs(self, obs_list):
        obj_constructor = obs_list[0].obj_constructor
        keys = list(obs_list[0].keys())
       
        for k in keys:
            res = []
            for obs in obs_list:
                res.append(obs[k])
            is_tensor = torch.is_tensor(obs_list[0][k])
            if is_tensor:
                res = torch.cat(res)
            else:
                res = np.concatenate(res)

            self.data[k] = res
            self.shape[k] = res.shape

        self.obj_constructor = obj_constructor
        self.len = len(obs_list)

    
    def init_from_list_generic_data(self, obs_list):
        # get class in a generic way
        if torch.is_tensor(obs_list[0]):
            self.obj_constructor = torch.tensor
            res = self.obj_constructor(obs_list)
        else:
            self.obj_constructor = np.array
            res = self.obj_constructor(obs_list)
            if len(res.shape) == 1:
                res = res.reshape(res.shape[0], 1).astype(np.float32)

        self.data['data'] = res
        self.len = len(res)
        self.shape['data'] = res.data.shape


    def _init_from_none_(self):
        self.__init__({})


    def __setitem__(self, key, value):
        if type(key) is str:
            self.data[key] = value
            len_v = len(value)
            if self.len == 0:
                self.len = len_v
            assert self.len == len_v, "cant set a value with differnet len"
        elif np.issubdtype(type(value), np.integer):
            for k in self.data.keys():
                self.data[k][key] = value
        else:
            assert self.data.keys() == value.keys(), "has to set item with same keys"
            for k in self.data.keys():
                self.data[k][key] = value[k]

        self.update_shape()


    def __iter__(self):
        return iter(self.data)


    def __getitem__(self, key):
        if self.obj_constructor == torch.tensor:
            return self.slice_tensors(key)
        if type(key) is str:
            return self.data[key]
        temp_dict = {}

        for k, v in self.data.items():
            if np.issubdtype(type(key), np.integer):
                temp_dict[k] = np.array([v.__getitem__(key)])
            else:
                temp_dict[k] = np.array(v.__getitem__(key))
        return ObsWraper(temp_dict)

    
    def slice_tensors(self, key):
        if type(key) is str:
            return self.data[key]
        temp_dict = {}
        
        for k, v in self.data.items():
            if np.issubdtype(type(key), np.integer):
                temp_dict[k] = v.__getitem__(key).clone().detach().requires_grad_(True).unsqueeze(-1).float()
            else:
                temp_dict[k] = v.__getitem__(key).clone().detach().requires_grad_(True).float()
        return ObsWraper(temp_dict, tensors=True)


    def keys(self):
        return self.data.keys()


    def items(self):
        return self.data.items()


    def values(self):
        return self.data.values()


    def __len__(self):
        return self.len


    def __str__(self) -> str:
        return self.data.__str__()


    def __repr__(self) -> str:
        return self.data.__repr__()


    def __mul__(self, other):

        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = v * other[k]
        return ObsWraper(temp_dict, keep_dims=True, tensors=other.tensors)


    def __add__(self, other):
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = v + other[k]
        return ObsWraper(temp_dict, keep_dims=True)

    def __neg__(self):
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = -v
        return ObsWraper(temp_dict, keep_dims=True)


    def __sub__(self, other):
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = v - other[k]
        return ObsWraper(temp_dict, keep_dims=True)
    

    def __truediv__(self, other):
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = v / other[k]
        return ObsWraper(temp_dict, keep_dims=True, tensors=other.tensors)


    def get_as_tensors(self, device='cpu'):
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = torch.tensor(v).float().to(device).float()
        return ObsWraper(temp_dict, keep_dims=True, tensors=True)


    def np_cat(self, other, axis=0):
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = np.concatenate([self.data[k], other[k]], axis)
        return ObsWraper(temp_dict)


    def np_append(self, other, axis=0):
        if self.len != 0:
            for k, v in self.data.items():
                self.data[k] = np.concatenate([self.data[k], other[k]], axis)
            self.len = self.len + len(other)
        else:
            self.data = copy.deepcopy(other.data)
            self.len = other.len
        
    def cat(self, other, axis=0):
        temp_dict = {}
        for k, v in self.data.items():
            if torch.is_tensor(v):
                temp_dict[k] = torch.cat([self.data[k], other[k]], axis)
            else:
                temp_dict[k] = np.concatenate([self.data[k], other[k]], axis)
        return ObsWraper(temp_dict)


    # def torch_cat(self, other, axis=0):
    #     temp_dict = {}
    #     for k, v in self.data.items():
    #         temp_dict[k] = torch.cat([self.data[k], other[k]], axis)
    #     return ObsWraper(temp_dict)


    # def torch_append(self, other, axis=0):
    #     if self.len != 0:
    #         for k, v in self.data.items():
    #             self.data[k] = torch.cat([self.data[k], other[k]], axis)
    #         self.len = self.len + len(other)
    #     else:
    #         self.data = copy.deepcopy(other.data)
    #         self.len = other.len

    def np_zero_roll(self, indx, axis=0, inplace=False):
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = np.concatenate([self.data[k][-indx:], np.zeros_like(self.data[k][:-indx])])
        if inplace:
            self.data = temp_dict
        else:
            return ObsWraper(temp_dict)
        

    def np_roll(self, indx, axis=0, inplace=False):
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = np.roll(self.data[k], indx, axis=axis)
        if inplace:
            self.data = temp_dict
        else:
            return ObsWraper(temp_dict)
        


class ExperienceReplay:
    # TODO: support continous infinte env by removeing long seeing states without dones
    def __init__(self, capacity, obs_shape, continous_mem=False):
        
        self.obs_shape = ObsShapeWraper(obs_shape)
        self.capacity = capacity
        self.init_buffers()
        self.continous_mem = continous_mem


    def __len__(self):
        return self.curr_size


    def append(self, curr_obs, action, reward, done, truncated, next_obs):
        # extra_exps

        curr_obs = ObsWraper(curr_obs)
        next_obs = ObsWraper(next_obs)
        
        num_samples = len(curr_obs)

        free_space = self.capacity - self.curr_size
        if num_samples > self.capacity:
            self.curr_size = 0
            curr_obs = curr_obs[:self.capacity]
            action = action[:self.capacity]
            reward = reward[:self.capacity]
            done = done[:self.capacity]
            truncated = truncated[:self.capacity]
            next_obs = next_obs[:self.capacity]

            done[-1] = True
            truncated[-1] = True
            num_samples = self.capacity

        elif self.curr_size + num_samples > self.capacity and not self.continous_mem:
            dones = np.where(self.all_buffers[self.dones_index] == True)[0]
            relevant_dones = np.where(dones > num_samples - free_space)[0]
            # roll more then memory needed:

            relevant_index = relevant_dones[len(relevant_dones)//2]
            done_index = dones[relevant_index]

            for i in range(len(self.all_buffers)):
                # self.all_buffers[i][:done_index+1] = 0
                if i == self.states_index or i == self.next_states_index:
                    self.all_buffers[i] = self.all_buffers[i].np_zero_roll(-done_index - 1, axis=0, inplace=False)
                else:
                    self.all_buffers[i] = np.concatenate([self.all_buffers[i][-(-done_index-1):], np.zeros_like(self.all_buffers[i][:-(-done_index-1)])]) # np.roll(self.all_buffers[i], -done_index - 1, axis=0)
            

            self.curr_size -= (done_index+1)

        self.all_buffers[self.states_index][self.curr_size:self.curr_size + num_samples] = curr_obs
        self.all_buffers[self.actions_index][self.curr_size:self.curr_size + num_samples] = action
        self.all_buffers[self.reward_index][self.curr_size:self.curr_size + num_samples] = reward
        self.all_buffers[self.dones_index][self.curr_size:self.curr_size + num_samples] = done
        self.all_buffers[self.truncated_index][self.curr_size:self.curr_size + num_samples] = truncated
        self.all_buffers[self.next_states_index][self.curr_size: self.curr_size+num_samples] = next_obs
    

        self.curr_size += num_samples


    def init_buffers(self):

        self.curr_size = 0
        actions_buffer = np.zeros((self.capacity), dtype=np.float32)
        reward_buffer = np.zeros((self.capacity), dtype=np.float32)
        dones_buffer = np.zeros((self.capacity), dtype=np.uint8)
        truncated_buffer = np.zeros((self.capacity), dtype=np.uint8)
        shape = (self.capacity, *self.obs_shape)

        states_buffer = ObsWraper()
        next_states_buffer = ObsWraper()
        for k in self.obs_shape:
            shape = (self.capacity, *self.obs_shape[k])
            states_buffer[k] = np.zeros(shape, dtype=np.float32)
            next_states_buffer[k] = np.zeros(shape, dtype=np.float32)

        self.all_buffers = [states_buffer, actions_buffer,
                            reward_buffer, dones_buffer, truncated_buffer, next_states_buffer]

        self.states_index = 0
        self.actions_index = 1
        self.reward_index = 2
        self.dones_index = 3
        self.truncated_index = 4
        self.next_states_index = 5


    def clear(self):
        self.init_buffers()


    def get_last_episodes(self, num_episodes):
        """return all last episode samples, or specified num samples"""
        episode_indices = [0]
        episode_indices.extend(np.where(self.all_buffers[self.dones_index] == True)[
                               0])  # last episode indx is done =1

        assert len(
            episode_indices) >= num_episodes, "requested more episodes then actual stored in mem"
        # it is a "False" done just for episode begin idx
        if episode_indices[-num_episodes - 1] == 0:
            num_samples = self.curr_size - episode_indices[-num_episodes - 1]
        else:  # we dont want the last done in our batch
            num_samples = self.curr_size - episode_indices[-num_episodes - 1] - 1  # exclude first done indice
        

        return self.get_last_samples(num_samples)


    def get_last_samples(self, num_samples=None):
        """return all last episode samples, or specified num samples"""

        if num_samples is None:
            "return last episode"

            dones = np.where(self.all_buffers[self.dones_index] == True)[0]
            if len(dones) > 1:
                # exclude the latest done sample
                first_sample_idx = dones[-2] + 1        # extra_exps

            else:
                # from 0 index to last done(which is also the first..)
                first_sample_idx = 0
                last_samples = [buff[first_sample_idx:self.curr_size]
                                for buff in self.all_buffers]

        else:
            last_samples = [buff[self.curr_size-num_samples:self.curr_size]
                            for buff in self.all_buffers]


        return last_samples


    def get_buffers_at(self, indices):
        buffers_at = tuple(buff[indices] for buff in self.all_buffers)
        return buffers_at


    def sample_random_batch(self, sample_size):
        sample_size = min(sample_size, self.curr_size)
        indices = np.random.choice(self.curr_size, sample_size, replace=False)
        return self.get_buffers_at(indices)


class ExperienceReplayBeta(ExperienceReplay):
    def __init__(self, capacity, obs_shape, continous_mem=False):
        super().__init__(capacity, obs_shape, continous_mem)
        self.latest_obs_start_idx = 0
        self.latest_obs_end_idx = -1
        self.num_episodes_added = 0
    
    def init_buffers(self):
        super().init_buffers()


    def append(self, curr_obs, action, reward, done, truncated, next_obs):
        self.num_episodes_added = sum(done)
        curr_obs = curr_obs
        next_obs = next_obs

        num_samples = len(curr_obs)
        if num_samples > self.capacity:
            self.curr_size = 0
            curr_obs = curr_obs[:self.capacity]
            action = action[:self.capacity]
            reward = reward[:self.capacity]
            done = done[:self.capacity]
            truncated = truncated[:self.capacity]
            next_obs = next_obs[:self.capacity]

            num_samples = self.capacity

            self.all_buffers[self.states_index][:] = curr_obs[:self.capacity]
            self.all_buffers[self.actions_index][:] = action[:self.capacity]
            self.all_buffers[self.reward_index][:] = reward[:self.capacity]
            self.all_buffers[self.dones_index][:] = done[:self.capacity]
            self.all_buffers[self.next_states_index][:] = next_obs[:self.capacity]
            self.all_buffers[self.dones_index][-1] = True
            self.all_buffers[self.truncated_index][-1] = True
            self.latest_obs_start_idx = 0
            self.latest_obs_end_idx = -1
            

        elif self.curr_size + num_samples > self.capacity and not self.continous_mem:
            placement_index = np.random.randint(0, self.capacity - num_samples)

            self.all_buffers[self.states_index][placement_index:placement_index + num_samples] = curr_obs
            self.all_buffers[self.actions_index][placement_index:placement_index + num_samples] = action
            self.all_buffers[self.reward_index][placement_index:placement_index + num_samples] = reward
            self.all_buffers[self.dones_index][placement_index:placement_index + num_samples] = done
            self.all_buffers[self.truncated_index][placement_index:placement_index + num_samples] = truncated
            self.all_buffers[self.next_states_index][placement_index:placement_index + num_samples] = next_obs
            self.curr_size = self.capacity
            self.latest_obs_start_idx = placement_index
            self.latest_obs_end_idx = placement_index + num_samples
        elif self.curr_size + num_samples > self.capacity and self.continous_mem:
            raise NotImplementedError()
        else:
            self.all_buffers[self.states_index][self.curr_size:self.curr_size + num_samples] = curr_obs
            self.all_buffers[self.actions_index][self.curr_size:self.curr_size + num_samples] = action
            self.all_buffers[self.reward_index][self.curr_size:self.curr_size + num_samples] = reward
            self.all_buffers[self.dones_index][self.curr_size:self.curr_size + num_samples] = done
            self.all_buffers[self.truncated_index][self.curr_size:self.curr_size + num_samples] = truncated
            self.all_buffers[self.next_states_index][self.curr_size: self.curr_size+num_samples] = next_obs
            self.latest_obs_start_idx = self.curr_size
            self.latest_obs_end_idx = self.curr_size + num_samples
            self.curr_size += num_samples


        
    def get_last_episodes(self, num_episodes):
        """return all last episode samples, or specified num samples"""

        assert self.num_episodes_added == num_episodes, "change expericne to deal with variating requests of latsets episodes"

        return [buff[self.latest_obs_start_idx:self.latest_obs_end_idx]
                            for buff in self.all_buffers]


class ForgettingExperienceReplayBeta(ExperienceReplayBeta):
    def __init__(self, capacity, obs_shape, continous_mem=False):
        super().__init__(capacity, obs_shape, continous_mem)


    def init_buffers(self):
        super().init_buffers()


    def append(self, curr_obs, action, reward, done, truncated, next_obs):
        num_samples = len(curr_obs)
        self.num_episodes_added = sum(done)
        curr_obs = ObsWraper(curr_obs)
        next_obs = ObsWraper(next_obs)
        self.all_buffers[self.states_index] = curr_obs #np.array(curr_obs).astype(np.float32)
        self.all_buffers[self.actions_index]= np.array(action).astype(np.float32)
        self.all_buffers[self.reward_index]= np.array(reward).astype(np.float32)
        self.all_buffers[self.dones_index] = np.array(done).astype(np.float32)
        self.all_buffers[self.truncated_index] = np.array(truncated).astype(np.float32)
        self.all_buffers[self.next_states_index] = next_obs #np.array(next_obs).astype(np.float32)
        self.curr_size = num_samples



    def get_last_episodes(self, num_episodes):
        """return all last episode samples, or specified num samples"""
        return self.all_buffers
        

# class UniqueExperienceReplay(ExperienceReplay):
#     # TODO: support continous infinte env by removeing long seeing states without dones
#     def __init__(self, capacity, obs_shape, continous_mem=False, unique=False):
#         super().__init__(capacity, obs_shape, continous_mem, unique)
    
#     def init_buffers(self):
#         super().init_buffers()
#         # self.priority = np.zeros((self.capacity), dtype=np.int32)
#         # self.all_buffers.append(self.priority)
#         # self.priority_index = len(self.all_buffers)-1 # last buffer

#     def append(self, curr_obs, action, reward, done, truncated, next_obs):
#         super().append(curr_obs, action, reward, done, truncated, next_obs)
#         new_state, new_index = np.unique(self.all_buffers[self.states_index], return_index=True)
#         for buff in self.all_buffers:
#             buff = buff[new_index]
#         self.curr_size = len(new_index)



# class PriorityExperienceReplay(ExperienceReplay):
#     # TODO: support continous infinte env by removeing long seeing states without dones
#     def __init__(self, capacity, obs_shape, continous_mem=False, unique=False):
#         super().__init__(capacity, obs_shape, continous_mem, unique)
    
#     def init_buffers(self):
#         super().init_buffers()
#         self.value_error = np.zeros((self.capacity), dtype=np.float32)

#     def append(self, curr_obs, action, reward, done, truncated, next_obs):
#         super().append(curr_obs, action, reward, done, truncated, next_obs)
        

def worker(env, conn, idx):
    proc_running = True
    try:
        env.set_random_seed(idx)
    except:
        pass
    try:
        env.seed(idx)
    except:
        pass

    done = False

    while proc_running:
        cmd, msg = conn.recv()

        if (cmd == "step"):
            if done:
                next_state, _ = env.reset()

            next_state, reward, terminated, truncated, _ = env.step(msg)
            conn.send((next_state, reward, terminated, truncated, _))
            done = terminated or truncated 
            if done:
                next_state = env.reset()

        elif (cmd == "reset"):
            conn.send(env.reset())

        elif (cmd == "clear_env"):
            next_state = env.clear_env()
            conn.send(next_state)

        elif (cmd == "step_generator"):
            next_state, reward, done, _ = env.step_generator(msg)
            conn.send((next_state, reward, done, _))

        elif(cmd == "get_env"):
            conn.send(env)

        elif (cmd == "close"):
            proc_running = False
            return conn.close()

        elif (cmd == "sample_random_state"):
            state = env.sample_random_state()
            conn.send(state)

        elif (cmd == "change_env"):
            env = msg
            gc.collect()
            done = False
        else:
            raise Exception("Command not implemented")


class ParallelEnv():
    def __init__(self, env, num_envs, for_val=False):
        self.num_envs = num_envs
        if num_envs > 1:
            self.p_env = ParallelEnv_m(env, num_envs, for_val)
        else:
            self.p_env = SingleEnv_m(env)

    def __del__(self):
        self.p_env.close_procs()
    
    def change_env(self, env):
        self.p_env.change_env(env)

    def get_envs(self):
        return self.p_env.get_envs()

    def reset(self):
        return self.p_env.reset()

    def step(self, actions):
        return self.p_env.step(actions)

    def step_generator(self, actions):
        return self.p_env.step_generator(actions)

    def clear_env(self):
        return self.p_env.clear_env()

    def close_procs(self):
        self.p_env.close_procs()

    def render(self):
        self.p_env.render()


class ParallelEnv_m():
    def __init__(self, env, num_envs, for_val=False):

        self.num_envs = num_envs
        self.process = namedtuple("Process", field_names=[
                                  "proc", "connection"])

        self.comm = []
        for idx in range(self.num_envs):
            parent_conn, worker_conn = Pipe()
            if for_val: #only for running orderd validation envs in parallel
                env.reset()
                # print(env.game_index)
            proc = Process(target=worker, args=((copy.deepcopy(env)), worker_conn, idx))
            proc.start()
            self.comm.append(self.process(proc, parent_conn))

    def change_env(self, env):
        [p.connection.send(("change_env", copy.deepcopy(env))) for p in self.comm]

    def get_envs(self):
        [p.connection.send(("get_env", "")) for p in self.comm]
        res = [p.connection.recv() for p in self.comm]
        return res

    def reset(self):
        [p.connection.send(("reset", "")) for p in self.comm]
        res = [p.connection.recv() for p in self.comm]
        return res

    def step(self, actions):
        # send actions to envs
        [p.connection.send(("step", action)) for i, p, action in zip(
            range(self.num_envs), self.comm, actions)]

        # Receive response from envs.
        res = [p.connection.recv() for p in self.comm]
        next_states, rewards, dones,truncated, _ = zip(*res)
        rewards = np.array(rewards)
        dones = np.array(dones)
        _ = np.array(_)

        return next_states, rewards, dones,truncated, _
    
    def render(self):
        print('Cant draw parallel envs [WIP]')

    # def step_generator(self, actions):
    #     # send actions to envs
    #     [p.connection.send(("step_generator", action)) for i, p, action in zip(
    #         range(self.num_envs), self.comm, actions)]

        # Receive response from envs.
        # res = [p.connection.recv() for p in self.comm]
        # next_states, rewards, dones, _ = zip(*res)
        # rewards = np.array(rewards)
        # dones = np.array(dones)
        # return next_states, rewards, dones, np.array(_)

    # def clear_env(self):
    #     [p.connection.send(("clear_env", "")) for p in self.comm]
    #     res = [p.connection.recv() for p in self.comm]
    #     return res

    # def sample_random_state(self):
    #     [p.connection.send(("sample_random_state", "")) for p in self.comm]
    #     res = [p.connection.recv() for p in self.comm]
    #     return

    def close_procs(self):
        # print("closed procs of env")
        for p in self.comm:
            try:
                p.connection.send(("close", ""))
            except:
                pass
            


class SingleEnv_m():
    def __init__(self, env):
        # print(env.game_index)
        self.env = copy.deepcopy(env)
        self.num_envs = 1

    def change_env(self, env):
        self.env = env

    def get_envs(self):
        return [self.env]

    def reset(self):
        s,info = self.env.reset()
        
        return [(np.array(s, ndmin=1), info)]

    def step(self, actions):
        action = None
        try:
          iter(actions)
          action = actions[0]
        except TypeError:
          action = actions

        next_states, rewards, terminated, trunc, _ = self.env.step(action)
        next_states = np.array(next_states, ndmin=1)

        next_states = next_states[np.newaxis, :]
        rewards = np.array(rewards, ndmin=1)
        terminated = np.array(terminated,ndmin=1)
        trunc = np.array(trunc, ndmin=1)
        return next_states, rewards, terminated, trunc, _

    # def step_generator(self, actions):
    #     next_states, rewards, dones, _ = self.env.step_generator(actions)
    #     next_states = next_states[np.newaxis, :]
    #     rewards = np.array(rewards).reshape(1, 1)
    #     dones = np.array(dones).reshape(1, 1)
    #     return next_states, rewards, dones, _

    # def clear_env(self):
        # return [self.env.clear_env()]

    # def sample_random_state(self):
    #     return [self.env.sample_random_state()]

    def render(self):
        self.env.render()

    def close_procs(self):
        pass
