from collections import namedtuple
from multiprocessing import Process, Pipe
import numpy as np
import torch
import copy
import gc
import gymnasium as gym
from collections import defaultdict


def calc_gaes(rewards, values, terminated, discount_factor=0.99, decay=0.9):
    """
    works with rewards vector which consitst of many epidsodes
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    device = rewards.device

    next_values = torch.cat([values[1:], torch.zeros(1, device=device)])

    td = rewards + discount_factor * (next_values) * (1 - terminated)
    deltas = td - values

    gaes = torch.zeros_like(deltas, device=device)
    gaes[-1] = rewards[-1]

    for i in reversed(range(len(deltas) - 1)):
        gaes[i] = deltas[i] + discount_factor * decay * gaes[i + 1] * (
            1 - terminated[i]
        )
        # warmup_td[i] = rewards[i] + discount_factor * warmup_td[i+1] * (1 - terminated[i])

    return gaes, td


def calc_returns(rewards, terminated, discount_factor=0.99):
    """
    works with rewards vector which consitst of many epidsodes
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    returns = np.zeros_like(rewards)
    returns[-1] = rewards[-1]

    for i in reversed(range(len(returns) - 1)):
        returns[i] = rewards[i] + discount_factor * returns[i + 1] * (1 - terminated[i])
    return returns


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
                if len(res) == 0:
                    res = (1,)
            except AttributeError:
                res = obs_shape
            super(ObsShapeWraper, self).__init__({"data": tuple([*res])})


class ObsWrapper:
    """
    A class for wrapping observations, the object is roughly a dict of np.arrays or torch.tensors
    A default key is 'data' for the main data if it in either a np.array or torch.tensor

    Example::

            obs = ObsWrapper({'data':np.array([1,2,3]), 'data2':np.array([4,5,6])})
            print(obs['data'])
            print(obs['data2'])
            print(obs['data'][0])
            obs = ObsWrapper(np.array([1,2,3]))
            print(obs['data'])
    """

    def __init__(
        self,
        data: (dict, np.array, torch.tensor) = None,
        keep_dims: bool = True,
        tensors: bool = False,
    ):
        """

        Args:
            data: The data to wrap
            keep_dims: Whether to keep the dimensions of the data, if False will add a dimension of batch to the data
            tensors: Whether to keep the data in torch.tensor
        """

        self.obj_constructor = None
        self.len = 0
        self.data = {}
        self.shape = {}
        if tensors:
            self.obj_constructor = torch.tensor
        if type(data) == ObsWrapper:
            self.data = copy.deepcopy(data.data)
            self.obj_constructor = data.obj_constructor
            self.len = data.len
            self.shape = copy.deepcopy(data.shape)
            return
        if type(data) == dict:
            return self.init_from_dict(data, keep_dims, tensors)

        if np.issubdtype(type(data), np.integer) or np.issubdtype(type(data), float):
            data = np.array(data, ndmin=1).astype(np.float32)

        if data is None:
            return self._init_from_none_(keep_dims, tensors)

        if type(data) == list or type(data) == tuple:
            if type(data[0]) == ObsWrapper:
                return self.init_from_list_obsWrapper_obs(data)
            else:
                return self.init_from_list_generic_data(data)
        else:
            if type(data) == dict:
                to_add = data
            else:
                if keep_dims:
                    to_add = np.array(data, ndmin=1)
                else:
                    to_add = np.expand_dims(data, axis=0)
                self.data = {"data": to_add}
                self.len = len(to_add)
                self.update_shape()
                return
            self.data = {}

            for k, v in to_add.items():
                if torch.is_tensor(v):
                    self.obj_constructor = torch.tensor
                    v = torch.atleast_1d(v)
                else:
                    self.obj_constructor = np.array
                    v = np.array(v, ndmin=1)

                self.data[k] = v
                len_v = len(self.data[k])

                if self.len == 0:
                    self.len = len_v
                assert (
                    self.len == len_v
                ), "cant init a dict with a value with different len"

        self.update_shape()

    def update_shape(self):
        """
        Updates the shape of the object
        """
        self.shape = {}
        for k, v in self.items():
            try:
                self.shape[k] = v.shape
            except AttributeError as e:
                try:
                    self.shape[k] = v.data.shape
                except:
                    self.shape[k] = None

    def init_from_dict(self, data, keep_dims, tensors):
        """
        Initializes from a dict

        Args:
            data: The data to initialize from
            keep_dims: Whether to keep the dimensions of the data, if False will add a dimension of batch to the data
            tensors: Whether to keep the data in torch.tensor
        """

        for k, v in data.items():
            if torch.is_tensor(v):
                self.obj_constructor = torch.tensor
                v = torch.atleast_1d(v)
            else:
                self.obj_constructor = np.array
                v = np.array(v, ndmin=1)

            self.data[k] = v
            len_v = len(self.data[k])

            if self.len == 0:
                self.len = len_v
            assert self.len == len_v, "cant init a dict with a value with different len"
        self.update_shape()

    def init_from_list_obsWrapper_obs(self, obs_list):
        """
        Initializes from a list of ObsWrapper objects
        Args:
            obs_list: The list of ObsWrapper objects
        """
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
        """
        Initializes from a list of generic data
        Args:
            obs_list: The list of generic data
        """
        # get class in a generic way
        if torch.is_tensor(obs_list[0]):
            self.obj_constructor = torch.tensor
            res = self.obj_constructor(obs_list)
        elif type(obs_list[0]) == dict:
            v = {k: [dic[k] for dic in obs_list] for k in obs_list[0]}
            return self.init_from_dict(v, keep_dims=False, tensors=False)

        else:
            self.obj_constructor = np.array
            res = self.obj_constructor(obs_list)
            if len(res.shape) == 1:
                res = res.reshape(res.shape[0], 1).astype(np.float32)

        self.data["data"] = res
        self.len = len(res)
        self.shape["data"] = res.data.shape

    def _init_from_none_(self, keep_dims, tensors):
        """
        Initializes an object without data
        """
        self.__init__({}, keep_dims, tensors)

    def __setitem__(self, key, value):
        """
        Sets an item in the object
        Args:
            key: The key to set
            value: The value to set
        """
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

    def __delitem__(self, key):
        """
        Deletes an item in the object

        Args:
            key: The key to delete
        """
        del self.data[key]
        self.update_shape()

    def __iter__(self):
        """
        Returns:
            an iterator over the object
        """
        return iter(self.data)

    def __getitem__(self, key):
        """
        Args:
            key: The key to get
        Returns:
            The relevant item in respect to the key
        """
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
        return ObsWrapper(temp_dict)

    def slice_tensors(self, key):
        """
        Args:
            key: The key to get
        Returns:
            The sliced tensors
        """
        if type(key) is str:
            return self.data[key]
        temp_dict = {}

        for k, v in self.data.items():
            if np.issubdtype(type(key), np.integer):
                temp_dict[k] = (
                    v.__getitem__(key)
                    .clone()
                    .detach()
                    .requires_grad_(True)
                    .unsqueeze(-1)
                    .float()
                )
            else:
                temp_dict[k] = (
                    v.__getitem__(key).clone().detach().requires_grad_(True).float()
                )
        return ObsWrapper(temp_dict, tensors=True)

    def keys(self):
        """
        Returns:
            the keys of the object
        """
        return self.data.keys()

    def items(self):
        """
        Returns:
            the items of the object
        """
        return self.data.items()

    def values(self):
        """
        Returns:
            the values of the object
        """
        return self.data.values()

    def __len__(self):
        """
        Returns:
            The length of the object
        """
        return self.len

    def __str__(self) -> str:
        """
        Returns the string representation of the object
        """
        return self.data.__str__()

    def __repr__(self) -> str:
        return self.data.__repr__()

    def __mul__(self, other):
        """
        Multiplies the object by another object
        Args:
            other: The other object to multiply by
            multiplies key by key using <*> pointwise operator
        """
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = v * other[k]
        return ObsWrapper(temp_dict, keep_dims=True)

    def __add__(self, other):
        """
        Adds the object by another object
        Args:
            other: The other object to add by
            adds key by key using <+> pointwise operator
        """
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = v + other[k]
        return ObsWrapper(temp_dict, keep_dims=True)

    def __neg__(self):
        """
        Negates the object
        """
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = -v
        return ObsWrapper(temp_dict, keep_dims=True)

    def __sub__(self, other):
        """
        Subtracts the object by another object
        Args:
            other: The other object to subtract by
            subtracts key by key using <-> pointwise operator
        """
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = v - other[k]
        return ObsWrapper(temp_dict, keep_dims=True)

    def __truediv__(self, other):
        """
        Divides the object by another object
        Args:
            other: The other object to divide by
            divides key by key using </> pointwise operator
        """
        temp_dict = {}
        if np.issubdtype(int, np.number):
            num = other
            other = defaultdict(lambda: float(num))
        for k, v in self.data.items():
            temp_dict[k] = v / other[k]
        return ObsWrapper(temp_dict, keep_dims=True)

    def unsqueeze(self, dim=0):
        """
        Args:
            dim: The device to put the tensors on
        Returns:
            The object as tensors
        """
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = v.unsqueeze(dim)
        return ObsWrapper(temp_dict, keep_dims=True, tensors=True)

    def squeeze(self, dim=0):
        """
        Args:
            dim: The device to put the tensors on
        Returns:
            The object as tensors
        """
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = v.squeeze(dim)
        return ObsWrapper(temp_dict, keep_dims=True, tensors=True)

    def get_as_tensors(self, device):
        """
        Args:
            device: The device to put the tensors on
        Returns:
            The object as tensors
        """
        temp_dict = {}
        for k, v in self.data.items():
            if type(v) == np.ndarray:
                temp_dict[k] = torch.from_numpy(v).float().to(device)
            elif type(v) == torch.Tensor:
                temp_dict[k] = v.detach().clone().to(device).float()

        return ObsWrapper(temp_dict, keep_dims=True, tensors=True)

    def to(self, device):
        """
        Args:
            device: The device to put the tensors on
        Returns:
            The object as tensors
        """
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = v.to(device)
        return ObsWrapper(temp_dict, keep_dims=True, tensors=True)

    def cat(self, other, axis=0):
        """
        Concatenates the object by another object
        Args:
            other: The other object to concatenate by
            concatenates key by key
        """
        temp_dict = {}
        for k, v in self.data.items():
            if torch.is_tensor(v):
                temp_dict[k] = torch.cat([self.data[k], other[k]], axis)
            else:
                temp_dict[k] = np.concatenate([self.data[k], other[k]], axis)
        return ObsWrapper(temp_dict)

    def np_roll(self, indx, inplace=False):
        """Rolls the data by indx and fills the empty space with zeros - only on axis 0
        Args:
            indx: The index to roll by
            inplace: Whether to do the roll inplace
        Returns:
            The rolled object
        """
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = np.concatenate(
                [self.data[k][-indx:], np.zeros_like(self.data[k][:-indx])]
            )
        if inplace:
            self.data = temp_dict
        else:
            return ObsWrapper(temp_dict)


class ExperienceReplay:
    # TODO: support continous infinte env by removeing long seeing states without dones
    """
    A class for storing expriences and sampling from them
    """

    def __init__(
        self,
        capacity: float,
        obs_shape: dict,
        n_actions: int,
        prioritize_high_reward=False,
    ):
        """
        Args:
            capacity: The max number of samples to store
            obs_shape: The shape of the obs
            n_actions: The number of actions
        """
        self.obs_shape = ObsShapeWraper(obs_shape)
        self.n_actions = n_actions
        self.capacity = capacity
        self.init_buffers()
        self.continous_mem = False
        self.prioritize_high_reward = prioritize_high_reward

    def __len__(self):
        """
        Returns:
            The current number of samples in the memory
        """
        return self.curr_size

    def append(
        self,
        curr_obs: ObsWrapper,
        actions: np.array,
        rewards: np.array,
        dones: np.array,
        truncateds: np.array,
    ):
        """
        Appends a new sample to the memory
        Args:
            curr_obs: The current observations
            actions: The action taken
            rewards: The reward recieved
            dones: Whether the episode is done
        """
        # extra_exps
        actions = np.array(actions).reshape(-1, self.n_actions)

        curr_obs = ObsWrapper(curr_obs)

        num_samples = len(curr_obs)

        free_space = self.capacity - self.curr_size
        if num_samples > self.capacity:
            self.curr_size = 0
            curr_obs = curr_obs[: self.capacity]
            actions = actions[: self.capacity]
            rewards = rewards[: self.capacity]
            dones = dones[: self.capacity]
            truncateds = truncateds[: self.capacity]
            dones[-1] = True
            truncateds[-1] = True
            num_samples = self.capacity
            self.episode_start_indices = self.get_episode_start_indices()

        elif self.curr_size + num_samples > self.capacity and not self.continous_mem:
            done_indices = self.get_episode_end_indices()
            relevant_dones = np.where(done_indices > num_samples - free_space)[0]
            # roll more then memory needed:

            relevant_index = relevant_dones[len(relevant_dones) // 2]
            done_index = done_indices[relevant_index]
            self.all_buffers[self.states_index] = self.all_buffers[
                self.states_index
            ].np_roll(-done_index - 1, inplace=False)
            self.all_buffers[self.actions_index] = np.concatenate(
                [
                    self.all_buffers[self.actions_index][-(-done_index - 1) :],
                    np.zeros_like(
                        self.all_buffers[self.actions_index][: -(-done_index - 1)]
                    ),
                ]
            )
            self.all_buffers[self.reward_index] = np.concatenate(
                [
                    self.all_buffers[self.reward_index][-(-done_index - 1) :],
                    np.zeros_like(
                        self.all_buffers[self.reward_index][: -(-done_index - 1)]
                    ),
                ]
            )
            self.all_buffers[self.dones_index] = np.concatenate(
                [
                    self.all_buffers[self.dones_index][-(-done_index - 1) :],
                    np.zeros_like(
                        self.all_buffers[self.dones_index][: -(-done_index - 1)]
                    ),
                ]
            )
            self.all_buffers[self.truncated_index] = np.concatenate(
                [
                    self.all_buffers[self.truncated_index][-(-done_index - 1) :],
                    np.zeros_like(
                        self.all_buffers[self.truncated_index][: -(-done_index - 1)]
                    ),
                ]
            )
            self.sampled_episodes_count = np.concatenate(
                [
                    self.sampled_episodes_count[-(-done_index - 1) :],
                    np.ones_like(self.sampled_episodes_count[: -(-done_index - 1)]),
                ]
            )
            self.curr_size -= done_index + 1

        self.all_buffers[self.states_index][
            self.curr_size : self.curr_size + num_samples
        ] = curr_obs
        self.all_buffers[self.actions_index][
            self.curr_size : self.curr_size + num_samples
        ] = actions
        self.all_buffers[self.reward_index][
            self.curr_size : self.curr_size + num_samples
        ] = rewards
        self.all_buffers[self.dones_index][
            self.curr_size : self.curr_size + num_samples
        ] = dones
        self.all_buffers[self.truncated_index][
            self.curr_size : self.curr_size + num_samples
        ] = truncateds

        self.curr_size += num_samples

    def init_buffers(self):
        """
        Initializes the buffers
        """
        self.sampled_episodes_count = np.ones(((self.capacity, 1)), dtype=np.float32)
        self.curr_size = 0
        try:
            actions_buffer = np.zeros(
                ((self.capacity, self.n_actions)), dtype=np.float32
            )  # .squeeze(-1)
        except:
            actions_buffer = np.zeros(
                ((self.capacity, self.n_actions)), dtype=np.float32
            )
        reward_buffer = np.zeros((self.capacity), dtype=np.float32)
        dones_buffer = np.zeros((self.capacity), dtype=np.uint8)
        truncated_buffer = np.zeros((self.capacity), dtype=np.uint8)
        shape = (self.capacity, *self.obs_shape)

        states_buffer = ObsWrapper()
        for k in self.obs_shape:
            shape = (self.capacity, *self.obs_shape[k])
            states_buffer[k] = np.zeros(shape, dtype=np.float32)

        self.all_buffers = [
            states_buffer,
            actions_buffer,
            reward_buffer,
            dones_buffer,
            truncated_buffer,
        ]

        self.states_index = 0
        self.actions_index = 1
        self.reward_index = 2
        self.dones_index = 3
        self.truncated_index = 4

    def clear(self):
        """
        Clears the memory
        """
        self.init_buffers()

    def get_episodes_accumulated_rewards(self):
        end_indices = self.get_episode_end_indices()
        summed_r = np.cumsum(self.all_buffers[self.reward_index])
        first_ep_r = summed_r[end_indices[0]]
        summed_r = summed_r[end_indices] - np.roll(summed_r[end_indices], 1)
        summed_r[0] = first_ep_r
        return summed_r

    def get_episode_end_indices(self):
        return np.where(self.all_buffers[self.dones_index] == True)[0]

    def get_episode_start_indices(self):
        episode_indices = [0]
        episode_indices.extend(self.get_episode_end_indices() + 1)
        return episode_indices[:-1]

    def get_num_samples_of_k_first_episodes(self, num_episodes):
        episode_indices = self.get_episode_start_indices()
        assert (
            len(episode_indices) >= num_episodes
        ), "requested more episodes then actual stored in mem"
        # it is a "False" done just for episode begin idx
        num_samples = episode_indices[num_episodes] + 1
        return num_samples

    def get_num_samples_of_k_last_episodes(self, num_episodes):
        episode_indices = [0]
        episode_indices.extend(
            self.get_episode_end_indices()
        )  # last episode indx is done =1
        assert (
            len(episode_indices) >= num_episodes
        ), "requested more episodes then actual stored in mem"
        # it is a "False" done just for episode begin idx
        if episode_indices[-num_episodes - 1] == 0:
            num_samples = self.curr_size - episode_indices[-num_episodes - 1]
        else:  # we dont want the last done in our batch
            num_samples = (
                self.curr_size - episode_indices[-num_episodes - 1] - 1
            )  # exclude done indice
        return num_samples

    def get_last_episodes(self, num_episodes):
        """
        Args:
            num_episodes: The number of episodes to return
        Returns:
            all last episode samples, or specified num samples
        """
        num_samples = self.get_num_samples_of_k_last_episodes(num_episodes)
        return self.get_last_samples(num_samples)

    def get_first_episodes(self, num_episodes):
        """
        Args:
            num_episodes: The number of episodes to return
        Returns:
            all last episode samples, or specified num samples
        """
        num_samples = self.get_num_samples_of_k_first_episodes(num_episodes)
        return self.get_first_samples(num_samples)

    def get_first_samples(self, num_samples):
        """return all last episode samples, or specified num samples"""
        assert (
            num_samples <= self.curr_size
        ), "requested more samples then actual stored in mem"
        last_samples = [buff[:num_samples] for buff in self.all_buffers]
        # Add next_obs:
        last_samples.append(
            self.all_buffers[self.states_index][:num_samples].np_roll(-1, inplace=False)
        )
        return last_samples

    def get_last_samples(self, num_samples=None):
        """return all last episode samples, or specified num samples"""

        if num_samples is None:
            "return last episode"

            dones = self.get_episode_end_indices()
            if len(dones) > 1:
                # exclude the latest done sample
                first_sample_idx = dones[-2] + 1  # extra_exps

            else:
                # from 0 index to last done(which is also the first..)
                first_sample_idx = 0
                last_samples = [
                    buff[first_sample_idx : self.curr_size] for buff in self.all_buffers
                ]

        else:
            first_sample_idx = self.curr_size - num_samples
            last_samples = [
                buff[first_sample_idx : self.curr_size] for buff in self.all_buffers
            ]

        # Add next_obs:
        last_samples.append(
            self.all_buffers[self.states_index][
                first_sample_idx : self.curr_size
            ].np_roll(-1, inplace=False)
        )

        return last_samples

    def get_all_buffers(self):
        """
        Returns:
            All the buffers
        """
        buffers = copy.deepcopy(self.all_buffers)
        next_obs = buffers[self.states_index].np_roll(-1, inplace=False)

        buffers.append(next_obs)
        return buffers

    def get_buffers_at(self, indices):
        """
        Args:
            indices: The indices to return
        Returns:
            The buffers at the given indices
        """
        buffers = self.get_all_buffers()
        buffers_at = (
            buffers[0][indices],
            buffers[1][indices],
            buffers[2][indices],
            buffers[3][indices],
            buffers[4][indices],
            buffers[5][indices],
        )
        return buffers_at

    def sample_random_episodes(self, num_episodes: int):
        """
        Args:
            num_episodes: The number of full episodes to return
        Returns:
            A batch ofrandom episodes samples
        """

        ###

        cumsum = self.get_episodes_accumulated_rewards()
        moved_cumsum = cumsum + np.abs(cumsum.min())
        moved_cumsum = moved_cumsum / (moved_cumsum.max() + 1e-3)
        reward_rank = 10 - moved_cumsum * 10

        ###

        end_done_indices = self.get_episode_end_indices()
        start_indices = self.get_episode_start_indices()
        stored_episodes = len(end_done_indices)

        weights = 1 / (
            self.sampled_episodes_count[start_indices].squeeze() + reward_rank
        )
        weights = weights / weights.sum()  # np.exp(weights)/np.exp(weights).sum()
        chosen_episodes_indices = np.random.choice(
            stored_episodes, num_episodes, replace=False, p=weights.flatten()
        )
        episoed_indices = []
        for i in chosen_episodes_indices:
            s = start_indices[i]
            e = end_done_indices[i]
            episoed_indices.append(np.arange(s, e + 1))  # to inclode the done also
            self.sampled_episodes_count[s] += 10  # reward_rank[i]
        return self.get_buffers_at(np.concatenate(episoed_indices).astype(np.int32))

    def sample_random_batch(self, sample_size):
        """
        Args:
            sample_size: The number of samples to return
        Returns:
            A random batch of samples
        """
        sample_size = min(sample_size, self.curr_size)
        indices = np.random.choice(self.curr_size, sample_size, replace=False)
        return self.get_buffers_at(indices)


class ForgettingExperienceReplay(ExperienceReplay):
    """
    This class is used to store and sample experience, it forgets old experience in every append.
    """

    def __init__(self, capacity, obs_shape, continous_mem=False):
        """
        Args:
            capacity: The capacity of the replay buffer.
            obs_shape: The shape of the observations.
            continous_mem: If true, the replay buffer will be continous, meaning that the oldest samples will be overwritten.
        """
        super().__init__(capacity, obs_shape, continous_mem)

    def init_buffers(self):
        """
        Initializes the buffers.
        """
        super().init_buffers()

    def append(
        self,
        curr_obs: ObsWrapper,
        actions: np.array,
        rewards: np.array,
        dones: np.array,
        truncateds: np.array,
    ):
        """
        Appends a new sample to the memory.
        Args:
            curr_obs: The current observations.
            actions: The action taken.
            rewards: The reward recieved.
            dones: Whether the episode is done.
        """

        num_samples = len(curr_obs)
        self.num_episodes_added = sum(dones)
        curr_obs = ObsWrapper(curr_obs)
        self.all_buffers[self.states_index] = (
            curr_obs  # np.array(curr_obs).astype(np.float32)
        )
        self.all_buffers[self.actions_index] = np.array(actions).astype(np.float32)
        self.all_buffers[self.reward_index] = np.array(rewards).astype(np.float32)
        self.all_buffers[self.dones_index] = np.array(dones).astype(np.float32)
        self.all_buffers[self.truncated_index] = np.array(truncateds).astype(np.float32)
        self.curr_size = num_samples

    def get_last_episodes(self, num_episodes):
        """return all last episode samples, or specified num samples"""
        return self.get_all_buffers()


def worker(env, conn):
    """
    This function is used to run an environment in a separate process.
    Args:
        env: The environment to run.
        conn: The connection to the main process.
    """
    proc_running = True

    done = False

    while proc_running:
        cmd, msg = conn.recv()

        if cmd == "step":
            if done:
                next_state, _ = env.reset()

            next_state, reward, terminated, truncated, _ = env.step(msg)
            conn.send((next_state, reward, terminated, truncated, _))
            done = terminated or truncated
            if done:
                next_state = env.reset()

        elif cmd == "reset":
            conn.send(env.reset())

        elif cmd == "get_env":
            conn.send(env)

        elif cmd == "close":
            proc_running = False
            return

        elif cmd == "change_env":
            env = msg
            gc.collect()
            done = False
        else:
            raise Exception("Command not implemented")


class ParallelEnv:
    """
    This class is used to run multiple environments in parallel.
    """

    def __init__(self, env: gym.Env, num_envs: int):
        """
        Args:
            env: The environment to run in parallel.
            num_envs: The number of environments to run in parallel.
            for_val: If true, the environments will be run in a fixed order to allow for deterministic evaluation.
        """
        self.num_envs = num_envs
        if num_envs > 1:
            self.p_env = ParallelEnv_m(env, num_envs)
        else:
            self.p_env = SingleEnv_m(env)

    def __del__(self):
        self.p_env.close_procs()

    def change_env(self, env):
        """
        Changes the environment to run in parallel.
        Args:
            env: The new environment.
        """
        self.p_env.change_env(env)

    def get_envs(self):
        """
        Returns:
            A list of the environments.
        """
        return self.p_env.get_envs()

    def reset(self):
        """
        Resets the environments.
        """
        return self.p_env.reset()

    def step(self, actions):
        """
        Takes a step in the environments.
        Args:
            actions: The actions (n, action_shape) to take in the environments.
        """
        return self.p_env.step(actions)

    def close_procs(self):
        """
        Closes the processes.
        """
        self.p_env.close_procs()

    def render(self):
        """
        Renders the environments. - not supported for multi envs - renders just the base env - good for single env case
        """
        self.p_env.render()


class ParallelEnv_m:
    def __init__(self, env, num_envs):
        """
        Args:
            env: The environment to run in parallel.
            num_envs: The number of environments to run in parallel.
            for_val: If true, the environments will be run in a fixed order to allow for deterministic evaluation.
        """
        self.num_envs = num_envs
        self.process = namedtuple(
            "Process", field_names=["proc", "connection", "worker_conn"]
        )
        self.comm = []
        for idx in range(self.num_envs):
            parent_conn, worker_conn = Pipe()
            proc = Process(target=worker, args=((copy.deepcopy(env)), worker_conn))
            proc.start()
            self.comm.append(self.process(proc, parent_conn, worker_conn))

    def change_env(self, env):
        """
        Changes the environment to run in parallel.
        Args:
            env: The new environment.
        """
        [p.connection.send(("change_env", copy.deepcopy(env))) for p in self.comm]

    def get_envs(self):
        """
        Returns:
            A list of the environments.
        """
        [p.connection.send(("get_env", "")) for p in self.comm]
        res = [p.connection.recv() for p in self.comm]
        return res

    def reset(self):
        """
        Resets the environments.
        """
        [p.connection.send(("reset", "")) for p in self.comm]
        res = [p.connection.recv() for p in self.comm]
        return res

    def step(self, actions):
        """
        Takes a step in the environments.
        Args:
            actions: The actions (n, action_shape) to take in the environments.
        """
        # send actions to envs
        [
            p.connection.send(("step", action))
            for i, p, action in zip(range(self.num_envs), self.comm, actions)
        ]

        # Receive response from envs.
        res = [p.connection.recv() for p in self.comm]
        next_states, rewards, terminated, truncated, _ = zip(*res)
        rewards = np.array(rewards)
        terminated = np.array(terminated)
        truncated = np.array(truncated)
        _ = np.array(_)

        return next_states, rewards, terminated, truncated, _

    def render(self):
        print("Cant draw parallel envs [WIP]")

    def __del__(self):
        self.close_procs()

    def close_procs(self):
        """
        Closes the processes.
        """
        for p in self.comm:
            try:
                p.connection.send(("close", ""))
                p.connection.close()
                p.worker_conn.close()
            except Exception as e:
                print("close failed", p)
                print("close failed -reason:", e)
                pass
        self.comm = []


class SingleEnv_m:
    """
    This class is used to run a single environment.
    """

    def __init__(self, env):
        """
        Args:
            env: The environment to run in parallel.
        """
        # print(env.game_index)
        self.env = copy.deepcopy(env)
        self.num_envs = 1

    def change_env(self, env):
        """
        Changes the environment to run.
        """
        self.env = env

    def get_envs(self):
        """
        Returns:
            A list of the environments - list with single item in this case.
        """
        return [self.env]

    def reset(self):
        """
        Resets the environment.
        """
        s, info = self.env.reset()
        if type(s) != dict:
            return [(np.array(s, ndmin=1), info)]
        else:
            return [(s, info)]

    def step(self, actions):
        """
        Takes a step in the environment.
        Args:
            actions: The actions (1, action_shape) to take in the environment.
        """
        action = None
        try:
            iter(actions)
            action = actions[0]
        except TypeError:
            action = actions
        next_states, rewards, terminated, trunc, _ = self.env.step(action)
        if type(next_states) != dict:
            next_states = np.array(next_states, ndmin=2)
        else:
            next_states = [next_states]
        rewards = np.array(rewards, ndmin=1)
        terminated = np.array(terminated, ndmin=1)
        trunc = np.array(trunc, ndmin=1)
        return next_states, rewards, terminated, trunc, _

    def render(self):
        """
        Renders the environment.
        """
        self.env.render()

    def close_procs(self):
        """
        Closes the processes[actually is a noop in this case].
        """
        pass


import pandas as pd


class TrainMetrics:
    def __init__(self):
        """
        Args:
            metrics: The metrics to store.
        """
        self.metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(list)

    def add(self, metric_name, value):
        """
        Adds a metric to the metrics.
        Args:
            metric_name: The name of the metric.
            value: The value of the metric.
        """
        self.epoch_metrics[metric_name].append(value)

    def on_epoch_end(self):
        """
        Adds a metric to the metrics.
        Args:
            metric_name: The name of the metric.
            value: The value of the metric.
        """
        for k in self.epoch_metrics.keys():
            self.metrics[k].append(
                sum(self.epoch_metrics[k]) / len(self.epoch_metrics[k])
            )
        self.epoch_metrics = defaultdict(list)

    def get_metrcis_df(self):
        """
        Returns:
            The metrics as a dataframe.
        """
        return pd.DataFrame(self.metrics)

    def __iter__(self):
        """
        Returns:
            An iterator over the metrics.
        """
        return iter(self.metrics)

    def __next__(self):
        """
        Returns:
        An iterator over the metrics.
        """
        return next(self.metrics)

    def __getitem__(self, key):
        """
        Returns:
        An iterator over the metrics.
        """
        return self.metrics[key]
