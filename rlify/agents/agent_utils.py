import logging
import numpy as np
import torch
import copy
import gymnasium as gym
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def pad_from_done_indices(data, dones):
    """
    Packs the data from the done indices to torch.nn.utils.rnn.PackedSequence

    """
    if isinstance(data, ObsWrapper):
        return pad_states_from_done_indices(data, dones)
    elif isinstance(data, torch.Tensor):
        return pad_tensors_from_done_indices(data, dones)


def pad_states_from_done_indices(data, dones):
    """
    Packs the data from the done indices to torch.nn.utils.rnn.PackedSequence

    """
    done_indices = torch.where(dones == True)[0].cpu().numpy().astype(np.int32)
    padded_obs = ObsWrapper(tensors=True)
    for k in data:
        temp = []
        curr_idx = 0
        for i, d_i in enumerate(done_indices):
            temp.append(data[k][curr_idx : d_i + 1])
            curr_idx = d_i + 1  #
        padded_seq_batch = torch.nn.utils.rnn.pad_sequence(temp, batch_first=True)
        padded_obs[k] = padded_seq_batch
    lengths = done_indices - np.roll(done_indices, 1)
    lengths[0] = done_indices[0]
    return padded_obs, lengths


def pad_tensors_from_done_indices(data, dones):
    """
    Packs the data from the done indices to torch.nn.utils.rnn.PackedSequence

    """
    done_indices = torch.where(dones == True)[0].cpu().numpy().astype(np.int32)
    temp = []
    curr_idx = 0
    for i, d_i in enumerate(done_indices):
        temp.append(data[curr_idx : d_i + 1])
        curr_idx = d_i + 1  #
    padded_seq_batch = torch.nn.utils.rnn.pad_sequence(temp, batch_first=True)
    padded_obs = padded_seq_batch
    lengths = done_indices - np.roll(done_indices, 1)
    lengths[0] = done_indices[0] + 1
    return padded_obs, lengths


def calc_gaes(rewards, values, terminated, discount_factor=0.99, decay=0.9):
    """
    works with rewards vector which consitst of many epidsodes
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    next_values = np.concatenate([values[1:], np.zeros(1)])
    td = rewards + discount_factor * (next_values) * (1 - terminated)
    deltas = td - values

    gaes = np.zeros_like(deltas).astype(np.float32)
    gaes[-1] = rewards[-1]

    for i in reversed(range(len(deltas) - 1)):
        gaes[i] = deltas[i] + discount_factor * decay * gaes[i + 1] * (
            1 - terminated[i]
        )
        # warmup_td[i] = rewards[i] + discount_factor * warmup_td[i+1] * (1 - terminated[i])

    return gaes.astype(np.float32), td.astype(np.float32)


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
        data: [dict, np.array, torch.Tensor] = None,
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

    def __copy__(self):
        return ObsWrapper(self)

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
                v = torch.atleast_1d(v).float()
            else:
                self.obj_constructor = np.array
                v = np.array(v, ndmin=1).astype(np.float32)

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
                res = torch.cat(res).float()
            else:
                res = np.concatenate(res).astype(np.float32)

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
            res = self.obj_constructor(obs_list).float()
        elif type(obs_list[0]) == dict:
            v = {k: [dic[k] for dic in obs_list] for k in obs_list[0]}
            return self.init_from_dict(v, keep_dims=False, tensors=False)

        else:
            self.obj_constructor = np.array
            res = self.obj_constructor(obs_list).astype(np.float32)
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
                    v.__getitem__(key).clone().detach().requires_grad_(False).float()
                )
            else:
                temp_dict[k] = (
                    v.__getitem__(key).clone().detach().requires_grad_(False).float()
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
        if np.issubdtype(type(other), np.number):
            num = other
            other = defaultdict(lambda: float(num))
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
        if np.issubdtype(type(other), np.number):
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

    def flatten(self, start_dim=None, env_dim=None):
        """
        Args:
            dim: The device to put the tensors on
        Returns:
            The object as tensors
        """
        temp_dict = {}

        try:
            for k, v in self.data.items():
                temp_dict[k] = v.flatten(start_dim, env_dim)
        except TypeError:
            raise Exception(
                "flatten with dims, not supported for ObsWrapper that is not torch.tensor"
            )
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

    def abs(self):
        """
        Returns the absolute value of the object
        """
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = v.abs()
        return ObsWrapper(temp_dict, keep_dims=True)

    def to(self, device, non_blocking=False):
        """
        Args:
            device: The device to put the tensors on
        Returns:
            The object as tensors
        """
        temp_dict = {}

        for k, v in self.data.items():
            temp_dict[k] = v.to(device, non_blocking=non_blocking)
        return ObsWrapper(temp_dict, keep_dims=True, tensors=True)

    def stack(obs_wrapper_list: list):
        """
        stack a list of objects
        """
        keys = list(obs_wrapper_list[0].keys())
        obs_dict = {}
        for k in keys:
            res = []
            for obs in obs_wrapper_list:
                res.append(obs[k])
            is_tensor = torch.is_tensor(obs_wrapper_list[0][k])
            if is_tensor:
                res = torch.stack(res)
            else:
                res = np.stack(res)
            obs_dict[k] = res
        return ObsWrapper(obs_dict)

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


from abc import ABC


class IData(ABC):
    """
    An abstract class for agents data
    """

    def __init__(self, dataset: Dataset, prepare_for_rnn):
        """
        Args:
            dataset: The dataset to use
            prepare_for_rnn: Whether to prepare the data for RNN

        """
        self.dataset = dataset
        self.prepare_for_rnn = prepare_for_rnn
        self.can_shuffle = False if self.prepare_for_rnn else True

    def get_dataloader(self, batch_size, shuffle, num_workers):
        """
        Args:
            batch_size: The batch size
            shuffle: Whether to shuffle the data
            num_workers: The number of workers
        Returns:
            A DataLoader object
        """
        if not self.can_shuffle and shuffle == True:
            logging.warning(
                "Shuffle is not allowed when preparing data for RNN, changing shuffle to False"
            )
            shuffle = False

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self.dataset.collate_fn,
        )


class LambdaDataset(Dataset):
    """
    A dataset class for general purposes
    """

    def __init__(
        self,
        obs_collection: tuple[ObsWrapper],
        tensor_collection: tuple[torch.tensor],
        dones: torch.tensor,
        prepare_for_rnn: bool,
    ):
        """
        Args:
            obs_collection: The observation collection
            tensor_collection: The tensor collection
            dones: The dones tensor
            prepare_for_rnn: Whether to prepare the data for RNN

        """
        obs_collection, tensor_collection, dones = self._prepare_data(
            obs_collection, tensor_collection, dones
        )
        self.obs_collection = obs_collection
        self.tensor_collection = tensor_collection
        self.dones = dones
        self.loss_flag = torch.ones_like(dones).bool()
        self.prepare_for_rnn = prepare_for_rnn
        self.num_items = len(dones)
        if self.prepare_for_rnn:
            (
                self.obs_collection,
                self.tensor_collection,
                self.dones,
                self.loss_flag,
                lengths,
            ) = self._pad_experiecne(self.obs_collection, self.tensor_collection, dones)
            self.num_items = (
                lengths.max()
            )  # 16 games , 200 sequential history, *obs_shape

    def __len__(self):
        return self.num_items

    def _prepare_data(self, obs_collection, tensor_collection, dones):
        """
        Prepares the data for the dataset in form of tensors
        Args:
            obs_collection: The observation collection
            tensor_collection: The tensor collection
            dones: The dones tensor

        Returns:
            The prepared data

        """
        obs_collection = [obs.get_as_tensors("cpu") for obs in obs_collection]
        tensor_collection = [torch.from_numpy(tensor) for tensor in tensor_collection]
        dones = torch.from_numpy(dones)
        return obs_collection, tensor_collection, dones

    def _pad_experiecne(self, obs_collection, tensor_collection, dones):
        """
        Pads the experience for RNN
        Args:
            obs_collection: The observation collection
            tensor_collection: The tensor collection
            dones: The dones tensor

        Returns:
            The padded experience and loss flag
            loss flag is a tensor of ones where the data are not padded
        """

        obs_collection = [
            pad_states_from_done_indices(obs, dones)[0] for obs in obs_collection
        ]
        tensor_collection = [
            pad_tensors_from_done_indices(tensor, dones)[0]
            for tensor in tensor_collection
        ]
        loss_flag, lengths = pad_tensors_from_done_indices(
            torch.ones_like(dones), dones
        )
        loss_flag = loss_flag.bool()
        dones, lengths = pad_tensors_from_done_indices(dones, dones)
        return (
            obs_collection,
            tensor_collection,
            dones,
            loss_flag,
            lengths,
        )

    def __getitems__(self, idx):
        if self.prepare_for_rnn:
            obs_collection = [obs[:, idx] for obs in self.obs_collection]
            tensor_collection = [tensor[:, idx] for tensor in self.tensor_collection]
            dones = self.dones[:, idx]
            loss_flag = self.loss_flag[:, idx].squeeze(0)
        else:
            obs_collection = [obs[idx] for obs in self.obs_collection]
            tensor_collection = [tensor[idx] for tensor in self.tensor_collection]
            dones = self.dones[idx]
            loss_flag = self.loss_flag[idx]

        return (
            obs_collection,
            tensor_collection,
            dones,
            loss_flag,
        )

    def __getitem__(self, idx):
        return self.__getitems__(idx)

    def collate_fn(self, batch):
        return batch


class LambdaData(IData):
    def __init__(
        self,
        obs_collection: tuple[ObsWrapper],
        tensor_collection: tuple[torch.tensor],
        dones: torch.tensor,
        prepare_for_rnn: bool,
    ) -> None:
        dataset = LambdaDataset(
            obs_collection, tensor_collection, dones, prepare_for_rnn
        )
        super().__init__(dataset, prepare_for_rnn)


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
