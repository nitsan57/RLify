import torch
import numpy as np
import copy
from rlify.agents.agent_utils import ObsWrapper, ObsShapeWraper


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
            prioritize_high_reward: Whether to prioritize high reward samples
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
            all last episode samples
        """
        num_samples = self.get_num_samples_of_k_last_episodes(num_episodes)
        last_samples = self.get_last_samples(num_samples)
        return last_samples

    def get_first_episodes(self, num_episodes):
        """
        Args:
            num_episodes: The number of episodes to return
            padded: Whether to pad the data
        Returns:
            all last episode samples
        """
        num_samples = self.get_num_samples_of_k_first_episodes(num_episodes)
        first_samples = self.get_first_samples(num_samples)
        return first_samples

    def get_first_samples(self, num_samples):
        """

        Args:
            num_samples: The number of samples to return
        returns:
            states, actions, rewards, dones, next_states

        """
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
        """

        Args:
            num_samples: The number of samples to return
        returns:
            states, actions, rewards, dones, next_states

        """
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
            A batch of random episodes samples
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
            episoed_indices.append(np.arange(s, e + 1))  # to include the done also
            self.sampled_episodes_count[s] += 10  # reward_rank[i]
        data = self.get_buffers_at(np.concatenate(episoed_indices).astype(np.int32))
        return data

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
        self.all_buffers[self.states_index] = curr_obs
        self.all_buffers[self.actions_index] = np.array(actions).reshape(
            -1, self.n_actions
        )  # np.array(actions).astype(np.float32)
        self.all_buffers[self.reward_index] = np.array(rewards).astype(np.float32)
        self.all_buffers[self.dones_index] = np.array(dones).astype(np.float32)
        self.all_buffers[self.truncated_index] = np.array(truncateds).astype(np.float32)
        self.curr_size = num_samples

    def get_last_episodes(self, num_episodes):
        """
        Args:
            num_episodes: The number of episodes to return.
            padded: Whether to pad the data.

        Returns:
            all last episode samples, or specified num samples and loss_flag if padded.
        """
        all_data = self.get_all_buffers()
        return all_data
