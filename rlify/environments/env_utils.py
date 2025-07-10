from collections import namedtuple
from multiprocessing import Process, Pipe
import gymnasium as gym
import gc
import copy
import numpy as np
from numpy.random import seed


def worker(env, conn):
    """
    This function is used to run an environment in a separate process.
    Args:
        env: The environment to run.
        conn: The connection to the main process.
    """
    proc_running = True
    done = False

    seed()

    while proc_running:
        if conn.poll(30):
            cmd, msg = conn.recv()
        else:
            return

        if cmd == "step":
            if done:
                next_state, _ = env.reset()

            next_state, reward, terminated, truncated, _ = env.step(msg)
            conn.send((next_state, reward, terminated, truncated, _))
            done = terminated or truncated

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
            proc = Process(target=worker, args=(env, worker_conn))
            proc.start()
            self.comm.append(self.process(proc, parent_conn, worker_conn))

    def change_env(self, env):
        """
        Changes the environment to run in parallel.
        Args:
            env: The new environment.
        """
        [p.connection.send(("change_env", env)) for p in self.comm]

    def get_responses(self):
        """
        Gets the responses from the environments.
        """
        timeout = 5
        got_data = all([p.connection.poll(timeout) for p in self.comm])
        if got_data:
            return [p.connection.recv() for p in self.comm]
        else:
            self.close_procs()
            raise Exception("Timeout in getting responses")

    def get_envs(self):
        """
        Returns:
            A list of the environments.
        """
        [p.connection.send(("get_env", "")) for p in self.comm]
        res = self.get_responses()
        return res

    def reset(self):
        """
        Resets the environments.
        """
        [p.connection.send(("reset", "")) for p in self.comm]
        res = self.get_responses()
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
        res = self.get_responses()
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
