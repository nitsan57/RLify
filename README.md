# RL Kit

RL Kit is a robust and flexible Reinforcement Learning (RL) framework. It provides a rich set of features to facilitate the development, training, and deployment of RL algorithms.

## Usage

For usage examples, please check `getting-started.ipynb`.

Also check the [docs](https://nitsan57.github.io/RLkit-docs/)

## Features

- **Algorithm Support**: Currently, RL Kit supports the following algorithms:

  - Proximal Policy Optimization (PPO)
  - Deep Q-Network (DDQN)
- **Gym API Compatibility**: RL Kit is designed to work seamlessly with environments that implement the OpenAI Gym API.
- **Customizable Agents**: You can create and extend your own agent by overriding certain functions.
- **Neural Network Flexibility**: It's straightforward to integrate your own neural network models, including recurrent models. RL Kit's agent implementations fully support these.
- **Customizable Intrinsic Reward**: Overriding agent's intrinsic reward is as simple as using `agent.set_intrinsic_reward_func()`.
- **Override Exploration Method Using heuristic functions**: Overriding exploration method from random choice to some user inputed heurstic to speed up train in certein env `HeuristicExplorer()`.
- **Observation Flexibility**: RL Kit supports dictionary format for observations.
- **Metrics:** Dumps all the metrics to tensorboard.

And many more features!

A short summery of curr supported:

| Algorithm  | Discrete action | Continuous action | Recurrent |
| ---------- | --------------- | ----------------- | --------- |
| cliped-PPO | Yes             | Yes               | Yes       |
| DQN (ddqn) | Yes             | No                | Yes       |

In this table, the first column represents the algorithm names (PPO and DQN). The second column indicates whether the algorithm supports discrete actions (Yes) or not (No). The third column indicates whether the algorithm supports continuous actions (Yes) or not (No). The last column represents whether the algorithm supports recurrent architectures (Yes) or not (No).

Please note that the table content is based on general information and may vary depending on the specific implementation or version of the algorithms.

## Work in Progress

- Add support for callbacks
- Add Suppor for MARL

## Contributions

We welcome contributions! If you're interested in enhancing the features, fixing bugs, or adding new RL algorithms, feel free to open a pull request or issue.

## License

RL Kit is MIT licensed.
