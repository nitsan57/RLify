# RL Kit

RL Kit is a robust and flexible Reinforcement Learning (RL) framework. It provides a rich set of features to facilitate the development, training, and deployment of RL algorithms.

## Usage

For usage examples, please check `rlkit.ipynb`.

## Features

- **Algorithm Support**: Currently, RL Kit supports the following algorithms:

  - Proximal Policy Optimization (PPO)
  - Deep Q-Network (DDQN)
- **Gym API Compatibility**: RL Kit is designed to work seamlessly with environments that implement the OpenAI Gym API.
- **Customizable Agents**: You can create and extend your own agent by overriding certain functions.
- **Neural Network Flexibility**: It's straightforward to integrate your own neural network models, including recurrent models. RL Kit's agent implementations fully support these.
- **Customizable Intrinsic Reward**: Overriding agent's intrinsic reward is as simple as using `agent.set_intrinsic_reward_func()`.
- **Observation Flexibility**: RL Kit supports dictionary format for observations.

And many more features!

## Work in Progress

- Adding support for callbacks.

## Contributions

We welcome contributions! If you're interested in enhancing the features, fixing bugs, or adding new RL algorithms, feel free to open a pull request or issue.

## License

RL Kit is MIT licensed.
