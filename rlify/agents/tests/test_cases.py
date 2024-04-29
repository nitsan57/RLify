def generate_test_cases(discrete_envs=True, continuous_envs=True):
    test_cases = []
    continiout_envs = [
        "MountainCarContinuous-v0",
        "LunarLanderContinuous-v2",
    ]
    discrete_environments = [
        "CartPole-v1",
        "Acrobot-v1",
        "Taxi-v3",
    ]
    assert (
        discrete_envs or continuous_envs
    ), "At least one of the environment types should be enabled"
    environments = discrete_environments if discrete_envs else []
    environments += continiout_envs if continuous_envs else []

    for env in environments:
        for num_parallel_envs in [1, 4]:
            for rnn_flag in [True, False]:
                test_cases.append((env, num_parallel_envs, rnn_flag))

    return test_cases
