from gymnasium.envs.registration import register

register(
    id='RLCagingEnv-v0',
    entry_point='gym_env.rl_caging_env:RLCagingEnv',
    max_episode_steps=1000,
)