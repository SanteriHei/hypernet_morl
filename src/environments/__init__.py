# Register custom environments

from gymnasium.envs.registration import register

register(
    id="mo-halfcheetah-v4-fork",
    entry_point="src.environments.halfcheetah:MoHalfCheetah",
    max_episode_steps=500,
    disable_env_checker=True
)

register(
    id="mo-hopper-v4-fork",
    entry_point="src.environments.hopper:MoHopper",
    max_episode_steps=500,
    disable_env_checker=True
)

register(
    id="mo-swimmer-v4-fork",
    entry_point="src.environments.swimmer:MoSwimmer",
    max_episode_steps=500,
    disable_env_checker=True
)
