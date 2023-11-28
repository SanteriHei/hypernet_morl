# Register custom environments

from gymnasium.envs.registration import register

register(
    id="mo-halfcheetah-v4-fork",
    entry_point="src.environments.halfcheetah:MoHalfCheetah",
    max_episode_steps=500
)

register(
    id="mo-hopper-v4-fork",
    entry_point="src.environments.hoppper:MoHopper",
    max_episode_steps=500
)
