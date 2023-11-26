# Register custom environments

from gymnasium.envs.registration import register

register(
    id="mo-halfcheetah-v4-fork",
    entry_point="src.environments.halfcheetah:MoHalfCheetah",
    max_episode_steps=1000
)
