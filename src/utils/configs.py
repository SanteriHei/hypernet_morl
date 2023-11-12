"""Utilities for handling structured configurations and registering them
with Hydra"""


from typing import Any

import mo_gymnasium as mo_gym
import numpy as np
import omegaconf
from hydra.core.config_store import ConfigStore

from .. import structured_configs


def register_resolvers():
    """Add some custom resolvers to omegaconf"""
    omegaconf.OmegaConf.register_new_resolver(
        name="env.obs_dim", resolver=_resolve_obs_dim, use_cache=True
    )

    omegaconf.OmegaConf.register_new_resolver(
        name="env.action_dim", resolver=_resolve_action_dim, use_cache=True
    )

    omegaconf.OmegaConf.register_new_resolver(
        name="env.reward_dim", resolver=_resolve_reward_dim, use_cache=True
    )

def register_configs(cs: ConfigStore):
    """Register the structured configurations to the given configuration store.

    Parameters
    ----------
    cs : ConfigStore
        The configuration store to register the data to.
    """


    cs.store(name="base_config", node=structured_configs.Config)
    cs.store(
        group="session_cfg", name="base_session",
        node=structured_configs.SessionConfig
    )
    cs.store(
        group="training_cfg", name="base_training",
        node=structured_configs.TrainingConfig
    )
    cs.store(
        group="msa_hyper_cfg", name="base_msa_hyper",
        node=structured_configs.MSAHyperConfig
    )
    cs.store(
        group="hypernet_cfg", name="base_hypernet",
        node=structured_configs.HypernetConfig
    )
    cs.store(
        group="policy_cfg", name="base_policy",
        node=structured_configs.PolicyConfig
    )


def _resolve_obs_dim(env_id: str) -> int:
    """Resolve the observation dimension of the currently used config.

    Parameters
    ----------
    env_id : str
        The id of the environment that is used for the experimemnt.

    Returns
    -------
    int
        The dimension of the observation space.
    """
    tmp_env = mo_gym.make(env_id)
    obs_dim = tmp_env.observation_space.shape[0]
    tmp_env.close()
    return obs_dim


def _resolve_action_dim(env_id: str) -> int:
    """Resolve the dimension of the used action space

    Parameters
    ----------
    env_id : str
        The id of the used environment.

    Returns
    -------
    int
        The dimension of the action space.
    """
    tmp_env = mo_gym.make(env_id)
    action_dim = tmp_env.action_space.shape[0]
    tmp_env.close()
    return action_dim


def _resolve_reward_dim(env_id: str) -> int:
    """Resolve the dimension of the used reward space

    Parameters
    ----------
    env_id : str
        The if of the used environment.

    Returns
    -------
    int
        The dimension of the reward space.
    """
    tmp_env = mo_gym.make(env_id)
    reward_dim = tmp_env.get_wrapper_attr("reward_space").shape[0]
    tmp_env.close()
    return reward_dim


def as_structured_config(cfg: omegaconf.DictConfig) -> Any:
    """Convert a omegaconf DictConfig into a structured config if the current
    configuratio is not already converted. Otherwise does nothing.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        The configuration to change

    Returns
    -------
    Any
        The converted config.
    """
    if isinstance(cfg, omegaconf.DictConfig):
        return omegaconf.OmegaConf.to_object(cfg)
    return cfg


def fill_missing_fields(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Fill any missing values from the configuration. Mostly used for filling
    values that require instantiating a gym env.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        The configuration to fill.

    Returns
    -------
    omegaconf.DictConfig
        The configuration with the missing values filled.
    """
    tmp_env = mo_gym.make(cfg.training_cfg.env_id)
    reward_dim = tmp_env.get_wrapper_attr("reward_space").shape[0]
    obs_dim = tmp_env.observation_space.shape[0]
    action_dim = tmp_env.action_space.shape[0]

    cfg.hypernet_cfg.reward_dim = reward_dim
    cfg.hypernet_cfg.obs_dim = obs_dim
    cfg.hypernet_cfg.action_dim = action_dim
    if (
            len(cfg.hypernet_cfg.resblock_arch) > 0 and
            omegaconf.OmegaConf.is_missing(
                cfg.hypernet_cfg.resblock_arch[0], "input_dim"
            )
    ):
        cfg.hypernet_cfg.resblock_arch[0].input_dim = obs_dim + reward_dim
    cfg.policy_cfg.obs_dim = obs_dim
    cfg.policy_cfg.reward_dim = reward_dim
    cfg.policy_cfg.output_dim = action_dim
    cfg.policy_cfg.action_space_high = tmp_env.action_space.high.tolist()
    cfg.policy_cfg.action_space_low = tmp_env.action_space.low.tolist()

    if omegaconf.OmegaConf.is_missing(cfg.training_cfg, "angle"):
        cfg.training_cfg.angle = np.pi * (22.5/100)

    tmp_env.close()
    assert len(missing_items := omegaconf.OmegaConf.missing_keys(cfg)) == 0, \
        ("Expected that all missing items are filled, found items "
         f"{missing_items} without a value")

    return cfg
