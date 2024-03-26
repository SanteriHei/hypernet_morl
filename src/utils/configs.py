"""Utilities for handling structured configurations and registering them
with Hydra"""
import json
import math
import pathlib
from typing import Any, List

import mo_gymnasium as mo_gym
import omegaconf
from hydra.core.config_store import ConfigStore

from .. import structured_configs
from ..structured_configs import HyperNetConfig, MlpConfig, ResblockConfig


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

    omegaconf.OmegaConf.register_new_resolver(
        name="env.action_space_low", resolver=_resolve_action_space_low, use_cache=True
    )

    omegaconf.OmegaConf.register_new_resolver(
        name="env.action_space_high",
        resolver=_resolve_action_space_high,
        use_cache=True,
    )
    omegaconf.OmegaConf.register_new_resolver(
            name="load_json", resolver=_resolve_json
    )

    omegaconf.OmegaConf.register_new_resolver(
            name="sum", resolver=lambda x, y: x + y
    )

    omegaconf.OmegaConf.register_new_resolver(
        name="imult", resolver=lambda x, y: math.floor(x * y)
    )

    omegaconf.OmegaConf.register_new_resolver(
        name="network.input_dim", resolver=_resolve_input_dim
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
        group="session_cfg", name="base_session", node=structured_configs.SessionConfig
    )
    cs.store(
        group="training_cfg",
        name="base_training",
        node=structured_configs.TrainingConfig,
    )
    cs.store(
        group="msa_hyper_cfg",
        name="base_msa_hyper",
        node=structured_configs.MSAHyperConfig,
    )
    cs.store(
        group="critic_cfg", name="base_critic", node=structured_configs.HyperCriticConfig
    )
    cs.store(
        group="policy_cfg", name="base_policy", node=structured_configs.PolicyConfig
    )


def _resolve_default_alpha(env_id: str) -> int:
    match env_id:
        case "mo-hopper-v4":
            return 0.2
        case "mo-swimmer-v4": 
            return 0.1
        case "mo-halfcheetah-v4":
            return 0.1
        case _:
            return 0.1


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
        The id of the used environment.

    Returns
    -------
    int
        The dimension of the reward space.
    """
    tmp_env = mo_gym.make(env_id)
    reward_dim = tmp_env.get_wrapper_attr("reward_space").shape[0]
    tmp_env.close()
    return reward_dim


def _resolve_action_space_low(env_id: str) -> List[float]:
    """Resolve the lower bound of the action space in the used environment.

    Parameters
    ----------
    env_id : str
        The id of the used environment.

    Returns
    -------
    List[float]
        The lower bound for each action component.
    """
    tmp_env = mo_gym.make(env_id)
    action_space_low = tmp_env.action_space.low.tolist()
    tmp_env.close()
    return action_space_low


def _resolve_action_space_high(env_id: str) -> List[float]:
    """Resolve the upper bound of the action space in the used environment.

    Parameters
    ----------
    env_id : str
        The id of the used environment.

    Returns
    -------
    List[float]
        The upper bound for each action component.
    """
    tmp_env = mo_gym.make(env_id)
    action_space_high = tmp_env.action_space.high.tolist()
    tmp_env.close()
    return action_space_high


def _resolve_json(filepath: str) -> List[List[float]]:
    """Resolves json to an 2D array.

    Parameters
    ----------
    filepath : str
        The path to the json file.

    Returns
    -------
    List[List[float]]
        The 2D array as list of lists.
    """
    filepath = pathlib.Path(filepath)
    if not filepath.exists() or not filepath.is_file():
        raise ValueError(f"{str(filepath)!r} does not point to a valid file!")

    payload = json.loads(filepath.read_text())
    return [[point["x"], point["y"]] for point in payload]


def _resolve_input_dim(input_list: List[str], env_id: str) -> int:
    """Resolve the input dimension of a network based on the named inputs
    for it.

    Parameters
    ----------
    input_list : List[str]
        The inputs the network is taking.
    env_id : str
        The id of the used environment.

    Returns
    -------
    int
        The dimensionality of the input for the network.
    """
    tmp_env = mo_gym.make(env_id)
    reward_dim = tmp_env.get_wrapper_attr("reward_space").shape[0]
    action_dim = tmp_env.action_space.shape[0]
    obs_dim = tmp_env.observation_space.shape[0]
    tmp_env.close()

    input_dim = 0
    for net_input in input_list:
        match net_input:
            case "prefs":
                input_dim += reward_dim
            case "obs":
                input_dim += obs_dim
            case "action":
                input_dim += action_dim
    return input_dim


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
        cfg = omegaconf.OmegaConf.to_object(cfg)

    if isinstance(cfg.critic_cfg.hypernet_cfg, dict):
        hypernet_cfg = cfg.critic_cfg.hypernet_cfg
        match cfg.critic_cfg.hypernet_type:
            case "resnet":
                embedding_layers = [
                    ResblockConfig(**layer_params)
                    for layer_params in hypernet_cfg["embedding_layers"]
                ]

                cfg.critic_cfg.hypernet_cfg = HyperNetConfig(
                    embedding_layers=embedding_layers,
                    head_hidden_dim=hypernet_cfg["head_hidden_dim"],
                    head_init_method=hypernet_cfg["head_init_method"],
                    head_init_stds=hypernet_cfg["head_init_stds"],
                )
            case "mlp":
                cfg.critic_cfg.hypernet_cfg = MlpConfig(**hypernet_cfg)

    return cfg


def validate(cfg: structured_configs.MSAHyperConfig):
    """Do some validations for the create configurations

    Parameters
    ----------
    cfg : structured_configs.MSAHyperConfig
        The configuration to validate.
    """
    if cfg.critic_cfg.reward_dim != len(cfg.training_cfg.ref_point):
        raise ValueError(
            (
                "Reward dim and ref-point mismatch! ref-point "
                f"{cfg.training_cfg.ref_point} vs "
                f"{cfg.critic_cfg.reward_dim}"
            )
        )

    if any(
        (elem := target_input) not in ("action", "prefs", "obs")
        for target_input in cfg.critic_cfg.target_net_inputs
    ):
        raise ValueError(
            (
                "'critic_cfg.target_net_inputs' can contain keys "
                "'action', 'prefs', 'obs'. Got unknown key "
                f"{elem!r} instead"
            )
        )

    if len(cfg.critic_cfg.target_net_inputs) > 3:
        raise ValueError(
            (
                "The critic target net can take in atmost 3 inputs "
                "('action', 'obs' and 'prefs'). Got "
                f"{cfg.critic_cfg.target_net_inputs} instead"
            )
        )
    if len(cfg.critic_cfg.target_net_inputs) == 0:
        raise ValueError(
            "'critic_cfg.target_net_inputs' must contain atleast one input"
        )

    missing_items = omegaconf.OmegaConf.missing_keys(cfg)
    if len(missing_items) > 0:
        raise ValueError(("Missing values for the following " f"keys {missing_items}"))
