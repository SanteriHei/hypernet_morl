import logging
import logging.config
import pathlib
import warnings
from typing import Any, List, Mapping

import numpy as np
import numpy.typing as npt
import torch
from ruamel.yaml import YAML

import wandb

from .. import structured_configs
from . import common, metrics, pareto

_CONFIG_SET: bool = False

_CONFIG_PATH = pathlib.Path(__file__).parents[2] / "configs" / "log_config.yml"

# Shorhand for this 
WandbRun = wandb.sdk.wandb_run.Run

def setup_wandb(
        session_cfg: structured_configs.SessionConfig,
        model_config: Mapping[str, Any]
) -> WandbRun:
    """Initialize the Wandb client to be used for logging the experiment
    results

    Parameters
    ----------
    session_cfg : structured_configs.SessionConfig
        The sesssion configuration.
    model_config : Mapping[str, Any]
        The configuration of the used model. This will be saved to the logged
        run.

    Returns:
        WandbRun
            The created run object.
    """
    run = wandb.init(
        project=session_cfg.project_name,
        entity=session_cfg.entity_name,
        group=session_cfg.experiment_group,
        name=session_cfg.run_name,
        config=model_config,
        sync_tensorboard=True
    )

    # Define an metric for using the global-steps
    wandb.define_metric("*", step_metric="global_step")
    return run

def log_losses(
        losses: Mapping[str, float], *,
        global_step: int,
        wandb_run: WandbRun | None = None,
        logger: logging.Logger | None = None
):

    """Log the losses of the components.

    Parameters
    ----------
    losses : Mapping[str, float]
        The losses to log. Should contain keys "critic" and "policy"
    global_step : int
        The global step during which the losses where calculated at.
    wandb_run : WandbRun | None, optional
        The wandn run handle. If None, no data is logged into wandb.
        Default None.
    logger : logging.Logger | None, optional
        THe logger used for logging. If None, no data is logged. Default None.
    """
    if wandb_run is None and logger is None:
        return

    if wandb_run is not None:
        wandb_run.log({
            "losses/critic_loss": losses["critic"],
            "losses/policy_loss": losses["policy"],
            "global_step": global_step
        })

    if logger is not None:
        logger.info(
            (f"Step {global_step} | Critic loss {losses['critic']:.3f} "
            f"| Policy loss {losses['policy']:.3f}")
        )

def log_eval_info(
        eval_info: Mapping[str, Any], *,
        global_step: int,
        wandb_run: WandbRun | None = None,
        logger: logging.Logger | None = None
):
    """Log information the agents evaluation to the wandb table and 
    possibly to the logger.

    Parameters
    ----------
    eval_info : Mapping[str, Any]
        The evaluation info. Should contain scalarized return, scalarized 
        discounted return, returns and discounted returns.
    global_step : int
        The global step at which the agent was evaluated at.
    wandb_run: WandbRun | None, optional
        The run handle to the wandb. I None, no data will be logged to wandb. 
        Default None.
    logger : logging.Logger | None, optional
        The logger used for logging. If None, no data will be logged. Default
        None
    """
    
    # If both logging endpoints are None, dont do anything
    if wandb_run is None and logger is None:
        return 

    log_info = {
        "eval/scalarized_return": eval_info["avg_scalarized_returns"],
        "eval/scalarized_discounted_return": eval_info[
            "avg_scalarized_discounted_returns"
        ],
        "global_step": global_step
    }

    return_info = {}
    for i in range(eval_info["avg_returns"].shape[0]):
        return_info[f"eval/vec_{i}"] = eval_info["avg_returns"][i]
        return_info[f"eval/discounted_vec_{i}"] =\
            eval_info["avg_discounted_returns"][i]


    log_info.update(return_info)

    if wandb_run is not None:
        wandb_run.log(log_info)

    if logger is not None:
        logger.info(
            (f"{global_step} | "
             f"Scalar return: {eval_info['avg_scalarized_returns']:.3f} | "
             "Discounted scalar return: "
             f"{eval_info['avg_scalarized_discounted_returns']:.3f}")
        )


def log_mo_metrics(
        current_front: List[npt.NDArray],
        ref_point: List[float],
        reward_dim: int,
        global_step: int,
        wandb_run: WandbRun | None = None,
        logger: logging.Logger | None = None,
        n_sample_prefs: int = 50
):
    """Log common multi-objective metrics.

    Parameters
    ----------
    current_front : List[npt.NDArray]
        The current approximation of the pareto-front.
    ref_point : List[float]
        The reference point for calculating hypervolume.
    reward_dim : int
        The reward dimension of the environment.
    global_step : int
        The current global step.
    wandb_run: WandbRun | None
        The run handle for the wandb run. If None, no data will be logged to
        wandb. Default None.
    logger : logging.Logger | None, optional
        The logger to which the data is written to. If None, logging is done 
        only to the wandb board. Default None.
    n_sample_prefs : int, optional
        The amount of preferences to sample for calculating the EUM. Default 50.
    """
    # If both logging endpoints are None, dont do anything
    if wandb_run is None and logger is None:
        return 

    filtered_front = pareto.filter_pareto_dominated(current_front)
    hypervol = metrics.get_hypervolume(np.asarray(ref_point), filtered_front)
    sparsity = metrics.get_sparsity(filtered_front)
    eum = metrics.get_expected_utility(
        filtered_front,
        prefs_set=common.get_equally_spaced_weights(
            reward_dim, n_points=n_sample_prefs
        )
    )

    if wandb_run is not None:
        wandb_run.log(
            {
                "eval/hypervolume": hypervol,
                "eval/sparsity": sparsity,
                "eval/eum": eum,
                "global-step": global_step
            }, commit=False
        )

        # Add the pareto-front as a table
        front = wandb.Table(
            columns=[f"objective_{i+1}" for i in range(reward_dim)],
            data=[point.tolist() for point in filtered_front]
        )
        wandb_run.log({"eval/front", front})

    # If logger is provided, log also to it.
    if logger is not None:
        logger.info(
            (f"{global_step} | Hypervolume: {hypervol:.3f} | Sparsity "
             f"{sparsity:.3f} | EUM: {eum:.3f}")
        )


def log_episode_stats(
        info: Mapping[str, Any], *,
        prefs: npt.NDArray,
        global_step: int,
        wandb_run: WandbRun | None = None,
        logger: logging.Logger | None = None
):
    """Log statistics about the last episode to the wandb-board and possibly
    to the "tradiotiona logger".

    Parameters
    ----------
    info : Mapping[str, Any]
        The information from the last episode as returned by the env.
    prefs : npt.NDArray
        The preferences over the objectives.
    global_step : int
        The current global step.
    wandb_run: WandbRun | None, optional
        The run instance for wandb used for logging. If None, no stats will be 
        logged to wandb. Default None.
    logger : logging.Logger | None, optional
        The logger to use for logging. If None, no stats will be logged.
        Default None
    """

    if wandb_run is None and logger is None:
        return
    # Lets check if the environment was using RecordEpisodeStatistics wrapper,
    # and if not, then we just raise a warning and do nothing
    if "episode" not in info:
        warnings.warn(
            ("Environment is missing MORecordEpisdodeStatistics wrapper. "
             "Cannot log episode statistics")
        )
        return

    # Otherwise, log the same metrics as in the morl-baselines for easy
    # comparisons
    episode_info = info["episode"]
    episode_ts = episode_info["l"]
    episode_time = episode_info["t"]
    episode_return = episode_info["r"]
    disc_episode_return = episode_info["dr"]

    if isinstance(prefs, torch.Tensor):
        prefs = prefs.detach().cpu().numpy()
    scalar_return = np.dot(episode_return, prefs)
    disc_scalar_return = np.dot(disc_episode_return, prefs)
    
    if wandb_run is not None:
        wandb_run.log(
            {
                "charts/timesteps_per_episode": episode_ts,
                "charts/episode_time": episode_time,
                "charts/scalarized_episode_return": scalar_return,
                "charts/discounted_scalarized_episode_return": disc_scalar_return,
                "global_step": global_step
            },
            commit=False
        )
        for i in range(episode_return.shape[0]):
            wandb_run.log(
                {
                    f"metrics/episode_return_obj_{i}": episode_return[i],
                    f"metrics/disc_episode_return_obj_{i}": disc_episode_return[i]
                }
            )

    # If the logger is provided, the stats will be also logged to it.
    if logger is not None:
        logger.info(
            (f"{global_step} | Timesteps/ep {episode_ts} | Ep scalar return "
             f"{scalar_return:.3f} | "
             f"Ep disc scalar return {disc_scalar_return:.3f}")
        )


def get_logger(logger_name: str) -> logging.Logger:
    '''
    Get a new logger (ensures that the configurations are set correctly)

    Parameters
    ----------
    logger_name: str
        The name of the logger. The corresponding file should be found on the 
        configuration file

    Returns
    -------
    logging.Logger: The configured logger.

    '''
    _maybe_set_config(_CONFIG_PATH)
    return logging.getLogger(logger_name)


def info_if(logger: logging.Logger, cond: bool, msg: str):
    '''
    Logs info msg if given condition is True

    Parameters
    ----------
    logger: logging.Logger
        The logger used for the logging
    cond: bool
        The condition to check
    msg: str
        The message to log
    '''
    if cond:
        logger.info(msg, stacklevel=2)


def debug_if(logger: logging.Logger, cond: bool, msg: str):
    '''
    Logs debug msg if given condition is True

    Parameters
    ----------
    logger: logging.Logger
        The logger used for the logging
    cond: bool
        The condition to check
    msg: str
        The message to log
    '''
    if cond:
        logger.debug(msg, stacklevel=2)


def warn_if(logger: logging.Logger, cond: bool, msg: str):
    '''
    Logs warning msg if given condition is True

    Parameters
    ----------
    logger: logging.Logger
        The logger used for the logging
    cond: bool 
        The condition to check
    msg: str
        The message to log
    '''
    if cond:
        logger.warning(msg, stacklevel=2)


def critical_if(logger: logging.Logger, cond: bool, msg: str):
    '''
    Logs critical msg if given condition is True

    Parameters
    ----------
    logger: logging.Logger
        The logger used for the logging
    cond: bool
        The condition to check
    msg: str
        The message to log
    '''
    if cond:
        logger.critical(msg, stacklevel=2)


def _maybe_set_config(path: str | pathlib.Path):
    '''
    Set the configuration for the logging lib if not set already.

    Parameters
    ----------
    path: str | pathlib.Path:
        The path to the logging configuration
    '''
    global _CONFIG_SET
    if _CONFIG_SET:
        return

    path = pathlib.Path(path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError((f"{str(path)!r} does not point to a valid "
                                 "logging configuration file!"))
    yaml = YAML(typ="safe")

    # rb is required to ensure correct decoding
    with path.open("r") as ifstream:
        conf = yaml.load(ifstream)
    logging.config.dictConfig(conf)
    _CONFIG_SET = True
