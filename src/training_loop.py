import logging
import pathlib
import time
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch

import wandb
from src.utils import common, envs, evaluation, log, pareto

from . import structured_configs
from .structured_configs import PrefSamplerFreq


def train_agent(cfg: structured_configs.Config, agent):
    """Train an agent with the specified configuration.

    Parameters
    ----------
    agent :
        The agent to train.
    cfg : structured_configs.Config
        The configuration for the training.
    """

    if cfg.training_cfg.num_envs == 1:
        training_env = envs.create_env(cfg.training_cfg.env_id, cfg.device)
    else:
        training_env = envs.create_vec_envs(
                cfg.training_cfg.env_id, device=cfg.device,
                n_envs = cfg.training_cfg.num_envs
        )

    if cfg.training_cfg.log_to_wandb:
        wandb_run = log.setup_wandb(cfg.session_cfg, cfg.summarize())

        if cfg.training_cfg.log_gradients:
            for critic in agent.critics:
                wandb_run.watch(critic, log="gradients", log_freq=1000)
            wandb_run.watch(cfg.policy, log="gradients", log_freq=1000)
    else:
        wandb_run = None

    logger = log.get_logger("train") if cfg.training_cfg.log_to_stdout else None

    # Construct the relevant buffers
    replay_buffer = common.ReplayBuffer(
        cfg.training_cfg.buffer_capacity,
        obs_dim=cfg.critic_cfg.obs_dim,
        reward_dim=cfg.critic_cfg.reward_dim,
        action_dim=cfg.policy_cfg.output_dim,
        device=agent.device,
        seed=cfg.seed,
    )

    # Only the "normal" sampler is using the angle, other samplers just ignore it
    preference_sampler = common.get_preference_sampler(
        cfg.training_cfg.sampler_type,
        cfg.critic_cfg.reward_dim,
        device=agent.device,
        seed=cfg.seed,
        angle_rad=common.deg_to_rad(cfg.training_cfg.angle_deg),
    )
    warmup_sampler = common.get_preference_sampler(
            "static",
            cfg.critic_cfg.reward_dim,
            device=agent.device,
            seed=cfg.seed,
            uneven_weighting=cfg.training_cfg.warmup_use_uneven_sampling,
            n_points=cfg.training_cfg.warmup_n_ref_points
    )

    # Ensure that the saving directory exists
    save_dir = pathlib.Path(cfg.training_cfg.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    (
        trained_agent,
        history_buffer,
        dynamic_net_params,
    ) = _gym_training_loop(
        agent,
        training_cfg=cfg.training_cfg,
        replay_buffer=replay_buffer,
        weight_sampler=preference_sampler,
        warmup_sampler=warmup_sampler,
        env=training_env,
        logger=logger,
        wandb_run=wandb_run,
        seed=cfg.seed,
    )

    if cfg.training_cfg.save_path is not None:

        # Record videos of the model performance
        evaluation.record_video(
                agent, cfg.training_cfg.env_id, cfg.training_cfg.save_path
        )

        save_dir_path = pathlib.Path(cfg.training_cfg.save_path)
    
        model_path = save_dir_path / "msa_hyper_final.tar"
        if logger is not None:
            logger.info(f"Saving trained model to {cfg.training_cfg.save_path}")

        trained_agent.save(model_path)
        pfront, non_dom_pfront = history_buffer.pareto_front_to_json()

        common.dump_json(save_dir_path / "pareto-front.json", pfront)
        common.dump_json(save_dir_path / "non_dom_pareto_front.json", non_dom_pfront)
        history_buffer.save_history(save_dir_path / "history.npz")

        for obj in dynamic_net_params:
            torch.save(
                obj["params"],
                save_dir_path / f"critic_net_params_{obj['global_step']}.tar",
            )

    if wandb_run is not None:
        wandb_run.finish()


def _gym_training_loop(
    agent,
    *,
    training_cfg: structured_configs.TrainingConfig,
    replay_buffer: common.ReplayBuffer,
    weight_sampler: common.PreferenceSampler,
    warmup_sampler: common.StaticSampler,
    env: gym.Env,
    logger: logging.Logger,
    wandb_run: log.WandbRun,
    seed: int | None = None,
):
    """A common training loop for the mo-gymnasiunm environments.

    Parameters
    ----------
    cfg : sconfigs.Config
        The configuration for the training run.
    """

    def _sample_preferences(step: int, n_samples: int) -> torch.Tensor:
        if (
                training_cfg.n_warmup_steps > 0 and 
                step < training_cfg.n_warmup_steps
        ):
            prefs = warmup_sampler.sample(n_samples=n_samples)
        else:
            prefs = weight_sampler.sample(n_samples=n_samples)
        return prefs.squeeze()



    # Initialize the run
    obs, info = env.reset(seed=seed)
    global_step = 0
    num_episodes = 0
    dims = envs.extract_env_dims(env)


    # Keep track of the history of the agent
    history_buffer = common.HistoryBuffer(
        total_timesteps=training_cfg.n_timesteps,
        eval_freq=training_cfg.eval_freq,
        n_eval_prefs=training_cfg.n_eval_prefs,
        obs_dim=dims["obs_dim"],
        reward_dim=dims["reward_dim"],
        action_dim=dims["action_dim"]
    )

    # Store the dynamic network weights
    dynamic_net_params = []
    static_pref = _get_static_preference(
            dims["reward_dim"], device=agent.device
    )
    static_obs = _get_static_obs(training_cfg.env_id, device=agent.device)

    # Create the preferences that are used later for evaluating the agent.
    eval_prefs = torch.tensor(
        common.get_equally_spaced_weights(
            dims["reward_dim"], training_cfg.n_eval_prefs
        ),
        device=agent.device,
        dtype=torch.float32,
    )
        
    start_time = time.perf_counter()
    for ts in range(0, training_cfg.n_timesteps, training_cfg.num_envs):
        global_step += training_cfg.num_envs
        
        # If this is the first step, or if the preferences are sampled 
        # at every timestep, select a preference
        if (
            global_step == training_cfg.num_envs
            or training_cfg.pref_sampling_freq == PrefSamplerFreq.timestep
        ):

            prefs = _sample_preferences(
                global_step, n_samples=training_cfg.num_envs
            )

        if global_step < training_cfg.n_random_steps:
            action = torch.tensor(
                env.action_space.sample(), device=agent.device, dtype=torch.float32
            )
        else:
            with torch.no_grad():
                action = agent.take_action(obs, prefs)

        next_obs, rewards, terminated, truncated, info = env.step(action)
        replay_buffer.append(obs, action, rewards, prefs, next_obs, terminated)

        # Store the whole action history
        history_buffer.append_step(
            obs, action, rewards, prefs, next_obs, terminated, num_episodes
        )

        #  === Update the agent ===
        if global_step > training_cfg.n_random_steps:
            batches = [
                replay_buffer.sample(training_cfg.batch_size)
                for _ in range(training_cfg.n_gradient_steps)
            ]
            critic_loss, policy_loss = agent.update(batches)

            # Log metrics
            if global_step % training_cfg.log_freq == 0:
                log.log_losses(
                    {"critic": critic_loss.item(), "policy": policy_loss.item()},
                    global_step=global_step,
                    wandb_run=wandb_run,
                    logger=logger,
                )

        #  === Store the parameters of the dynamic network ===
        if (
            training_cfg.save_dynamic_weights
            and global_step % training_cfg.dynamic_net_save_freq == 0
        ):
            params = agent.get_critic_dynamic_params(static_obs, static_pref)
            dynamic_net_params.append({"global_step": global_step, "params": params})

        #  === Evaluate the current policy ===
        if global_step % training_cfg.eval_freq == 0:
            # If using multiple environments, there are multiple preferences,
            # so just choose one of them.
            tmp_prefs = prefs[0, :] if prefs.ndim == 2 else prefs
            eval_info = evaluation.eval_policy(
                agent,
                training_cfg.env_id,
                prefs=tmp_prefs,
                n_episodes=training_cfg.n_eval_episodes,
            )
            log.log_eval_info(
                eval_info, global_step=global_step, wandb_run=wandb_run, logger=logger
            )

        # === evaluate the policy on the evaluation preferences ===
        if global_step % (training_cfg.eval_freq * 5) == 0:

            # Steps per second
            stop_time = time.perf_counter()
            sps = global_step / (stop_time - start_time)


            eval_data = [
                evaluation.eval_policy(
                    agent,
                    training_cfg.env_id,
                    prefs=prefs,
                    n_episodes=training_cfg.n_eval_episodes,
                )
                for prefs in eval_prefs
            ]

            avg_returns, sd_returns = zip(
                *map(
                    lambda elem: [
                        elem["avg_discounted_returns"],
                        elem["std_discounted_returns"],
                    ],
                    eval_data,
                )
            )
            history_buffer.append_avg_returns(avg_returns, sd_returns, global_step)

            log.log_mo_metrics(
                current_front=avg_returns,
                ref_point=training_cfg.ref_point,
                reward_dim=dims["reward_dim"],
                global_step=global_step,
                total_timesteps=training_cfg.n_timesteps,
                sps=sps,
                ref_set=np.asarray(training_cfg.ref_set),
                wandb_run=wandb_run,
                logger=logger,
            )

        # === Store a model checkpoint ===
        if (
            training_cfg.save_path is not None
            and global_step % training_cfg.model_save_freq == 0
            and global_step > 0
        ):

            if logger is not None:
                logger.info(f"Saving model at {global_step}")
            path = pathlib.Path(training_cfg.save_path)
            agent.save(path / f"msa_hyper_{global_step}.tar")

        if isinstance(terminated, np.ndarray):
            done = terminated.any() or truncated.any()
            new_episodes = (terminated | truncated).sum()
        else:
            done = terminated or truncated
            new_episodes = 1
        
        if done:
            num_episodes += new_episodes
            log.log_episode_stats(
                info, prefs=prefs, global_step=global_step, logger=logger
            )

            if training_cfg.pref_sampling_freq == PrefSamplerFreq.episode:
                prefs = _sample_preferences(
                        global_step, n_samples=training_cfg.num_episodes
                )
            obs, info = env.reset()
        else:
            obs = next_obs
    
    pfront, non_dom_pfront = history_buffer.pareto_front_to_json()
    # Finally, store the pareto-front to the wandb
    if wandb_run is not None:
        _log_pareto_front(pfront, non_dom_pfront, wandb_run)

    return (
        agent,
        history_buffer,
        dynamic_net_params,
    )


def _get_static_preference(pref_dim: int, device: torch.device | str) -> torch.Tensor:
    """Get a static preference from the preference space
    (NOTE: will ALWAYS be the same preference)

    Parameters
    ----------
    pref_dim : int
        The dimensionality of the preference space.
    device: torch.device | str
        The device where the results will be stored to.
    Returns
    -------
    torch.Tensor
        The static preference.
    """
    return torch.full((pref_dim,), 1, dtype=torch.float32, device=device) / pref_dim


def _get_static_obs(env_id: str, device: torch.device | str) -> torch.Tensor:
    """Get a static observation from the observation space of the given
    environment. NOTE: The observation will be ALWAYS the same!

    Parameters
    ----------
    env_id : str
        The environment id.
    device: torch.device | str
        The device where the results will be stored to.

    Returns
    -------
    npt.NDArray
        The static obseration.
    """
    tmp_env = gym.make(env_id)
    obs, info = tmp_env.reset(seed=4)
    tmp_env.close()
    return torch.from_numpy(obs).float().to(device)


def _pfront_to_json(
    pfront_table: List[Dict[str, List[float] | int]],
) -> Tuple[List[Dict[str, int | float]], List[Dict[str, int | float]]]:
    """Convert the stored pareto-front table into json.

    Parameters
    ----------
    pfront_table : List[Dict[str, List[float] | int]]
        The pareto-data that should be converted.

    Returns
    -------
    Tuple[List[Dict[str, int | float]], List[Dict[str, int | float]]]
        The list containing all the points of pareto-front, and list that
        contains only the non-dominated points for each iteration.
    """
    pfront = []
    filtered_pfront = []
    for obj in pfront_table:
        front = np.asarray(obj["front"])
        sds = np.asarray(obj["sd"])
        step = obj["global_step"]

        non_dominated_inds = pareto.get_non_pareto_dominated_inds(
            front, remove_duplicates=True
        )

        filtered_points = front[non_dominated_inds]
        filtered_sds = sds[non_dominated_inds]

        for point, sd in zip(front, sds):
            pfront.append(
                {
                    "global_step": step,
                    "avg_disc_return_0": float(point[0]),
                    "avg_disc_return_1": float(point[1]),
                    "std_disc_return_0": float(sd[0]),
                    "std_disc_return_1": float(sd[1]),
                }
            )

        for point, sd in zip(filtered_points, filtered_sds):
            filtered_pfront.append(
                {
                    "global_step": step,
                    "avg_disc_return_0": float(point[0]),
                    "avg_disc_return_1": float(point[1]),
                    "std_disc_return_0": float(sd[0]),
                    "std_disc_return_1": float(sd[1]),
                }
            )
    return pfront, filtered_pfront


def _log_pareto_front(
    pfront_table: List[Dict[str, float | int]],
    non_dom_pfront_table: List[Dict[str, float | int]],
    wandb_run: log.WandbRun,
    step_freq: int = int(5e4)
):
    """Logs the pareto-front to the W&B dashboard.

    Parameters
    ----------
    pfront_table : List[Dict[str, float | int]]
        The pareto-front data that contains all the points (incl. the dominated points)
    non_dom_pfront_table : List[Dict[str, float | int]]
        The pareto-front data that contains only the non-dominated points.
    wandb_run : log.WandbRun
        The wandb run used for logging.
    """

    def _row_to_list(row):
        return [
                row["global_step"],
                row["avg_disc_return_0"], 
                row["avg_disc_return_1"],
                row["std_disc_return_0"],
                row["std_disc_return_1"]
        ]
        
    # Filter certain amount of rows from the dataset to make them fit to the 
    # wandb-tables
    pfront_table = filter(
            lambda x: x["global_step"] % step_freq == 0, pfront_table
    )

    pareto_data = list(map(_row_to_list, pfront_table))
    non_dom_pareto_data = list(map(_row_to_list, non_dom_pfront_table))

    pareto_table = wandb.Table(
        columns=["step", "avg_obj1", "avg_obj2", "std_obj1", "std_obj2"],
        data=pareto_data,
    )
    non_dom_pareto_table = wandb.Table(
        columns=["step", "avg_obj1", "avg_obj2", "std_obj1", "std_obj2"],
        data=non_dom_pareto_data,
    )

    wandb_run.log({"eval/pareto-front": pareto_table})
    wandb_run.log({"eval/non-dom-pareto-front": non_dom_pareto_table})

    wandb_run.plot_table(
        vega_spec_name="santeriheiskanen/pareto-front/v3",
        data_table=pareto_table,
        fields={
            "x": "avg_obj1",
            "y": "avg_obj2",
        },
        string_fields={
            "title": "Pareto-front",
        },
    )
