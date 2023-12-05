import logging
import pathlib

import gymnasium as gym
import torch
import numpy as np

import wandb
from src.utils import common, envs, evaluation, log, pareto

from . import structured_configs


def train_agent(cfg: structured_configs.Config, agent):
    """Train an agent with the specified configuration.

    Parameters
    ----------
    agent : 
        The agent to train.
    cfg : structured_configs.Config
        The configuration for the training.
    """
    training_env = envs.create_env(cfg.training_cfg.env_id, cfg.device)

    if cfg.training_cfg.log_to_wandb:
        wandb_run = log.setup_wandb(cfg.session_cfg, cfg.summarize())
    else:
        wandb_run = None

    logger = log.get_logger(
        "train") if cfg.training_cfg.log_to_stdout else None

    # Construct the relevant buffers
    replay_buffer = common.ReplayBuffer(
        cfg.training_cfg.buffer_capacity,
        obs_dim=cfg.critic_cfg.obs_dim,
        reward_dim=cfg.critic_cfg.reward_dim,
        action_dim=cfg.policy_cfg.output_dim,
        seed=cfg.seed
    )

    weight_sampler = common.WeightSampler(
        reward_dim=cfg.critic_cfg.reward_dim,
        angle_rad=common.deg_to_rad(cfg.training_cfg.angle_deg),
        device=agent.device
    )

    trained_agent, pareto_front_table = _gym_training_loop(
        agent,
        training_cfg=cfg.training_cfg,
        replay_buffer=replay_buffer,
        weight_sampler=weight_sampler,
        env=training_env,
        logger=logger,
        wandb_run=wandb_run,
        seed=cfg.seed
    )

    if cfg.training_cfg.save_path is not None:
        if logger is not None:
            logger.info(
                f"Saving trained model to {cfg.training_cfg.save_path}"
            )
        trained_agent.save(cfg.training_cfg.save_path)
        common.dump_json(
            pathlib.Path(cfg.training_cfg.save_path) / "pareto-front.json",
            pareto_front_table
        )

    if wandb_run is not None:
        wandb_run.finish()


def _gym_training_loop(
        agent, *,
        training_cfg: structured_configs.TrainingConfig,
        replay_buffer: common.ReplayBuffer,
        weight_sampler: common.WeightSampler,
        env: gym.Env,
        logger: logging.Logger,
        wandb_run: log.WandbRun,
        seed: int | None = None
):
    """A common training loop for the mo-gymnasiunm environments.

    Parameters
    ----------
    cfg : sconfigs.Config
        The configuration for the training run.
    """
    # Initialize the run
    obs, info = env.reset(seed=seed)
    global_step = 0
    num_episodes = 0
    reward_dim = env.get_wrapper_attr("reward_space").shape[0]

    # Keep track of the pareto-front
    pareto_front_table = []

    # Create the preferences that are used later for evaluating the agent.
    eval_prefs = torch.tensor(
        common.get_equally_spaced_weights(
            reward_dim, training_cfg.n_eval_prefs
        ), device=agent.device, dtype=torch.float32
    )

    for ts in range(training_cfg.n_timesteps):
        global_step += 1
        prefs = weight_sampler.sample(n_samples=1)
        prefs = prefs.squeeze()

        if global_step < training_cfg.n_random_steps:
            action = torch.tensor(
                env.action_space.sample(), device=agent.device,
                dtype=torch.float32
            )
        else:
            with torch.no_grad():
                action = agent.take_action(obs, prefs)

        next_obs, rewards, terminated, truncated, info = env.step(action)
        replay_buffer.append(obs, action, rewards, prefs, next_obs, terminated)

        # Update the agent
        if global_step > training_cfg.n_random_steps:
            batches = [
                replay_buffer.sample(training_cfg.batch_size)
                for _ in range(training_cfg.n_gradient_steps)
            ]
            critic_loss, policy_loss = agent.update(batches)

            # Log metrics
            if global_step % training_cfg.log_every_nth == 0:
                log.log_losses(
                    {
                        "critic": critic_loss.item(),
                        "policy": policy_loss.item()
                    },
                    global_step=global_step,
                    wandb_run=wandb_run, logger=logger
                )

        # Evaluate the current policy after some timesteps.
        if global_step % training_cfg.eval_every_nth == 0:
            eval_info = evaluation.eval_policy(
                agent, training_cfg.env_id,
                prefs=prefs, n_episodes=training_cfg.n_eval_episodes
            )
            log.log_eval_info(
                eval_info, global_step=global_step,
                wandb_run=wandb_run, logger=logger
            )

        # Similarly, we evaluate the policy on the evaluation preferences
        if global_step % (training_cfg.eval_every_nth * 10) == 0:
            eval_data = [
                evaluation.eval_policy(
                    agent, training_cfg.env_id,
                    prefs=prefs, n_episodes=training_cfg.n_eval_episodes
                ) for prefs in eval_prefs
            ]
            
            current_front, current_stds = zip(
                *map(lambda elem: [
                    elem["avg_discounted_returns"], elem["std_discounted_returns"]
                ], eval_data)
            )
        
            # Filter the non-dominated elements
            non_dominated_inds = pareto.get_non_pareto_dominated_inds(
                    current_front, remove_duplicates=True
            )
            current_front = np.asarray(current_front)
            current_stds = np.asarray(current_stds)
            current_front = current_front[non_dominated_inds].tolist()
            current_stds = current_stds[non_dominated_inds].tolist()

            # Store the current pareto front
            for avg_disc_return, std_disc_return in zip(current_front, current_stds):
                assert (n_elems := len(avg_disc_return)) == 2, \
                    ("Expected two elements in eval data, got "
                     f"{n_elems} instead!")
                assert (n_elems := len(std_disc_return)) == 2, \
                    ("Expected two elements in eval data, got "
                     f"{n_elems} instead!")
                pareto_front_table.append({
                    "global_step": global_step,
                    "avg_disc_return_0": avg_disc_return[0],
                    "avg_disc_return_1": avg_disc_return[1],
                    "std_disc_return_0": std_disc_return[0],
                    "std_disc_return_1": std_disc_return[1]
                })

            log.log_mo_metrics(
                current_front=current_front, ref_point=training_cfg.ref_point,
                reward_dim=reward_dim, global_step=global_step,
                wandb_run=wandb_run, logger=logger
            )

        if terminated or truncated:
            num_episodes += 1
            log.log_episode_stats(
                info, prefs=prefs, global_step=global_step, logger=logger
            )
            obs, info = env.reset()
        else:
            obs = next_obs

    # Finally, store the pareto-front to the wandb
    if wandb_run is not None:
        # NOTE: bit sketchy to convert the dicts to a lists using values(),
        # eventhough the order of the values is quaranteed to be correct in
        # python 3.7+
        pareto_data = list(
            map(lambda row: list(row.values()), pareto_front_table)
        )
        eval_table = wandb.Table(
            columns=[
                "step", "avg_obj1", "avg_obj2", "std_obj1", "std_obj2"
            ],
            data=pareto_data
        )

        wandb_run.log({"eval/pareto-front": eval_table})
        wandb_run.plot_table(
            vega_spec_name="santeriheiskanen/test",
            data_table=eval_table,
            fields={
                "x": "avg_obj1", "y": "avg_obj2", "color": "step",
                "tooltip_1": "std_obj1", "tooltip_2": "std_obj2"
            }
        )
    return agent, pareto_front_table
