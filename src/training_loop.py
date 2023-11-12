import logging
import pprint

import torch

from src.utils import common, envs, evaluation, log

from . import structured_configs


def train(cfg: structured_configs.Config, agent, logger: logging.Logger):
    """A common training loop for the mo-gymnasiunm environments.

    Parameters
    ----------
    cfg : sconfigs.Config
        The configuration for the training run.
    """
    env = envs.create_env(cfg.training_cfg.env_id, cfg.device)

    replay_buffer = common.ReplayBuffer(
        cfg.training_cfg.buffer_capacity,
        obs_dim=cfg.hypernet_cfg.obs_dim,
        reward_dim=cfg.hypernet_cfg.reward_dim,
        action_dim=cfg.policy_cfg.output_dim,
        seed=cfg.seed
    )
    weight_sampler = common.WeightSampler(
        reward_dim=cfg.hypernet_cfg.reward_dim,
        angle=cfg.training_cfg.angle
    )

    obs, info = env.reset(seed=cfg.seed)
    global_step = 0
    num_episodes = 0

    for ts in range(cfg.training_cfg.n_timesteps):
        global_step += 1
        prefs = weight_sampler.sample(n_samples=1)
        prefs = prefs.squeeze()

        if global_step < cfg.training_cfg.n_random_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = agent.take_action(obs, prefs)

        next_obs, rewards, terminated, truncated, info = env.step(action)
        replay_buffer.append(obs, action, rewards, prefs, next_obs, terminated)

        if global_step > cfg.training_cfg.n_random_steps:
            batch = replay_buffer.sample(cfg.training_cfg.batch_size)
            critic_loss, policy_loss = agent.update(batch)
            log.debug_if(
                logger, global_step % cfg.training_cfg.log_every_nth == 0,
                (f"step {global_step} | critic loss {critic_loss:.3f} | "
                 f"Policy loss {policy_loss:.3f}")
            )

        if global_step % cfg.training_cfg.eval_every_nth == 0:
            eval_info = evaluation.eval_policy(
                agent, cfg.training_cfg.env_id,
                prefs=prefs, n_episodes=cfg.training_cfg.n_eval_episodes
            )
            logger.debug(
                (f"step {global_step} | Eval info "
                 f"{pprint.pformat(eval_info, compact=True, indent=4)}")
            )

        if terminated or truncated:
            obs, info = env.reset()
            num_episodes += 1
        else:
            obs = next_obs
