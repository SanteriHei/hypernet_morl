import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp

from .. import structured_configs as sconfigs
from ..models.capql import Capql
from ..utils import common, envs, evaluation


# TODO: make the gym training loop to work with the warmup
def _warmup_training_loop(
    worker_id: int,
    agent,
    *,
    buffer_capacity: int,
    batch_size: int,
    n_gradient_steps: int,
    n_timesteps: int,
    n_random_steps: int,
    env: gym.Env,
    pref: torch.Tensor,
    log_freq: int = 500,
    seed: int | None = None,
):
    """
    Train an given agent in a simplified training loop without much instrumentation

    Parameters
    ----------
    worker_id: int
        The process worker id.
    agent
        The agent to train.
    buffer_capacity: int
        The replay buffer capacity
    batch_size: int
        The batch size used for sampling the buffer
    n_gradient_steps: int
        The amount of gradient steps to use when updating the agent.
    n_timesteps: int
        The amount of timesteps to train the agent for.
    n_random_steps: int
        The amount of random steps to take during the beginning of the training.
    env: gym.Env
        The environment used for the training
    pref: torch.Tensor
        The preference used for training the policy.
    log_freq: int
        The frequency at which the loss information is logged.
    seed: int | None
        The seed used for reseting the environment and replay bufffer.
    """
    obs, info = env.reset(seed=seed)
    warmup_step = 0
    num_episodes = 0

    dims = envs.extract_env_dims(env)
    num_envs = dims["num_envs"]

    replay_buffer = common.ReplayBuffer(
        buffer_capacity,
        obs_dim=dims["obs_dim"],
        reward_dim=dims["reward_dim"],
        action_dim=dims["action_dim"],
        device=agent.device,
        seed=seed,
    )

    if num_envs > 1:
        pref = pref.expand(num_envs, -1)

    for ts in range(0, n_timesteps, num_envs):
        warmup_step += num_envs

        if warmup_step < n_random_steps:
            action = torch.from_numpy(env.action_space.sample()).to(
                device=agent.device, dtype=torch.float32
            )
        else:
            with torch.no_grad():
                action = agent.take_action(obs, pref)

        next_obs, rewards, terminated, truncated, info = env.step(action)
        replay_buffer.append(obs, action, rewards, pref, next_obs, terminated)

        if warmup_step > n_random_steps:
            batches = [
                replay_buffer.sample(batch_size) for _ in range(n_gradient_steps)
            ]

            critic_loss, policy_loss, _, _ = agent.update(
                batches, return_individual_losses=False
            )

            if warmup_step % log_freq == 0:
                print(
                    f"Worker {worker_id} -> warmup_step: {warmup_step} | "
                    f"Policy loss {policy_loss.item():.2f} | Critic loss "
                    f"{critic_loss.item():.2f}"
                )

        if isinstance(terminated, np.ndarray):
            done = terminated.any() or truncated.any()
            num_episodes += (terminated | truncated).sum()
            obs = next_obs
        else:
            done = terminated or truncated
            if done:
                num_episodes += 1
                obs, info = env.reset()
            else:
                obs = next_obs
    return agent


def _warmup_worker(
    worker_id: int,
    done_event,
    queue,
    cfg: sconfigs.MSAHyperConfig,
    policy_cfg: sconfigs.PolicyConfig,
    critic_cfg: sconfigs.HyperCriticConfig,
    env_id: str,
    pref: torch.Tensor,
    n_timesteps: int,
    utopia_point: float,
    seed: int | None = None,
):
    agent = Capql(cfg, policy_cfg=policy_cfg, critic_cfg=critic_cfg)
    env = envs.create_env(env_id, device=cfg.device, gamma=cfg.gamma)

    agent = _warmup_training_loop(
        worker_id,
        agent,
        buffer_capacity=int(0.8 * n_timesteps),
        batch_size=128,
        n_gradient_steps=1,
        n_timesteps=n_timesteps,
        n_random_steps=3000,
        env=env,
        pref=pref,
        seed=seed,
    )

    eval_data = evaluation.eval_policy(agent, env_id, prefs=pref)

    relative_dist = (
            (utopia_point - eval_data["avg_scalarized_returns"]) / utopia_point
    )

    queue.put(dict(worker_id=worker_id, pref=pref, dist=relative_dist))
    done_event.wait()


def create_warmup_distribution(
    cfg: sconfigs.MSAHyperConfig,
    policy_cfg: sconfigs.PolicyConfig,
    critic_cfg: sconfigs.CriticConfig,
    *, 
    env_id: str,
    n_timesteps: int,
    prefs: torch.Tensor,
    utopia_points: torch.Tensor,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Selects the parameters for the preference distribution used during the 
    sampling.

    Parameters
    ----------
    cfg: sconfigs.MSAHyperConfig
        The configuration for the agent.
    policy_cfg: sconfigs.PolicyConfig
        The configuration for the policy network
    critic_cfg: sconfigs.CriticConfig
        The configuration for the critic network
    env_id: str
        The id for the used environment
    n_timesteps: int
        The amount of timesteps to train the agent for.
    prefs: torch.Tensor
        The preferences that are used for training the different policies.
    utopia_points: torch.Tensor
        The utopia points for the different preferences.
    seed: int | None, optional
        The seed used for setting the randomness. Default None
    
    Returns
    -------
    torch.Tensor
        The "alpha" / "concentration" parameters for a Dirichlet distribution.
    """
    common.set_thread_count("cpu", 1)
    n_threads = utopia_points.shape[0]

    if isinstance(prefs, np.ndarray):
        prefs = torch.from_numpy(prefs).to(
                device=cfg.device, dtype=torch.float32
        )

    if isinstance(utopia_points, np.ndarray):
        utopia_points = torch.from_numpy(utopia_points).to(
            device=cfg.device, dtype=torch.float32
        )

    ctx = mp.get_context("spawn")
    done_event = ctx.Event()
    procs = []
    queue = ctx.Queue()

    for i in range(n_threads):
        print(f"Starting process {i}")
        p = ctx.Process(
            target=_warmup_worker,
            args=(i, done_event, queue),
            kwargs={
                "cfg": cfg,
                "policy_cfg": policy_cfg,
                "critic_cfg": critic_cfg,
                "env_id": env_id,
                "pref": prefs[i, :],
                "n_timesteps": n_timesteps,
                "utopia_point": utopia_points[i, :],
                "seed": seed,
            },
        )
        p.start()
        procs.append(p)
    # Extract the results
    ready_procs = 0
    results = {}
    while ready_procs < len(procs):
        warmup_res = queue.get()
        results[warmup_res["worker_id"]] = warmup_res
        ready_procs += 1

    done_event.set()

    # Wait for the processses to finish
    should_terminate = False
    for i, proc in enumerate(procs):
        proc.join(timeout=int(60 * 10))
        print(f"Process {i} finished with exit code {proc.exitcode}")
        should_terminate = proc.exitcode != 0 or should_terminate

    if should_terminate:
        raise ValueError("The warmup failed!")

    # TODO: Convert the results into a dirichlet distribution parameters

    # Start from a uniform distribution over the values
    alphas = torch.ones(prefs.shape[1])

    # Add the relative distances to the alpha values
    # -> The objectives that where furthest away will get a larger
    # alpha value 
    # -> More samples selected from there.
    for i in range(n_threads):
        alphas[i] += torch.abs(results[i]["dist"])
    return alphas
