import os
import pathlib
from typing import Dict

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import numpy.typing as npt
import pymoo.util.ref_dirs
import torch


def record_video(
        agent, env_id: str, save_dir: str | pathlib.Path, n_prefs: int = 5
):
    """Record video of the given agent with different preferences.

    Parameters
    ----------
    agent : Any
        The agent to train.
    env_id : str
        The id of the used environment.
    save_dir : str | pathlib.Path
        Path to the directory where the videos will be stored to.
    n_prefs : int, optional
        The amount of preferences to test the agent on. Default 5.
    """

    def _record_video_impl():
        dummy_env = mo_gym.make(env_id, render_mode="rgb_array")
        eval_prefs = pymoo.util.ref_dirs.get_reference_directions(
            "energy", dummy_env.get_wrapper_attr("reward_space").shape[0], 
            n_prefs, seed=42
        )
        dummy_env.close()

        for i, pref in enumerate(eval_prefs):
            prefix = f"pref_{i}_"
            env = gym.wrappers.RecordVideo(
                mo_gym.make(env_id, render_mode="rgb_array"),
                name_prefix=prefix,
                video_folder=save_dir,
            )
            th_pref = torch.atleast_2d(
                    torch.from_numpy(pref).float().to(agent.device)
            )
            _ = eval_agent(agent, env, th_pref)
            env.close()
    
    # Check if the program is run on headless mode or not
    display = os.environ.get("DISPLAY", None)
    
    # If DISPLAY is set, the environment should not be headless,
    # and video can be recorded as usual
    if display is not None:
        _record_video_impl()
    
    # Otherwise, try to run using a virtual X Frame-buffer
    else:
        # This can fail in many ways, os use the catch all exception handler
        try:
            from xvfbwrapper import Xvfb
            with Xvfb() as xvfb:
                _record_video_impl()
        except Exception as e:
            print(f"Error while trying to record a video in headless: ({e}).")


def eval_agent(
    agent, env: gym.Env, prefs: torch.Tensor, gamma: float = 1.0
) -> Dict[str, float | npt.NDArray]:
    """Evaluate the given agent for a single episode in the given environment.

    Parameters
    ----------
    agent :
        The agent to evaluate.
    env_id : str
        The id of the evaluation environment.
    prefs : torch.Tensor
        The preferences over the objectives.
    gamma : float
        The discount factor. Default 1.0

    Returns
    -------
    Dict[str, float | npt.NDArray]
        Returns the scalarized returns, scalarized discounted returns,
        returns and the discounted returns.
    """
    # Reset env
    obs, info = env.reset()

    # Ensure that inputs for agent have batch dimension
    th_obs = torch.atleast_2d(torch.tensor(obs).float().to(agent.device))

    done = False

    np_prefs = prefs.detach().cpu().numpy()
    returns = np.zeros_like(np_prefs.squeeze())
    discounted_returns = np.zeros_like(np_prefs.squeeze())
    disc_factor = 1.0
    while not done:
        action = agent.eval_action(th_obs, prefs)
        np_action = action.detach().cpu().numpy().squeeze()
        obs, reward, terminated, truncated, info = env.step(np_action)
        done = terminated or truncated
        th_obs = torch.atleast_2d(torch.tensor(obs).float().to(agent.device))

        returns += reward
        discounted_returns += disc_factor * reward
        disc_factor *= gamma

    scalarized_returns = np.dot(np_prefs, returns)
    scalarized_disc_returns = np.dot(np_prefs, discounted_returns)

    return {
        "scalarized_returns": scalarized_returns,
        "scalarized_discounted_returns": scalarized_disc_returns,
        "returns": returns,
        "discounted_returns": discounted_returns,
    }


def eval_policy(
    agent, env_id: str, *, prefs: torch.Tensor, n_episodes: int = 5
) -> Dict[str, float | npt.NDArray]:
    """Evaluate the given policy for 'n_reps' times, and calculate the
    average statistics from the evaluation.

    Parameters
    ----------
    agent :
        The agent to evaluate.
    env_id : str
        The id of the evaluation environment.
    prefs : torch.Tensor
        The preferences over the objectives.
    n_episodes : int
        The amount of times the agent will be evaluated. Default 5.

    Returns
    -------
    Dict[str, float | npt.NDArray]
        The average and standard deviation of scalarized returns,
        the average and standard deviation of scalarized discounted returns,
        the average and standard deviation of returns (as vector),
        and the average and standard deviation of discounted returns (as vector)
    """
    agent.set_mode(train_mode=False)
    prefs = torch.atleast_2d(prefs)
    gamma = agent.config.gamma

    eval_env = mo_gym.make(env_id)
    evals = [eval_agent(agent, eval_env, prefs, gamma=gamma) for _ in range(n_episodes)]
    eval_env.close()
    agent.set_mode(train_mode=True)

    # Convert list of dicts into dict of lists
    evals = {key: [obj[key] for obj in evals] for key in evals[0].keys()}
    out = {}
    for key, data in evals.items():
        out[f"avg_{key}"] = np.mean(data, axis=0)
        out[f"std_{key}"] = np.std(data, axis=0)
    return out
