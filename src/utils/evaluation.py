from typing import Dict

import mo_gymnasium as mo_gym
import numpy as np
import numpy.typing as npt
import torch


def eval_agent(
        agent, env_id: str,
        prefs: torch.Tensor, gamma: float = 1.0
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
    prefs = prefs.squeeze()
    env = mo_gym.make(env_id)
    obs, info = env.reset()
    th_obs = torch.tensor(obs).float().to(agent.device)
    
    done = False
        
    np_prefs = prefs.detach().cpu().numpy()
    returns = np.zeros_like(np_prefs)
    discounted_returns = np.zeros_like(np_prefs)
    disc_factor = 1.0
    while not done:
        action = agent.eval_action(th_obs, prefs)
        np_action = action.detach().cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(np_action)
        done = terminated or truncated
        th_obs = torch.tensor(obs).float().to(agent.device)

        returns += reward
        discounted_returns += disc_factor * reward
        disc_factor *= gamma
    scalarized_returns = np.dot(np_prefs, returns)
    scalarized_disc_returns = np.dot(np_prefs, returns)

    return {
        "scalarized_returns": scalarized_returns,
        "scalarized_discounted_returns": scalarized_disc_returns,
        "returns": returns,
        "discounted_returns": returns
    }


def eval_policy(
        agent, env_id: str,  *, prefs: torch.Tensor, n_episodes: int = 5
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
    gamma = agent.config.gamma
    evals = [
            eval_agent(agent, env_id, prefs, gamma=gamma)
            for _ in range(n_episodes)
    ]

    # Convert list of dicts into dict of lists
    evals = {key: [obj[key] for obj in evals] for key in evals[0].keys()}
    out = {}
    for key, data in evals.items():
        out[f"avg_{key}"] = np.mean(data, axis=0)
        out[f"std_{key}"] = np.std(data, axis=0)
    return out
