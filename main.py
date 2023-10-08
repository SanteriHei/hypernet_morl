from dataclasses import dataclass
import mo_gymnasium as mo_gym
import numpy as np
import mlflow

from src.utils import log


@dataclass 
class SessionConfig:
    """
    Defines parameters for the MLflow session

    Attributes
    ----------
    experiment_name: str
        The name of the experiment
    run_name: str
        The name of the run
    uri: str
        The URI used for the parameter tracking
    env: str
        The name of the used environment
    """
    experiment_name: str
    run_name: str
    uri: str
    env: str
    





def main():
    '''
    '''
    mlflow.log_param({"test": 10})
    rng = np.random.default_rng()
    logger = log.get_logger("main")
    logger.info("Test log")

    env = mo_gym.make("mo-hopper-v4", render_mode="human")
    obs, info = env.reset()

    for i in range(100):
        action = rng.uniform(size=(3, ))
        next_obs, rewar, terminated, truncated, info = env.step(action)
        env.render()


if __name__ == "__main__":
    main()
