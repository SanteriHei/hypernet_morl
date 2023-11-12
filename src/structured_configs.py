""" Defines all the structured configs for the models """

from dataclasses import dataclass, field
from typing import List, Tuple

from omegaconf import MISSING


@dataclass
class MSAHyperConfig:
    """A configuration for the MSA-hyper algorithm

    Attributes
    ----------
    n_networks: int The amount of critic networks to use. Default 2.
    alpha: float The augmentation factor from the CAPQL. Should be in range
        [0, 1]. If 0, the optimization task becomes the traditional RL
        optimization task.
    tau: float The factor used for the update of the target networks.
        Should be in range (0.0, 1.0]. if tau == 1.0, we copy the 
        parameters directly.
    critic_optim: str The optimizer for the critic. Default adam.
    critic_lr: float The learning rate for the critic optimizer. Default 3e-4.
    policy_optim: str The optimizer for the policy. Default adam.
    critic_lr: float The learning rate for the policy optimizer. Default 3e-4

    """
    n_networks: int = 2
    alpha: float = MISSING
    tau: float = MISSING
    gamma: float = MISSING
    critic_optim: str = "adam"
    critic_lr: float = 3e-4
    policy_optim: str = "adam"
    policy_lr: float = 3e-4
    device: str = MISSING


@dataclass
class ResblockConfig:
    """
    A configuration for a Residual block

    Attributes
    ----------
    n_resblocks: int The amount of residual blocks to add in a single block. 
        Default 2
    input_dim: int The input dimension of the network.
    output_dim: int The ouput dimension of the network.
    activation_fn: Callable | str The activation function to use. Default "relu"
    network_arch: Tuple[int, ...] The network architecture. Default (128, 128)
    (i.e. two linear layers with 128 neurons)

    """
    n_resblocks: int = 2
    input_dim: int = MISSING
    output_dim: int = MISSING
    activation_fn: str = "relu"
    network_arch: Tuple[int, ...] = (128, 128)


@dataclass
class HypernetConfig:
    """
    A configuration for the Q-Hypernetwork

    Attributes
    ----------
    resblock_arch: Tuple[ResblockConfig, ...] The configuration for the residual
        blocks
    layer_dims: Tuple[int, ...] The dimensions for the dynamic network.
    reward_dim: int The reward dimension of the environment. Default 3.
    obs_dim: int The observation dimension of the  environment. Default 3 
    head_hidden_dim: int The hidden dimension of the "Heads". Default 1024.
    activation_fn: Callable | str: The activation function used in the dynamic
        network. Default "relu"
    """
    resblock_arch: Tuple[ResblockConfig, ...] = field(
        default_factory=lambda: (ResblockConfig, )
    )
    layer_dims: Tuple[int, ...] = MISSING
    reward_dim: int = MISSING
    obs_dim: int = MISSING
    action_dim: int = MISSING
    head_hidden_dim: int = 1024
    activation_fn: str = "relu"


@dataclass
class PolicyConfig:
    """Configuration for the policy network.

    Attributes
    ----------
    obs_dim: int The observation dimension of the used environment.
    reward_dim: int The reward dimension of the used environment.
    output_dim: int The output dimesion of the network
        (i.e. the action dimension)
    network_arch: Tuple[int, ...] The architecture of the network as a 
        list of neuron amounts for each layer.
    activation_fn: str The activation function to be used in the neural network.
        Default "relu"
    action_space_high: List[float] The upper bound of the used action space.
    action_space_low: List[float] The lower bound of the used action space.

    """
    obs_dim: int = MISSING
    reward_dim: int = MISSING
    output_dim: int = MISSING
    network_arch: Tuple[int, ...] = (256, 256)
    activation_fn: str = "relu"
    action_space_high: List[float] = MISSING
    action_space_low: List[float] = MISSING


@dataclass
class SessionConfig:
    """
    The configuration for logging the session.

    Attributes
    ----------
    experiment_name: str The name of the experiment/project.
    experiment_group: str The name of the group of the runs.
    run_name: str The name of current run.

    """
    experiment_name: str = MISSING
    experiment_group: str = MISSING
    run_name: str = MISSING


@dataclass
class TrainingConfig:
    """
    The configuration for the training algorithm

    Attributes:
        n_timesteps: int The amount of timesteps to train for. Default 1000
        n_random_steps: int The amount of random actions to make during the 
            start of the training before starting to follow the policy.
        env_id: str The environment at which the model is trained at.
        save_path: str The path where the results will be saved to.
        batch_size: int The batch size used for training the MSA-hyper.
            Default 1000
        buffer_capacity: int The maximum capacity of the buffer. Default 10_000
        angle: float The angle for the data
        eval_every_nth: int The frequency at which the trained policy is
            evaluated at. Default 100.
        log_every_nth: int The frequency at which the metrics are logged.
            Default 100.
        n_eval_episodes: int The amount of episodes to evaluate the policy for
            during each evaluation pass in the training
    """
    n_timesteps: int = 1_000
    n_random_steps: int = 200
    env_id: str = MISSING
    save_path: str = MISSING
    batch_size: int = 100
    buffer_capacity: int = 10_000
    angle: float = MISSING
    eval_every_nth: int = 100
    log_every_nth: int = 100
    n_eval_episodes: int = 5


@dataclass
class Config:
    """
    The main configuration for the run
    
    Attributes
    ----------
    training_cfg: TrainingConfig The used training configuration.
    session_cfg: SessionConfig The used session configuration.
    policy_cfg: PolicyConfig The used policy configuration.
    hypernet_cfg: HypernetConfig The used hypernet configuration.
    msa_hyper_cfg: MSAHyperConfig: The used MSA-hyper configuration.
    seed: int | None The seed used in the run. Default None.
    device: str The used device for the torch computations. Default cpu.
    """
    training_cfg: TrainingConfig = MISSING
    session_cfg: SessionConfig = MISSING
    policy_cfg: PolicyConfig = MISSING
    hypernet_cfg: HypernetConfig = MISSING
    msa_hyper_cfg: MSAHyperConfig = MISSING
    seed: int | None = None
    device: str = "cpu"
