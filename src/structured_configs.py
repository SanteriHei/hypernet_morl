""" Defines all the structured configs for the models """

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

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
    use_action: bool
        Determines if the critic is given the action as an input or not.
        Default True.
    use_prefs: bool
        Determines if the critic is given the preferences as an input or not.
        Default False.
    use_obs: bool
        Determines if the critic is given the observation as an input or not.
        Default False.
    """
    resblock_arch: Tuple[ResblockConfig, ...] = field(
        default_factory=lambda: (ResblockConfig, )
    )
    layer_dims: Tuple[int, ...] = MISSING
    head_init_stds: Tuple[float, ...] = MISSING
    reward_dim: int = MISSING
    obs_dim: int = MISSING
    action_dim: int = MISSING
    head_hidden_dim: int = 1024
    activation_fn: str = "relu"
    use_action: bool = True 
    use_prefs: bool = False
    use_obs: bool = False
    


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
    entity_name: str The name of the entity. Usually the username.
    experiment_name: str The name of the experiment/project.
    experiment_group: str The name of the group of the runs.
    run_name: str The name of current run.

    """
    entity_name: str = MISSING
    project_name: str = MISSING
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
        batch_size: int The batch size used for training the MSA-hyper.
            Default 1000
        buffer_capacity: int The maximum capacity of the buffer. Default 10_000
        angle_deg: float The angle for the data (in degrees). Default 22.5

        save_path: str The path where the results will be saved to.
        log_to_stdout: bool If set to True, data will be logged to stdout/stderr
            using standard Pythob logging facilities. Default True
        log_to_wandb: bool If set to True, data will be logged to wandb.
        Default True
        eval_every_nth: int The frequency at which the trained policy is
            evaluated at. Default 100.
        log_every_nth: int The frequency at which the metrics are logged.
            Default 100.
        n_eval_episodes: int The amount of episodes to evaluate the policy for
            during each evaluation pass in the training
        n_eval_prefs: int The amount of preferences that are used to evaluate
            the agent.
    """

    # Simulation env stuff
    n_timesteps: int = 1_000
    n_random_steps: int = 200
    env_id: str = MISSING

    # buffer parameters
    batch_size: int = 100
    buffer_capacity: int = 10_000
    angle_deg: float = 45
    
    # Logging parameters
    save_path: str = MISSING
    log_to_stdout: bool = True
    log_to_wandb: bool = True
    eval_every_nth: int = 100
    log_every_nth: int = 100
    n_eval_episodes: int = 5
    n_eval_prefs: int = 1000
    ref_point: List[float] = MISSING


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


    def summarize(self) -> Dict[str, Any]:
        """Summarize the currently used configurations as a single 
        dict that contains the most relevant options.

        Returns
        -------
        Dict[str, Any]
            The most relevant configuration options.
        """
        return {
            # Training 
            "env_id": self.training_cfg.env_id,
            "n_timesteps:": self.training_cfg.n_timesteps,
            "batch_size": self.training_cfg.batch_size,
            "buffer_capacity": self.training_cfg.buffer_capacity,

            # MSA-hyper 
            "n_networks": self.msa_hyper_cfg.n_networks,
            "alpha": self.msa_hyper_cfg.alpha,
            "tau": self.msa_hyper_cfg.tau,
            "gamma": self.msa_hyper_cfg.gamma,
            "critic_lr": self.msa_hyper_cfg.critic_lr,
            "critic_optim": self.msa_hyper_cfg.critic_optim,
            "policy_lr": self.msa_hyper_cfg.policy_lr,
            "policy_optim": self.msa_hyper_cfg.policy_optim,

            # Policy
            "policy/network_arch": self.policy_cfg.network_arch,
            "policy/activation_fn": self.policy_cfg.activation_fn,

            # Hyper-Q net
            "q-net/layer_dims": self.hypernet_cfg.layer_dims,
            "q-net/activation_fn": self.hypernet_cfg.activation_fn,
    
            # Common stuff
            "seed": self.seed
        }
