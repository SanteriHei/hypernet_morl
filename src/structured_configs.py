""" Defines all the structured configs for the models """

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple
from enum import Enum

from omegaconf import MISSING


PrefSamplerFreq = Enum("PrefSamplerFreq", ["timestep", "episode"])


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
    activation_fn: Callable | str The activation function to use. Default "relu"
    layer_features: Tuple[int, ...] The desired number of features per layer.
        The last value indicates the ouput size of the network.
        Default (128, 128) (i.e. two linear layers with 128 neurons)
    dropout_rates: Tuple[float | None, ...] Controls the used dropout rate.
        If the dropout_rate is None, no dropout is applied after that layer.

    """

    n_resblocks: int = 2
    input_dim: int = MISSING
    activation_fn: str = "relu"
    layer_features: Tuple[int, ...] = (128, 128)
    dropout_rates: Tuple[float | None, ...] = (None, None)


@dataclass
class HyperNetConfig:
    """
    Configuration for a Hyper network. Contains the relevant information
    to create the embedding and the heads for the network.

    Atrributes
    ----------
    embedding_layers: Tuple[ResblockConfig, ...] The configuration for the
        embedding network.
    head_hidden_dim: int The hidden dimension for the head network. Should match
        the ouput of the embedding layer.
    head_init_method {"uniform", "normal"} The initialization method used for
        the head layers. Default uniform.
    head_init_stds: Tuple[float, ...] | float. The standard deviations used for
        initializing the network.
    use_weight_norm: bool Controls if the weights are renormalized. Default False
    """

    embedding_layers: Tuple[ResblockConfig, ...] = field(
        default_factory=lambda: (ResblockConfig,)
    )
    head_hidden_dim: int = MISSING
    head_init_method: str = "uniform"
    head_init_stds: Tuple[float, ...] = MISSING


@dataclass
class CriticConfig:
    """s
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
    use_obs: boolg
        Determines if the critic is given the observation as an input or not.
        Default False.
    """

    layer_dims: Tuple[int, ...] = MISSING
    reward_dim: int = MISSING
    obs_dim: int = MISSING
    action_dim: int = MISSING
    activation_fn: str = "relu"
    use_action: bool = True
    use_prefs: bool = False
    use_obs: bool = False

    hypernet_cfg: HyperNetConfig = MISSING


@dataclass
class PolicyConfig:
    """Configuration for the policy network.

    Attributes
    ----------
    policy_type: {"gaussian", "hyper-gaussian"}, optional. The type of the used
        policy. Can be either Gaussian or Hyper-Gaussian.
    obs_dim: int The obse00rvation dimension of the used environment.
    reward_dim: int The reward dimension of the used environment.
    output_dim: int The output dimesion of the network
        (i.e. the action dimension)
    layer_features: Tuple[int, ...] The architecture of the network as a
        list of the desired number of neurons in each layer.
    activation_fn: str The activation function to be used in the neural network.
        Default "relu"
    action_space_high: List[float] The upper bound of the used action space.
    action_space_low: List[float] The lower bound of the used action space.
    ResblockConfig: Tuple[ResblockConfig, ...] The configuration for the
        residual network. Used only if 'policy_type' == 'hyper-gaussian'
    """

    policy_type: str = "gaussian"
    obs_dim: int = MISSING
    reward_dim: int = MISSING
    output_dim: int = MISSING
    layer_features: Tuple[int, ...] = (256, 256)
    activation_fn: str = "relu"
    action_space_high: List[float] = MISSING
    action_space_low: List[float] = MISSING

    target_use_obs: bool = True
    target_use_prefs: bool = True

    hypernet_use_obs: bool = True
    hypernet_use_prefs: bool = True

    hypernet_cfg: HyperNetConfig = MISSING


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
        sampler_type: str The used type of sampler. Can be either "normal",
            "uniform" or "static". Default "normal"
        angle_deg: float The angle for the data (in degrees). Default 22.5
        n_gradient_steps: int The amount of update steps to take during
            each time the model is updated.
        save_path: str The path where the results will be saved to.
        log_to_stdout: bool If set to True, data will be logged to stdout/stderr
            using standard Python logging facilities. Default True
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

    # Sampler
    sampler_type: str = "normal"
    pref_sampling_freq: PrefSamplerFreq = "timestep"
    angle_deg: float = 45

    # updates
    n_gradient_steps: int = 1

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
    critic_cfg: CriticConfig = MISSING
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
            "obs_dim": self.policy_cfg.obs_dim,
            "reward_dim": self.policy_cfg.reward_dim,
            "action_dim": self.critic_cfg.action_dim,
            "n_timesteps:": self.training_cfg.n_timesteps,
            "sampler_type": self.training_cfg.sampler_type,
            "pref_sampling_freq": self.training_cfg.pref_sampling_freq.name,
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
            "policy/type": self.policy_cfg.policy_type,
            "policy/layer_features": self.policy_cfg.layer_features,
            "policy/activation_fn": self.policy_cfg.activation_fn,
            "policy/hypernet_cfg": (
                asdict(self.policy_cfg.hypernet_cfg)
                if self.policy_cfg.policy_type == "hyper-gaussian"
                else None
            ),
            # Critic
            "critic/layer_dims": self.critic_cfg.layer_dims,
            "critic/activation_fn": self.critic_cfg.activation_fn,
            "critic/use_action_input": self.critic_cfg.use_action,
            "critic/use_obs_input": self.critic_cfg.use_obs,
            "critic/use_prefs_input": self.critic_cfg.use_prefs,
            "critic/hypernet_cfg": asdict(self.critic_cfg.hypernet_cfg),
            # Common stuff
            "seed": self.seed,
        }
