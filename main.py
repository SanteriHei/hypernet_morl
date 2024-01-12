import warnings

import hydra
import omegaconf
from hydra.core.config_store import ConfigStore

# Just to make sure that the custom envs are registered
import src.environments  # noqa
from src import training_loop
from src.models import msa_hyper
from src.utils import configs
from src.utils.common import set_global_rng_seed

# Ignore warning of accessing reward space directly from gymnasium.
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*WARN: env.reward_space.*"
)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: omegaconf.DictConfig):
    print(omegaconf.OmegaConf.to_yaml(cfg, resolve=True))
    cfg = configs.as_structured_config(cfg)
    configs.validate(cfg)

    set_global_rng_seed(cfg.seed)

    agent = msa_hyper.MSAHyper(
        cfg.msa_hyper_cfg,
        policy_cfg=cfg.policy_cfg,
        critic_cfg=cfg.critic_cfg
    )
    training_loop.train_agent(cfg, agent)


if __name__ == "__main__":
    cs = ConfigStore.instance()

    configs.register_resolvers()
    configs.register_configs(cs)
    main()
