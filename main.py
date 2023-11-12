
import hydra
import omegaconf

from hydra.core.config_store import ConfigStore

from src.models import msa_hyper
from src.training_loop import train
from src.utils import configs, log


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: omegaconf.DictConfig):
    cfg = configs.fill_missing_fields(cfg)
    cfg = configs.as_structured_config(cfg)
    logger = log.get_logger("main")

    agent = msa_hyper.MSAHyper(
            cfg.msa_hyper_cfg,
            policy_cfg=cfg.policy_cfg,
            hypernet_cfg=cfg.hypernet_cfg
    )
    train(cfg, agent, logger)




  # rng = np.random.default_rng()
  # logger = log.get_logger("main")
  # logger.info("Test log")
  # env = mo_gym.make("mo-hopper-v4", render_mode="human")
  # obs, info = env.reset()
  # for i in range(100):
  #     action = rng.uniform(size=(3, ))
  #     next_obs, rewar, terminated, truncated, info = env.step(action)
  #     env.render()
if __name__ == "__main__":
    cs = ConfigStore.instance()

    configs.register_resolvers()
    configs.register_configs(cs)
    main()
