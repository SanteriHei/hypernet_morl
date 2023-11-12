""" Define the MSA-hyper algorithm"""

import dataclasses
import itertools
import pathlib
from typing import Tuple

import torch
import torch.nn.functional as F

from .. import structured_configs as sconfigs
from ..utils import common, configs, nets
from .hypernet import HyperNet
from .policy import GaussianPolicy


class MSAHyper:
    def __init__(
            self,
            cfg: sconfigs.MSAHyperConfig, *,
            policy_cfg: sconfigs.PolicyConfig,
            hypernet_cfg: sconfigs.HypernetConfig,
    ):
        """Create a MSA-hyper network, that uses a modified CAPQL, with
        hypernetworks for (hopefully) generalizing the learned information
        from policies.

        Parameters
        ----------
        config : structured_configs.MSAHyperConfig
            The configuration for the MSAHyper.
        """
        self._cfg = configs.as_structured_config(cfg)
        self._device = (
            torch.device(self._cfg.device)
            if isinstance(self._cfg.device, str) else self._cfg.device
        )

        self._policy = GaussianPolicy(policy_cfg).to(self._device)

        self._critics = [
            HyperNet(hypernet_cfg).to(self._device)
            for _ in range(self._cfg.n_networks)
        ]

        self._critic_targets = [
            HyperNet(hypernet_cfg).to(self._device)
            for _ in range(self._cfg.n_networks)
        ]

        # Note: The target parameters are not updated through direct gradient
        # descent, so we disable the gradient tracking for them.
        for critic, critic_target in zip(self._critics, self._critic_targets):
            # Copy the initial parameters of critics to targets
            critic_target.load_state_dict(critic.state_dict())
            for param in critic_target.parameters():
                param.requires_grad_(False)

        self._critic_optim = nets.get_optim_by_name(self._cfg.critic_optim)(
            itertools.chain(*[critic.parameters() for critic in self._critics]),
            lr=self._cfg.critic_lr
        )

        self._policy_optim = nets.get_optim_by_name(self._cfg.policy_optim)(
            self._policy.parameters(), lr=self._cfg.policy_lr
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def save(self, dir_path: pathlib.Path | str):
        """Saves the current state of the MSA-hyper and its submodules.

        Parameters
        ----------
        dir_path : pathlib.Path | str
            The path to the directory where the model will be saved to.
        """
        dir_path = pathlib.Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        state = {}
        # NOTE: Will likely fail if the config is actually duck-typed
        # omegaconf.DictConfig

        # Critics
        for i, critic in enumerate(self._critics):
            state[f"critic_{i}"] = {
                "state": critic.state_dict(),
                "config": dataclasses.asdict(critic.config)
            }

        # Critic targets
        for i, critic in enumerate(self._critic_targets):
            state[f"critic_target_{i}"] = {
                "state": critic.state_dict(),
                "config": dataclasses.asdict(critic.config)
            }

        # Policy
        state["policy"] = {
            "state": self._policy.state_dict(),
            "config": self._policy.config
        }

        state["msa_hyper"] = {
            "policy_optim": self._policy_optim.state_dict(),
            "critic_optim": self._critic_optim.state_dict(),
            "config": dataclasses.as_dict(self._cfg)
        }
        torch.save(state, dir_path / "msa_hyper.tar")

    def take_action(
            self, obs: torch.Tensor, prefs: torch.Tensor
    ) -> torch.Tensor:
        return self._policy.take_action(obs, prefs)

    def update(
            self, replay_sample: common.ReplaySample
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the Q-networks and the policy network.

        Parameters
        ----------
        replay_sample : common.ReplaySample
            The sample from the Replay buffer.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The loss of the critic and the policy.
        """
        replay_sample = replay_sample.as_tensors(self._device)
        # Get the Q-target values
        with torch.no_grad():
            next_state_action, next_state_log_prob, next_state_mean =\
                self._policy.sample_action(
                    replay_sample.next_obs, replay_sample.prefs
                )

            state_value_targets = torch.stack([
                target_net(
                    replay_sample.obs, replay_sample.actions, replay_sample.prefs
                ) for target_net in self._critic_targets
            ])
            # Find the minimum values among the networks and add the rewards
            # to the minimum q-values.
            # (third line in CAPQL pseudo-code training loop)
            min_targets = (
                torch.min(state_value_targets, dim=0).values
                - self._cfg.alpha * next_state_log_prob.view(-1, 1)
            )

            target_q_values = (
                replay_sample.rewards + self._cfg.gamma *
                (1 - replay_sample.dones.unsqueeze(1)) * min_targets
            ).detach()

        # Update the networks
        critic_loss = self._update_q_network(target_q_values, replay_sample)
        policy_loss = self._update_policy(replay_sample)

        # Lastly, update the q-target networks
        for critic, critic_target in zip(self._critics, self._critic_targets):
            nets.polyak_update(critic, critic_target, self._cfg.tau)

        return critic_loss, policy_loss

    def _update_q_network(
            self, target_q_values: torch.Tensor,
            replay_sample: common.ReplaySample
    ) -> torch.Tensor:
        """Update the critic networks.

        Parameters
        ----------
        target_q_values : torch.Tensor
            The Q-values given by the target critic network.
        replay_sample : common.ReplaySample
            The sample from the replay buffer.

        Returns
        -------
        torch.Tensor
            The loss of the critc network.
        """
        self._critic_optim.zero_grad()
        q_vals = [
            critic(
                replay_sample.obs, replay_sample.actions,
                replay_sample.prefs
            ) for critic in self._critics
        ]
        critic_loss = (1 / self._cfg.n_networks) * sum(
                F.mse_loss(q_value, target_q_values) for q_value in q_vals
        )
        critic_loss.backward()
        self._critic_optim.step()
        return critic_loss

    def _update_policy(
            self, replay_sample: common.ReplaySample
    ) -> torch.Tensor:
        """Update the Policy network.

        Parameters
        ----------
        replay_sample : common.ReplaySample
            The sample from the Replay buffer.

        Returns
        -------
        torch.Tensor
            The loss of the policy network.
        """
        action, log_prob, _ = self._policy.sample_action(
            replay_sample.obs, replay_sample.prefs
        )

        q_values = torch.stack([
            target_net(replay_sample.obs, action, replay_sample.prefs)
            for target_net in self._critic_targets
        ])
        min_q_val = torch.min(q_values, dim=0).values
        min_q_val = (min_q_val * replay_sample.prefs).sum(dim=-1, keepdim=True)

        policy_loss = ((self._cfg.alpha * log_prob) - min_q_val).mean()

        self._policy_optim.zero_grad()
        policy_loss.backward()
        self._policy_optim.step()
        return policy_loss
