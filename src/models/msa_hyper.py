""" Define the MSA-hyper algorithm"""

import itertools
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..utils import common, nets
from .hypernet import HyperNet
from .policy import GaussianPolicy


@dataclass
class MSAHyperConfig:
    n_networks: int
    alpha: float
    tau: float
    critic_optim: str = "adam"
    critic_lr: float = 3e-4
    policy_optim: str = "adam"
    policy_lr: float = 3e-4
    device: str = "cpu"


class MSAHyper:
    def __init__(self, config: MSAHyperConfig):
        """Create a MSA-hyper network, that uses a modified CAPQL, with
        hypernetworks for (hopefully) generalizing the learned information
        from policies.

        Parameters
        ----------
        config : MSAHyperConfig
            The configuration for the MSAHyper.
        """
        self._config = config
        if isinstance(self._config.device, str):
            self._config.device = torch.device(self._config.device)

        self._policy = GaussianPolicy(
            self._config.policy_config
        )

        self._critics = [
            HyperNet(self._config.critic_config) for _ in range(self._config.n_networks)
        ]

        self._critic_targets = [
            HyperNet(self._config.critic_config) for _ in range(self._config.n_networks)
        ]

        # Note: The target parameters are not updated through direct gradient
        # descent, so we disable the gradient tracking for them.
        for critic, critic_target in zip(self._critics, self._critic_targets):
            # Copy the initial parameters of critics to targets
            critic_target.load_state_dict(critic.state_dict())
            for param in critic_target.parameters():
                param.requires_grad_(False)

        self._critic_optim = nets.get_optim_by_name(self._config.critic_optim)(
            itertools.chain(critic.parameters() for critic in self._critics),
            lr=self._config.critic_lr
        )

        self._policy_optim = nets.get_optim_by_name(self._config.policy_optim)(
            self._policy.parameters(), lr=self._config.policy_lr
        )

    def update(self, replay_sample: common.ReplaySample):
        """Update the Q-networks and the policy network.

        Parameters
        ----------
        replay_sample : common.ReplaySample
            The sample from the Replay buffer.
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

            # Find the minimum values among the network
            min_targets = (
                state_value_targets.min(dim=1)
                - self._config.alpha * next_state_log_prob
            )

            target_q_values = (
                replay_sample.rewards + self._config.gamma *
                (1 - replay_sample.dones) * min_targets
            ).detach()

        # Update the networks
        critic_loss = self._update_q_network(target_q_values, replay_sample)
        policy_loss = self._update_policy()

        # Lastly, update the q-target networks
        for critic, critic_target in zip(self._critics, self._critic_targets):
            nets.polyak_update(critic, critic_target, self._config.tau)

        print(f"critic loss | {critic_loss}")
        print(f"Policy loss | {policy_loss}")

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
            ) for critic in self._critics()
        ]
        critic_loss = (1 / self._config.num_nets) * sum(F.mse_loss(
            q_value, target_q_values) for q_value in q_vals
        )
        critic_loss.backwards()
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
        min_q_val = q_values.min(dim=0)
        min_q_val = (min_q_val * replay_sample.prefs).sum(dim=-1, keepdim=True)

        policy_loss = ((self._config.alpha * log_prob) - min_q_val).mean()

        self._policy_optim.zero_grad()
        policy_loss.backwards()
        self._policy_optim.step()
        return policy_loss
