from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch

from causaldt_ad.drawings.schema import RegulationConfig, RoutingConfig
from causaldt_ad.regulation.abstraction import therapy_reward
from causaldt_ad.regulation.weir import GaussianPolicy, TwinCritic
from causaldt_ad.routing.reservoir import CausalWorldModel


@dataclass(frozen=True, slots=True)
class PolicyReport:
    cumulative_reward: float
    delta_cognition: float
    biomarker_norm: float


class _Replay:
    def __init__(self, capacity: int, latent_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.latent = torch.zeros(capacity, latent_dim)
        self.action = torch.zeros(capacity, action_dim)
        self.reward = torch.zeros(capacity)
        self.nxt = torch.zeros(capacity, latent_dim)
        self.done = torch.zeros(capacity)
        self.position = 0
        self.size = 0

    def push(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        nxt: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        count = latent.shape[0]
        for offset in range(count):
            index = (self.position + offset) % self.capacity
            self.latent[index] = latent[offset]
            self.action[index] = action[offset]
            self.reward[index] = reward[offset]
            self.nxt[index] = nxt[offset]
            self.done[index] = done[offset]
        self.position = (self.position + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch: int, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
        index = torch.randint(0, self.size, (batch,), generator=generator)
        return (
            self.latent[index],
            self.action[index],
            self.reward[index],
            self.nxt[index],
            self.done[index],
        )


class TherapyOptimizer:
    def __init__(
        self,
        model: CausalWorldModel,
        biomarker_index: tuple[int, ...],
        reference: tuple[float, ...],
        config: RegulationConfig,
        routing: RoutingConfig,
        seed: int,
    ) -> None:
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.config = config
        self.routing = routing
        self.horizon = routing.horizon
        self.generator = torch.Generator().manual_seed(seed)
        latent = model.latent_dim
        self.policy = GaussianPolicy(latent, config.action_dim, config.policy_hidden)
        self.critic = TwinCritic(latent, config.action_dim, config.q_hidden)
        self.target = copy.deepcopy(self.critic)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.target_entropy = -float(config.action_dim) * config.target_entropy_scale
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)
        self.biomarker_index = torch.tensor(biomarker_index, dtype=torch.long)
        self.reference = torch.tensor(reference, dtype=torch.float32)
        self.buffer = _Replay(config.buffer_size, latent, config.action_dim)

    def _start(self, batch: int) -> torch.Tensor:
        return torch.randn(batch, self.model.latent_dim, generator=self.generator)

    @torch.no_grad()
    def _step(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, cog_prev = self.model.decoder(latent)
        nxt, cog_next, summary = self.model.rollout_step(latent, action)
        biomarkers = summary.index_select(-1, self.biomarker_index)
        reward = therapy_reward(
            cog_next - cog_prev, biomarkers, self.reference, action, self.config.reward_weights
        )
        return nxt, reward

    @torch.no_grad()
    def _rollout(self, batch: int, deterministic: bool) -> PolicyReport:
        latent = self._start(batch)
        total = torch.zeros(batch)
        first_cog = self.model.decoder(latent)[1]
        last_cog = first_cog
        norm_acc = torch.zeros(batch)
        discount = 1.0
        for step in range(self.horizon):
            action = self.policy.act(latent) if deterministic else self.policy.sample(latent)[0]
            nxt, reward = self._step(latent, action)
            total = total + discount * reward
            discount *= self.config.gamma
            done = torch.full((batch,), 1.0 if step == self.horizon - 1 else 0.0)
            self.buffer.push(latent, action, reward, nxt, done)
            summary = self.model.node_summary(nxt).index_select(-1, self.biomarker_index)
            norm_acc = (1.0 - (summary - self.reference).abs()).sum(dim=-1)
            last_cog = self.model.decoder(nxt)[1]
            latent = nxt
        return PolicyReport(
            cumulative_reward=float(total.mean()),
            delta_cognition=float((last_cog - first_cog).mean()),
            biomarker_norm=float(norm_acc.mean()),
        )

    def _update(self) -> None:
        if self.buffer.size < self.config.batch_size:
            return
        latent, action, reward, nxt, done = self.buffer.sample(
            self.config.batch_size, self.generator
        )
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_action, next_logp = self.policy.sample(nxt)
            target_q1, target_q2 = self.target(nxt, next_action)
            target_q = torch.min(target_q1, target_q2) - alpha * next_logp
            backup = reward + self.config.gamma * (1.0 - done) * target_q
        q1, q2 = self.critic(latent, action)
        critic_loss = ((q1 - backup) ** 2).mean() + ((q2 - backup) ** 2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        fresh_action, logp = self.policy.sample(latent)
        q1p, q2p = self.critic(latent, fresh_action)
        policy_loss = (alpha * logp - torch.min(q1p, q2p)).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        for online, target in zip(self.critic.parameters(), self.target.parameters(), strict=False):
            target.data.mul_(1.0 - self.config.tau).add_(self.config.tau * online.data)

    def train(self) -> PolicyReport:
        rollout_batch = max(8, self.config.batch_size // 4)
        per_collect = rollout_batch * self.horizon
        collects = max(1, math.ceil(self.config.steps / max(1, per_collect)))
        for _ in range(collects):
            self._rollout(rollout_batch, deterministic=False)
            for _ in range(self.horizon):
                self._update()
        return self.evaluate(rollout_batch)

    @torch.no_grad()
    def evaluate(self, batch: int) -> PolicyReport:
        return self._rollout(batch, deterministic=True)
