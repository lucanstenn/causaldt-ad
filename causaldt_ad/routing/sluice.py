from __future__ import annotations

import torch

from causaldt_ad.routing.reservoir import CausalWorldModel


def descendants(parents: tuple[tuple[int, ...], ...], source: int) -> frozenset[int]:
    children: dict[int, list[int]] = {j: [] for j in range(len(parents))}
    for child, mothers in enumerate(parents):
        for parent in mothers:
            children[parent].append(child)
    reached: set[int] = set()
    frontier = [source]
    while frontier:
        current = frontier.pop()
        for child in children[current]:
            if child not in reached:
                reached.add(child)
                frontier.append(child)
    return frozenset(reached)


def _block(latent: torch.Tensor, node: int, per_node: int) -> slice:
    return slice(node * per_node, (node + 1) * per_node)


def counterfactual(
    model: CausalWorldModel,
    latent0: torch.Tensor,
    actions: torch.Tensor,
    node: int,
    value: float,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    affected = descendants(_parents_of(model), node) | {node}
    factual = latent0.clone()
    counter = latent0.clone()
    per_node = model.per_node
    counter[..., _block(counter, node, per_node)] = value
    factual_path = [factual]
    counter_path = [counter]
    for step in range(horizon):
        action = actions[:, step] if step < actions.shape[1] else actions[:, -1]
        factual_mean, _ = model.transition(factual, action)
        counter_mean, _ = model.transition(counter, action)
        factual = factual_mean
        merged = factual_mean.clone()
        for affected_node in affected:
            merged[..., _block(merged, affected_node, per_node)] = counter_mean[
                ..., _block(counter_mean, affected_node, per_node)
            ]
        merged[..., _block(merged, node, per_node)] = value
        counter = merged
        factual_path.append(factual)
        counter_path.append(counter)
    return torch.stack(factual_path, dim=1), torch.stack(counter_path, dim=1)


def _parents_of(model: CausalWorldModel) -> tuple[tuple[int, ...], ...]:
    mask = model.transition.parent_mask
    per_node = model.per_node
    parents: list[tuple[int, ...]] = []
    for j in range(model.node_dim):
        found: list[int] = []
        for i in range(model.node_dim):
            if i == j:
                continue
            if float(mask[j, i * per_node]) > 0.0:
                found.append(i)
        parents.append(tuple(found))
    return tuple(parents)
