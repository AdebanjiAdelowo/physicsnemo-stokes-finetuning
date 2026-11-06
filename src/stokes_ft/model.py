# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from examples/cfd/stokes_mgn/train.py (MGNTrainer.__init__) of NVIDIA PhysicsNeMo,
# commit b45a5c810c741e6b41f8515be24c51121f8fc21f. Modified by Adebanji Adelowo (2026): the
# model construction is a function.

"""MeshGraphNet construction with the arguments of the upstream training script, and two
receptive-field variants for the controlled architecture experiment (G12).

``model_variant`` (absent from the upstream configurations, default ``baseline``):

* ``baseline``: upstream MeshGraphNet, 15 processor blocks, receptive field 15 hops.
* ``global_latent``: the same blocks; after every node block, the mean of the node latents is
  passed through one MLP shared by all blocks and added to every node (a "virtual node", Gilmer
  et al. 2017). Every node sees a summary of the whole graph from the first block. The last MLP
  layer starts at zero, so the untrained model equals the baseline with the same weights.
  Parameters +33,024 (+1.3 % at the upstream width).
* ``shared_depth``: the 15 blocks are applied ``processor_repeats`` times in sequence (weights
  shared), so the receptive field is 15 x repeats hops with exactly the baseline parameter
  count; compute per forward pass grows by the same factor.

The two variants separate the receptive field from the parameter count; the cost in compute is
measured, not matched (global latent about 1x, shared depth about repeats x).
"""

import torch
from physicsnemo.models.meshgraphnet import MeshGraphNet

VARIANTS = ("baseline", "global_latent", "shared_depth")


class GlobalLatentMeshGraphNet(MeshGraphNet):
    """MeshGraphNet with a mean-pooled global latent added to the nodes after every node block."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width = self.processor.input_dim_node
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(width, width), torch.nn.ReLU(), torch.nn.Linear(width, width)
        )
        torch.nn.init.zeros_(self.global_mlp[-1].weight)
        torch.nn.init.zeros_(self.global_mlp[-1].bias)

    def _pool(self, node_features, graph):
        batch = getattr(graph, "batch", None)
        if batch is None:
            return node_features.mean(0, keepdim=True).expand_as(node_features)
        count = torch.bincount(batch).clamp(min=1).to(node_features.dtype)
        sums = torch.zeros(len(count), node_features.shape[1], device=node_features.device, dtype=node_features.dtype)
        sums.index_add_(0, batch, node_features)
        return (sums / count[:, None])[batch]

    def forward(self, node_features, edge_features, graph, **kwargs):
        edge_features = self.edge_encoder(edge_features)
        node_features = self.node_encoder(node_features)
        layers = self.processor.processor_layers
        for edge_block, node_block in zip(layers[0::2], layers[1::2]):
            edge_features, node_features = edge_block(edge_features, node_features, graph)
            edge_features, node_features = node_block(edge_features, node_features, graph)
            node_features = node_features + self.global_mlp(self._pool(node_features, graph))
        return self.node_decoder(node_features)


class SharedDepthMeshGraphNet(MeshGraphNet):
    """MeshGraphNet whose processor blocks are applied ``processor_repeats`` times."""

    def __init__(self, *args, processor_repeats: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor_repeats = processor_repeats

    def forward(self, node_features, edge_features, graph, **kwargs):
        edge_features = self.edge_encoder(edge_features)
        node_features = self.node_encoder(node_features)
        for _ in range(self.processor_repeats):
            # each pass starts from the encoded edge features; the node latents carry over
            node_features = self.processor(node_features, edge_features, graph)
        return self.node_decoder(node_features)


def build_meshgraphnet(cfg) -> MeshGraphNet:
    """The data-driven model. Every argument not listed keeps the PhysicsNeMo default,
    including 15 message-passing layers of width 128."""
    variant = cfg.get("model_variant", "baseline")
    if variant not in VARIANTS:
        raise ValueError(f"model_variant must be one of {VARIANTS}")
    cls = {"baseline": MeshGraphNet, "global_latent": GlobalLatentMeshGraphNet, "shared_depth": SharedDepthMeshGraphNet}[variant]
    extra = {"processor_repeats": int(cfg.get("processor_repeats", 2))} if variant == "shared_depth" else {}
    return cls(
        cfg.input_dim_nodes,
        cfg.input_dim_edges,
        cfg.output_dim,
        aggregation=cfg.aggregation,
        hidden_dim_node_encoder=cfg.hidden_dim_node_encoder,
        hidden_dim_edge_encoder=cfg.hidden_dim_edge_encoder,
        hidden_dim_node_decoder=cfg.hidden_dim_node_decoder,
        **extra,
    )


def receptive_field_hops(model) -> int | None:
    """Hops a node's output can depend on; ``None`` when unbounded (a global latent)."""
    if isinstance(model, GlobalLatentMeshGraphNet):
        return None
    depth = len(model.processor.processor_layers) // 2
    return depth * getattr(model, "processor_repeats", 1)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
