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

"""MeshGraphNet construction with the arguments of the upstream training script."""

from physicsnemo.models.meshgraphnet import MeshGraphNet


def build_meshgraphnet(cfg) -> MeshGraphNet:
    """The data-driven model. Every argument not listed keeps the PhysicsNeMo default,
    including 15 message-passing layers of width 128."""
    return MeshGraphNet(
        cfg.input_dim_nodes,
        cfg.input_dim_edges,
        cfg.output_dim,
        aggregation=cfg.aggregation,
        hidden_dim_node_encoder=cfg.hidden_dim_node_encoder,
        hidden_dim_edge_encoder=cfg.hidden_dim_edge_encoder,
        hidden_dim_node_decoder=cfg.hidden_dim_node_decoder,
    )


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
