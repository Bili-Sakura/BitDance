from __future__ import annotations

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from transformers.activations import ACT2FN


class BitDanceProjector(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_act: str = "gelu_pytorch_tanh",
    ) -> None:
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
