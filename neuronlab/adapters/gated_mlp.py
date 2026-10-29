"""Qwen2 / Llama / Mistral / Gemma: mlp(x) = down_proj(act(gate_proj(x)) * up_proj(x))."""
import torch

from .base import DownProjAdapter


class GatedMLPAdapter(DownProjAdapter):
    layers_path = "model.layers"
    final_norm_path = "model.norm"

    def _down_weight_as_out_dirs(self, w: torch.Tensor) -> torch.Tensor:
        # nn.Linear weight is (d_model, d_mlp): neuron n's output direction is column n.
        return w.t()

    @property
    def d_model(self) -> int:
        return int(self.config.hidden_size)

    def d_mlp(self, layer):
        return int(self.neuron_site(layer).in_features)
