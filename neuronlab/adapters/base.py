"""ModelAdapter: the one place that knows where neurons live in a given architecture."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


def residual_tensor(output):
    """A decoder layer returns a tensor or a tuple whose first element is the hidden state."""
    return output[0] if isinstance(output, tuple) else output


def replace_residual(output, new_hidden):
    if isinstance(output, tuple):
        return (new_hidden,) + tuple(output[1:])
    return new_hidden


class ModelAdapter(ABC):
    """Neuron = element n of the input to the MLP down projection.

    Attention heads are out of scope for v1; `attn_module` and `n_heads` exist so a
    head-level view can be added without changing this interface.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.config = model.config

    # structure -------------------------------------------------------------
    @property
    @abstractmethod
    def n_layers(self) -> int: ...

    @property
    def d_model(self) -> int:
        return int(self.config.hidden_size)

    @abstractmethod
    def d_mlp(self, layer: int) -> int: ...

    @abstractmethod
    def layer_module(self, i: int) -> nn.Module: ...

    @abstractmethod
    def mlp_module(self, i: int) -> nn.Module: ...

    @abstractmethod
    def attn_module(self, i: int) -> nn.Module: ...

    @abstractmethod
    def neuron_site(self, i: int) -> nn.Module:
        """Module whose *input* (forward pre-hook) holds the neuron activations."""

    @abstractmethod
    def neuron_out_dirs(self, i: int) -> torch.Tensor:
        """[d_mlp, d_model]: row n is neuron n's output direction in the residual stream."""

    def residual_site(self, i: int) -> nn.Module:
        """Module whose *output* is the residual stream after layer i."""
        return self.layer_module(i)

    @abstractmethod
    def final_norm(self) -> nn.Module: ...

    @abstractmethod
    def unembed(self) -> torch.Tensor:
        """[vocab, d_model] output embedding matrix."""

    @property
    def n_heads(self) -> int:
        return int(getattr(self.config, "num_attention_heads", getattr(self.config, "n_head", 0)))

    @property
    def n_neurons(self) -> int:
        return sum(self.d_mlp(i) for i in range(self.n_layers))

    def describe(self) -> dict:
        def n_params(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        layers = []
        for i in range(self.n_layers):
            layers.append(
                {
                    "layer": i,
                    "attention_heads": self.n_heads,
                    "mlp_neurons": self.d_mlp(i),
                    "attn_params": n_params(self.attn_module(i)),
                    "mlp_params": n_params(self.mlp_module(i)),
                    "layer_params": n_params(self.layer_module(i)),
                }
            )
        embed_params = n_params(self.model.get_input_embeddings())
        return {
            "adapter": type(self).__name__,
            "model_type": getattr(self.config, "model_type", "?"),
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "d_mlp": [self.d_mlp(i) for i in range(self.n_layers)],
            "n_neurons": self.n_neurons,
            "n_heads": self.n_heads,
            "vocab_size": int(self.unembed().shape[0]),
            "tied_embeddings": bool(getattr(self.config, "tie_word_embeddings", False)),
            "embedding_params": embed_params,
            "total_params": n_params(self.model),
            "layers": layers,
        }


class DownProjAdapter(ModelAdapter):
    """Shared implementation: a list of decoder layers, each with an MLP and a down projection."""

    layers_path: str = ""
    mlp_name: str = "mlp"
    attn_name: str = "self_attn"
    down_name: str = "down_proj"
    final_norm_path: str = ""

    def _get(self, path: str) -> nn.Module:
        obj = self.model
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj

    @property
    def layers(self) -> nn.ModuleList:
        return self._get(self.layers_path)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def layer_module(self, i):
        return self.layers[i]

    def mlp_module(self, i):
        return getattr(self.layers[i], self.mlp_name)

    def attn_module(self, i):
        return getattr(self.layers[i], self.attn_name)

    def neuron_site(self, i):
        return getattr(self.mlp_module(i), self.down_name)

    def _down_weight_as_out_dirs(self, w: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def d_mlp(self, layer):
        return int(self.neuron_out_dirs(layer).shape[0])

    def neuron_out_dirs(self, i):
        site = self.neuron_site(i)
        return self._down_weight_as_out_dirs(site.weight).detach()

    def final_norm(self):
        return self._get(self.final_norm_path)

    def unembed(self):
        return self.model.get_output_embeddings().weight.detach()
