"""GPT-2 family. The neuron site is the input of mlp.c_proj (= the output of mlp.act)."""
import torch

from .base import DownProjAdapter


class GPT2Adapter(DownProjAdapter):
    layers_path = "transformer.h"
    attn_name = "attn"
    down_name = "c_proj"
    final_norm_path = "transformer.ln_f"

    def _down_weight_as_out_dirs(self, w: torch.Tensor) -> torch.Tensor:
        # Conv1D weight is (d_mlp, d_model), transposed relative to nn.Linear:
        # neuron n's output direction is row n, so no transpose here.
        return w

    @property
    def d_model(self) -> int:
        return int(self.config.n_embd)
