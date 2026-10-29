import pytest
import torch
from torch import nn

from neuronlab.adapters import AdapterError, get_adapter
from neuronlab.adapters.registry import discover_adapter
from neuronlab.model_loader import build_lab


def _capture(adapter, ids, layer):
    got = {}
    h1 = adapter.neuron_site(layer).register_forward_pre_hook(lambda m, a: got.update(acts=a[0].detach().clone()))
    h2 = adapter.mlp_module(layer).register_forward_hook(lambda m, a, o: got.update(mlp=o.detach().clone()))
    with torch.no_grad():
        adapter.model(ids)
    h1.remove(); h2.remove()
    return got


def test_adapter_structure(lab_any):
    ad = lab_any.adapter
    cfg = ad.config
    assert ad.n_layers == 3
    assert ad.d_model == (cfg.n_embd if cfg.model_type == "gpt2" else cfg.hidden_size)
    assert ad.d_mlp(0) == 64
    assert ad.neuron_out_dirs(0).shape == (64, ad.d_model)
    d = ad.describe()
    assert d["n_neurons"] == 3 * 64 and len(d["layers"]) == 3


def test_down_proj_of_neuron_acts_equals_mlp_output(lab_any):
    """mlp(x) = sum_n a_n * out_dir_n (+ bias): proves the site and the (transposed) out dirs are right."""
    ad = lab_any.adapter
    ids = torch.randint(5, 100, (2, 9))
    for layer in range(ad.n_layers):
        got = _capture(ad, ids, layer)
        site = ad.neuron_site(layer)
        recon = got["acts"] @ ad.neuron_out_dirs(layer)
        if getattr(site, "bias", None) is not None:
            recon = recon + site.bias
        assert torch.allclose(recon, got["mlp"], atol=1e-5), f"layer {layer}"
        assert torch.allclose(site(got["acts"]), got["mlp"], atol=1e-5)


def test_gpt2_conv1d_is_not_transposed(lab_any):
    ad = lab_any.adapter
    if ad.config.model_type != "gpt2":
        pytest.skip("gpt2 only")
    w = ad.neuron_site(0).weight  # Conv1D: (d_mlp, d_model)
    assert w.shape == (64, 32)
    assert torch.equal(ad.neuron_out_dirs(0), w)


def test_residual_site_handles_tensor_or_tuple(lab_any):
    from neuronlab.adapters import residual_tensor
    ad = lab_any.adapter
    got = {}
    h = ad.residual_site(0).register_forward_hook(lambda m, a, o: got.update(o=o))
    ad.model(torch.randint(5, 100, (1, 5)))
    h.remove()
    assert residual_tensor(got["o"]).shape == (1, 5, ad.d_model)
    assert residual_tensor((torch.zeros(1), 1)).shape == (1,)


def test_discovery_matches_registered_adapter(lab_any):
    found = discover_adapter(lab_any.model)
    ad = lab_any.adapter
    for i in range(ad.n_layers):
        assert torch.equal(found.neuron_out_dirs(i), ad.neuron_out_dirs(i))
        assert found.neuron_site(i) is ad.neuron_site(i)


def test_discovery_failure_lists_module_tree():
    class Cfg:
        model_type = "weird"
        num_hidden_layers = 2
        hidden_size = 8

    class Weird(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()
            self.stack = nn.ModuleList([nn.ModuleDict({"foo": nn.Linear(8, 8)}) for _ in range(2)])

    with pytest.raises(AdapterError) as e:
        get_adapter(Weird())
    assert "Module tree" in str(e.value) and "stack" in str(e.value)
