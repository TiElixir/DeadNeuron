"""Every figure builder runs for both themes on realistic inputs, and the numbers in them are the inputs."""
import numpy as np
import pytest

from neuronlab.scoring import contrast_scores
from neuronlab.viz.figures import (architecture_figure, eval_bars_figure, grid_shape, layer_grid_figure, layers_figure,
                                   score_histogram, separation_figure, strip_figure)
from neuronlab.viz.html import token_html


@pytest.fixture(scope="module")
def res():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(20, 4, 30)).astype(np.float32); neg = rng.normal(size=(20, 4, 30)).astype(np.float32)
    pos[:, 2, 5] += 3
    return contrast_scores(pos, neg, pos[:8], n_perm=40)


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_all_figures_build(mode, res, lab):
    desc = lab.adapter.describe()
    assert architecture_figure(desc, None, mode).data
    sig = res.significant_per_layer()
    assert architecture_figure({**desc, "n_layers": 3}, np.array([1, 5, 2]), mode).data
    f = layers_figure(res, mode)
    assert list(f.data[0].y) == res.significant_per_layer(True).tolist()
    assert score_histogram(res, mode).data and separation_figure(np.array([0.6, 0.9, 0.7]), mode).data
    assert strip_figure(np.arange(5.), np.arange(4.), np.arange(3.), mode).data
    assert layer_grid_figure(np.arange(64.), "d", True, mode, {3, 4}).data
    stats = {k: dict(real=-0.5, rand_mean=0.0, rand_std=0.1) for k in ("d_pos", "d_neg", "d_ctl")}
    assert eval_bars_figure(stats, mode).data


def test_grid_shape_tiles_exactly():
    for n in (4864, 896, 64, 17, 11008):
        r, c = grid_shape(n)
        assert r * c == n and r <= c
    assert grid_shape(4864) == (64, 76)


def test_grid_cells_map_to_neuron_ids():
    f = layer_grid_figure(np.arange(4864.), "d", False, "dark")
    z, ids = np.array(f.data[0].z), np.array(f.data[0].customdata)
    assert z.shape == (64, 76) and ids[3, 5] == 3 * 76 + 5 and z[3, 5] == ids[3, 5]
    sc = f.data[1]
    assert (np.array(sc.y)[3 * 76 + 5], np.array(sc.x)[3 * 76 + 5]) == (3, 5)


def test_token_html_escapes_and_dims_uncounted_tokens():
    out = token_html(["<script>", "a", "b"], np.array([5.0, -1.0, 2.0]), np.array([False, True, True]))
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert out.count("background:rgba(") == 2  # only counted tokens are shaded
    assert "217,89,38" in out and "57,135,229" in out  # orange for negative, blue for positive
