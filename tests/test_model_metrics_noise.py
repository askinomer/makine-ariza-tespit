"""Model boyutları, ölçütler ve gürültü ekleme."""
import numpy as np
import pytest
import torch

from src.detectors import aggregate
from src.metrics import all_metrics, bootstrap_ci, threshold_metrics
from src.models import ConvAutoencoder
from src.noise import add_white_noise, file_rng


def test_autoencoder_shapes():
    m = ConvAutoencoder()
    x = torch.randn(3, 1, 64, 64)
    assert m(x).shape == x.shape
    assert m.encode(x).shape == (3, 128)
    assert ConvAutoencoder(bottleneck=32).encode(x).shape == (3, 32)
    with pytest.raises(ValueError):
        ConvAutoencoder(n_mels=60)


def test_autoencoder_output_unbounded():
    """Son katmanda sigmoid yok: negatif değer üretebilmeli."""
    torch.manual_seed(0)
    m = ConvAutoencoder()
    out = m(torch.randn(8, 1, 64, 64))
    assert out.min() < 0


def test_autoencoder_can_overfit_tiny_batch():
    torch.manual_seed(0)
    x = torch.randn(4, 1, 64, 64)
    m = ConvAutoencoder()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    first = None
    for _ in range(150):
        loss = ((m(x) - x) ** 2).mean()
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first


def test_metrics_perfect_and_far():
    y = np.array([0] * 10 + [1] * 10)
    s = np.r_[np.linspace(0, 1, 10), np.linspace(2, 3, 10)]
    m = all_metrics(y, s, threshold=1.5)
    assert m["auc"] == 1.0 and m["pauc"] == 1.0
    assert m["recall"] == 1.0 and m["far"] == 0.0 and m["f1"] == 1.0
    t = threshold_metrics(y, s, threshold=0.55)   # normallerin 5'i eşiği aşar
    assert t["fp"] == 5 and t["far"] == 0.5


def test_metrics_single_class_nan():
    m = all_metrics(np.zeros(5), np.arange(5.0), 2.0)
    assert np.isnan(m["auc"])


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(100), np.ones(100)]
    s = np.r_[rng.normal(0, 1, 100), rng.normal(1, 1, 100)]
    m = all_metrics(y, s, 1.5)
    ci = bootstrap_ci(y, s, 1.5, n_boot=200)
    lo, hi = ci["auc_ci"]
    assert lo < m["auc"] < hi


def test_aggregate():
    w = np.array([1.0, 2.0, 3.0, 10.0])
    assert aggregate(w, "mean") == 4.0
    assert aggregate(w, "max") == 10.0
    assert aggregate(w, "top2") == 6.5


@pytest.mark.parametrize("snr", [20, 10])
def test_white_noise_snr(snr):
    y = np.sin(np.linspace(0, 2000, 160000)).astype(np.float32) * 0.2
    yn = add_white_noise(y, snr, np.random.default_rng(0))
    noise = yn - y
    measured = 20 * np.log10(np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(noise ** 2)))
    assert abs(measured - snr) < 0.01


def test_noise_deterministic_per_file():
    y = np.ones(1000, np.float32)
    a = add_white_noise(y, 10, file_rng(1234, 10, "test/x"))
    b = add_white_noise(y, 10, file_rng(1234, 10, "test/x"))
    c = add_white_noise(y, 10, file_rng(1234, 10, "test/y"))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
