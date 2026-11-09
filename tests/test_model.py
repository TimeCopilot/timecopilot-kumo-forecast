import numpy as np
import pytest
import torch

from kumo_forecast.model import KumoForecastModel, Standardizer, _build_backbone

TINY_T5 = {
    "d_ff": 64,
    "d_kv": 8,
    "d_model": 32,
    "feed_forward_proj": "gated-gelu",
    "model_type": "t5",
    "num_decoder_layers": 2,
    "num_heads": 4,
    "num_layers": 2,
    "vocab_size": 128,
}
SEQ_LEN = 32
HORIZON = 6


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    module = _build_backbone(TINY_T5, SEQ_LEN, HORIZON, use_cross_channel=True)
    return KumoForecastModel(
        module, Standardizer(mean=-5.0, std=9.0), seq_len=SEQ_LEN, horizon=HORIZON
    )


def make_series(n_series=3, length=SEQ_LEN):
    rng = np.random.default_rng(0)
    t = np.arange(length)
    return 100 + 10 * np.sin(t / 3)[None] + rng.normal(size=(n_series, length))


def test_output_shape_and_finite(model):
    out = model.forecast(make_series(), h=HORIZON)
    assert out.shape == (3, HORIZON)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("h", [1, HORIZON - 1, HORIZON, HORIZON + 1, 3 * HORIZON + 2])
def test_horizons_use_rollout_beyond_native_horizon(model, h):
    out = model.forecast(make_series(), h=h)
    assert out.shape == (3, h)
    assert np.isfinite(out).all()


def test_rollout_prefix_matches_direct_forecast(model):
    series = make_series()
    direct = model.forecast(series, h=HORIZON)
    rolled = model.forecast(series, h=2 * HORIZON)
    np.testing.assert_allclose(rolled[:, :HORIZON], direct, rtol=1e-5, atol=1e-5)


def test_short_series_are_left_padded(model):
    out = model.forecast(make_series(length=10), h=HORIZON)
    assert out.shape == (3, HORIZON)
    assert np.isfinite(out).all()


def test_long_series_use_only_the_last_context(model):
    long = make_series(length=3 * SEQ_LEN)
    np.testing.assert_allclose(
        model.forecast(long, h=HORIZON),
        model.forecast(long[:, -SEQ_LEN:], h=HORIZON),
        rtol=1e-5,
        atol=1e-5,
    )


def test_series_are_independent(model):
    series = make_series()
    batched = model.forecast(series, h=HORIZON)
    single = model.forecast(series[1:2], h=HORIZON)
    np.testing.assert_allclose(batched[1:2], single, rtol=1e-4, atol=1e-4)


def test_nan_values_are_masked(model):
    series = make_series()
    series[:, 5:8] = np.nan
    out = model.forecast(series, h=HORIZON)
    assert np.isfinite(out).all()


def test_invalid_inputs(model):
    with pytest.raises(ValueError, match="`h`"):
        model.forecast(make_series(), h=0)
    with pytest.raises(ValueError, match="shape"):
        model.forecast(np.ones(SEQ_LEN), h=2)
    with pytest.raises(ValueError, match="observed"):
        model.forecast(np.full((1, SEQ_LEN), np.nan), h=2)


def test_standardizer_roundtrip():
    s = Standardizer(mean=3.0, std=2.0)
    x = np.array([1.0, 2.0, 5.0])
    np.testing.assert_allclose(s.inverse(s.transform(x)), x, rtol=1e-6)
