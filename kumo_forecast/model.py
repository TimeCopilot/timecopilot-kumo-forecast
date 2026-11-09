"""Inference API for Kumo-Forecast.

The checkpoint (`nvidia/Kumo-Forecast`) contains every weight, so the model is built
from the bundled T5 configuration and the MOMENT-1-large weights are never downloaded.
"""

from __future__ import annotations

import warnings
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from ._backbone import BackboneModel

DEFAULT_REPO_ID = "nvidia/Kumo-Forecast"
DEFAULT_CHECKPOINT = "run8_best_model_cr.pt"
DEFAULT_STANDARDIZER = "standardizer.pkl"

# Context and native horizon of the published checkpoint. `load_kumo_forecast`
# reads the real values from the checkpoint; these are the expected defaults.
SEQ_LEN = 512
MODEL_HORIZON = 72

PATCH_LEN = 8

# google/flan-t5-large encoder used by MOMENT-1-large (see its `config.json`).
MOMENT_T5_CONFIG: dict[str, Any] = {
    "architectures": ["T5ForConditionalGeneration"],
    "d_ff": 2816,
    "d_kv": 64,
    "d_model": 1024,
    "decoder_start_token_id": 0,
    "dropout_rate": 0.1,
    "eos_token_id": 1,
    "feed_forward_proj": "gated-gelu",
    "initializer_factor": 1.0,
    "is_encoder_decoder": True,
    "layer_norm_epsilon": 1e-06,
    "model_type": "t5",
    "n_positions": 512,
    "num_decoder_layers": 24,
    "num_heads": 16,
    "num_layers": 24,
    "output_past": True,
    "pad_token_id": 0,
    "relative_attention_max_distance": 128,
    "relative_attention_num_buckets": 32,
    "tie_word_embeddings": False,
    "use_cache": True,
    "vocab_size": 32128,
}


@dataclass(frozen=True)
class Standardizer:
    """Global affine scaling stored next to the checkpoint (`standardizer.pkl`)."""

    mean: float = 0.0
    std: float = 1.0
    eps: float = 1e-8

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + self.eps)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std + self.eps) + self.mean


def _build_backbone(
    t5_config: dict[str, Any],
    seq_len: int,
    forecast_horizon: int,
    use_cross_channel: bool,
) -> BackboneModel:
    config = Namespace(
        task_name="forecasting",
        model_name="MOMENT",
        transformer_type="encoder_only",
        transformer_backbone="google/flan-t5-large",
        d_model=None,
        seq_len=seq_len,
        patch_len=PATCH_LEN,
        patch_stride_len=PATCH_LEN,
        forecast_horizon=forecast_horizon,
        head_dropout=0.0,
        freeze_encoder=True,
        freeze_embedder=True,
        freeze_head=True,
        use_cross_channel=use_cross_channel,
        cross_channel_heads=8,
        cross_channel_dropout=0.1,
        enable_gradient_checkpointing=False,
        t5_config=t5_config,
    )
    with warnings.catch_warnings():
        # "Only reconstruction head is pre-trained": the forecasting head comes
        # from the Kumo checkpoint.
        warnings.simplefilter("ignore", UserWarning)
        return BackboneModel(config)


class KumoForecastModel:
    """Point forecaster around a Kumo-Forecast backbone.

    Each series is one independent channel. Series are standardized with the global
    scaler shipped with the checkpoint (the backbone also normalizes every series
    internally), left-padded and masked up to `seq_len`, and forecast `horizon` steps
    at a time; longer horizons roll the forecast forward autoregressively.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        standardizer: Standardizer,
        seq_len: int,
        horizon: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.module = module.eval()
        self.standardizer = standardizer
        self.seq_len = seq_len
        self.horizon = horizon
        self.device = torch.device(device)

    @torch.inference_mode()
    def forecast(self, series: np.ndarray, h: int) -> np.ndarray:
        """Forecast `h` steps for each row of `series`.

        Args:
            series: Array of shape `[n_series, length]`. NaN marks missing values;
                only the last `seq_len` columns are used and shorter inputs are
                left-padded with NaN.
            h: Number of steps to forecast.

        Returns:
            Array of shape `[n_series, h]` in the original scale.
        """
        if h < 1:
            raise ValueError(f"`h` must be a positive integer, got {h}.")
        values = np.asarray(series, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("`series` must have shape [n_series, length].")
        values = values[:, -self.seq_len :]
        padded = np.full((values.shape[0], self.seq_len), np.nan, dtype=np.float32)
        padded[:, self.seq_len - values.shape[1] :] = values
        observed = ~np.isnan(padded)
        if not observed.any(axis=1).all():
            raise ValueError("Every series needs at least one observed value.")

        scaled = self.standardizer.transform(padded)
        x = torch.from_numpy(np.where(observed, scaled, 0.0).astype(np.float32))
        mask = torch.from_numpy(observed.astype(np.int64))
        dtype = next(self.module.parameters()).dtype
        x = x.to(device=self.device, dtype=dtype)[:, None, :]
        mask = mask.to(self.device)

        preds: list[torch.Tensor] = []
        remaining = h
        while remaining > 0:
            out = self.module(x_enc=x, input_mask=mask).forecast  # [B, 1, horizon]
            step = out[:, :, : min(self.horizon, remaining)]
            preds.append(step)
            remaining -= step.shape[-1]
            if remaining > 0:
                x = torch.cat([x, step], dim=-1)[..., -self.seq_len :]
                ones = torch.ones(
                    step.shape[0], step.shape[-1], dtype=mask.dtype, device=mask.device
                )
                mask = torch.cat([mask, ones], dim=-1)[..., -self.seq_len :]

        forecast = torch.cat(preds, dim=-1)[:, 0, :].float().cpu().numpy()
        return self.standardizer.inverse(forecast).astype(np.float32)


def _load_standardizer(path: str | Path) -> Standardizer:
    artifact = joblib.load(path)
    if isinstance(artifact, dict):
        mean, std = artifact["mean"], artifact["std"]
    else:
        mean, std = artifact.mean, artifact.std
    mean = float(np.asarray(mean, dtype=np.float64).reshape(-1)[0])
    std = float(np.asarray(std, dtype=np.float64).reshape(-1)[0])
    return Standardizer(mean=mean, std=std if std >= 1e-8 else 1.0)


def load_kumo_forecast(
    repo_id: str = DEFAULT_REPO_ID,
    checkpoint: str = DEFAULT_CHECKPOINT,
    standardizer: str = DEFAULT_STANDARDIZER,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> KumoForecastModel:
    """Download (or find locally) a Kumo-Forecast checkpoint and build the model.

    Args:
        repo_id: Hugging Face repository id, or a local directory containing the
            checkpoint and the standardizer.
        checkpoint: Checkpoint file name inside `repo_id`.
        standardizer: Standardizer file name inside `repo_id`.
        device: Torch device. Defaults to CUDA when available, otherwise CPU.
        dtype: Weights dtype.
        token: Hugging Face token.
        local_files_only: Only use files already in the local cache.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    local = Path(repo_id)
    if local.is_dir():
        ckpt_path, std_path = local / checkpoint, local / standardizer
    else:
        kwargs = {"token": token, "local_files_only": local_files_only}
        ckpt_path = Path(hf_hub_download(repo_id, checkpoint, **kwargs))
        std_path = Path(hf_hub_download(repo_id, standardizer, **kwargs))

    try:
        state = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
    except RuntimeError:  # legacy (non-zip) checkpoint, cannot be memory mapped
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    d_model = MOMENT_T5_CONFIG["d_model"]
    horizon, head_in = state["head.linear.weight"].shape
    seq_len = (head_in // d_model) * PATCH_LEN
    use_cross_channel = any("cross_channel" in key for key in state)
    module = _build_backbone(MOMENT_T5_CONFIG, seq_len, horizon, use_cross_channel)
    module.load_state_dict(state, strict=True)
    del state
    module = module.to(device=device, dtype=dtype)
    return KumoForecastModel(
        module,
        _load_standardizer(std_path),
        seq_len=seq_len,
        horizon=horizon,
        device=device,
    )
