# timecopilot-kumo-forecast

Inference-only packaging of [NVIDIA Kumo-Forecast](https://github.com/NVIDIA/Kumo-TS)
(`nvidia/Kumo-Forecast` on Hugging Face), used by
[FoundationForecast](https://github.com/TimeCopilot/foundationforecast).

Upstream is clone-only and installs top-level modules named `sdk`, `backbone` and `model`.
This package ships the same backbone under the single `kumo_forecast` namespace, without the
anomaly, interpretability, DARR and fine-tuning code, and without downloading the
MOMENT-1-large weights (the Kumo checkpoint contains every weight).

```python
import numpy as np

from kumo_forecast import load_kumo_forecast

model = load_kumo_forecast()  # downloads nvidia/Kumo-Forecast
y = np.sin(np.arange(300) * 2 * np.pi / 24)[None]  # [n_series, length], NaN allowed
forecast = model.forecast(y, h=24)  # [n_series, 24]
```

The checkpoint has a fixed 512-step context and a native horizon of 72; shorter series are
left-padded and masked, longer ones truncated, and horizons above 72 use an autoregressive
rollout. Forecasts are point forecasts only.

Licensed under Apache-2.0. See `NOTICE`.
