# seizure-detection-ll

This code implements line-length short/long window seizure detector for multiple channel LFP data. 

## Install
```bash
pip install -e .
```

## Minimal working example
### Assumes your LFP dataframe is named lfp_df_SC38

```
from seizure_detection_ll import seizure_detector_LL_shortlongtrend
import matplotlib.pyplot as plt
import numpy as np

results = seizure_detector_LL_shortlongtrend(
    lfp_df=lfp_df_SC38,
    lfp_sample_rate=1000,
    short_window_size_sec=1,
    long_window_size_sec=60,
    percent_overlap=50,
    ratio_threshold=1.5,
    num_channels=5,
    duration_threshold_sec=10,
)

X = lfp_df_SC38.to_numpy(dtype=float)
time_vals = np.asarray(lfp_df_SC38.index, dtype=float)

spike_window_df = results["spike_window_df"].copy()

channel_range = np.nanpercentile(X, 99) - np.nanpercentile(X, 1)
offset_step = channel_range * 0.8 if channel_range > 0 else 1000

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(18, 12), sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

for i, col in enumerate(lfp_df_SC38.columns):
    ax1.plot(time_vals, X[:, i] + i * offset_step, linewidth=0.5)

ax1.set_title("LFP traces")

ax2.plot(
    spike_window_df["window_center_time"],
    spike_window_df["n_channels_above_threshold"],
    linewidth=1.5,
)
ax2.axhline(results["params"]["num_channels"], linestyle="--")
ax2.set_xlabel("Time")
ax2.set_ylabel("# channels > threshold")
ax2.set_title("LL detector")

seiz_mask = spike_window_df["seizure_window"].fillna(False).to_numpy()
win_start = np.asarray(spike_window_df["window_start_time"], dtype=float)
win_end = np.asarray(spike_window_df["window_end_time"], dtype=float)

for s, e, flag in zip(win_start, win_end, seiz_mask):
    if flag:
        ax1.axvspan(s, e, alpha=0.2)
        ax2.axvspan(s, e, alpha=0.2)

plt.tight_layout()
plt.show()
```




