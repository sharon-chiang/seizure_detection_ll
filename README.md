# seizure-detection-ll

Line-length short/long window detectors for seizures and interictal spikes in multiple channel LFP recordings.

## Install

```bash
pip install -e .
```

## Minimal working example

Assumes your LFP dataframe is named `lfp_df_SC38`, with timestamps (seconds) as rows and channels as columns.

```python
from seizure_detection_ll import (
    seizure_detector_LL_shortlongtrend,
    spike_detector_LL_shortlongtrend,
)
import matplotlib.pyplot as plt

# Seizure detection
seizures = seizure_detector_LL_shortlongtrend(
    lfp_df=lfp_df_SC38,
    lfp_sample_rate=1000,
    short_window_size_sec=1,
    long_window_size_sec=60,
    percent_overlap=50,
    ratio_threshold=1.5,
    num_channels=5,
    duration_threshold_sec=10,
    max_gap_sec=1,
)

# Interictal spike detection
spikes = spike_detector_LL_shortlongtrend(
    lfp_df=lfp_df_SC38,
    lfp_sample_rate=1000,
    seizure_intervals_df=seizures["seizure_intervals_df"],
    short_window_size_sec=0.05,
    long_window_size_sec=60,
    percent_overlap=50,
    ratio_threshold=1.5,
    max_derivative_threshold=75000,
    amplitude_threshold=150,
    num_channels=5,
)

# Detected event intervals
seizure_intervals_df = seizures["seizure_intervals_df"]
spike_intervals_df = spikes["spike_intervals_df"]

print("Seizures:", len(seizure_intervals_df))
print("Spikes:", len(spike_intervals_df))
```

## Visualization

Plot the number of channels exceeding the seizure LL threshold.

```python
w = seizures["spike_window_df"]

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(w["window_center_time"], w["n_channels_above_threshold"])
ax.axhline(5, linestyle="--", color="gray", label="Channel threshold")

for _, event in seizure_intervals_df.iterrows():
    ax.axvspan(event["start_time"], event["end_time"], color="red", alpha=0.2)

ax.set(xlabel="Time (s)", ylabel="# channels above threshold",
       title="Line-length seizure detection")
plt.tight_layout()
plt.show()
```

## Detection methods

- **Seizures:** Short/long line-length ratio exceeding a threshold across a minimum number of channels, sustained for a minimum duration. Short gaps between qualifying windows permitted.
- **Interictal spikes:** Separate line-length ratio, max derivative, and peak to peak amplitude thresholds across a minimum number of channels. Windows overlapping detected seizures must be pre-excluded.

Thresholds should be adjusted for the recording. Amplitude thresholds should be in LFP units (presumed uV). Derivative thresholds should be in those units per second.