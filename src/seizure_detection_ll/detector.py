import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d, maximum_filter1d, minimum_filter1d


"""Line-length-based seizure and interictal spike detection.
"""

def _ll_window_features(
    lfp_df, lfp_sample_rate, short_window_size_sec,
    long_window_size_sec, percent_overlap, spikes=False,
    channel_batch_size=8
):
    """Compute LL ratios and spike features. LFP time stamps should be in seconds. 
    """
    n_time, n_channels = lfp_df.shape
    short_n = max(2, int(round(short_window_size_sec * lfp_sample_rate)))
    long_n = max(2, int(round(long_window_size_sec * lfp_sample_rate)))
    hop = max(1, int(round(short_n * (1 - percent_overlap / 100))))

    if n_time < long_n:
        raise ValueError(f"Need at least {long_n} samples to calculate long window")

    starts = np.arange(0, n_time - short_n + 1, hop)
    valid = starts >= long_n
    short_centers = starts + (short_n - 1) // 2
    long_centers = starts[valid] - long_n + (long_n - 1) // 2
    n_windows = len(starts)
    has_valid = valid.any()
    amp_centers = starts + short_n // 2 if spikes else None

    short_ll = np.empty((n_windows, n_channels))
    long_ll = np.full((n_windows, n_channels), np.nan)
    ratio = np.full((n_windows, n_channels), np.nan)
    deriv = np.empty((n_windows, n_channels)) if spikes else None
    amp = np.empty((n_windows, n_channels)) if spikes else None
    X = lfp_df.to_numpy(copy=False)

    for a in range(0, n_channels, channel_batch_size):
        b = min(a + channel_batch_size, n_channels)
        x = np.ascontiguousarray(X[:, a:b], dtype=np.float64)
        diff = np.abs(np.diff(x, axis=0))
        bad_diff = ~np.isfinite(diff)
        has_bad_diff = bad_diff.any()

        if has_bad_diff:
            diff[bad_diff] = 0

        s = uniform_filter1d(
            diff, size=short_n - 1, axis=0
        )[short_centers]

        if has_valid:
            l = uniform_filter1d(
                diff, size=long_n - 1, axis=0
            )[long_centers]

        if has_bad_diff:
            bad_s = maximum_filter1d(
                bad_diff, size=short_n - 1, axis=0
            )[short_centers]
            s[bad_s] = np.nan

            if has_valid:
                bad_l = maximum_filter1d(
                    bad_diff, size=long_n - 1, axis=0
                )[long_centers]
                l[bad_l] = np.nan

        short_ll[:, a:b] = s

        if has_valid:
            long_ll[valid, a:b] = l
            ratio[valid, a:b] = s[valid] / (l + 1e-12)

        if spikes:
            d = maximum_filter1d(
                diff, size=short_n - 1, axis=0
            )[short_centers] * lfp_sample_rate

            if has_bad_diff:
                d[bad_s] = np.nan

            deriv[:, a:b] = d

            bad_x = ~np.isfinite(x)
            if bad_x.any():
                x[bad_x] = 0

            a_vals = (
                maximum_filter1d(
                    x, size=short_n, axis=0
                )[amp_centers]
                - minimum_filter1d(
                    x, size=short_n, axis=0
                )[amp_centers]
            )

            if bad_x.any():
                bad_amp = maximum_filter1d(
                    bad_x, size=short_n, axis=0
                )[amp_centers]
                a_vals[bad_amp] = np.nan

            amp[:, a:b] = a_vals

    return starts, short_n, hop, valid, short_ll, long_ll, ratio, deriv, amp


def _event_intervals(mask, starts, short_n, index_vals, lfp_sample_rate):
    """Collect consecutive windows into event intervals."""
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    first = np.flatnonzero(changes == 1)
    last = np.flatnonzero(changes == -1) - 1

    if len(first) == 0:
        return pd.DataFrame()

    return pd.DataFrame({
        "start_window_idx": first,
        "end_window_idx": last,
        "start_time": index_vals[starts[first]],
        "end_time": index_vals[starts[last] + short_n - 1],
        "duration_sec": (
            starts[last] - starts[first] + short_n
        ) / lfp_sample_rate,
    })


def seizure_detector_LL_shortlongtrend(
    lfp_df,
    lfp_sample_rate,
    short_window_size_sec=1,
    long_window_size_sec=60,
    percent_overlap=50,
    ratio_threshold=1.5,
    num_channels=5,
    duration_threshold_sec=10,
    max_gap_sec=1,
    channel_batch_size=8,
):
    """LL seizure detector
    """
    (
        starts, short_n, hop, valid,
        short_ll, long_ll, ratio, _, _
    ) = _ll_window_features(
        lfp_df,
        lfp_sample_rate,
        short_window_size_sec,
        long_window_size_sec,
        percent_overlap,
        channel_batch_size=channel_batch_size,
    )

    n_above = np.count_nonzero(ratio > ratio_threshold, axis=1)
    candidate = (n_above >= num_channels) & valid

    step_sec = hop / lfp_sample_rate
    duration_windows = max(
        1, int(np.ceil(duration_threshold_sec / step_sec))
    )
    max_gap_windows = max(
        0, int(np.floor(max_gap_sec / step_sec))
    )

    seizure_window = np.zeros(len(starts), dtype=bool)
    consecutive_count = np.zeros(len(starts), dtype=int)

    run_start = last_positive = None
    n_positive = gap_len = 0

    for i, positive in enumerate(candidate):
        if positive:
            if run_start is None:
                run_start = i
                n_positive = 0

            last_positive = i
            n_positive += 1
            gap_len = 0

        elif run_start is not None:
            gap_len += 1

            if gap_len > max_gap_windows:
                if n_positive >= duration_windows:
                    seizure_window[
                        run_start:last_positive + 1
                    ] = True

                run_start = last_positive = None
                n_positive = gap_len = 0

        consecutive_count[i] = n_positive

    if run_start is not None and n_positive >= duration_windows:
        seizure_window[run_start:last_positive + 1] = True

    index_vals = np.asarray(lfp_df.index)
    start_time = index_vals[starts]
    end_time = index_vals[starts + short_n - 1]
    center_time = (start_time + end_time) / 2
    window_index = pd.Index(
        center_time, name="window_center_time"
    )

    return {
        "ratio_df": pd.DataFrame(
            ratio, index=window_index, columns=lfp_df.columns
        ),
        "short_ll_df": pd.DataFrame(
            short_ll, index=window_index, columns=lfp_df.columns
        ),
        "long_ll_df": pd.DataFrame(
            long_ll, index=window_index, columns=lfp_df.columns
        ),
        "spike_window_df": pd.DataFrame({
            "window_start_sample": starts,
            "window_end_sample": starts + short_n - 1,
            "window_start_time": start_time,
            "window_end_time": end_time,
            "window_center_time": center_time,
            "valid_long_window": valid,
            "n_channels_above_threshold": n_above,
            "spike_window": candidate,
            "consecutive_spike_windows": consecutive_count,
            "seizure_window": seizure_window,
        }),
        "seizure_intervals_df": _event_intervals(
            seizure_window,
            starts,
            short_n,
            index_vals,
            lfp_sample_rate,
        ),
    }


def spike_detector_LL_shortlongtrend(
    lfp_df,
    lfp_sample_rate,
    seizure_intervals_df,
    short_window_size_sec=0.05,
    long_window_size_sec=60,
    percent_overlap=50,
    ratio_threshold=1.5,
    max_derivative_threshold=75000,
    amplitude_threshold=150,
    num_channels=5,
    channel_batch_size=4,
):
    """LL interictal spike detector. Requires (1) channel must exceed the LL ratio, (2) maximum absolute derivative threshold, (3) peak-to-peak amplitude thresholds in the same short window.
    Note: Make sure to pre-exclude seizures by running seizure detector before this. 
    """
    (
        starts, short_n, _, valid,
        short_ll, long_ll, ratio, deriv, amp
    ) = _ll_window_features(
        lfp_df,
        lfp_sample_rate,
        short_window_size_sec,
        long_window_size_sec,
        percent_overlap,
        spikes=True,
        channel_batch_size=channel_batch_size,
    )

    above_ratio = ratio > ratio_threshold
    above_deriv = deriv > max_derivative_threshold
    above_amp = amp > amplitude_threshold

    n_ratio = np.count_nonzero(above_ratio, axis=1)
    n_deriv = np.count_nonzero(above_deriv, axis=1)
    n_amp = np.count_nonzero(above_amp, axis=1)

    n_meeting = np.count_nonzero(
        above_ratio & above_deriv & above_amp,
        axis=1,
    )

    index_vals = np.asarray(lfp_df.index)
    start_time = index_vals[starts]
    end_time = index_vals[starts + short_n - 1]
    center_time = (start_time + end_time) / 2
    window_index = pd.Index(
        center_time, name="window_center_time"
    )

    seizure_window = np.zeros(len(starts), dtype=bool)

    for _, row in seizure_intervals_df.iterrows():
        seizure_window |= (
            (start_time <= row["end_time"])
            & (end_time >= row["start_time"])
        )

    spike_window = (
        (n_meeting >= num_channels)
        & valid
        & ~seizure_window
    )

    return {
        "ratio_df": pd.DataFrame(
            ratio, index=window_index, columns=lfp_df.columns
        ),
        "short_ll_df": pd.DataFrame(
            short_ll, index=window_index, columns=lfp_df.columns
        ),
        "long_ll_df": pd.DataFrame(
            long_ll, index=window_index, columns=lfp_df.columns
        ),
        "max_derivative_df": pd.DataFrame(
            deriv, index=window_index, columns=lfp_df.columns
        ),
        "amplitude_df": pd.DataFrame(
            amp, index=window_index, columns=lfp_df.columns
        ),
        "spike_window_df": pd.DataFrame({
            "window_start_sample": starts,
            "window_end_sample": starts + short_n - 1,
            "window_start_time": start_time,
            "window_end_time": end_time,
            "window_center_time": center_time,
            "valid_long_window": valid,
            "n_channels_above_threshold": n_ratio,
            "n_channels_above_derivative_threshold": n_deriv,
            "n_channels_above_amplitude_threshold": n_amp,
            "n_channels_meeting_spike_criteria": n_meeting,
            "seizure_window": seizure_window,
            "spike_window": spike_window,
        }),
        "spike_intervals_df": _event_intervals(
            spike_window,
            starts,
            short_n,
            index_vals,
            lfp_sample_rate,
        ),
    }
