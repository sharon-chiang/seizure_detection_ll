import numpy as np
import pandas as pd

def seizure_detector_LL_shortlongtrend(
    lfp_df,
    lfp_sample_rate,
    short_window_size_sec=1,
    long_window_size_sec=60,
    percent_overlap=50,
    ratio_threshold=0.5,
    num_channels=5,
    duration_threshold_sec=10.0,
):
    """
    Implement line length detector that 1) compute short window LL and long window LL for every channel
    2) computes ratio = short_LL / long_LL
    3) mark a shots window as having a spike if ratio exceeds threshold in at least num_channels channels (spike detection)
    4) mark a candidate seizure if consecutive windows have sepikes for duration_threshold_sec( (seizure detector)

    Inputs
    ----------
    lfp_df : pd.DataFrame
        Rows are time points, columns are channels. Requires that lfp_df.index is timestamps. If not, need to update code. 
    lfp_sample_rate : float
        Sampling frequency in Hz.
    short_window_size_sec : float
        Short-term window size in seconds.
    long_window_size_sec : float
        Long-term window size in seconds.
    percent_overlap : float
        Percent overlap for both short and long windows.
    ratio_threshold : float
        Direct threshold for short_LL / long_LL.
    num_channels : int
        Minimum number of channels exceeding ratio threshold
        for a short window to count as spike-positive.
    duration_threshold_sec : float
        Require spike-positive windows for this many consecutive seconds.

    Ouputs
    -------
    results : dict
    """

    X = lfp_df.to_numpy(dtype=float)  
    n_time, n_channels_total = X.shape

    # Calculate number of samples requires in the short window and long window
    short_window_samples = max(2, int(round(short_window_size_sec * lfp_sample_rate)))
    long_window_samples = max(2, int(round(long_window_size_sec * lfp_sample_rate)))

    # How many samples to move each shrt window forward based on sliding window overlap
    short_hop_samples = max(1, int(round(short_window_samples * (1.0 - percent_overlap / 100.0))))

    if n_time < long_window_samples:
        raise ValueError(f"Need at least {long_window_samples} samples to calculate long window")

    # Get starting points of short windows
    short_window_starts = np.arange(0, n_time - short_window_samples + 1, short_hop_samples)
    n_short_windows = len(short_window_starts)

    # Initialize arrays to store short window LL and long window LL
    short_ll = np.empty((n_short_windows, n_channels_total))
    long_ll = np.empty((n_short_windows, n_channels_total))

    # Some arrays don't have long enough time to also have a long window. True if also can calculate a long window
    valid_mask = np.zeros(n_short_windows, dtype=bool)

    # Calculate line length for hsort and long windows
    for i, s_start in enumerate(short_window_starts):
        s_end = s_start + short_window_samples 

        short_seg = X[s_start:s_end, :]
        short_ll[i, :] = np.mean(np.abs(np.diff(short_seg, axis=0)), axis=0)
        
        l_end = s_end
        l_start = l_end - long_window_samples

        if l_start >= 0:
            long_seg = X[l_start:l_end, :]
            long_ll[i, :] = np.mean(np.abs(np.diff(long_seg, axis=0)), axis=0)
            valid_mask[i] = True
        else:
            long_ll[i, :] = np.nan

    # Calculate ratio and make it nan if no long window 
    ratio_short_long = short_ll / (long_ll + 1e-12)
    ratio_short_long[~valid_mask, :] = np.nan

    # Number of channels above threshold. We probably need to tune this per rat? I don't know if each rat always has >90% above thrhoesl. 
    n_channels_above_threshold = np.sum(ratio_short_long > ratio_threshold, axis=1)
    spike_window = n_channels_above_threshold >= num_channels
    spike_window[~valid_mask] = False

    # Convert duration into # of consecutive windows to meet criteria for a seizure
    step_sec = short_hop_samples / lfp_sample_rate
    duration_windows = max(1, int(np.ceil(duration_threshold_sec / step_sec)))

    # Counter for consecutive windows to meet criteria
    seizure_window = np.zeros(n_short_windows, dtype=bool)
    consecutive_count = np.zeros(n_short_windows, dtype=int)

    run_len = 0
    for i in range(n_short_windows):
        if spike_window[i]:
            run_len += 1
        else:
            run_len = 0
        consecutive_count[i] = run_len
        seizure_window[i] = run_len >= duration_windows

    # Get times of start and stop of each short window. 
    index_vals = np.asarray(lfp_df.index, dtype=float)
    short_start_time = index_vals[short_window_starts]
    short_end_idx = np.minimum(short_window_starts + short_window_samples - 1, len(index_vals) - 1)
    short_end_time = index_vals[short_end_idx]
    short_center_time = 0.5 * (short_start_time + short_end_time)

    # Save out
    ratio_df = pd.DataFrame(ratio_short_long,
                            index=pd.Index(short_center_time, name="window_center_time"), 
                            columns=lfp_df.columns)

    spike_window_df = pd.DataFrame({
        "window_start_sample": short_window_starts,
        "window_end_sample": short_window_starts + short_window_samples - 1,
        "window_start_time": short_start_time,
        "window_end_time": short_end_time,
        "window_center_time": short_center_time,
        "valid_long_window": valid_mask,
        "n_channels_above_threshold": n_channels_above_threshold,
        "spike_window": spike_window,
        "consecutive_spike_windows": consecutive_count,
        "seizure_window": seizure_window,
    })

    # Candidate seizure start/stop times. Merge ones that overlap. 
    seizure_intervals = []
    in_event = False
    event_start_idx = None

    for i, is_seiz in enumerate(seizure_window):
        if is_seiz and not in_event:
            in_event = True
            event_start_idx = i
        elif not is_seiz and in_event:
            in_event =False
            event_end_idx = i-1
            seizure_intervals.append((event_start_idx, event_end_idx))

    if in_event:
        seizure_intervals.append((event_start_idx, n_short_windows - 1))

    seizure_intervals_rows = []
    for start_i, end_i in seizure_intervals:
        seizure_intervals_rows.append({"start_window_idx": start_i, "end_window_idx": end_i,
                                       "start_time": spike_window_df.loc[start_i, "window_start_time"],
                                       "end_time": spike_window_df.loc[end_i, "window_end_time"],
                                       "duration_sec": (spike_window_df.loc[end_i, "window_end_sample"] 
                                                        - spike_window_df.loc[start_i, "window_start_sample"] + 1) / lfp_sample_rate})
    seizure_intervals_df = pd.DataFrame(seizure_intervals_rows)

    return {"ratio_df": ratio_df,
            "short_ll_df": pd.DataFrame(short_ll,
                                        index=pd.Index(short_center_time, name="window_center_time"),
                                        columns=lfp_df.columns),
            "long_ll_df": pd.DataFrame(long_ll,
                                       index=pd.Index(short_center_time, name="window_center_time"),
                                       columns=lfp_df.columns),
            "spike_window_df": spike_window_df,
            "seizure_intervals_df": seizure_intervals_df}