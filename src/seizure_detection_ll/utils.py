import numpy as np
import pandas as pd
from .detector import seizure_detector_LL_shortlongtrend, spike_detector_LL_shortlongtrend


def find_probe_dropouts(lfp_df, fs, channels_per_probe=128, flat_fraction=0.90, flat_tolerance=0.01, min_flat_sec=0.25, pad_sec=0.5,block_size=10000):
    X = lfp_df.to_numpy(copy=False)
    n, n_channels = X.shape

    t = lfp_df.index.to_numpy(dtype=float)
    dt = np.diff(t)
    gap_idx = np.flatnonzero(
        ~np.isfinite(dt) | (dt <= 0) | (dt > 1.5 / fs))

    n_flat = max(1, round(min_flat_sec * fs))
    pad = max(0, round(pad_sec * fs))
    masks = []

    for c0 in range(0, n_channels, channels_per_probe):
        c1 = c0 + channels_per_probe
        threshold = int(np.ceil(flat_fraction * (c1 - c0)))
        invalid = np.zeros(n, dtype=bool)
        unchanged = np.zeros(n - 1, dtype=bool)

        for a in range(0, n, block_size):
            b = min(a + block_size, n)
            x = X[a:b, c0:c1]
            invalid[a:b] = (
                np.count_nonzero(~np.isfinite(x), axis=1) >= threshold)

        for a in range(0, n - 1, block_size):
            b = min(a + block_size, n - 1)
            x0, x1 = X[a:b, c0:c1], X[a + 1:b + 1, c0:c1]
            flat = (~np.isfinite(x0) | ~np.isfinite(x1) |
                    (np.abs(x1 - x0) <= flat_tolerance))
            unchanged[a:b] = (
                np.count_nonzero(flat, axis=1) >= threshold)

        unchanged[gap_idx] = False
        changes = np.diff(np.r_[False, unchanged, False].astype(np.int8))

        for a, b in zip(
            np.flatnonzero(changes == 1),
            np.flatnonzero(changes == -1)
        ):
            if b - a >= n_flat:
                invalid[max(0, a - pad):min(n, b + pad + 1)] = True

        masks.append(invalid)

    return masks
    
def detect_global_epoch(lfp_df, fs, seizure_params, spike_params, channels_per_probe=128, **dropout_kwargs):
    t = lfp_df.index.to_numpy(dtype=float)
    dt = np.diff(t)

    gap_cuts = np.flatnonzero(
        ~np.isfinite(dt) | (dt <= 0) | (dt > 1.5 / fs)) + 1

    masks = find_probe_dropouts(
        lfp_df, fs, channels_per_probe=channels_per_probe,
        **dropout_kwargs)
    any_valid = np.zeros(len(lfp_df), dtype=bool)
    all_valid = np.ones(len(lfp_df), dtype=bool)

    for invalid in masks:
        any_valid |= ~invalid
        all_valid &= ~invalid
    valid_cuts = np.flatnonzero(np.diff(any_valid.astype(np.int8))) + 1
    bounds = np.unique(np.r_[0, gap_cuts, valid_cuts, len(lfp_df)])

    seizure_dfs, spike_dfs = [], []
    n_segments = 0
    seizure_min = max(2, round(seizure_params["long_window_size_sec"] * fs))
    spike_min = max(2, round(spike_params["long_window_size_sec"] * fs))
    empty_seizures = pd.DataFrame(columns=["start_time", "end_time"])

    for a, b in zip(bounds[:-1], bounds[1:]):
        if not any_valid[a] or b - a < min(seizure_min, spike_min):
            continue

        n_segments += 1
        segment = lfp_df.iloc[a:b].copy()
        for p, invalid in enumerate(masks):
            bad = invalid[a:b]
            if bad.any():
                segment.iloc[
                    bad, p * channels_per_probe:(p + 1) * channels_per_probe
                ] = np.nan

        seizures = empty_seizures

        if len(segment) >= seizure_min:
            result = seizure_detector_LL_shortlongtrend(
                segment, fs, **seizure_params)
            seizures = result["seizure_intervals_df"]

            if not seizures.empty:
                seizure_dfs.append(seizures)
            del result

        if len(segment) >= spike_min:
            result = spike_detector_LL_shortlongtrend(
                segment, fs, seizures, **spike_params)

            spikes = result["spike_intervals_df"]
            if not spikes.empty:
                spike_dfs.append(spikes)
            del result

        del segment

    cols = ["start_time", "end_time", "duration_sec"]
    seizure_df = (pd.concat(seizure_dfs, ignore_index=True)
                  if seizure_dfs else pd.DataFrame(columns=cols))
    spike_df = (pd.concat(spike_dfs, ignore_index=True)
                if spike_dfs else pd.DataFrame(columns=cols))

    return {
        "seizure_intervals_df": seizure_df,
        "spike_intervals_df": spike_df,
        "duration_sec": np.count_nonzero(any_valid) / fs,
        "duration_all_probes_sec": np.count_nonzero(all_valid) / fs,
        "excluded_all_probes_sec": np.count_nonzero(~any_valid) / fs,
        "timestamp_gap_sec": np.sum(
            np.maximum(dt[gap_cuts - 1] - 1 / fs, 0)),
        "n_timestamp_gaps": len(gap_cuts),
        "n_detection_segments": n_segments,
    }


def bipolar_montage(df, bad_channels=()):
    out = {}
    for shank in range(4):
        x = df.iloc[:, 32*shank:32*(shank+1)]
        pairs = [(a, b) for a, b in zip(x.columns[::2], x.columns[1::2])
                 if a not in bad_channels and b not in bad_channels]
        out[f"shank{shank+1}"] = pd.concat({f"{a}-{b}": x[a]-x[b] for a, b in pairs}, axis=1)
    return pd.concat(out, axis=1, names=["shank", "channel"])


def probe_bad_channels(nwb_file_name):
    import spyglass.common as sgc
    bad = {}
    for i, group in enumerate((2, 3, 4)):
        ids = (sgc.Electrode & {"nwb_file_name": nwb_file_name,
                "electrode_group_name": str(group), "bad_channel": "True"}).fetch("electrode_id")
        bad[i] = {int(ch) - 128*group - 1 + 128*i for ch in ids}
    return bad


def create_lfp_seizure_spike_figurl(lfp_df, seizure_intervals_df, spike_intervals_df,
                                   downsample=20, channel_spacing=None, url_label="lfp_seizure_spike"):
    import figpack.views as vv
    t = lfp_df.index.to_numpy(dtype=float)
    x = lfp_df.to_numpy(dtype=float)
    dt = np.diff(t)
    good_dt = dt[np.isfinite(dt) & (dt > 0)]
    cuts = np.flatnonzero(~np.isfinite(dt) | (dt <= 0) | (dt > 1.5*np.median(good_dt))) + 1
    bounds = np.r_[0, cuts, len(t)]
    segments = list(zip(bounds[:-1], bounds[1:]))
    if channel_spacing is None:
        z = x[::downsample]
        channel_spacing = 3*np.nanmedian(np.nanpercentile(z, 95, axis=0) - np.nanpercentile(z, 5, axis=0))

    palette = ["blue", "red", "green", "black"]
    graph = vv.TimeseriesGraph()
    for i, col in enumerate(lfp_df.columns):
        color = palette[int(col[0].replace("shank", "")) - 1]
        for j, (a, b) in enumerate(segments):
            idx = np.arange(a, b, downsample)
            ok = np.isfinite(x[idx, i])
            runs = np.diff(np.r_[False, ok, False].astype(int))
            for k, (start, end) in enumerate(zip(np.flatnonzero(runs == 1), np.flatnonzero(runs == -1))):
                ii = idx[start:end]
                if len(ii) >= 2:
                    graph.add_line_series(name=f"{col}_{j}_{k}", t=t[ii],
                                          y=(x[ii, i] + i*channel_spacing).astype(np.float32),
                                          color=color, width=1)

    events_graph = vv.TimeseriesGraph()
    for name, events, color in [("seizure", seizure_intervals_df, "red"),
                                ("interictal spike", spike_intervals_df, "green")]:
        starts, ends = [], []
        for s, e in events[["start_time", "end_time"]].itertuples(index=False, name=None):
            for a, b in segments:
                s0, e0 = max(s, t[a]), min(e, t[b-1])
                if s0 < e0:
                    starts.append(s0)
                    ends.append(e0)
        if starts:
            events_graph.add_interval_series(name=name, t_start=np.array(starts), t_end=np.array(ends), color=color)

    view = vv.Box(direction="vertical", show_titles=True, items=[
        vv.LayoutItem(graph, stretch=8), vv.LayoutItem(events_graph, stretch=1)])
    return view.show(title=url_label, open_in_browser=True, upload=True)
