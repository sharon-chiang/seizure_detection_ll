"""Line-length-based seizure and spike detectors."""

from .detector import (
    seizure_detector_LL_shortlongtrend,
    spike_detector_LL_shortlongtrend,
)

__all__ = [
    "seizure_detector_LL_shortlongtrend",
    "spike_detector_LL_shortlongtrend",
]
