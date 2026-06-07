# Information: Diagnostics metrics helper.
# Importance: Provides high-precision duration trackers (latency calculation) and maps raw float probability arrays to class name dictionaries.

import time
import numpy as np
from typing import Dict

class LatencyTracker:
    """Utility class to track processing/inference latency."""
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()

    @property
    def duration_ms(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000.0

def format_probabilities(probs: np.ndarray, class_mapping: Dict[int, str]) -> Dict[str, float]:
    """Helper to convert a flat array of probabilities to a class name map."""
    return {class_mapping[i]: float(probs[i]) for i in range(len(probs))}
