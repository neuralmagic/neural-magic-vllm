import numpy
from threading import RLock
from contextlib import contextmanager
import time

__all__ = ["Timer"]


class Timer:
    """Timer to log timings during inference requests."""
    _instance = None

    def __init__(self, avg_after_iterations: int):
        self.measurements = {}
        self.iter_counts = {}
        self.avg_after_iterations = avg_after_iterations
        self.lock = RLock()

    @contextmanager
    def time(self, func_name: str):
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            with self.lock:
                self._update(func_name, end-start)
                self._compute_average(func_name)

    def _update(self, func_name: str, time_elapased: float):
        if self.measurements.get(func_name) is None:
            self.measurements[func_name] = [time_elapased]
        else:
            self.measurements[func_name].append(time_elapased)

    def _set_iter_counts(self, func_name: str):
        current_iter = self.iter_counts.get(func_name)
        if current_iter is not None:
            self.iter_counts[func_name] += 1
        else:
            self.iter_counts[func_name] = 0

    # Maybe just show current average for all of them after x iterations?
    def _compute_average(self, func_name: str):
        self._set_iter_counts(func_name)
        current_iter = self.iter_counts.get(func_name)
        if current_iter == self.avg_after_iterations:
            print(func_name, numpy.average(self.measurements.get(func_name)))
            self.iter_counts[func_name] = 0
            self.measurements[func_name].clear()
