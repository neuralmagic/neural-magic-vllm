import logging
import time
from contextlib import contextmanager
from threading import RLock

import numpy

__all__ = ["Timer"]

_LOGGER = logging.getLogger(__name__)


class Timer:
    """Timer to log timings during inference requests. Should be used through 
        timings.utils.get_singleton_manager()
    """
    _instance = None

    def __init__(self, avg_after_iterations: int, enable_logging: bool):
        """
        :param avg_after_iterations: the number of iterations that have to occur
            before the time measurements are averaged and displayed to the user
        :param enable_logging: whether or not time logging is enabled
        """
        self.enable_logging = enable_logging
        self.measurements = {}
        self.avg_after_iterations = avg_after_iterations
        self.current_iter = {}
        self.lock = RLock()

    @contextmanager
    def time(self, func_name: str):
        """
        Time the given function, add the time to the dictionary of lists 
        tracking each function and calculate the average, if 
        self.avg_after_iterations number of iterations have passed.

        :param func_name: name of the function that will be used to log the 
            measurements
        """
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            with self.lock:
                self._update(func_name, end - start)
                self._compute_average(func_name)

    def _update(self, func_name: str, time_elapased: float):
        """
        Update the dictionary of measurements and counter for each function call

        :param func_name: name of the function that will be used to log the 
            measurements
        :param time_elapsed: time taken for function execution  
        """
        if self.current_iter.get(func_name) is None:
            self.current_iter[func_name] = 1
        else:
            self.current_iter[func_name] += 1

        if self.measurements.get(func_name) is None:
            self.measurements[func_name] = [time_elapased]
        else:
            self.measurements[func_name].append(time_elapased)

    def _compute_average(self, func_name: str):
        """
        Determine the number of iterations. Display averages if this value
        is equivalent to self.avg_after_iterations. After the average is
        calculated, clear the measurements.

        :param func_name: name of the function that will be used to log the 
            measurements

        """
        if self.current_iter.get(func_name) == self.avg_after_iterations:
            _LOGGER.info(f"Average time for {func_name}: ")
            _LOGGER.info(str(numpy.average(self.measurements.get(func_name))))
            self.current_iter[func_name] = 0
            self.measurements[func_name].clear()
