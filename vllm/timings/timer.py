import time
from contextlib import contextmanager
from threading import RLock

import numpy

__all__ = ["Timer"]


class Timer:
    """Timer to log timings during inference requests. Should be used through 
        timings.utils.get_singleton_manager()
    """
    _instance = None

    def __init__(self, avg_after_iterations: int, log: bool):
        """
        :param avg_after_iterations: the number of iterations that have to occur
            before the time measurements are averaged and displayed to the user
        :param log: whether or not time logging is enabled
        """
        self.log = log
        self.measurements = {}
        self.avg_after_iterations = avg_after_iterations
        self.current_iter = 0
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
                self._compute_average()

    def _update(self, func_name: str, time_elapased: float):
        """
        Update the dictionary of measurements. 

        :param func_name: name of the function that will be used to log the 
            measurements
        :param time_elapsed: time taken for function execution  
        """
        self.current_iter += 1
        if self.measurements.get(func_name) is None:
            self.measurements[func_name] = [time_elapased]
        else:
            self.measurements[func_name].append(time_elapased)

    def _compute_average(self):
        """
        Determine the number of iterations. Display averages if this value
        is equivalent to self.avg_after_iterations.

        TODO: Do we want to keep track of each function individually and only 
        display the average when a specific function reaches a certain number 
        of iterations or one counter for all timer calls? Do we want to clear
        the values after the average is calculated?
        """
        if self.current_iter == self.avg_after_iterations:
            for func_name in self.measurements:
                print(func_name,
                      numpy.average(self.measurements.get(func_name)))
            self.current_iter = 0
