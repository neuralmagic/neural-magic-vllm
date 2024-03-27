from timer import Timer

__all__ = ["log_time", "get_singleton_manager"]


def get_singleton_manager(avg_after_iterations: int = 100):
    if Timer._instance is None:
        Timer._instance = Timer(avg_after_iterations = avg_after_iterations)
    return Timer._instance


def log_async_time(func):
    TIMER_MANAGER = get_singleton_manager()
    async def wrapper(self, *arg, **kwargs):
        func_name = f"{self.__class__.__name__}.{func.__name__}"
        with TIMER_MANAGER.time(func_name):
            return await func(self, *arg, **kwargs)

    return wrapper

def log_time(func):
    TIMER_MANAGER = get_singleton_manager()
    def wrapper(self, *arg, **kwargs):
        func_name = f"{self.__class__.__name__}.{func.__name__}"
        with TIMER_MANAGER.time(func_name):
            return func(self, *arg, **kwargs)

    return wrapper

