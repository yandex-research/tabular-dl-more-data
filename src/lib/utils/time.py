import time
from copy import deepcopy
from typing import Any


# Inspired by: https://github.com/Yura52/delu/blob/a76e3139f83703b02500f96b9031cd592619d28e/delu/tools.py#L163
class Timer:
    """A pickle-friendly timer.

    **Usage**

    Creating a timer:

    >>> timer = Timer()

    A just created timer is not running:

    >>> import time
    >>>
    >>> JOB_TIME = 0.001
    >>>
    >>> time.sleep(JOB_TIME)
    >>> assert timer.elapsed() == 0.0

    Running a timer:

    >>> timer.run()
    >>> time.sleep(JOB_TIME)
    >>> assert timer.elapsed() > 0.0

    Pausing a timer:

    >>> timer.pause()
    >>> elapsed_so_far = timer.elapsed()
    >>> time.sleep(JOB_TIME)
    >>> assert timer.elapsed() == elapsed_so_far

    Resuming a timer:

    >>> timer.run()
    >>> time.sleep(JOB_TIME)
    >>> assert timer.elapsed() > elapsed_so_far

    Pickling and unpickling a timer:

    >>> import pickle
    >>>
    >>> # Pausing is not required for pickling, it is needed only for the demonstration.
    >>> timer.pause()
    >>> elapsed_before_pickling = timer.elapsed()
    >>>
    >>> pickled_timer = pickle.dumps(timer)
    >>> time.sleep(JOB_TIME)
    >>> timer = pickle.loads(pickled_timer)

    A timer is not running while it is in a pickled state:

    >>> assert timer.elapsed() == elapsed_before_pickling

    A just unpickled timer is not running and has to be resumed explicitly:

    >>> time.sleep(JOB_TIME)
    >>> assert timer.elapsed() == elapsed_before_pickling
    >>>
    >>> timer.run()
    >>> time.sleep(JOB_TIME)
    >>> assert timer.elapsed() > elapsed_before_pickling
    """

    def __init__(self) -> None:
        self._shift = 0.0
        self._start_time = None
        self._pause_time = None

    def run(self) -> None:
        if self._start_time is None:
            self._start_time = time.perf_counter()
        elif self._pause_time is not None:
            self._shift -= time.perf_counter() - self._pause_time
            self._pause_time = None

    def pause(self) -> None:
        if self._start_time is None:
            return

        if self._pause_time is None:
            self._pause_time = time.perf_counter()

    def elapsed(self) -> float:
        if self._start_time is None:
            return self._shift
        else:
            now = time.perf_counter() if self._pause_time is None else self._pause_time
            return now - self._start_time + self._shift

    def __getstate__(self) -> dict[str, Any]:
        state = deepcopy(self.__dict__)
        state['_shift'] = self.elapsed()
        state['_start_time'] = None
        state['_pause_time'] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
