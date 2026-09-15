import functools
import logging
import time

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def timing_decorator(func):
    """
    A simple decorator that prints the execution time of the decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Record the start time before calling the function
        start_time = time.perf_counter()

        # Call the original function and store its result
        result = func(*args, **kwargs)

        # Record the end time
        end_time = time.perf_counter()

        # Calculate the duration
        duration = end_time - start_time

        # Print the timing information
        function_name = func.__name__.replace("_", " ").title()
        logging.info(f"\033[93m{function_name} took {duration:.4f} seconds to execute\033[0m")

        # Return the original function's result
        return result

    return wrapper
