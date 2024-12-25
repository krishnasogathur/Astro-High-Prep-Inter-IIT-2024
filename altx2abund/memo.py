from functools import wraps
import numpy as np
def memoize_last(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = [*args, *tuple(sorted(kwargs.items()))]  # Create a unique key for the current input
        variable = True
        if wrapper.previnput == None:
            wrapper.previnput = key  # Update the most recent input
            wrapper.prevans = func(*args, **kwargs)  # Compute and store the most recent output
            return wrapper.prevans
        for a,b in zip(key,wrapper.previnput):
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray): 
                if (a!=b).any():
                    variable = False
                    break
            else:
                if a!=b:
                    variable = False
                    break
        # exit(0)
        if variable:
                return wrapper.prevans
        wrapper.previnput = key  # Update the most recent input
        wrapper.prevans = func(*args, **kwargs)  # Compute and store the most recent output
        return wrapper.prevans

    wrapper.previnput = None  # Initial value for the last input
    wrapper.prevans = None    # Initial value for the last output
    return wrapper
