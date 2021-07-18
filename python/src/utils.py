import numpy as np

from .types import Matrix

# 
# Python decorators
# 
def memoize(func):
    cache = dict()
    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return memoized_func

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

# 
# Matrix operations
# 
def adjoint(M: np.ndarray) -> np.ndarray:
    return M.conj().T

def commutator(A: Matrix, B: Matrix) -> Matrix:
    return A@B - B@A
