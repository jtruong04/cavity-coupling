import numpy as np

from .types import Matrix, Vector
from .utils import static_vars, adjoint


@static_vars(dimensions=0)
def Ket(i: int, N: int = 0) -> Vector:
    """Get a column vector with a single a single element equal to 1 and the rest 0

    Args:
        i (int): Position of the element to be equal to 1.
        N (int, optional): Size of the vector. Defaults to 0.

    Raises:
        ValueError: The dimensionality of the vector cannot be 0.

    Returns:
        Vector: Column vector of length N and single element equal to 1.
    """    
    if N != 0:
        Ket.dimensions = N
    if i >= Ket.dimensions:
        Ket.dimensions = i + 1
    if Ket.dimensions == 0:
        raise ValueError("Dimesionality should be greater than 0")
    res = np.zeros((Ket.dimensions, 1))
    res[i, 0] = 1
    return res

def Bra(i: int, N: int = 0) -> Vector:
    return adjoint(Ket(i, N))
