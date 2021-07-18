import numpy as np

from .types import Matrix

GAMMA_E = 27.970                                     # GHz/T
GAMMA_N = 0.01723                                    # GHz/T
DELTA_GAMMA = -0.002                                 # Dimensionless
HYPERFINE = 0.1170                                   # GHz
SILICON_DIELECTRIC = 11.7                            # Dimensionless
DELTA = GAMMA_E * DELTA_GAMMA / (GAMMA_E + GAMMA_N)  # Dimensionless
HBAR = 1.0 / (2.0 *np.pi)                            # Dimensionless


def getPauliVector(use_pm: bool = False):
    if use_pm:
        return (np.array([[1, 0], [0, 1]]),  
                np.array([[0, 1], [1, 0]]) + np.array([[0, -1j], [1j, 0]])*1j,  
                np.array([[0, 1], [1, 0]]) - np.array([[0, -1j], [1j, 0]])*1j, 
                np.array([[1, 0], [0, -1]]))
    else:
        return (np.array([[1, 0], [0, 1]]),
                np.array([[0, 1], [1, 0]]) ,
                np.array([[0, -1j], [1j, 0]]),
                np.array([[1, 0], [0, -1]]))


def Pauli2(i: int, n: int, N: int, use_pm: bool = False) -> Matrix:
    """Get a Pauli matrix projected onto an 2^N dimensional space

    Ex: Pauli2(3, 2, 3) = I * I * Z
        Pauli2(1, 0, 4) = X * I * I * I

    Args:
        i (int): The index of the Pauli matrix to use (0: I, 1: X, 2: Y, 3: Z)
        n (int): Position of the Pauli matrix, remaining ones are equal to I.
        N (int): Number of matrices to tensor product
        use_pm (bool): Use the basis of I, sigma_plus, sigma_minus, sigma_z instead of IXYZ

    Returns:
        Matrix: Matrix of dimensionality 2^N x 2^N
    """    
    assert( i <= 3 and i >= 0 and n < N and N >= 0)
    pauli = getPauliVector(use_pm)
    res = np.identity(1)
    res = np.kron(res, np.identity(2**n))
    res = np.kron(res, pauli[i])
    res = np.kron(res, np.identity(2**(N-n-1)))
    return res

def Pauli4(i: int, j: int, n: int, N: int, use_pm: bool = False) -> Matrix:
    """Get a Pauli matrix projected onto an 4^N dimensional space

    Ex: Pauli4(3, 1, 2, 3) = I4 * I4 * (ZX)

    Args:
        i (int): The index of the Pauli matrix to use (0: I, 1: X, 2: Y, 3: Z)
        j (int): The index of the Pauli matrix to use (0: I, 1: X, 2: Y, 3: Z)
        n (int): Position of the Pauli matrix, remaining ones are equal to I.
        N (int): Number of matrices to tensor product
        use_pm (bool): Use the basis of I, sigma_plus, sigma_minus, sigma_z instead of IXYZ

    Returns:
        Matrix: Matrix of dimensionality 4^N x 4^N
    """
    assert( i <= 3 and i >= 0 and j <= 3 and j >= 0 and n < N and N >= 0)
    pauli = getPauliVector(use_pm)
    res = np.identity(1)
    res = np.kron(res, np.identity(4**n))
    sigma = np.kron(pauli[i], pauli[j])
    res = np.kron(res, sigma)
    res = np.kron(res, np.identity(4**(N-n-1)))
    return res
