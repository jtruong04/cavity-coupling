import multiprocessing
import time
from typing import List

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.linalg import expm

from .BraKet import Bra, Ket
from .Hamiltonian import Hamiltonian, NoiseHamiltonian
from .types import Matrix, Vector


def kronDelta(i: int, j: int) -> int:
    '''Standard Kronecker Delta function'''  
    return 1 if i == j else 0

def kronSuper(i: int, j: int, p: int, q: int, dims: int) -> Matrix:
    '''Returns superoperator |i><j| * |p><q|''' 
    return np.kron(Ket(i, dims) @ Bra(j, dims), Ket(p, dims) @ Bra(q, dims))

# Use parallel processing for some performance gains.
try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 2  # arbitrary default

class DensityMatrix:
    def __init__(self, initialState: Vector, H0: Hamiltonian, Hn: List[NoiseHamiltonian]) -> None:
        """Instantiate the density operator

        Solves for the density operator of a system defined by

            H = H_0 + \sum_n f_n(t)h_n.

        The solution is given by the following equation

            rho(t) = \sum_{jk} exp(-i\omega_{jk}t)(R*R)(|j><j| * |k><k|)*exp(-\sum_n K_n(t)) (R^{-1} * R^{-1}) rho_0
        
        where the summation over j,k is over all states.

        Args:
            initialState (Vector): Vector containing the initial state of the system |\Psi_0>
            H0 (Hamiltonian): The noiseless part of the system H_0
            Hn (List[NoiseHamiltonian]): List containing the h_n
        """
        # Constants
        self.dimensions = len(H0())
        self.hbar = H0.hbar
        # Initial density matrix
        self.initialState = initialState
        self.initialDensityMatrix = np.kron(initialState.transpose(), initialState)
        self.initialDensityMatrixVec = self.initialDensityMatrix.reshape(
            (self.dimensions ** 2, 1)
        )
        # System Hamiltonian
        self.H0 = H0
        self.Hn = Hn
        # Compute Eigenbasis
        self.eigenvalues, self.eigenvectors = self.H0.eigh()
        self.RR = np.kron(self.eigenvectors, self.eigenvectors)
        self.RRinv = np.kron(
            np.linalg.inv(self.eigenvectors), np.linalg.inv(self.eigenvectors)
        )
        # Rotate Noise into new basis
        for hn in self.Hn:
            hn.rotate(self.eigenvectors)
        # Compute energy differences
        self.Omega = np.zeros((self.dimensions ** 2, self.dimensions ** 2))
        for j in range(self.dimensions):
            for p in range(self.dimensions):
                for k in range(self.dimensions):
                    for q in range(self.dimensions):
                        self.Omega += (
                            kronDelta(j, p) * kronDelta(k, q) * kronSuper(j, p, k, q, self.dimensions) \
                                * (self.eigenvalues[j] - self.eigenvalues[k]) / self.hbar
                        )

    def KD(self, t: float) -> Matrix:
        '''Get dephasing part of K(t)'''
        res = np.zeros((self.dimensions ** 2, self.dimensions ** 2)).astype(np.complex128)
        for j in range(self.dimensions):
            for k in range(self.dimensions):
                factor = 0
                for hn in self.Hn:
                    factor += hn.J(t, 0, 0) * (hn.H[j, j] - hn.H[k, k]) ** 2
                res += factor * kronSuper(j, j, k, k, self.dimensions)
        return res / (self.hbar ** 2)

    def KM(self, t: float) -> Matrix:
        '''Get Markovian part of K(t). Equivalent to KMR + KMD'''
        res = np.zeros((self.dimensions ** 2, self.dimensions ** 2)).astype(np.complex128)
        for a in range(self.dimensions):
            for b in range(self.dimensions):
                if a==b:
                    continue
                for hn in self.Hn:
                    res += (np.kron(Ket(a,self.dimensions)@Bra(a,self.dimensions),np.identity(self.dimensions))
                                +np.kron(np.identity(self.dimensions),Ket(b,self.dimensions)@Bra(b,self.dimensions))
                                -kronSuper(a,b,a,b,self.dimensions)
                                -kronSuper(b,a,b,a,self.dimensions)
                            )*hn.H[a,b]*hn.H[b,a]*hn.J(t,
                            (self.eigenvalues[b] - self.eigenvalues[a]) / self.hbar,
                            -(self.eigenvalues[b] - self.eigenvalues[a]) / self.hbar)
        return res / (self.hbar ** 2)
    
    def KMR(self, t: float) -> Matrix:
        '''Get Markovian relaxation part of K(t)'''
        res = np.zeros((self.dimensions ** 2, self.dimensions ** 2)).astype(np.complex128)
        for a in range(self.dimensions):
            for b in range(self.dimensions):
                if a == b:
                    continue
                for hn in self.Hn:
                    res += (kronSuper(a,a,a,a,self.dimensions)+kronSuper(b,b,b,b,self.dimensions)-kronSuper(a,b,a,b,self.dimensions)-kronSuper(b,a,b,a,self.dimensions)) \
                                    *hn.H[a,b]*hn.H[b,a]*hn.J(t,(self.eigenvalues[a] - self.eigenvalues[b]) / self.hbar,-(self.eigenvalues[a] - self.eigenvalues[b]) / self.hbar) 
        return res / (self.hbar ** 2)

    def KMD(self, t: float) -> Matrix:
        '''Get Markovian dephasing part of K(t)'''
        res = np.zeros((self.dimensions ** 2, self.dimensions ** 2)).astype(np.complex128)
        for a in range(self.dimensions):
            for b in range(self.dimensions):
                for c in range(self.dimensions):
                    if a == b or a==c:
                        continue
                    for hn in self.Hn:
                        res += (kronSuper(a,a,c,c,self.dimensions)*hn.J(t,(self.eigenvalues[b] - self.eigenvalues[a]) / self.hbar,-(self.eigenvalues[b] - self.eigenvalues[a]) / self.hbar)*hn.H[a,b]*hn.H[b,a]
                                + kronSuper(c,c,a,a,self.dimensions)*hn.J(t,(self.eigenvalues[a] - self.eigenvalues[b]) / self.hbar,-(self.eigenvalues[a] - self.eigenvalues[b]) / self.hbar)*hn.H[a,b]*hn.H[b,a])
        return res / (self.hbar ** 2)

    def KnDnM(self, t:float) -> Matrix :
        '''Get non-dephasing and non-Markovian part of K(t)'''
        res = np.zeros((self.dimensions ** 2, self.dimensions ** 2)).astype(np.complex128)
        # TODO
        return res

    def K(self, t: float) -> Matrix:
        '''Get K(t)'''
        res = np.zeros((self.dimensions ** 2, self.dimensions ** 2)).astype(np.complex128)
        for hn in self.Hn:
            for a in range(self.dimensions):
                for b in range(self.dimensions):
                    for c in range(self.dimensions):
                        res += hn.H[b, c] * hn.H[a, b] * hn.J(t,(self.eigenvalues[b] - self.eigenvalues[a]) / self.hbar,(self.eigenvalues[c] - self.eigenvalues[b]) / self.hbar,)  \
                                        * np.kron(Ket(a, self.dimensions) @ Bra(c, self.dimensions),np.identity(self.dimensions),) \
                                + hn.H[b, a] * hn.H[c, b] * hn.J(t,(self.eigenvalues[a] - self.eigenvalues[b]) / self.hbar,(self.eigenvalues[b] - self.eigenvalues[c]) / self.hbar,)  \
                                        * np.kron(np.identity(self.dimensions),Ket(a, self.dimensions) @ Bra(c, self.dimensions),)
                        for d in range(self.dimensions):
                            res -= (2.0* hn.H[a, b]* hn.H[d, c]* hn.J.JS(t,(self.eigenvalues[b] - self.eigenvalues[a])/ self.hbar,(self.eigenvalues[c] - self.eigenvalues[d])/ self.hbar,) \
                                            * kronSuper(a, b, c, d, self.dimensions))
        return res / (self.hbar ** 2)

    def __call__(self, timeRange: Vector, includeNoise: bool = True) -> Matrix:
        '''Get rho(t) given a range of time values'''
        # U(t)
        self.U = [expm(-1j * self.Omega * t) for t in timeRange]
        if not includeNoise:
            return np.reshape(self.RR @ self.U @ self.RRinv @ self.initialDensityMatrixVec,(len(timeRange), self.dimensions, self.dimensions),)

        """
        Notice here that where are can write exp(-(KD + KM)) = exp(-KD)exp(-KM) because KD and KM
        commute (for the most part). If we also include KnDnM, then that is no longer the case and 
        we have to compute exp(K) as a whole, and that is not easy to do. Numerical matrix 
        exponentials in general are pretty hard to compute.
        """

        # D(t) = exp(-KD(t))
        pool = Pool(processes=cpus)
        def expD(t):
            return expm(-1 * self.KD(t))
        self.D = pool.map(expD, timeRange)

        # L(t) = exp(-KM(t))
        # def expL(t):
        #     return expm(-1 * self.KM(t))
        # self.L = pool.map(expL, timeRange)   

        # LD(t) = exp(-KMD)
        def expLD(t):
            return expm(-1 * self.KMD(t))
        self.LD = pool.map(expLD, timeRange)

        # LR(t) = exp(-KMR)
        # def expLR(t):
        #     return expm(-1 * self.KMR(t))
        # self.LR = pool.map(expLR, timeRange)

        # exp(-K)
        # def expmk(t):
        #     return expm(-1 * self.K(t))
        # self.expK = pool.map(expmk, timeRange)  # [expm(-K) for K in self.Kall]

        # Multiply everything together
        return np.reshape(
            self.RR @ 
            self.U @ 
            # self.expK @ 
            self.D @
            # self.L @
            self.LD @
            # self.LR @
            self.RRinv @ 
            self.initialDensityMatrixVec,
            (len(timeRange), self.dimensions, self.dimensions),
        )
