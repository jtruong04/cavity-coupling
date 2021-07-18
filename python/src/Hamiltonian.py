from typing import Tuple

import numpy as np

from .constants import Pauli2, Pauli4
from .Noise import Noise
from .Noise_Pink import PinkNoise
from .types import Matrix, Vector


class Hamiltonian:
    def __init__(self, H: Matrix, hbar: float = 1.0 / (2.0 * np.pi)):
        self.H = H
        self.hbar = hbar

    def __call__(self):
        '''Get the Hamiltonian in matrix form'''
        return self.H

    def eigh(self) -> Tuple[Vector, Matrix]:
        """Solve for the Eigensystem of the Hamiltonian.

        Returns:
            Tuple[Vector, Matrix]: [List of Eigenvalues, List of Eigenvectors]
        """
        return np.linalg.eigh(self.H)

    def rotate(self, rotationMatrix: Matrix, inplace: bool = True) -> None:
        """Apply a transformation to the system in the form R^{-1}.H.R

        The transformation is applied in place.

        Args:
            rotationMatrix (Matrix): Transformation matrix. Needs to be the same saze as H.
        """        
        rotated_hamiltonian = np.linalg.inv(rotationMatrix) @ self.H @ rotationMatrix
        if inplace:
            self.H = rotated_hamiltonian
        return rotated_hamiltonian


class NoiseHamiltonian(Hamiltonian):
    """
        A noise Hamiltonian describes the interaction of the system Hamiltonian with some
        outside source of noise. A classical noise Hamiltonian is in the following form

            H_n = f(n) * h_n

        where f(n) is a dimensionless fluctuating parameter with zero mean and standard
        deviation of one.

            <f(t)>   = 0
            <f(t)^2> = 1

    """
    def __init__(self, H: Matrix, J: Noise, hbar: float = 1.0 / (2.0 * np.pi)):
        """
        Args:
            H (Matrix): Interaction between the system and the noise. h_n in the above equation.
            J (Noise): Noise spectral properties. See Noise class.
            hbar (float, optional): Defaults to 1.0/(2.0 * np.pi).
        """        
        super().__init__(H, hbar)
        self.J = J


class FlipFlopCavityHamiltonian(Hamiltonian):
    def __init__(self, electric_field_energy: Vector, zeeman_energy: Vector, tunnel_coupling: Vector, cavity_energy: float, cavity_occupation: int, cavity_coupling_strength: float):
        # Set constants
        self.dimensions = 4 ** len(electric_field_energy)
        self.num_qubits = len(electric_field_energy)
        self.hbar = 1.0 / (2.0 * np.pi)
        self.hyperfine = 0.117
        self.Delta = -0.00199
        # Compute 
        self.charge_mixing_angle = np.arctan2(tunnel_coupling, electric_field_energy)
        self.bare_charge_energy = np.sqrt(electric_field_energy ** 2 + tunnel_coupling ** 2)
        self.spin_energy = (
            zeeman_energy
            + 0.5 * self.Delta * zeeman_energy * (1.0 + np.cos(self.charge_mixing_angle))
            + self.hyperfine**2 / (8.0*zeeman_energy) * (1.0 - 2.0 * np.cos(self.charge_mixing_angle) + np.cos(self.charge_mixing_angle)**2 - zeeman_energy**2 * np.sin(self.charge_mixing_angle)**2 * ( 1.0 / (zeeman_energy**2 - self.bare_charge_energy**2) - self.Delta / (self.hyperfine * self.bare_charge_energy) ))
        )
        self.charge_energy = (
            self.bare_charge_energy
            + self.Delta * zeeman_energy * np.cos(self.charge_mixing_angle) / 2.0
            - self.hyperfine * np.cos(self.charge_mixing_angle) / 4.0
            + self.hyperfine**2 * (-np.cos(self.charge_mixing_angle)/(4.0*zeeman_energy) + np.sin(self.charge_mixing_angle)**2/(32.0*self.bare_charge_energy) + np.sin(self.charge_mixing_angle)**2/(16.0*(self.bare_charge_energy+zeeman_energy)) + np.sin(self.charge_mixing_angle)**2/(16.0*(self.bare_charge_energy-zeeman_energy)) )
            - self.hyperfine * self.Delta * zeeman_energy * np.sin(self.charge_mixing_angle)**2 / (8.0 * self.bare_charge_energy)
            + self.Delta**2 + zeeman_energy**2 * np.sin(self.charge_mixing_angle)**2 / (8.0 * self.bare_charge_energy)
        )
        self.charge_cavity_detuning = self.charge_energy - cavity_energy
        self.spin_cavity_detuning = self.spin_energy - cavity_energy
        # Compute coefficients of Z
        self.z03 = (
            self.hyperfine ** 2
            * self.bare_charge_energy ** 3
            * np.sin(self.charge_mixing_angle) ** 2
            * np.cos(self.charge_mixing_angle)
            / (4.0 * zeeman_energy * (self.bare_charge_energy ** 2 - zeeman_energy ** 2) ** 2)
        )
        self.z30 = np.cos(self.charge_mixing_angle) - self.hyperfine * np.sin(self.charge_mixing_angle) ** 2 / (4.0 * self.bare_charge_energy)
        self.z11 = (
            -self.hyperfine
            * self.bare_charge_energy
            * np.sin(self.charge_mixing_angle)
            * np.cos(self.charge_mixing_angle)
            / (2.0 * (self.bare_charge_energy ** 2 - zeeman_energy ** 2))
        )
        self.z22 = self.z11 * (self.bare_charge_energy / zeeman_energy)
        self.z31 = (
            self.hyperfine * self.bare_charge_energy * np.sin(self.charge_mixing_angle) ** 2 / (2.0 * (self.bare_charge_energy ** 2 - zeeman_energy ** 2))
        )
        self.z33 = self.Delta * zeeman_energy * np.sin(self.charge_mixing_angle) ** 2 / (2.0 * self.bare_charge_energy)
        self.z10 = np.sin(self.charge_mixing_angle) + self.hyperfine * np.cos(self.charge_mixing_angle) * np.sin(self.charge_mixing_angle) / (
            4.0 * self.bare_charge_energy
        )
        # Compute coupling strengths
        self.gt = cavity_coupling_strength * self.z31
        self.gs = cavity_coupling_strength * self.z10
        self.gp = cavity_coupling_strength * (self.z11 + self.z22)
        self.gm = cavity_coupling_strength * (self.z11 - self.z22)
        self.Ss = self.gs ** 2 / (2.0 * self.charge_cavity_detuning)
        self.St = self.gt ** 2 / (2.0 * self.spin_cavity_detuning)
        self.Jt = (0.5* self.gt[0]* self.gt[1]* (1.0 / self.spin_cavity_detuning[0] + 1.0 / self.spin_cavity_detuning[1]))
        # Set Hamiltonian
        self.H = cavity_energy * cavity_occupation * np.identity(2 ** self.num_qubits) \
                     - 0.5 * np.sum(self.charge_energy - 2 * (2 * cavity_occupation + 1) * self.Ss) * np.identity(2 ** self.num_qubits)
        for q in range(self.num_qubits):
            self.H -= (
                0.5
                * (self.spin_energy[q] - 2 * (2 * cavity_occupation + 1) * self.St[q])
                * Pauli2(3, q, self.num_qubits)
            )
        self.H = self.H.astype(np.complex128) + 0.5 * self.Jt * (
            Pauli4(1, 1, 0, 1) + Pauli4(2, 2, 0, 1)
        )


class ChargeNoiseHamiltonian(NoiseHamiltonian):
    def __init__(self,wn: float,z03: float,z33: float,z31: float,qubit: int,totalQubits: int,):
        self.gammatz = 0.5 * wn * z03
        self.gammastz = 0.5 * wn * z33
        self.gammat = 0.5 * wn * z31
        hbar = 1.0 / (2.0 * np.pi)
        super().__init__(
            (self.gammatz + self.gammastz) * Pauli2(3, qubit, totalQubits)
            + (self.gammat) * Pauli2(1, qubit, totalQubits),
            PinkNoise(),
            hbar,
        )
