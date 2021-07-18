from dataclasses import dataclass

import numpy as np

from .Noise import Noise

Matrix = np.ndarray
Vector = np.ndarray

class NoiseHamiltonian:
    hamiltonian: Matrix
    noise: Noise

@dataclass
class Diagram:
    operator: Matrix
    energy: float = 0
    label: str = ""

@dataclass
class StateLabel:
    label: str = ""
    index: int = 0
