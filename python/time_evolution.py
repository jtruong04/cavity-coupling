import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy.linalg import expm

from src.BraKet import Bra, Ket
from src.constants import *
from src.DensityMatrix import *
from src.Hamiltonian import *
from src.Noise_Pink import *
from src.types import Diagram, StateLabel
from src.utils import adjoint, commutator

UBColors = ("#005bbb", "#002f56", "#990000", "#e56a54", "#6da04b", "#ad841f","#00a69c","#006570","#ebec00")
LineStyles = ('-', '--', '-.', ':')
rc('text', usetex=True)
rc("font", size=18)
matplotlib.rcParams["font.serif"] = "Times New Roman"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["grid.alpha"] = 0.5


def plotEvolution(timeRange,
                    density_matrix=[None, None],
                    include_legend=True, 
                    states: StateLabel = [],
                    column_titles=['Populations', 'Coherences'],
                    row_labels=['Noiseless', 'Noisy'],
                    file_name="plot.png"):

    assert(len(density_matrix) == len(row_labels))
    assert(len(column_titles) == 2)

    fig, axs = plt.subplots(nrows=len(row_labels), ncols=len(column_titles), squeeze=False,
                            sharex=True, sharey=True, figsize=(12, 8))

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    axs[0, 0].set_ylim([0, 1])
    axs[0, 0].set_xlim([0, timeRange[-1] + timeRange[1]])
    plt.tight_layout()
    
    for col, col_title in enumerate(column_titles):
        axs[0, col].set_title(col_title)

    for row, col in [(row, col) for row in range(len(row_labels)) for col in range(len(column_titles))]:
        axs[row, col].set_ylabel(row_labels[row])
        axs[row, col].set_xlabel(r"Time ($\mu$s)")

    for ax in axs.flat:
        ax.label_outer()

    for row_index, density_matrix in enumerate(density_matrix):

        for plot_index, state in enumerate(states):
            axs[row_index, 0].plot(
                timeRange,
                np.real(density_matrix[:, state.index, state.index]),
                color=UBColors[plot_index % len(UBColors)],
                linestyle=LineStyles[plot_index % len(LineStyles)],
                linewidth=2.0,
                label=rf"$P_{{{state.label}}}$",
            )

        for plot_index, state_vals in enumerate([(state_row, state_col) for state_row in states for state_col in states]):
            (state_row, state_col) = state_vals
            if(state_row.index >= state_col.index):
                continue
            axs[row_index, 1].plot(
                timeRange,
                np.abs(density_matrix[:, state_row.index, state_col.index]),
                color=UBColors[plot_index % len(UBColors)],
                linestyle=LineStyles[plot_index % len(LineStyles)],
                linewidth=2.0,
                label=rf"$\rho_{{{state_row.label},{state_col.label}}}$",
            )
    if include_legend:
        axs[0, 0].legend(loc=1, framealpha=1)
        axs[0, -1].legend(loc=1, framealpha=1)
    
    fig.savefig(file_name)


def getRho(eps, wB, Vt, wc, N, g, wn, maxT=1000, numPoints=100):
    unperturbedHamiltonian = FlipFlopCavityHamiltonian(eps, wB, Vt, wc, N, g)
    numQubits = len(eps)

    hbar = unperturbedHamiltonian.hbar

    ws = unperturbedHamiltonian.charge_energy
    wt = unperturbedHamiltonian.spin_energy

    z03 = unperturbedHamiltonian.z03
    z33 = unperturbedHamiltonian.z33
    z31 = unperturbedHamiltonian.z31
    z11 = unperturbedHamiltonian.z11
    z22 = unperturbedHamiltonian.z22
    z10 = unperturbedHamiltonian.z10
    z30 = unperturbedHamiltonian.z30

    ws = unperturbedHamiltonian.charge_energy
    wt = unperturbedHamiltonian.spin_energy

    noiseInteractions = [ChargeNoiseHamiltonian(
        wn[q], z03[q], z33[q], z31[q], q, 2) for q in range(2)]
    noiseInteractions[0].H = np.kron(noiseInteractions[0].H, np.identity(4))
    noiseInteractions[1].H = np.kron(np.identity(4), noiseInteractions[1].H)

    gt = g * z31
    gs = g * z10
    gp = g * (z11 + z22)
    gm = g * (z11 - z22)
    gzs = g * z30
    gzt = g * z03
    gzst = g * z33

    diagrams = [[
        Diagram(Pauli4(3, 2, q, numQubits, use_pm=True) * gt[q]  ,       -wt[q]),
        Diagram(Pauli4(2, 0, q, numQubits, use_pm=True) * gs[q]  , -ws[q]      ),
        # Diagram(Pauli4(2, 1, q, numQubits, use_pm=True) * gp[q]  , -ws[q]+wt[q]),
        # Diagram(Pauli4(1, 2, q, numQubits, use_pm=True) * gp[q]  ,  ws[q]-wt[q]),
        # Diagram(Pauli4(2, 2, q, numQubits, use_pm=True) * gm[q]  , -ws[q]-wt[q]),
        # Diagram(Pauli4(0, 0, q, numQubits, use_pm=True) * g      ,       0     ),
        # Diagram(Pauli4(0, 3, q, numQubits, use_pm=True) * gzt[q] ,       0     ),
        # Diagram(Pauli4(3, 0, q, numQubits, use_pm=True) * gzs[q] ,       0     ),
        # Diagram(Pauli4(3, 3, q, numQubits, use_pm=True) * gzst[q],       0     ),
        # Diagram(Pauli4(3, 1, q, numQubits, use_pm=True) * gt[q]  ,        wt[q]),
        # Diagram(Pauli4(1, 0, q, numQubits, use_pm=True) * gs[q]  ,  ws[q]      ),
        # Diagram(Pauli4(1, 1, q, numQubits, use_pm=True) * gm[q]  ,  ws[q]+wt[q])
    ] for q in range(numQubits)]

    H0 = sum([0.5*ws[q]*Pauli4(3, 0, q, numQubits) +
             0.5*wt[q]*Pauli4(0, 3, q, numQubits) for q in range(numQubits)]) + hbar * wc * N * Pauli4(0, 0, 0, 2)

    psi0 = Ket(1, 16)
    psi0 /= np.linalg.norm(psi0)
    
    print("Cavity Energy: ", wc)
    print("Spin Energy", wt)
    print("Charge Energy", ws)

    for i,diagram_i in enumerate(diagrams[0]):
        for j,diagram_j in enumerate(diagrams[1]):
            print(i,j,)
            start = time.time()
            V1 = (adjoint(diagram_j.operator)@diagram_i.operator + commutator(adjoint(diagram_j.operator), diagram_i.operator)@Pauli4(0,0,0,2)*N) * 0.5 * (1.0/(diagram_i.energy + hbar*wc) +
                          1.0/(diagram_j.energy + hbar*wc))
            V2 = (adjoint(diagram_i.operator)@diagram_j.operator + commutator(adjoint(diagram_i.operator), diagram_j.operator)@Pauli4(0, 0, 0, 2)*N) * 0.5 * (1.0/(diagram_i.energy + hbar*wc) +
                                                                                                                                                  1.0/(diagram_j.energy + hbar*wc))
            effective_hamiltonian = Hamiltonian(H0 + V1 + V2, hbar)
            rho = DensityMatrix(psi0, effective_hamiltonian, noiseInteractions)
            dt = maxT/numPoints
            timeRange = np.arange(0, maxT, dt)
            rhoTNoNoise = rho(timeRange, False)
            rhoT = rho(timeRange, True)

            np.savetxt(f"data/rho_no_noise_{i}_{j}.csv", rhoTNoNoise.reshape(
                (numPoints, 16**numQubits)), delimiter=',')
            np.savetxt(f"data/rho_noisy_{i}_{j}.csv", rhoT.reshape(
                (numPoints, 16**numQubits)), delimiter=',')
            end = time.time()
            state_labels:List(StateLabel) = [
                StateLabel("00", 0),
                StateLabel("01", 1),
                # StateLabel("02", 2),
                # StateLabel("03", 3),
                StateLabel("10", 4),
                StateLabel("11", 5),
                # StateLabel("12", 6),
                # StateLabel("13", 7),
                # StateLabel("20", 8),
                # StateLabel("21", 9),
                # StateLabel("22", 10),
                # StateLabel("23", 11),
                # StateLabel("30", 12),
                # StateLabel("31", 13),
                # StateLabel("32", 14),
                # StateLabel("33", 15),
            ]
            plotEvolution(timeRange/1000, [rhoTNoNoise, rhoT], True,
                          state_labels, file_name=f"plots/rho_noise_{i}_{j}.png")
            print(end - start)


getRho(eps=np.array([0, 0]), wB=np.array([10.8, 10.8]), Vt=np.array([11.4, 11.4]), wc=10.8085, N=5, g=0.05, wn=np.array([0.323,0.323]), maxT=10000, numPoints=20)
