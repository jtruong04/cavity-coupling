import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

rc("text", usetex=True)
UBColors = ( "#005bbb", "#002f56", "#990000","#e56a54", "#6da04b", "#ad841f")


def plotEvolution(timeRange, rho, rhoNoNoise, stateLabels, includeLegend = True):
    rc("font", size=18)
    matplotlib.rcParams["font.serif"] = "Times New Roman"
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["axes.grid"] = True
    matplotlib.rcParams["grid.alpha"] = 0.5

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 8))

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    axs[0, 0].set_ylim([0, 1])
    axs[0, 0].set_xlim([0, timeRange[-1] + timeRange[1]])

    plt.tight_layout()
    axs[0, 0].set_title("Populations")
    axs[0, 1].set_title("Off-Diagonals")
    axs[0, 0].set_ylabel("Without Noise")
    axs[1, 0].set_ylabel("With Noise")
    axs[1, 0].set_xlabel(r"Time ($\mu$s)")
    axs[1, 1].set_xlabel(r"Time ($\mu$s)")

    for index, state in enumerate(stateLabels):
        if(state == ""):
            continue
        axs[0, 0].plot(
            timeRange,
            np.real(rhoNoNoise[:, index, index]),
            color=UBColors[index % len(UBColors)],
            linewidth=2.0,
            label=rf"$P_{{{state}}}$",
        )

    colorIndex = 0
    for i, si in enumerate(stateLabels):
        for j, sj in enumerate(stateLabels):
            if(si == "" or sj==""):
                continue
            if j <= i:
                continue
            axs[0, 1].plot(
                timeRange,
                np.abs(rhoNoNoise[:, i, j]),
                color=UBColors[colorIndex % len(UBColors)],
                linewidth=2.0,
                label=rf"$\rho_{{{si},{sj}}}$",
            )
            colorIndex += 1

    for index, state in enumerate(stateLabels):
        if(state == ""):
            continue
        axs[1, 0].plot(
            timeRange,
            np.real(rho[:, index, index]),
            color=UBColors[index % len(UBColors)],
            linewidth=2.0,
            label=rf"$P_{{{state}}}$",
        )

    colorIndex = 0
    for i, si in enumerate(stateLabels):
        for j, sj in enumerate(stateLabels):
            if(si == "" or sj==""):
                continue
            if j <= i:
                continue
            axs[1, 1].plot(
                timeRange,
                np.abs(rho[:, i, j]),
                color=UBColors[colorIndex % len(UBColors)],
                linewidth=2.0,
                label=rf"$\rho_{{{si},{sj}}}$",
            )
            colorIndex += 1

    if(includeLegend):
        axs[0, 0].legend(loc=1, framealpha=1)
        axs[0, 1].legend(loc=1, framealpha=1)
    
def plotEvolutionSingle(timeRange, rho, stateLabels, includeLegend = True):
    rc("font", size=24)
    matplotlib.rcParams["font.serif"] = "Times New Roman"
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["axes.grid"] = True
    matplotlib.rcParams["grid.alpha"] = 0.5

    fig, axs = plt.subplots(1,1, squeeze = False,sharex=True, sharey=True, figsize=(12, 8))

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    axs[0, 0].set_ylim([0, 1])
    axs[0, 0].set_xlim([0, timeRange[-1] + timeRange[1]])

    plt.tight_layout()
    # axs[0, 0].set_title("Populations")
    # axs[0, 1].set_title("Off-Diagonals")
    axs[0, 0].set_ylabel("Population")
    # axs[1, 0].set_ylabel("With Noise")
    axs[0, 0].set_xlabel(r"Time ($\mu$s)")
    # axs[1, 1].set_xlabel(r"Time ($\mu$s)")

    for index, state in enumerate(stateLabels):
        if(state == ""):
            continue
        axs[0, 0].plot(
            timeRange,
            np.real(rho[:, index, index]),
            color=UBColors[index % len(UBColors)],
            linewidth=2.0,
            label=rf"$P_{{{state}}}$",
        )

    # colorIndex = 0
    # for i, si in enumerate(stateLabels):
    #     for j, sj in enumerate(stateLabels):
    #         if(si == "" or sj==""):
    #             continue
    #         if j <= i:
    #             continue
    #         axs[0, 1].plot(
    #             timeRange,
    #             np.abs(rhoNoNoise[:, i, j]),
    #             color=UBColors[colorIndex % len(UBColors)],
    #             linewidth=2.0,
    #             label=rf"$\rho_{{{si},{sj}}}$",
    #         )
    #         colorIndex += 1

    # for index, state in enumerate(stateLabels):
    #     if(state == ""):
    #         continue
    #     axs[1, 0].plot(
    #         timeRange,
    #         np.real(rho[:, index, index]),
    #         color=UBColors[index % len(UBColors)],
    #         linewidth=2.0,
    #         label=rf"$P_{{{state}}}$",
    #     )

    # colorIndex = 0
    # for i, si in enumerate(stateLabels):
    #     for j, sj in enumerate(stateLabels):
    #         if(si == "" or sj==""):
    #             continue
    #         if j <= i:
    #             continue
    #         axs[1, 1].plot(
    #             timeRange,
    #             np.abs(rho[:, i, j]),
    #             color=UBColors[colorIndex % len(UBColors)],
    #             linewidth=2.0,
    #             label=rf"$\rho_{{{si},{sj}}}$",
    #         )
    #         colorIndex += 1

    if(includeLegend):
        axs[0, 0].legend(loc=1, framealpha=1)
    

def plotEvolutionDouble(timeRange, rho, rhoNoNoise, stateLabels, includeLegend = True):
    rc("font", size=18)
    matplotlib.rcParams["font.serif"] = "Times New Roman"
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["axes.grid"] = True
    matplotlib.rcParams["grid.alpha"] = 0.5
    styles = ['-', '-', '-', '-', '--', ':']
    fig, axs = plt.subplots(1,2, squeeze = False,sharex=True, sharey=True, figsize=(12, 4))

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    axs[0, 0].set_ylim([0, 1])
    axs[0, 0].set_xlim([0, timeRange[-1] + timeRange[1]])

    plt.tight_layout()
    axs[0, 0].set_title("Noiseless")
    axs[0, 1].set_title("Noisy")
    axs[0, 0].set_ylabel("Population")
    # axs[1, 0].set_ylabel("With Noise")
    axs[0, 0].set_xlabel(r"Time ($\mu$s)")
    axs[0, 1].set_xlabel(r"Time ($\mu$s)")

    for index, state in enumerate(stateLabels):
        if(state == ""):
            continue
        axs[0, 0].plot(
            timeRange,
            np.real(rhoNoNoise[:, index, index]),
            color=UBColors[index % len(UBColors)],
            linestyle=styles[index % len(styles)],
            linewidth=2.0,
            label=rf"$P_{{{state}}}$",
        )

    for index, state in enumerate(stateLabels):
        if(state == ""):
            continue
        axs[0, 1].plot(
            timeRange,
            np.real(rho[:, index, index]),
            color=UBColors[index % len(UBColors)],
            linestyle=styles[index % len(styles)],
            linewidth=2.0,
            label=rf"$P_{{{state}}}$",
        )

    if(includeLegend):
        axs[0, 1].legend(loc=1, framealpha=1)
    


