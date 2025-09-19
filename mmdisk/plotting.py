from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np


def create_bree_diagram_figure(
    omega_0: float,
    dT_0: float,
    limits: tuple[float, float] = (4, 4),
    ax: Optional[plt.Axes] = None,
    labelsize: int = 16,
    ticksize: int = 14,
) -> tuple[plt.Axes, plt.Axes, plt.Axes]:
    """Create an empty plot for a Bree diagram for a given set of parameters.

    Args:
        omega_0 (float): Omega normalization factor.
        dT_0 (float): _description_
        limits (tuple, optional): Upper x and y axis limits. Defaults to (4, 4).
        ax (plt.Axes, optional): Axes object to plot the Bree diagram. If None,
                                 a new figure is created. Defaults to None.

    Returns:
        tuple[plt.Axes, plt.Axes, plt.Axes]: Main axe and secondary axes objects for the Bree diagram.
    """
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)

    ax.set_xlim(0, limits[0])
    ax.set_ylim(0, limits[1])

    ax.set_xlabel(r"$\omega^2 / \omega_{0}^{2}$", size=labelsize)
    ax.set_ylabel(r"$\Delta T / \Delta T_{0}$", size=labelsize)

    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1))
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.2))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=1))
    ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.2))
    ax.grid(which="major", color="lightgray", linestyle="-")
    ax.grid(which="minor", color="whitesmoke", linestyle="-")
    ax.tick_params(axis="x", labelsize=f"{ticksize}")
    ax.tick_params(axis="y", labelsize=f"{ticksize}")
    ax.set_axisbelow(True)
    ax.minorticks_on()

    ax_x = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    ax_x.set_xlabel(r"$\omega$ (rad/s)", size=labelsize)
    ax_x.tick_params(axis="x", labelsize=f"{ticksize}")
    ax_x.xaxis.set_major_locator(plticker.MultipleLocator(base=1))
    ax_x.xaxis.set_major_formatter(
        plticker.FuncFormatter(lambda x, _: f"{np.sqrt(np.abs(x))*omega_0:.0f}")
    )

    ax_y = ax.secondary_yaxis(
        "right", functions=(lambda x: x * dT_0, lambda x: x / dT_0)
    )
    ax_y.set_ylabel(r"$\Delta T$ (K)", size=labelsize)
    ax_y.tick_params(axis="y", labelsize=f"{ticksize}")
    ax_y.yaxis.set_major_locator(plticker.MultipleLocator(base=dT_0))

    ax.set_box_aspect(1)

    return ax, ax_x, ax_y
