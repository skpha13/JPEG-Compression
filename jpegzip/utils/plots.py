import os

import matplotlib.pyplot as plt

os.makedirs(f"{os.getcwd()}/../plots", exist_ok=True)


def save_fig(fig: plt.Figure, plot_name: str) -> None:
    fig.savefig(fname=f"../plots/{plot_name}.png", format="png")
