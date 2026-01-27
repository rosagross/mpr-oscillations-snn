import seaborn as sns
import matplotlib as plt
import numpy as np
import tkinter as tk

def figure_style():
    """
    Set style for plotting figures.
    """
    sns.set_theme(style="ticks", context="paper",
            font="Arial",
            rc={"font.size": 11,
                "figure.titlesize": 11,
                "axes.titlesize": 11,
                "axes.labelsize": 11,
                "axes.linewidth": 0.5,
                "lines.linewidth": 1,
                "lines.markersize": 3,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "savefig.transparent": True,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                "xtick.minor.width": 0.5,
                "ytick.minor.width": 0.5,
                'legend.fontsize': 10,
                'legend.title_fontsize': 10,
                'legend.frameon': False,
                 })
    
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # choose colors
    colors = {
                'snn': sns.color_palette('Dark2')[0],
                'mpr': sns.color_palette('Dark2')[1],
                'snn_ref': sns.color_palette('Dark2')[2],
                'mpr_fit': sns.color_palette('Dark2')[3],
                'snn_exc': sns.color_palette('colorblind')[0],
                'snn_inh': sns.color_palette('colorblind')[1],
            }

    sns.despine(trim=True)
    screen_width = tk.Tk().winfo_screenwidth()
    dpi = screen_width / 10

    return colors, dpi