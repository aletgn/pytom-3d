from pytom3d.core import Topography
# from util import summation, distance, distance2
from matplotlib import pyplot as plt
from typing import List
import matplotlib
from matplotlib import ticker
import numpy as np

class Viewer:
    
    def __init__(self, name: str = "unnamed") -> None:
        self.name = name
    
    def views2D(self, data: List[Topography]) -> None:
        """
        Generate 2D scatter plots for the XY, XZ, and YZ planes of multiple Topography objects.

        Parameters
        ----------
        data : List[Topography]
            A list of Topography objects for which 2D scatter plots will be generated.

        Returns
        -------
        None

        """
        plt.figure(dpi=300)
        for d in data: 
            # XY plane
            plt.subplot(221)
            plt.scatter(d.P[:,0], d.P[:,1], s=3, alpha=1)
            plt.title('xy plane')
            plt.xlabel('x')
            plt.ylabel('y')
             
            # XZ plane
            plt.subplot(222)
            plt.scatter(d.P[:,0], d.P[:,2], s=3, alpha=1)
            plt.title('xz plane')
            plt.xlabel('x')
            plt.ylabel('z')
             
            # YZ plane
            plt.subplot(223)
            plt.scatter(d.P[:,1], d.P[:,2], s=3, alpha=1)
            plt.title('yz plane')
            plt.xlabel('y')
            plt.ylabel('z')
        plt.gcf().tight_layout(pad=1)
        plt.show()
    
    def scatter3D(self, data: List[Topography], x_lim: List[float] = None,
                  y_lim: List[float] = None, z_lim: List[float] = None, colour = False) -> None:
        """
        Generate a 3D scatter plot for the given Topography data.

        Parameters
        ----------
        data : List[Topography]
            A list of Topography objects for which a 3D scatter plot will be generated.
        x_lim : List[float], optional
            Limits for the x-axis. Default is None.
        y_lim : List[float], optional
            Limits for the y-axis. Default is None.
        z_lim : List[float], optional
            Limits for the z-axis. Default is None.
        colour : bool, optional
            If True, use color mapping based on z-coordinate. Default is False.

        Returns
        -------
        None

        """
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if z_lim is not None:
            ax.set_zlim(z_lim)
            vmin = z_lim[0]
            vmax = z_lim[1]
        else:
            vmin = np.array([h.m[2] for h in data]).min()
            vmax = np.array([h.M[2] for h in data]).max()
            ax.set_zlim([vmin, vmax])

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")

        ax.xaxis.pane.set_color('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.xaxis.pane.set_color('w')
        ax.yaxis.pane.set_color('w')
        ax.zaxis.pane.set_color('w')

        plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        plt.gca().zaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
    
        ax.grid(True)

        for d in data:
            sc = ax.scatter3D(d.P[:, 0], d.P[:, 1], d.P[:, 2], s=2, alpha=1,
                              vmin=vmin, vmax=vmax, c=d.P[:, 2])

        cbar = fig.colorbar(sc, ax=ax, orientation="vertical",
                            pad=0.12, format="%.2f",
                            ticks=list(np.linspace(vmin,
                                                   vmax, 11)),
                            label='Altitude')
        cbar.ax.tick_params(direction='in', right=1, left=1, size=2.5)
        
        ax.axis('tight')
        plt.show()

    def scatter3DRegression(self, regression: Topography, reference: Topography = None) -> None:
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")

        ax.xaxis.pane.set_color('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.xaxis.pane.set_color('w')
        ax.yaxis.pane.set_color('w')
        ax.zaxis.pane.set_color('w')

        plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        plt.gca().zaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))

        ax.grid(True)
        vmin = regression.unc.min()
        vmax = regression.unc.max()

        # ax.scatter3D(regression.P[:, 0], regression.P[:, 1], regression.P[:, 2], s=2, alpha=1, c=regression.unc)
        ax.plot_trisurf(regression.P[:, 0], regression.P[:, 1], regression.P[:, 2],
                             alpha=1, cmap="RdYlBu", edgecolor=None, antialiased=True)

        sm = plt.cm.ScalarMappable(cmap="RdYlBu")
        sm.set_array(regression.unc)

        cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                            pad=0.12, format="%.3f",
                            ticks=list(np.linspace(vmin, vmax, 11)),
                            label='Uncertainty')
        cbar.ax.tick_params(direction='in', right=1, left=1, size=2.5)
        
        if reference is not None:
            ax.scatter3D(reference.P[:, 0], reference.P[:, 1], reference.P[:, 2], s=2, alpha=1)

        ax.axis('tight')
        plt.show()

def cfg_matplotlib(font_size: int = 12, font_family: str = 'sans-serif', use_latex: bool = False) -> None:
    """
    Set Matplotlib RC parameters for font size, font family, and LaTeX usage.

    Parameters
    ----------
    font_size : int, optional
        Font size. The default is 12.

    font_family : str, optional
        Font family. The default is 'sans-serif'.

    use_latex : bool, optional
        Enable LaTeX text rendering. The default is False.

    Returns
    -------
    None

    """
    matplotlib.rcParams['font.size'] = font_size
    matplotlib.rcParams['font.family'] = font_family
    matplotlib.rcParams['text.usetex'] = use_latex
    