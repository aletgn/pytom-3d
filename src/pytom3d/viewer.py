from pytom3d.core import Topography
# from util import summation, distance, distance2
from matplotlib import pyplot as plt
from typing import List

class Viewer:
    
    def __init__(self, name: str = "unnamed"):
        self.name = name
    
    def views2D(self, data: List[Topography]):
        plt.figure(dpi=300)
        for d in data: 
            # XY plane
            plt.subplot(221)
            plt.scatter(d.P[:,0], d.P[:,1], s=3)
            plt.title('xy plane')
            plt.xlabel('x')
            plt.ylabel('y')
             
            # XZ plane
            plt.subplot(222)
            plt.scatter(d.P[:,0], d.P[:,2], s=3)
            plt.title('xz plane')
            plt.xlabel('x')
            plt.ylabel('z')
             
            # YZ plane
            plt.subplot(223)
            plt.scatter(d.P[:,1], d.P[:,2], s=3)
            plt.title('yz plane')
            plt.xlabel('y')
            plt.ylabel('z')
        plt.gcf().tight_layout(pad=1)
        plt.show()
    
    def scatter3D(self, data: List[Topography]):
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        for d in data:
            ax.scatter3D(d.P[:,0], d.P[:,1], d.P[:,2], s=3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()