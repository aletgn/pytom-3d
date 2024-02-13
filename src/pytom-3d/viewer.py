from core import Topography, Grid, summation, distance
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
            plt.xlabel('x-axis')
            plt.ylabel('z-axis')
             
            # YZ plane
            plt.subplot(223)
            plt.scatter(d.P[:,1], d.P[:,2], s=3)
            plt.title('yz plane')
            plt.xlabel('y')
            plt.ylabel('z')
        
        plt.subplots_adjust(wspace=0.8, hspace=0.8)
    
    def scatter3D(self, data: List[Topography]):
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        for d in data:
            ax.scatter3D(d.P[:,0], d.P[:,1], d.P[:,2])
        ax.set_xlabel("x")
        ax.set_xlabel("y")
        ax.set_xlabel("z")
        
        
        
        
        
if __name__ == "__main__":
    g = Grid()
    g.make([-5,5], [-5,5], 9, 9)
    g.add(distance, None)
    
    v = Viewer()
    
    v.views2D([g])
    
    