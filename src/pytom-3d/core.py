import numpy as np
import pandas as pd
from numpy import cos, sin
from typing import List, Tuple
from util import update
# from numpy.linalg import svd, det
# from fancy_log import milestone as ml
# from fancy_log import single_operation as so
# import matplotlib.pyplot as plt
# from matplotlib import ticker

class Cloud:
    
    def __init__(self, file_path: str, reader: callable, name: str = "unnamed", **reader_args) -> None:
        """
        Initialize the Cloud instance.

        Parameters
        ----------
        file_path : str
            The path to the file.
        reader : callable
            A callable pandas reader function to read data from the file.
        name : str, optional
            The name of the instance, by default "unnamed".
        **reader_args
            Additional arguments to pass to the reader.

        Returns
        -------
        None
        """
        self.name = name
        self.file_path = file_path
        self.P = reader(self.file_path, **reader_args).to_numpy(dtype=np.float128)
        self.N = self.P.shape[0]
        self.m = None
        self.M = None
        self.G = None
        self.events = []
    
    def edges(self) -> None:
        """
        Update the minimum and maximum values along each dimension.

        Returns
        -------
        None
        """
        self.m = self.P.min(axis=0)
        self.M = self.P.max(axis=0)
        
    def centroid(self) -> None:
        """
        Update the centroid of the data.

        Returns
        -------
        None
        """
        self.G = self.P.sum(axis=0)/self.N
    
    @update
    def translate(self, v: np.ndarray = np.array([0,0,0])) -> List[Tuple]:
        """
        Translate the data points by the given vector.
    
        Parameters
        ----------
        v : np.ndarray, optional
            The translation vector, by default np.array([0, 0, 0]).
    
        Returns
        -------
        List[Tuple]
            A list containing information about the translation event.
        """
        self.P += v
        return [(len(self.events), self.translate.__name__), ("vector", v)]
    
    @update
    def rotate(self, v: np.ndarray = np.array([0.,0.,0.]),
               t_deg: np.ndarray = np.array([0.,0.,0.])) -> List[Tuple]:
        """
        Rotate the data points about a specified center.
    
        Parameters
        ----------
        v : np.ndarray, optional
            The center of rotation, by default np.array([0., 0., 0.]).
        t_deg : np.ndarray, optional
            The rotation angles in degrees around x, y, and z axes,
            by default np.array([0., 0., 0.]).
    
        Returns
        -------
        List[Tuple]
            A list containing information about the rotation event.

        """
        t = np.deg2rad(t_deg)
        
        rx = np.array([[1, 0, 0],
                      [0, cos(t[0]), sin(t[0])],
                      [0, -sin(t[0]), cos(t[0])]])
        
        ry = np.array([[cos(t[1]), 0, sin(t[1])],
                      [0, 1, 0],
                      [-sin(t[1]), 0, cos(t[1])]])
        
        rz = np.array([[cos(t[2]), sin(t[2]), 0],
                      [-sin(t[2]), cos(t[2]), 0],
                      [0, 0, 1]])
        
        R = np.matmul(np.matmul(rx, ry), rz)
        self.translate(v)
        np.matmul(self.P, R)
        self.translate(np.array([-h for h in v]))
        return [(len(self.events), self.rotate.__name__),
                ("centre", v), ("angles", t_deg)]
    
    @update
    def flip(self, v: np.ndarray = np.array([1.,1.,1.])) -> List[Tuple]:
        """
        Flip the data points along each axis.
    
        Parameters
        ----------
        v : np.ndarray, optional
            The scaling factors along each axis, by default np.array([1., 1., 1.]).
            
        Returns
        -------
        List[Tuple]
            A list containing information about the flip event.
    
        """
        self.P *= v
        return [(len(self.events), self.flip.__name__), ("flip", v)]
    
    @update
    def cut(self, ax: str = None, lo: float = -np.inf, up: float = np.inf,
            tol=1e-8, out=False) -> None:
        """
        Cut data points along a specified axis within a given range.
    
        Parameters
        ----------
        ax : str, optional
            The axis to cut along (choose from 'x', 'y', 'z'), by default None.
        lo : float, optional
            The lower bound for cutting, by default -np.inf.
        up : float, optional
            The upper bound for cutting, by default np.inf.
        tol : float, optional
            The tolerance for considering values close to bounds, by default 1e-8.
        out : bool, optional
            If True, keep the points outside the specified range, by default False.
    
        Returns
        -------
        List[Tuple]
            A list containing information about the cut event.
    
        Raises
        ------
        KeyError
            If the specified axis is not valid.
        ValueError
            If the resulting cloud has no points after cutting.
    
        """
        ax2id = {"x": 0, "y": 1, "z": 2}
        try:
            iax = ax2id[ax]
        except KeyError as KE:
            raise KeyError("Axis is not valid") from KE
        
        c1 = np.where((self.P[:, iax] > lo) & (self.P[:, iax] < up))[0]
        c2 = np.where(np.isclose(self.P[:, iax], lo, atol=tol))[0]
        c3 = np.where(np.isclose(self.P[:, iax], up, atol=tol))[0]
        met = np.concatenate([c1, c2, c3])
        if out:
            met = np.array(list(set(range(0, self.N)) - set(met)))
        
        self.P = self.P[met]
        if self.P.shape[0] == 0:
            raise ValueError("The cloud has no points.")
        else:        
            return [(len(self.events), self.cut.__name__), ("axis", ax),
                    ("lower bnd", lo),
                    ("upper bnd", up),
                    ("exterior", out)]
    
    def history(self):
        for h in self.events:
            print(h)
    
    def __len__(self):
        return self.P.shape[0]
    
    def __repr__(self):
        s_name = f"NAME: {self.name}\n"
        s_len = f"LEN: {self.N}\n"
        s_min = f"MIN: {self.m}\n"
        s_max = f"MAX: {self.M}\n"
        s_ = f"{self.P}\n"
        return s_name+s_len+s_min+s_max+s_

def main():
    pass

if __name__ == "__main__":
    
    c = Cloud("../../data/test_data.csv", pd.read_csv, name="test",
              header=None, index_col=False)
    # print(c)
    c.translate(v=[1.,2.,3.])
    # print(c)
    c.flip([1.,-1.,-1.])
    # print(c)
    c.rotate([0,0,0], [90, 90 ,0])
    # print(c)
    c.cut("y")
    # print(c)
    print(c)
    c.history()
# class point_cloud:
#     def __init__(self, file_name, col_x, col_y, col_z, col_sep, **add_params):
#         """
#         Parameters
#         ----------
#         data_file : str
#             name of the input file containing the data set of the point cloud. 
#             file_name must be structured by columns. On the other hand, each 
#             row of the file must identify a single sample of the point cloud.            
#             Although file_name can include any number of columns, three of them 
#             should clearly identify x-,y- and z-coordinates of the point cloud,
#             regardless of their order. In fact, one can extract the 
#             correspondent column by specifying the appropriate col_* index, 
#             see below. Please note that the input file must not contain any 
#             header line.
#         col_x : int
#             index of the column containing the x-coordinates of the point cloud.
#         col_y : int
#             index of the column containing the y-coordinates of the point cloud.
#         col_z : int
#             index of the column containing the z-coordinates of the point cloud.
#         col_sep : str
#             column separator
#         add_params: dictionary
#             this argument represents additional parameters. The following can 
#             be specified:
#                 - cloud_label, in order to name the acquired point cloud

#         Returns
#         -------
#         None.

#         Notes
#         ------
#         For the sake of the numerical implementation bear in mind:
#             - x -> 0;
#             - y -> 1;
#             - z -> 2.
#         Raw data is stored in the attribute raw_data, which is a Nx3 numpy
#         array (N is the total number of rows in data_file).
#         The attribute data contains the manipulated data, which is consistently
#         updated throughout the manipulation process. 
#         """
#         self.file_name = file_name
#         self.raw_data  = np.loadtxt(self.file_name, delimiter=col_sep)
#         self.data = np.array([self.raw_data[:,col_x],
#                               self.raw_data[:,col_y],
#                               self.raw_data[:,col_z]]).T
#         if add_params.get('cloud_label'):
#             self.cloud_label = add_params['cloud_label']
#         else:
#             self.cloud_label = file_name
#         #
#         # cloud extrema
#         #
#         self.x_min = None
#         self.y_min = None
#         self.z_min = None
#         self.x_max = None
#         self.y_max = None
#         self.z_max = None
#         #
#         # cloud centroid
#         #
#         self.x_g = None
#         self.y_g = None
#         self.z_g = None       

    
#     def cloud_svd(self):
#         """
#         Let the point cloud data stored in self.data be represented by the 
#         matrix M. This method carries out a sequence of operations:                
        
#         1) update centroid 
#         2) translation to the -centroid
#         3) Singular Value Decomposition (SVD): M = U*S*V
#         4) Rotation according to the right principal matrix (V)
#         5) translation to centroid to restore the cloud position 
    
#         Returns
#         -------
#         None.

#         """
#         log_level = 1
#         self.translate_cloud(-np.array([self.x_g,self.y_g,self.z_g]))
#         ml('PERFORMING SINGULAR VALUE DECOMPOSITION', log_level)
#         U,S,V = svd(self.data)
#         so('Right principal matrix\n', log_level,V)
#         so('Determinant of right principal matrix', log_level, det(V))
#         so('Singular values', log_level, S)
#         self.data = np.array([np.matmul(V,p) for p in self.data])
#         self.translate_cloud(np.array([self.x_g,self.y_g,self.z_g]))
#         self.compute_extrema()
#         self.compute_centroid()
        

# def cloud_views_2d(*cloud_list):
#     msize = 0.4
#     fig, ax = plt.subplots(nrows=2, ncols=2, dpi=300)
#     for c in cloud_list:
#         coord = c.data
#         ax[0,0].plot(coord[:,0],coord[:,1],'o',
#                       markersize=msize,
#                       picker=True,
#                       pickradius=5,label = c.cloud_label)
#         ax[0,0].set_xlabel('x [mm]')
#         ax[0,0].set_ylabel('y [mm]')
#         ax[0,0].grid('True')
#         #
#         ax[0,1].plot(coord[:,1],coord[:,2],'o',
#                       markersize=msize,
#                       picker=True,
#                       pickradius=5)
#         ax[0,1].set_xlabel('y [mm]')
#         ax[0,1].set_ylabel('z [mm]')
#         ax[0,1].grid('True')
#         #
#         ax[1,0].plot(coord[:,0],coord[:,2],'o',
#                       markersize=msize,
#                       picker=True,
#                       pickradius=5)
#         ax[1,0].set_xlabel('x [mm]')
#         ax[1,0].set_ylabel('z [mm]')
#         ax[1,0].grid('True')
#     ax[0,0].legend(loc='upper center',ncol=len(cloud_list)+1, bbox_to_anchor=(0.5,1.5))
#     fig.delaxes(ax[1,1])
#     fig.tight_layout(pad=1)
#     plt.show()
    
# def cloud_view_3d(*cloud_list):
#     fig = plt.figure(figsize=plt.figaspect(1.2),dpi=300)#figsize=plt.figaspect(1))
#     ax = fig.add_subplot(1, 1, 1, projection='3d')
#     ax.set_xlabel('$x_i$ [mm]')
#     ax.set_ylabel('$y_i$ [mm]')
#     ax.set_zlabel('$z_i$ [mm]')
#     ax.xaxis.pane.fill = False
#     ax.yaxis.pane.fill = False
#     ax.zaxis.pane.fill = False
#     ax.xaxis.pane.set_edgecolor('w')
#     ax.yaxis.pane.set_edgecolor('w')
#     ax.zaxis.pane.set_edgecolor('w')
#     ax.grid(True)
#     plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
#     plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
#     plt.gca().zaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
#     for c in cloud_list:
#         coord = c.data
#         ax.scatter(coord[:,0], coord[:,1],coord[:,2],
#                         s=0.1, #c=coord[:,2], cmap='jet',
#                         marker="o",label=c.cloud_label)
#     ax.legend(loc='upper center',ncol=len(cloud_list)+1)#, bbox_to_anchor=(0.5,1.5))
#     plt.show()

# if __name__ == '__main__':
#     cloud = point_cloud('data_set.txt', 2, 1, 0, ',', cloud_label = 'cloud_1')
#     cloud.compute_extrema()
#     cloud.compute_centroid()
#     cloud1 = point_cloud('data_set.txt', 2, 1, 0, ',')
#     cloud1.flip_cloud(1)
#     # cloud.flip_cloud(2)
#     # cloud.translate_cloud(np.array([50,100,-100]))
#     # cloud.rotate_cloud([0, 0, 0], 1, 0)
#     cloud_view_3d(cloud,cloud1)
#     cloud.cloud_svd()
#     cloud_view_3d(cloud,cloud1)
#     # cloud.cutoff_cloud(1, [0, 25])
#     cloud_views_2d(cloud,cloud1) 
