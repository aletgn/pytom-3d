import numpy as np
# import pandas as pd
from numpy import cos, sin
from typing import List, Tuple
from pytom3d.util import update

class Topography:
    
    def __init__(self, name: str = "unnamed")-> None:
        """
        Initialize the Cloud instance.

        Parameters
        ----------
        name : str, optional
            The name of the instance, by default "unnamed".

        Returns
        -------
        None
        """
        self.name = name
        self.file_path = None
        self.P = None
        self.N = None
        self.m = None
        self.M = None
        self.G = None
        self.history_ = []
        
        self.regressor = None
        
    def read(self, file_path: str, reader: callable, **reader_args):
        """
        Read data from file.

        Parameters
        ----------
        file_path : str
            The path to the file.
        reader : callable
            A callable pandas reader function to read data from the file.
        **reader_args
            Additional arguments to pass to the reader.

        Returns
        -------
        None
        """
        self.file_path = file_path
        self.P = reader(self.file_path, **reader_args)
        
        try:
            self.P = reader(self.file_path, **reader_args).to_numpy(dtype=np.float64)
        except:
            pass
        
        self.cardinality()
        self.edges()
        self.centroid()
        
    def make_grid(self, x_bounds: List[float], y_bounds: List[float],
             x_res: int = 10, y_res: int = 10) -> None:
        """
        Initializes the grid within specified x and y bounds with given resolution.

        Parameters
        ----------
        x_bounds : list of float
            The bounds for the x-axis [x_min, x_max].
        y_bounds : list of float
            The bounds for the y-axis [y_min, y_max].
        x_res : int, optional
            The resolution of the grid along the x-axis (default is 10).
        y_res : int, optional
            The resolution of the grid along the y-axis (default is 10).
            
        Notes
        -----
        z-value is initialli set to zero.
        
        Returns
        -------
        None
        """
        x, y = np.meshgrid(np.linspace(x_bounds[0], x_bounds[1], x_res),
                            np.linspace(y_bounds[0], y_bounds[1], y_res))
        x = x.flatten()
        y = y.flatten()
        z = np.zeros(shape=x.shape)
        self.P = np.vstack([x,y,z]).T
        self.cardinality()
        self.edges()
        self.centroid()
        
    def add_points(self, fxy: callable, std_noise = None):
        """
        Adds a function-generated z-coordinate to the grid points.

        Parameters
        ----------
        fxy : callable
            A function that takes x and y coordinates and returns z.
        std_noise : float or None, optional
            Standard deviation of Gaussian noise to be added to z (default is None).

        Returns
        -------
        None
        """
        self.P[:,2] = fxy(self.P[:,0], self.P[:,1])
        if std_noise is not None:
            self.P[:,2] += np.random.normal(loc=0, scale=std_noise, size=self.P.shape[0])
        self.cardinality()
        self.edges()
        self.centroid()
        
    def get_topography(self, x, y, z, unc: np.ndarray = None):
        self.P = np.vstack([x, y, z]).T
        self.unc = unc
        self.cardinality()
        self.edges()
        self.centroid()
    
    def cardinality(self):
        self.N = self.P.shape[0]
    
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
    def translate(self, v: np.ndarray = np.array([0,0,0]), aux: bool = False) -> List[Tuple]:
        """
        Translate the data points by the given vector.

        Parameters
        ----------
        v : np.ndarray, optional
            The translation vector, by default np.array([0, 0, 0]).
        aux : bool, optional
            If True, indicates an auxiliary translation, by default False.
    
        Returns
        -------
        List[Tuple]
            A list containing information about the translation event.

        """
        self.P += v
        return [(len(self.history_), self.translate.__name__), ("vector", v)]
    
    @update
    def rotate(self, v: np.ndarray = np.array([0.,0.,0.]),
               t_deg: np.ndarray = np.array([0.,0.,0.]), rot_mat: np.ndarray = None) -> List[Tuple]:
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
        if rot_mat is not None:
            R = rot_mat        
        self.translate(v)
        self.P = np.matmul(self.P, R)
        self.translate(np.array([-h for h in v]))
        return [(len(self.history_), self.rotate.__name__),
                ("centre", v), ("angles", t_deg), ("rot_mat", R)]
    
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
        return [(len(self.history_), self.flip.__name__), ("flip", v)]
    
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
        # c2 = np.where(np.isclose(self.P[:, iax], lo, atol=tol))[0]
        # c3 = np.where(np.isclose(self.P[:, iax], up, atol=tol))[0]
        # met = np.concatenate([c1, c2, c3])
        met = c1
        if out:
            met = np.array(list(set(range(0, self.N)) - set(met)))
        
        self.P = self.P[met]
        if self.P.shape[0] == 0:
            raise ValueError("The cloud has no points.")
        else:        
            return [(len(self.history_), self.cut.__name__), ("axis", ax),
                    ("lo", lo),
                    ("up", up),
                    ("exterior", out)]
    @update
    def svd(self) -> List[Tuple]:
        """
        Perform Singular Value Decomposition (SVD) on the data points.
    
        Returns
        -------
        List[Tuple]
            A list containing information about the SVD event.
    
        Notes
        -----
        The SVD decomposes the data matrix into three matrices U, S, and V,
        such that P = U * S * V^T. The rotation is applied to align the
        principal components with the coordinate axes.
    
        """
        U,S,V = np.linalg.svd(self.P)
        self.rotate(-self.G, rot_mat=V)
        return [(len(self.history_), self.svd.__name__),
                ("U", U),
                ("V", V),
                ("S", S),
                ("det_S", np.linalg.det(V)),]   
    
    def history(self):
        """
        Print the event history.
    
        Prints each recorded event in the history list, displaying key-value pairs.
    
        Returns
        -------
        None
    
        Notes
        -----
        Each event is separated by a line of dashes for better readability.
        """
        for h in self.history_:
            print("-------------------------------------------------------")
            for k in h.keys():
                print( k, h[k])
            print("-------------------------------------------------------")
    
    def export(self, path_to_file: str, filename: str, extension: str = ".csv", delimiter: str = ",") -> None:
        """
        Export the grid points to a file in CSV format.
    
        Parameters
        ----------
        path_to_file : str
            The path to the directory where the file will be saved.
        filename : str
            The name of the file (without extension).
        extension : str, optional
            The file extension (default is ".csv").
        delimiter : str, optional
            The delimiter used in the CSV file (default is ",").
    
        Returns
        -------
        None
        """
        np.savetxt(path_to_file + filename + extension, self.P, delimiter=delimiter)

    def fit(self, regressor, **args):
        self.regressor = regressor
        self.regressor.fit(self.P[:,0:2], self.P[:,2])
        
    def pred(self, X):
        return self.regressor.predict(X, return_std=True)
    
    def __len__(self):
        return self.P.shape[0]
    
    def __repr__(self):
        s_name = f"NAME: {self.name}\n"
        s_len = f"LEN: {self.N}\n"
        s_min = f"MIN: {self.m}\n"
        s_max = f"MAX: {self.M}\n"
        s_g = f"G: {self.G}\n"
        s_ = f"{self.P}\n"
        return s_name+s_len+s_min+s_max+s_g+s_
    
    def __add__(self, topography):
        """
        Concatenate the points of the current grid with the points of another topography.
    
        Parameters
        ----------
        topography : Grid
            Another instance of the Grid class whose points will be concatenated with the current grid.
    
        Returns
        -------
        None
        """
        self.P = np.concatenate([self.P, topography.P])
