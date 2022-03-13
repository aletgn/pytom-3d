"""
    pyCloM (PoY(i)nt Cloud Manipulation) a Python module to manipulate
    three-dimensional point clouds.
    
    Copyright (C) 2022  Alessandro Tognan

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
# print(__doc__)

import numpy as np
from numpy import cos, sin
from fancy_log import milestone as ml
from fancy_log import single_operation as so
import matplotlib.pyplot as plt
from matplotlib import ticker

class point_cloud:
    def __init__(self, file_name, col_x, col_y, col_z, col_sep, **add_params):
        """
        Parameters
        ----------
        data_file : str
            name of the input file containing the data set of the point cloud. 
            file_name must be structured by columns. On the other hand, each 
            row of the file must identify a single sample of the point cloud.            
            Although file_name can include any number of columns, three of them 
            should clearly identify x-,y- and z-coordinates of the point cloud,
            regardless of their order. In fact, one can extract the 
            correspondent column by specifying the appropriate col_* index, 
            see below. Please note that the input file must not contain any 
            header line.
        col_x : int
            index of the column containing the x-coordinates of the point cloud.
        col_y : int
            index of the column containing the y-coordinates of the point cloud.
        col_z : int
            index of the column containing the z-coordinates of the point cloud.
        col_sep : str
            column separator
        add_params: dictionary
            this argument represents additional parameters. The following can 
            be specified:
                - cloud_label, in order to name the acquired point cloud

        Returns
        -------
        None.

        Notes
        ------
        For the sake of the numerical implementation bear in mind:
            - x -> 0;
            - y -> 1;
            - z -> 2.
        Raw data is stored in the attribute original_data, which is a Nx3 numpy
        array (N is the total number of rows in data_file).
        The attribute data contains the manipulated data, which is consistently
        updated throughout the manipulation process. 
        """
        self.file_name = file_name
        self.raw_data  = np.loadtxt(self.file_name, delimiter=col_sep)
        self.original_data = np.array([self.raw_data[:,col_x],
                                       self.raw_data[:,col_y],
                                       self.raw_data[:,col_z]]).T
        self.data = self.original_data
        if add_params.get('cloud_label'):
            self.cloud_label = add_params['cloud_label']
        else:
            self.cloud_label = file_name
        #
        # cloud extrema
        #
        self.x_min = None
        self.y_min = None
        self.z_min = None
        self.x_max = None
        self.y_max = None
        self.z_max = None
        #
        # cloud centroid
        #
        self.x_g = None
        self.y_g = None
        self.z_g = None

    def compute_extrema(self):
        """
        Description:
        -------
        This method allows for computing the extremum points of the point cloud
        with respect all the axis, i.e. x,y and z.

        Returns
        -------
        None.
        The extremum of the point cloud are stored (and updated throughout the
        manipulation) in *_min *_max, where * can assume either x or y or z.

        """
        log_level = 1
        self.x_min = self.data[:, 0].min()
        self.y_min = self.data[:, 1].min()
        self.z_min = self.data[:, 2].min()
        self.x_max = self.data[:, 0].max()
        self.y_max = self.data[:, 1].max()
        self.z_max = self.data[:, 2].max()
        ml('COMPUTING EXTREMA OF THE POINT CLOUD', log_level)
        so('x_min', log_level, self.x_min)
        so('x_max', log_level, self.x_max)
        so('y_min', log_level, self.y_min)
        so('y_max', log_level, self.y_max)
        so('z_min', log_level, self.z_min)
        so('z_max', log_level, self.z_max)

    def compute_centroid(self):
        """
        Description:
        -------
        This method allows for computing the centroid of the point cloud
        with respect all the axis, i.e. x,y and z.

        Returns
        -------
        None.
        The extremum of the point cloud are stored (and updated throughout the
        manipulation) in *_g where * can assume either x or y or z.

        """
        log_level = 1

        def coord_centroid(coord):
            return coord.sum()/len(coord)
        ml('COMPUTING THE CENTROID OF THE POINT CLOUD', log_level)
        self.x_g = coord_centroid(self.data[:, 0])
        self.y_g = coord_centroid(self.data[:, 1])
        self.z_g = coord_centroid(self.data[:, 2])
        so('x_g', log_level, self.x_g)
        so('y_g', log_level, self.y_g)
        so('z_g', log_level, self.z_g)

    def flip_cloud(self, ax):
        """
        Parameters
        ----------
        ax : int
            ax is an index that selects which axis to flip. In other words, 
            this method mirrors the data along the specified axis. The 
            following convention holds:
                - 0 -> flip x-axis;
                - 1 -> flip y-axis;
                - 2 -> flip z-axis.
        Returns
        -------
        None.

        Note
        ----
        At the end of the flipping operation, the extrema are updated

        """
        log_level = 1
        x_ax = 1
        y_ax = 1
        z_ax = 1
        if ax == 0:
            x_ax = -1
        elif ax == 1:
            y_ax = -1
        elif ax == 2:
            z_ax = -1
        else:
            raise Exception(
                f"The axis you specified ({ax:d}) doesn't belong to [0,1,2]")
        ml('FLIPPING DATA', log_level)
        self.data = np.array([[p[0]*x_ax, p[1]*y_ax, p[2]*z_ax]
                             for p in self.data])
        so('Data flipped along the axis', 1, ax)
        self.compute_extrema()
        self.compute_centroid()

    def translate_cloud(self, v):
        """
        Parameters
        ----------
        v : tuple,list, or numpy array representing the vector
                v = [x_t, y_t, z_t] in R^3
            The vector is summed to each row of data, thus performing a
            translation of the point cloud as follows:
                data[j,:] = data[j,:]+ v

        Returns
        -------
        None.

        """
        log_level = 1
        if isinstance(v, np.ndarray):
            pass
        else:
            v = np.array(v)
        if not len(v) == 3:
            raise Exception(
                f"Vector must be three dimensional. Specified dimension: {len(v):d}")
        ml('TRANSLATING DATA', log_level)
        self.data = np.array([[p[0]+v[0], p[1]+v[1], p[2]+v[2]]
                             for p in self.data])
        so('Data translated through the vector', 1, v)
        self.compute_extrema()
        self.compute_centroid()

    def rotate_cloud(self, pole, ax, angle):
        """
        Parameters
        ----------
        pole : tuple,list, or numpy array representing the vector
                v = [x_t, y_t, z_t] \in R^3
                which is the pole with respect to the point cloud is rotated
        ax : axis of rotation
            - 0 -> rotate about x-axis
            - 1 -> rotate about y-axis
            - 2 -> rotate about z-axis
        angle : float
                rotation angle in degrees

        Returns
        -------
        None.

        Note
        Firstly, the point cloud is translated to the pole (using 
        translate_cloud). Secondly the cloud is rotate with respect to the
        pole. Finally, the point cloud is restored to its original position by
        translating it through -pole.

        """
        log_level = 1
        if isinstance(pole, np.ndarray):
            pass
        else:
            pole = np.array(pole)
        if not len(pole) == 3:
            raise Exception(
                f"The pole must be three dimensional. Specified dimension: {len(pole):d}")
        angle = np.pi*angle/180
        if ax == 0:
            r = np.array([[1, 0, 0],
                          [0, cos(angle), sin(angle)],
                          [0, -sin(angle), cos(angle)]]).T
        elif ax == 1:
            r = np.array([[cos(angle), 0, sin(angle)],
                          [0, 1, 0],
                          [-sin(angle), 0, cos(angle)]]).T
        elif ax == 2:
            r = np.array([[cos(angle), sin(angle), 0],
                          [-sin(angle), cos(angle), 0],
                          [0, 0, 1]]).T
        else:
            raise Exception(
                f'{ax:d} is not a valid axis. Plese pass a value in [0,1,2]')
        ml('ROTATING DATA', log_level)
        self.data = np.array([np.matmul(r, p)
                             for p in self.data])
        so('Rotation matrix\n', 1, r)
        self.compute_extrema()
        self.compute_centroid()

    def cutoff_cloud(self, ax, value_range):
        """
        Parameters
        ----------
        ax : int
            index that select the axis for the cut-off
        value_range : list
            extrema of the interval in which data are supposed to be kept,
            thus not discarded
        Returns
        -------
        None.

        """
        log_level = 1
        tol = 0.001
        ml('CUTTING_OFF DATA', log_level)
        self.data = np.array(
            [p for p in self.data if p[ax] > value_range[0] and 
                                     p[ax] < value_range[1] or 
                                     abs(p[ax] - value_range[0]) < tol or 
                                     abs(p[ax] - value_range[1]) < tol])
        so('Keeping values in', 1, np.array(value_range))
        self.compute_extrema()
        self.compute_centroid()
        pass

def cloud_views_2d(*cloud_list):
    msize = 0.4
    fig, ax = plt.subplots(nrows=2, ncols=2, dpi=300)
    for c in cloud_list:
        coord = c.data
        ax[0,0].plot(coord[:,0],coord[:,1],'o',
                      markersize=msize,
                      picker=True,
                      pickradius=5,label = c.cloud_label)
        ax[0,0].set_xlabel('x [mm]')
        ax[0,0].set_ylabel('y [mm]')
        ax[0,0].grid('True')
        #
        ax[0,1].plot(coord[:,1],coord[:,2],'o',
                      markersize=msize,
                      picker=True,
                      pickradius=5)
        ax[0,1].set_xlabel('y [mm]')
        ax[0,1].set_ylabel('z [mm]')
        ax[0,1].grid('True')
        #
        ax[1,0].plot(coord[:,0],coord[:,2],'o',
                      markersize=msize,
                      picker=True,
                      pickradius=5)
        ax[1,0].set_xlabel('x [mm]')
        ax[1,0].set_ylabel('z [mm]')
        ax[1,0].grid('True')
    ax[0,0].legend(loc='upper center',ncol=len(cloud_list)+1, bbox_to_anchor=(0.5,1.5))
    fig.delaxes(ax[1,1])
    fig.tight_layout(pad=1)
    plt.show()
    
def cloud_view_3d(*cloud_list):
    fig = plt.figure(figsize=plt.figaspect(1.2),dpi=300)#figsize=plt.figaspect(1))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('$x_i$ [mm]')
    ax.set_ylabel('$y_i$ [mm]')
    ax.set_zlabel('$z_i$ [mm]')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(True)
    plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    plt.gca().zaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    for c in cloud_list:
        coord = c.data
        ax.scatter(coord[:,0], coord[:,1],coord[:,2],
                        s=0.1, #c=coord[:,2], cmap='jet',
                        marker="o",label=c.cloud_label)
    ax.legend(loc='upper center',ncol=len(cloud_list)+1)#, bbox_to_anchor=(0.5,1.5))
    plt.show()

if __name__ == '__main__':
    cloud = point_cloud('data_set.txt', 2, 1, 0, ',', cloud_label = 'cloud_1')
    cloud.compute_extrema()
    cloud.compute_centroid()
    cloud1 = point_cloud('data_set.txt', 1, 2, 0, ',')
    cloud1.flip_cloud(1)
    # cloud.flip_cloud(2)
    # cloud.translate_cloud(np.array([50,100,-100]))
    # cloud.rotate_cloud([0, 0, 0], 1, 0)
    cloud_view_3d(cloud,cloud1)
    # cloud.cutoff_cloud(1, [0, 25])
    cloud_views_2d(cloud,cloud1)
