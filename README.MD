# poY(i)nt Cloud Manipulation (pyCloM)
pyCloM is simple Python module to ease the manipulation of 3-dimensional point clouds. It provides methods to perform traditional geometric transformations and visualise the results as well. The module is actually in its very early stage and it is currently under development. The documentation provided herein should still be revised; I apologise for typos and inaccuracies.

# Table of Contents
- [Requirements and Installation](#requirements-and-installation)
- [Point Cloud Initialisation](#point-cloud-initialisation)
- [Point Cloud Manipulation](#point-cloud-manipulation)
  * [Preliminary Inspection](#preliminary-inspection)
  * [Geometric Transformations](#geometric-transformations)
    - [Translation](#translation)
    - [Rigid Rotation](#rigid-rotation)
    - [Flip](#flip)
    - [Cut-off](#flip)
- [TO-DO List](#todo-list)

# Requirements and Installation <a name="requirements-and-installation"></a>
Before using pyCloM in your work, a few standar Python modules need to be installed: [```Numpy```](https://numpy.org/) and [```Matplotlib```](https://matplotlib.org/). For instance, if you're keen on [```pypi```](https://pypi.org/project/pip/), just run:

```
pip3 install --user numpy
pip3 install --user matplotlib
```

or else, choose the way you prefer.


Next, be sure to copy in your working directory the following files:
- pyCloM.py
- fancy_log.py

Finally, include ```cloud``` in your Python script:

```python
from pyCloM import point_cloud
```

which is the core of pyCloM module, and possibly the plotting functions:

```python
from pyCloM import cloud_views_2d
from pyCloM import cloud_view_3d
```

That's it, you are ready!

# Point Cloud Initialisation <a name="point-cloud-initialisation"></a>
You should have available a 3-dimensional point cloud. If not, use ```synthetic_data.py``` to generate a fictitious point cloud. The correspondent data should be set up in a file looking like this:

```
...
-31.8367,-50.0000,59.2754
-28.3673,-50.0000,57.4866
-24.8980,-50.0000,55.8561
...
```

Importantly, each column must be clearly separated by a specific character, e.g. commas. The point is used as a decimal separator, so it must not be used as a column separator. The column separator must be indicated when initialising ```point_cloud``` objects.

The file containing the data, can included any number of columns, since those that have to be read are picked through a specific index.

Let us read the point cloud:

```python
cl = point_cloud('data_set.txt', 1, 2, 0, ',')
```

where:
- ```data_set.txt``` is the file of the point cloud
- ```1,2,0``` are the column selectors, which correspond to ```x```, ```y``` and ```z``` coordinates of the point cloud, respectively
- ```,``` is the column separator

In this instance it is worth observing that the column:
- ```1``` will be considered as the column containing ```x``` data 
- ```2``` will be considered as the column containing ```y``` data
- ```0``` will be considered as the column containing ```z``` data

Therefore, it does not matter how the columns are sorted in ```data_set.txt```, since the user can select them appropriately.

The raw coordinates of the point cloud have been acquired, and they are now stored in the attribute:

```python
cl.raw_data
```

Furthermore, a copy of ```original_data``` is also stored in:

```python
cl.data
```

and this attribute is consistently updated throughout the manipulations/transformation of the point cloud. Both ```raw_data``` and ```data``` are treated as a ```numpy array``` of dimension ```N x 3``` where ```N``` is the number of acquired points.


# Point Cloud Manipulation <a name="point-cloud-manipulation"></a>

## Preliminary Inspection <a name="preliminary-inspection"></a>

Once the data has been acquired, it is possible to check how many points are there. This is accomplished by the (overloaded) function ```__len__```:

```python
len(cloud)
```

Also, one can quickly display the result through:

```python
cloud_view_3d(cl)
```

and display different views of the point cloud by using:

```python
cloud_views_2d(cl)
```

Additionally, it is possible to compute both the extrema (min/max of x,y, and z) and the centroid of the point cloud; just use:

```python
cl.compute_extrema()
cl.compute_centroid()
```

The first method will store (or update) the extrema of the point cloud in:

```python
data.x_min
data.y_min
data.z_min
data.x_max
data.y_max
data.z_max
```

whereas the second will store (or update) the coordinates of the centroid of the point cloud in:

```python
data.x_g
data.y_g
data.z_g
```

and the related outcomes will be displayed in the terminal.

## Geometric Transformations <a name="geometric-transformations"></a>

pyCloM enables the following geometric transformation to be performed:
- Translation
- Rotation
- Flip
- Cut-off

In order to understand each geometric transformation remember that ```data``` is a ```N x 3``` matrix, say **M**, and assume **M**<sub>i</sub> represents the i-th row of **M**. Therefore, **M**<sub>i</sub> is the triplet [x<sub>i</sub>,y<sub>i</sub>,z<sub>i</sub>] which corresponds to the i-th point of the cloud.

### Translation <a name="translation"></a>
Let us define a vector **v** = [x<sub>v</sub>,y<sub>v</sub>,z<sub>v</sub>]. The method:

```python
cl.translate_cloud(v)
```

performs a rigid translation of the point cloud by adding **v** to each **M**<sub>i</sub>:

**M**<sub>i</sub>   <-|   **M**<sub>i</sub> + **v** for all i

and updates the point cloud stored in ```data```.

### Rigid Rotation <a name="rigid-rotation"></a>
Let **v** = [x<sub>p</sub>,y<sub>p</sub>,z<sub>p</sub>], *a* and *ax* be a pole, and angle (in degrees) and the axis identifier (0=x, 1=y, 2=z), respectively. Furthermore, according to the chosen axis, assume the correspondent rotation matrix **R**(*a*). The method:

```python
cl.rotate_cloud(p, ax, a)
```
performs a rigid rotation of the point cloud about the *ax* axis, with respect to the pole **p**,  by rotating **M**<sub>i</sub> in agreement with **R**(*a*):

**M**<sub>i</sub>   <-|   **R**(*a*)[**M**<sub>i</sub>] for all i

and updates the point cloud stored in ```data```.

### Flip <a name="flip"></a>
This method allows flipping to be performed. Herein, flipping indicates a reflection of the point cloud with respect to a selected axis. Therefore, the coordinate of each point along the selected axis will be replaced by their inverse:
- x |-> -x
- y |-> -y
- z |-> -z

You can exploit this method by using:

```python
cl.flip_cloud(ax)
```

where *ax* is the selected axis, which can assume either 0 (x) or 1 (y) or 2 (z). Formally, let **M**<sub>i</sub>=[x<sub>i</sub>,y<sub>i</sub>,z<sub>i</sub>] the i-th point of the point cloud **M**. The method in question carries out the following operation:
- [x<sub>i</sub>,y<sub>i</sub>,z<sub>i</sub>] <-| [-x<sub>i</sub>,y<sub>i</sub>,z<sub>i</sub>] for all i
- [x<sub>i</sub>,y<sub>i</sub>,z<sub>i</sub>] <-| [x<sub>i</sub>,-y<sub>i</sub>,z<sub>i</sub>] for all i
- [x<sub>i</sub>,y<sub>i</sub>,z<sub>i</sub>] <-| [x<sub>i</sub>,y<sub>i</sub>,-z<sub>i</sub>] for all i

in agreement with the selected axis.


### Cut-off <a name="cut-off"></a>
Suppose you would like to remove some outliers from your point cloud. For the sake of clarity consider the z-axis, the same procedure applies to the other axes identically. Also, assume two threshold values gathered in the following vector [z<sub>min</sub>, z<sub>max</sub>]. This method enables to keep those points whose z-value belongs to [z<sub>min</sub>, z<sub>max</sub>], thus peforming a cutt-off.

In order to utilise this method, use:
```python
cl.cutoff_cloud(ax, [ax_min,ax_max])
```

where ```ax``` is the reference axis for the cut-off and ```[ax_min,ax_max]``` is the vector of the threshold values.


## TO-DO List <a name="todo-list"></a>
- [ ] Enhance data visualisation
- [ ] Add methods to export data
