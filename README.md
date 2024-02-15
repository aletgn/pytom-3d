# PyToM-3D: Python Topography Manipulator in 3D

PyToM-3D is a Python package to transform and fit 3-dimensional topographies. It provides methods to perform traditional geometric transformations and visualise the results as well.

# Table of Contents

[Geometric Transformations](#geometric-transformations)

- [Translation](#translation)
  
  [Rigid Rotation](#rigid-rotation)
  
  [Flip](#flip)
  
  [Cut-off](#flip)
  
  [Algebraic Transformation](#algebraic-transformation)
  
  [Singular Value Decomposition (SVD)](#svd)
  
  - [Brief Definition](#svd-def)
  - [Using SVD](#using-svd) 

- [TO-DO List](#todo-list)

# # Initialising a Topography

Initially we need to istantiate a `Topography`:

```python
t = Topography()
```

We have a multitude of options:

- reading from a `.csv` file;

- generating a grid and add synthetic data.

# Geometric Transformations <a name="geometric-transformations"></a>

In order to understand each geometric transformation remember that the points of the topography are stored in the attribute ```P```, which is a $N\times 3$ matrix, say $\mathbf{P}=\left[\mathbf{p}_1\quad \mathbf{p}_2\quad\mathbf{p}_N\right]^\top$, where $\mathbf{p}_i = \left[x_i\quad y_i \quad z_i\right]$ is the *i*th point of the topography. In, practice the altitude $z_i$ is determined upon a latent function of the coordinates, hence $z_i = f(x_i, y_i)$.

## Translation <a name="translation"></a>

Let us define a vector $\mathbf{v} = \left[x_v\quad y_v\quad z_v \right]$. By calling:

```python
t.translate([xv,yv,zv])
```

we translate $\mathbf{P}$ by $\mathbf{v}$ as:

$\mathbf{p}_i \leftarrow \mathbf{p}_i + \mathbf{v}\quad\forall\ i = 1,2,\dots,N.$

### Rigid Rotation <a name="rigid-rotation"></a>

Let **v** = [x<sub>p</sub>,y<sub>p</sub>,z<sub>p</sub>], *a* and *ax* be a pole, and angle (in degrees) and the axis identifier (0=x, 1=y, 2=z), respectively. Furthermore, according to the chosen axis, assume the correspondent rotation matrix **R**(*a*). The method:

```python
cl.rotate_cloud(p, ax, a)
```

performs a rigid rotation of the point cloud about the *ax* axis, with respect to the pole **p**,  by rotating **M**<sub>i</sub> in agreement with **R**(*a*):

**M**<sub>i</sub>   <-|   **R**(*a*)[**M**<sub>i</sub>] for all i

and updates the point cloud stored in ```data```.

## Flip <a name="flip"></a>

This method allows mirroring data with respect to a given vector $\mathbf{v}=\left[v_x\quad v_y\quad v_z \right]$ which represent the outward normal of the intended flipping plane. Therefore each component must be $-1$ or $1$. Assuming one wishes to flip the data about the $yz$ plane, they would define $\mathbf{v}=\left[-1\quad 1\quad 1 \right]$, and call:

```python
t.flip([-1,1,1])
```

This operation performs $(\mathbf{p}_i = \left[x_i\quad y_i \quad z_i\right])$:

$\left[x_i\quad y_i \quad z_i\right] \leftarrow \left[x_i\cdot v_x\quad y_i\cdot vy\quad z_i\cdot vz\right] \quad\forall\ i = 1,2,\dots,N.$

## Cut-off <a name="cut-off"></a>

Suppose you would like to remove some outliers from your point cloud. For the sake of clarity consider the z-axis, the same procedure applies to the other axes identically. Also, assume two threshold values gathered in the following vector [z<sub>min</sub>, z<sub>max</sub>]. This method enables to keep those points whose z-value belongs to [z<sub>min</sub>, z<sub>max</sub>], thus peforming a cutt-off.

In order to utilise this method, use:

```python
cl.cutoff_cloud(ax, [ax_min,ax_max])
```

where ```ax``` is the reference axis for the cut-off and ```[ax_min,ax_max]``` is the vector of the threshold values.

## Algebraic Transformation <a name="algebraic-transformation"></a>

### Singular Value Decomposition (SVD) <a name="svd"></a>

#### Brief Definition <a name="svd-def"></a>

Let **M** be a generic matrix belonging to *R*<sup> m x n</sup>. SVD allows **M** to be decomposed into the product of three matrices:

**M** = **U** **S** **V**

where **U** (in *R*<sup> m x m</sup>), and **V** (in *R*<sup> n x n</sup>) are called respectively *left-* and *right-principal* matrix (both orthonormal), whereas **S** (in *R*<sup> m x n</sup>) is the so-called matrix of the Singular Values.

#### Using SVD <a name="using-svd"></a>

Suppose that the points of the cloud are distributed according to three preferred directions, which are identifiable by visual inspection. Assume that these directions define a reference frame *{x<sub> S</sub>,y<sub> S</sub>,z<sub> S</sub>}*. Conversely, assume that the points forming the point cloud have been acquired with respect to another (arbitrary) reference frame *{x,y,z}*, which (unfortunately) is not aligned with *{x<sub> S</sub>,y<sub> S</sub>,z<sub> S</sub>}*. However, expressing the acquired points with respect to *{x<sub> S</sub>,y<sub> S</sub>,z<sub> S</sub>}* would be of considerable interest, e.g. for further numerical computations. From a mathematical standpoint, a change of reference frame, i.e. change of basis, from *{x,y,z}* to *{x<sub> S</sub>,y<sub> S</sub>,z<sub> S</sub>}* is sought as well as the associated matrix.

In this instance, it is therefore evident that finding this matrix by inspection, intuition or "trial & error" becomes preposterous. SVD holds the potential to compute such a matrix almost automatically. Let **M** (in *R*<sup> N x 3 </sup>) be the matrix representing the point cloud. In order to apply the SVD and obtain satisfactory results, the whole point cloud should be translated to -centroid. Following, the SVD applied to **M** provides **U** (in *R*<sup> N x N </sup>), **S** (in *R*<sup> N x 3 </sup>), and **V** (in *R*<sup> 3 x 3 </sup>). In particular, **V** is the matrix that realises the change of reference frame. Hence, each point of **M**, namely **M**<sub>i</sub>, will be rotated according to **V** as follows:

**M**<sub> i, S </sub> = **V** **M**<sub> i </sub>

where **M**<sub> i, S </sub> is **M**<sub> i </sub> but expressed with respect to *{x<sub> S</sub>,y<sub> S</sub>,z<sub> S</sub>}*. Finally, **M** is restored to its original position by translating it to +centroid.

All these operation are condensed and performed by the method:

```python
cl.cloud_svd()
```

**Remark**. Given that **V** is orthonormal the point cloud are subjected to a *rigid* rotation: neither stretching nor deformations occur.

**Remark**. If the determinant of **V** is -1, the transformed cloud (through SVD) may need flipping. To overcome this use ```flip_cloud``` ([Flip](#flip)).

# TO-DO List <a name="todo-list"></a>

- [ ] Enhance data visualisation
- [ ] Add methods to export data
