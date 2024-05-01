Introduction
------------

In order to understand each geometric transformation remember that the points of the topography are stored in the attribute ``P``, which is a :math:`N\times 3` matrix:

    .. math::
        \mathbf{P} = \begin{bmatrix}  x_1 & y_1 & z_1 \\ x_2 & y_2 & z_2 \\ \vdots & \vdots & \vdots \\ x_N & y_N & z_N\end{bmatrix}

where :math:`\mathbf{p}_i = \left[x_i\quad y_i \quad z_i\right]` is the *i*th point of the topography. In practice the altitude :math:`z_i` is determined upon a latent function of the coordinates, hence :math:`z_i = f(x_i, y_i)`.

Translation
-----------

Let us define a vector :math:`\mathbf{v} = \left[x_v\quad y_v\quad z_v \right]`. By calling:

    .. code-block:: python
        
        t.translate([xv,yv,zv])

we translate :math:`\mathbf{P}` by :math:`\mathbf{v}` as:

    .. math::
        \mathbf{p}_i \leftarrow \mathbf{p}_i + \mathbf{v}\quad\forall\ i = 1,2,\dots,N.

Rotation
--------
This method performs the rotation about the centre given three rotation angles :math:`\theta_x`, :math:`\theta_y`, :math:`\theta_z`. The method builds the corresponding rotation matrices:

    .. math::
        \mathbf{R}_x = \begin{bmatrix}  1 & 0 & 0 \\\ 0 & \cos(\theta_x) & \sin(\theta_x) \\\ 0 & -\sin(\theta_x) & \cos(\theta_x) \\\ \end{bmatrix},

    .. math::
        \mathbf{R}_y = \begin{bmatrix}  \cos(\theta_y) & 0 & \sin(\theta_y) \\\ 0 & 1 & 0 \\\ -\sin(\theta_y) & 0 & \cos(\theta_y) \end{bmatrix},
        
    .. math::
        \mathbf{R}_z = \begin{bmatrix} \cos(\theta_z) & \sin(\theta_z) & 0 \\\  -\sin(\theta_z) & \cos(\theta_z) & 0 \\\ 0 & 0 & 1 \end{bmatrix},

    .. math::
        \mathbf{R} = \mathbf{R}_x\mathbf{R}_y\mathbf{R}_z.

Then each point is rotated as:

    .. math::
        \mathbf{p}_i \leftarrow \mathbf{R}\mathbf{p}_i.

This is accomplished via:

    .. code-block:: python
            
            t.rotate(t_deg=[t_x, t_y, t_z])

The method supports passing a rotation matrix too:

    .. code-block:: python

        t.rotate(rot_mat=R)


In case on wishes to rotate about a given centre :math:`c=\left[c_x\quad c_y\quad c_z\right]^\top`, they would call the wrapper method:

    .. code-block::
    
        t.rotate_about_centre(c=[c_x, c_y, c_z]t_deg=[t_x, t_y, t_z])


or by providing a rotation matrix ``rot_mat``.


Flip
----
This method allows mirroring data with respect to a given vector :math:`\mathbf{v}=\left[v_x\quad v_y\quad v_z \right]` which represent the outward normal of the intended flipping plane. Assuming one wishes to flip the data about the $yz$ plane, they would define :math:`\mathbf{v}=\left[-1\quad 1\quad 1 \right]`, and call:

    .. code-block:: python
        
        t.flip([-1,1,1])


This operation performs :math:`(\mathbf{p}_i = \left[x_i\quad y_i \quad z_i\right])`:

    .. math::
        \left[x_i\quad y_i \quad z_i\right] \leftarrow \left[x_i\cdot v_x\quad y_i\cdot vy\quad z_i\cdot vz\right] \quad\forall\ i = 1,2,\dots,N.

Cut
---

Suppose you would like to remove some outliers from topography. Although the same procedure applies to the other axes identically, we focus on the z-axis. We also assume two threshold :math:`l` and :math:`u`, whereby we filter each *i*th datum using the criterion:

    .. math::
        z_i \leftarrow z_i:\quad z_i > l\quad \text{and}\quad z_i < u.

To do so, we call:

    .. code-block:: python
        
        t.cut(ax="z", lo=l, up=u, out=False)


If ``out=True`` the method keep the points complying with:

    .. math::
        z_i \leftarrow z_i:\quad z_i < l\quad \text{and}\quad z_i > u.