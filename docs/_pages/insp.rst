Introduction
------------
PyToM-3D offers afew visualisation utilities to inspect the data. Particularly, it exploits :py:mod:`pytom3d.viewer.Viewer` class. So we need one istance of this class before proceeding:

    .. code-block:: python
        
        from pytom3d.Viewer import Viewer
        v = Viewer()


2D Views
--------

We can easily display the three Cartesian  views of the topography ``t``, by using:

    .. code--block:: python
        
        v.views2D(t)


The method can take an indefinite number of input topographies.

3D Scatter
----------

Similarly we can easily the 3D scatter plot of ``t`` with:

    .. code-block:: python
        
        v.scatter(t)


The method accepts an indefinite number of input topographies.