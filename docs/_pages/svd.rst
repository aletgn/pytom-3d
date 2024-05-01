Introduction
------------

Let :math:`\mathbf{P}` be a generic matrix belonging to :math:`\mathbb{R}^{m\times n}`. SVD allows :math:`\mathbf{P}` to be decomposed into the product of three matrices:

    .. math::
        \mathbf{P} = \mathbf{U}\mathbf{S}\mathbf{V^\top},

where :math:`\mathbf{U}\in \mathbb{R}^{m\times m}` , and :math:`\mathbf{V}^\top\in \mathbb{R}^{n\times n}` are called respectively *left-* and *right-principal* matrix (both orthonormal), whereas :math:`\mathbf{S}\in \mathbb{R}^{m\times n}` is the so-called Singular Value Matrix.

We do SVD via:

    .. code-block:: python
        
        t.svd()


Using SVD
---------

Suppose that the points of the topography are distributed according to three preferred directions, which are identifiable by visual inspection. Assume that these directions define a reference frame :math:`\{x_s,y_s,z_s\}`. Conversely, assume that the points forming the point topography have been acquired with respect to another (arbitrary) reference frame :math:`\{x,y,z\}`, which (unfortunately) is not aligned with :math:`\{x_s,y_s,z_s\}`. However, expressing the acquired points with respect to :math:`\{x_s,y_s,z_s\}` would be of considerable interest, e.g. for further numerical computations. From a mathematical standpoint, a change of reference frame, i.e. change of basis, from :math:`\{x,y,z\}` to :math:`\{x_s,y_s,z_s\}` is sought as well as the associated matrix.

In this instance, it is therefore evident that finding this matrix by inspection, intuition or 'trial & error' becomes preposterous. SVD holds the potential to compute such a matrix almost automatically. Let :math:`\mathbf{P}\in\mathbb{R}^{N\times 3}` be the matrix representing the topography. In order to apply the SVD and obtain satisfactory results, the whole point topography should be translated to :math:`-G` (the centroid). Following, the SVD applied to :math:`\mathbf{P}` provides :math:`\mathbf{U}\in \mathbb{R}^{N\times N}` , :math:`\mathbf{S}\in \mathbb{R}^{N\times 3}`, and :math:`\mathbf{V}^\top\in \mathbb{R}^{3\times 3}`.  In particular, :math:`\mathbf{V}` realises the change of basis. Hence, each point of :math:`\mathbf{P}`, namely :math:`\mathbf{p}_i`, will be rotated according to :math:`\mathbf{V}^\top`:

    .. math::
        \mathbf{p}_i \leftarrow \mathbf{V}^\top\mathbf{p}_i.

To perform this, just invoke:

    .. code-block::
        
        t.rotate_by_svd()

which wraps ``rotate_about_centre``.

Since :math:`\mathbf{V}^\top` is orthonormal, the topography is subjected to a *rigid* rotation: neither stretching nor deformations occur. If the :math:`\det{V^\top} \simeq -1`, the transformed topography (through SVD) may need flipping. To overcome this, just use ``Flip``.