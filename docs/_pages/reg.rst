Gaussian Process Regression (GPR)
---------------------------------

PyToM-3D wraps the regressors of scikit-learn, amongst which GPR. In this instance, the topography is modelled as the following regression model:

    .. math::
        (x,y) = f(x,y) + \epsilon(0, \sigma),

where :math:`f(x,y)` is a latent function modelling the topography and :math:`\epsilon` is Gaussian noise having null mean and :math:`\sigma` as the standard deviation. Next, a Gaussian Process is placed over :math:`f(x,y)`:

    .. math::
        f \sim GP(M(x,y), K(x,y)),

where :math:`M(x,y)` is the mean, and :math:`K(x,y)` is the kernel (covariance function). Initially, we need to define the kernel:

    .. code-block:: python
        
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
        kernel = ConstantKernel() * RBF([1.0, 1.0], (1e-5, 1e5)) \
                        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e5))


which represents a typical squared exponential kernel with noise:

    .. math::
        K(\mathbf{x},\mathbf{x}') = C \exp{\left(\frac{\Vert \mathbf{x} - \mathbf{x}'\Vert^2}{l^2}\right)} + \sigma_{ij},

where:

    .. math::
        \sigma_{ij} = \delta_{ij} \epsilon,\quad \mathbf{x}=\left[x,y\right],

and :math:`\delta_{ij}` is Kronecker's delta applied to any pair of points :math:`\mathbf{x}_{i}`, :math:`\mathbf{x}_j`.

Finally, we invoke:

    .. code-block:: python
        
        from sklearn.gaussian_process import GaussianProcessRegressor as gpr
        t.fit(gpr(kernel=kernel))

and the GPR is fit. To predict data:

    .. code-block:: python
        
        t.pred(X)


where :math:`X` is :math:`M \times 2` a numpy array containing the :math:`x` and :math:`y` coordinates of an evaluation grid.

Spline fitting
--------------

PyToM-3D allows user to fit data by bi-variate splines specifically wrapping ``scipy.interpolate.bisplrep``. Modelling the topography as a two-variable function :math:`f(x,y)`, we approximate it as:

    .. math::
        f(x,y) \approx \sum_{i=1}^{m} \sum_{j=1}^{n} C_{ij} Q_{i,r}(x) Q_{j,s}(y),

where :math:`Q_{i,r}` and :math:`Q_{j,s}` are polynomials of degree :math:`r` and :math:`s` respectively. Additionally, :math:`m` and :math:`n` are the number of control points (aka knots) distributed along the :math:`x`- and :math:`y`-axis. Lastly, :math:`C_{ij}` are the trainable coefficients, which are determined upon the points of the topography.

To fit a bi-variate spline to the data, we call:

    .. code-block:: python
        
        from scipy.interpolate import bisplrep
        t.fit(bisplrep, kx, ky, tx, ty)


where ``kx``, ``ky`` represent the above-mentioned :math:`r` and :math:`s` (degree of the polynomials), and ``tx`` and ``ty`` are the control points, which are expected to be ``np.ndarray``-like objects. Once the fitting is done we forecast the topography elsewhere by:

    .. code-block:: python
        
        t.pred(X)