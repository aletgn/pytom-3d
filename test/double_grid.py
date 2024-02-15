from sys import path as syspath
from os import path as ospath
# syspath.append(ospath.join(ospath.expanduser("~"),
#                             '/home/ale/pythonEnv/bayes/lib/python3.8/site-packages/'))

syspath.append(ospath.join(ospath.expanduser("~"), '/home/ale/Desktop/pytom-3d/src/'))
# import numpy as np; np.random.seed(0)
from pytom3d.core import Topography
from pytom3d.viewer import Viewer, cfg_matplotlib
from pytom3d.util import save, load, prediction_wrapper

import numpy as np
np.random.seed(100)
def distance3(x,y):
    return (x**2)/100000

def distance4(x,y):
    return -(x**2)/100000

l = 200/2
H = 37.5/2
h = 31.5/2
x_res = 25
y_res = 5

# cfg_matplotlib(font_family="serif", use_latex=True)

"""
v = Viewer()
g_top = Topography()
g_top.make_grid(x_bounds=[-l,l], y_bounds=[h,H], x_res=x_res, y_res=y_res)
g_top.add_points(distance3, std_noise=0.01)
# v.scatter3D([g_top])

# g_top_ = Topography()
# g_top_.make_grid(x_bounds=[-l,l], y_bounds=[-H,-h], x_res=x_res, y_res=y_res)
# g_top_.add_points(distance4, std_noise=0.005)
# v.scatter3D([g_top, g_top_])

# g_top + g_top_

from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
kernel = ConstantKernel(1.0, (1e-5, 1e5)) \
    * RBF([1.0, 1.0], (1e-5, 1e5)) \
    + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e5))

g_top.fit(gpr(kernel=kernel, n_restarts_optimizer=100))

g_test = Topography()
g_test.make_grid(x_bounds=[-l,l], y_bounds=[h,H], x_res=x_res*2, y_res=y_res*2)
# g_top.regressor.kernel_.get_params() # list parameters

z, sigma = g_top.pred(g_test.P[:,0:2])

# v.scatter3D([g_test])

g_eval = Topography()
g_eval.get_topography(g_test.P[:,0], g_test.P[:,1], z, sigma)
# v.scatter3DRegression(g_eval, g_top)

save(g_top.regressor, filename="reg_top", extension=".gpr")
"""

gpr = load(filename="reg_top.gpr", folder="./")
x = -100
y = 18.5
p, s = prediction_wrapper(gpr, x, y)

print(p,s)