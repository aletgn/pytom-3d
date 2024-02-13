from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),
                           '/home/ale/Desktop/pytom-3d/src/'))
import numpy as np; np.random.seed(0)
from pytom3d.core import Grid
from pytom3d.viewer import Viewer

def distance3(x,y):
    return (x**2)/100000

def distance4(x,y):
    return -(x**2)/100000

l = 200/2
H = 37.5/2
h = 31.5/2
x_res = 25
y_res = 5

v = Viewer()
g_bot = Grid()
g_bot.make(x_bounds=[-l,l], y_bounds=[h,H], x_res=x_res, y_res=y_res)
g_bot.add(distance3, std_noise=0.005)
g_top = Grid()
g_top.make(x_bounds=[-l,l], y_bounds=[-H,-h], x_res=x_res, y_res=y_res)
g_top.add(distance4, std_noise=0.005)
# v.scatter3D([g_bot, g_top])
g_bot + g_top


from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0, 1.0], (1e-2, 1e2)) + WhiteKernel()

# 'n_features_in_': 2
# regressor
g_bot.fit(gpr(kernel=kernel, n_restarts_optimizer=10))
z, sigma = g_bot.pred(g_bot.P[:,0:2])

g_test = Grid()
g_test.get_grid(g_bot.P[:,0:2], z)

v.scatter3D([g_bot, g_test])