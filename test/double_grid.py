from sys import path as syspath
from os import path as ospath
# syspath.append(ospath.join(ospath.expanduser("~"),
#                             '/home/ale/pythonEnv/bayes/lib/python3.8/site-packages/'))

syspath.append(ospath.join(ospath.expanduser("~"), '/home/ale/Desktop/pytom-3d/src/'))
# import numpy as np; np.random.seed(0)
from pytom3d.core import Topography
from pytom3d.viewer import Viewer, cfg_matplotlib
from pytom3d.util import save, load, prediction_wrapper, trials, predict_at_node
import pandas as pd

import numpy as np
np.random.seed(100)
def distance3(x,y):
    return (x**2)/100000

def distance4(x,y):
    return -(x**2)/100000

l = 220/2
H = 37.5/2
h = 31.5/2
x_res = 25
y_res = 5

# cfg_matplotlib(font_family="serif", use_latex=True)

v = Viewer()
g = Topography()
g.make_grid(x_bounds=[-l,l], y_bounds=[h,H], x_res=x_res, y_res=y_res)
g.add_points(distance3, std_noise=0.01)

# g.cut("x", lo=-50, up=50, out=True)
# g.rotate(t_deg=[0,0,90])

g.rotate_about_centre(c=[0,0,0], t_deg=[0,0,90])

v.views2D([g])

print(g.history_)







"""
v = Viewer()
g_top = Topography()
g_top.make_grid(x_bounds=[-l,l], y_bounds=[h,H], x_res=x_res, y_res=y_res)
g_top.add_points(distance3, std_noise=0.01)
# v.scatter3D([g_top])

# g_bot = Topography()
# g_bot.make_grid(x_bounds=[-l,l], y_bounds=[-H,-h], x_res=x_res, y_res=y_res)
# g_bot.add_points(distance4, std_noise=0.005)
# v.scatter3D([g_top, g_bot])

# g_top + g_bot

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

"""
v = Viewer()
gpr = load(filename="reg_top.gpr", folder="./")
x = -100
y = 18.5
p, s = prediction_wrapper(gpr, x, y)

g_top = Topography()
g_top.regressor = gpr
g_top.make_grid(x_bounds=[-l,l], y_bounds=[h,H], x_res=x_res, y_res=y_res)
g_top.add_points(distance3)


# mesh = Topography()
# mesh.read("/home/ale/Desktop/cm/cm_tube.mail", reader=np.loadtxt, skiprows=1)
# mesh.cut(ax="z", lo=-0.1, out=False)
# # mesh.P[:,2] = 0
# # v.scatter3D([mesh])

# z, sigma = g_top.pred(mesh.P[:,0:2])
# g_eval = Topography()
# g_eval.get_topography(mesh.P[:,0], mesh.P[:,1], z, sigma)

# v.scatter3D([mesh])
# # v.scatter3DRegression(g_eval, g_top)

# trials(gpr, mesh, 5)


gpr = np.loadtxt("./unnamed_1.txt")

ver = Topography()
ver.get_topography(gpr[:,0], gpr[:,1], gpr[:,3])

# a = apply_bc(110, -18.75, gpr)

disp = np.load("./cm_tube_disp.dat")
res = Topography()
res.get_topography(disp[:,0], disp[:,1], disp[:,3])

a = res.P[res.P[:,1].argsort()]
b = ver.P[ver.P[:,1].argsort()][1:]
c = a-b

# # res.P = c
# ver.unc = np.array([9])
# v.scatter3D([ver,res])
"""



