from sys import path as syspath
from os import path as ospath
# syspath.append(ospath.join(ospath.expanduser("~"),
#                             '/home/ale/pythonEnv/bayes/lib/python3.8/site-packages/'))

syspath.append(ospath.join(ospath.expanduser("~"), '/home/ale/Desktop/pytom-3d/src/'))
# import numpy as np; np.random.seed(0)
from pytom3d.core import Topography
from pytom3d.viewer import Viewer
from pytom3d.util import save, load
# 
def distance3(x,y):
    return (x**2)/100000

def distance4(x,y):
    return -(x**2)/100000

l = 200/2
H = 37.5/2
h = 31.5/2
x_res = 25*2
y_res = 5*2

v = Viewer()
g_bot = Topography()
g_bot.make_grid(x_bounds=[-l,l], y_bounds=[h,H], x_res=x_res, y_res=y_res)
g_bot.add_points(distance3, std_noise=0.005)
# v.scatter3D([g_bot], x_lim=[-50,50])

g_top = Topography()
g_top.make_grid(x_bounds=[-l,l], y_bounds=[-H,-h], x_res=x_res, y_res=y_res)
g_top.add_points(distance4, std_noise=0.005)



v.scatter3D([g_bot, g_top])











# # 
# # g_bot + g_top

# from sklearn.gaussian_process import GaussianProcessRegressor as gpr
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
# kernel = C(1.0, (1e-3, 1e3)) \
#     * RBF([1.0, 1.0], (1e-2, 1e3)) \
#         #+ WhiteKernel(noise_level=0.005, noise_level_bounds=(1e-5, 1e3))
        
# g_bot.fit(gpr(kernel=kernel, n_restarts_optimizer=1))

# # 'n_features_in_': 2
# g_test = Topography()
# # g_test.make(x_bounds=[-l,l], y_bounds=[-H,H], x_res=x_res, y_res=50)
# z, sigma = g_bot.pred(g_bot.P[:,0:2])

# # g_test = Grid()
# # z, sigma = g_bot.pred(g_tt.P[:,0:2])
# g_test.get_grid(g_bot.P[:,0:2], z)

# v.scatter3D([g_bot, g_test])


# save(g_bot.regressor, extension=".gpr")

