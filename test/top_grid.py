from sys import path as syspath
from os import path as ospath
# syspath.append(ospath.join(ospath.expanduser("~"),
#                             '/home/ale/pythonEnv/bayes/lib/python3.8/site-packages/'))

syspath.append(ospath.join(ospath.expanduser("~"), '/home/ale/Desktop/pytom-3d/src/'))
# import numpy as np; np.random.seed(0)
from pytom3d.core import Topography
from pytom3d.viewer import Viewer, cfg_matplotlib
from pytom3d.util import save, load, prediction_wrapper, trials, apply_bc
import pandas as pd
import numpy as np

l = 220/2
H = 37.5/2
h = 31.5/2
x_res = 25
y_res = 5

v = Viewer()
top = Topography("top")
top.read("/home/ale/Desktop/cm/cm_tube.mail", reader=np.loadtxt, skiprows=1)
top.cut(ax="z", lo=-0.1, out=False)
top.cut(ax="y", lo=0.1, out=False)


gpr = load(filename="reg_top.gpr", folder="./")
trials(gpr, top, 5)


gpt = np.loadtxt("./top_1.txt")
ttop = Topography()
ttop.get_topography(gpt[:,0], gpt[:,1], gpt[:,3])

fem = np.load("./top_grid.dat")
rtop = Topography()
rtop.get_topography(fem[:,0], fem[:,1], fem[:,3])


v.scatter3D([ttop, rtop])