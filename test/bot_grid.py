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
bot = Topography("bot")
bot.read("/home/ale/Desktop/cm/cm_tube.mail", reader=np.loadtxt, skiprows=1)
bot.cut(ax="z", lo=-0.1, out=False)
bot.cut(ax="y", up=0, out=False)
v.scatter3D([bot])

gpr = load(filename="reg_top.gpr", folder="./")
trials(gpr, bot, 1)


gpb = np.loadtxt("./bot_1.txt")
tbot = Topography()
tbot.get_topography(gpb[:,0], gpb[:,1], gpb[:,3])

bem = np.load("./bot_grid.dat")
rbot = Topography()
rbot.get_topography(bem[:,0], bem[:,1], bem[:,3])


v.scatter3D([tbot, rbot])