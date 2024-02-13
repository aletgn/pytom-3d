from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"),'/home/ale/Desktop/pytom-3d/src/'))

# import pytom3d.core, pytom3d.util
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
g_bot.add(distance3)
g_top = Grid()
g_top.make(x_bounds=[-l,l], y_bounds=[-H,-h], x_res=x_res, y_res=y_res)
g_top.add(distance4)
v.scatter3D([g_bot, g_top])

