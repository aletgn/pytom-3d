#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    synthetic_data is part of pyCloM and it used to generate synthetic data to
    test the pyCloM module.
    
    Copyright (C) 2022  Alessandro Tognan

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from matplotlib import pyplot as plt 

x = np.linspace(-70,100, 50)
y = np.linspace(-50,100, 50)
X,Y = np.meshgrid(x, y)
Z = (X**2+Y**2)**0.5
# fig = plt.figure(figsize=plt.figaspect(1.2),dpi=300)#figsize=plt.figaspect(1))
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.scatter(X,Y,Z)
xf = X.flatten()
yf = Y.flatten()
zf = Z.flatten()

p = np.array([xf,yf,zf]).T

np.savetxt('data_set.txt', p, delimiter=',', fmt=['%.4f']*3)

