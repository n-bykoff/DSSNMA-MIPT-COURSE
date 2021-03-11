# -*- coding: utf-8 -*-
"""
Example: how to use mesh module
"""

import mesh as flowmod_mesh
from mesh import Mesh
import numpy as np

import time


path = 'mesh-cyl/'

mesh = Mesh()
t1 = time.time()
mesh.read_starcd(path)
t2 = time.time()

print('Time for mesh reading = {0:5.2e} s'.format(t2 - t1))


# Create random data and write it using 'write_tecplot' function 
# After that check that Tecplot opens this file and properly shows mesh

data = np.random.uniform(0, 255, (mesh.nc, 1))
var_names = ['random']

flowmod_mesh.write_tecplot(mesh, data, 'testing.tec', var_names)