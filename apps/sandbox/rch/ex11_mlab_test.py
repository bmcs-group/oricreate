'''
Created on Jan 24, 2017

@author: rch
'''

from mayavi import mlab

import numpy as np

x, y, z = np.ogrid[-10:10:20j, -10:10:20j, -10:10:20j]
s = np.sin(x * y * z) / (x * y * z)
mlab.pipeline.volume(mlab.pipeline.scalar_field(s))

mlab.show()