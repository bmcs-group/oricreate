'''
Created on 13.04.2016

@author: jvanderwoerd
'''
from oricreate.api import YoshimuraCPFactory
import matplotlib.pyplot as plt


cp_factory = YoshimuraCPFactory(n_x=1, n_y=2, L_x=2, L_y=4)
cp = cp_factory.formed_object

print 'Integration point positions r within facets:\n', cp.Fa_r
print 'Normal vectors at integration positions r\n', cp.Fa_normals

fig, ax = plt.subplots()
cp.plot_mpl(ax, facets=True, lines=True, linewidth=2, fontsize=30)

plt.show()