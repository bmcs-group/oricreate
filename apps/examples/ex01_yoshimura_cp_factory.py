'''
Created on Apr 3, 2015

@author: rch
'''


import matplotlib.pyplot as plt
from oricreate.api import YoshimuraCPFactory
cp_factory = YoshimuraCPFactory(L_x=4, L_y=1, n_x=2, n_y=2)
cp = cp_factory.formed_object
print 'list of sector angle arrays\n', cp.iN_theta
print 'facet area', cp.F_area
print 'potential energy', cp.V
print 'gradient of potential energy with respect to node displacements u\n', cp.V_du

fig, ax = plt.subplots()
cp.plot_mpl(ax, facets=True, lines=False, linewidth=2, fontsize=30)
fig.patch.set_visible(False)
ax.axis('off')
# plt.tight_layout()
plt.show()
