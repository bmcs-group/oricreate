'''
Created on Apr 3, 2015

@author: rch
'''


import matplotlib.pyplot as plt
from oricreate.api import YoshimuraCPFactory
cp_factory = YoshimuraCPFactory(L_x=0.634, n_x=6, L_y=0.406, n_y=4)
cp = cp_factory.formed_object

fig, ax = plt.subplots()
cp.plot_mpl(ax, facets=True, lines=True, linewidth=4, fontsize=30)
fig.patch.set_visible(False)
ax.axis('on')
# plt.tight_layout()
plt.show()
