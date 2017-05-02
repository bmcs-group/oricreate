'''
Created on 13.04.2016

@author: jvanderwoerd
'''
from oricreate.api import CreasePatternState, CustomCPFactory
import matplotlib.pyplot as plt

cp = CreasePatternState(X=[[0, 0, 0],
                           [1, 0, 0],
                           [1, 1, 0],
                           [0, 1, 0],
                           [0.5, 0.5, 0]
                          ],
                        L=[[0, 1], [1, 2], [2, 3], [3, 0],
                           [0, 4], [1, 4], [2, 4], [3, 4]],
                        F=[[0, 1, 4], [1, 2, 4], [4, 3, 2], [4, 3, 0]]
                        )

cp_factory = CustomCPFactory(formed_object=cp)

fig, ax = plt.subplots()
cp.plot_mpl(ax, facets=True, lines=True, linewidth=2, fontsize=30)

plt.show()

