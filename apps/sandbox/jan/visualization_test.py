<<<<<<< master
from oricreate.api import YoshimuraCPFactory
import matplotlib.pyplot as plt

cp_factory = YoshimuraCPFactory(n_x=4, n_y=2, L_x=4, L_y=2)
cp = cp_factory.formed_object

print('Nodes of facets enumerated counter clock-wise\n', cp.F_N)
print('Lines of facets enumerated counter clock-wise\n', cp.F_L)

fig, ax = plt.subplots()
cp.plot_mpl(ax, facets=True, lines=True, linewidth=2, fontsize=30)

=======
from oricreate.api import YoshimuraCPFactory
import matplotlib.pyplot as plt

cp_factory = YoshimuraCPFactory(n_x=4, n_y=2, L_x=4, L_y=2)
cp = cp_factory.formed_object

print('Nodes of facets enumerated counter clock-wise\n', cp.F_N)
print('Lines of facets enumerated counter clock-wise\n', cp.F_L)

fig, ax = plt.subplots()
cp.plot_mpl(ax, facets=True, lines=True, linewidth=2, fontsize=30)

>>>>>>> interim stage 1
plt.show()