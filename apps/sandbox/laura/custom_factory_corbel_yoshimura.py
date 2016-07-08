r'''

Construct a general crease pattern configuration,
used in the examples below demonstrating the evaluation of goal functions
and constraints.
'''
# from oricreate.api import CreasePatternState, CustomCPFactory
# from oricreate.api import YoshimuraCPFactory
#   
# def create_cp_factory():
#     
#     cp_factory = YoshimuraCPFactory(L_x=4, L_y=1, n_x=2, n_y=2)
#     cp = cp_factory.formed_object
#     
#     cp_factory = CustomCPFactory(formed_object=cp)
#     return cp_factory
    
from oricreate.api import CreasePatternState, CustomCPFactory
import numpy as np
        
def create_cp_factory(n=2):
      
    x, y = np.mgrid[0:n + 1, 0:2]
    z = np.zeros_like(x)
    X = np.c_[x.flatten(), y.flatten(), z.flatten()]
    N = np.arange((2*n)+2).reshape(-1, 2)
    L1 = np.c_[N[:-1, 0], N[1:, 1]]
    L2 = np.c_[N[0:, 0], N[0:, 1]]
    L3 = np.c_[N[:-1, 0], N[1:, 0]]
    L4 = np.c_[N[:-1, 1], N[1:, 1]]
    L = np.vstack([L1, L2, L3, L4])
    F1 = np.c_[N[:-1, 0], N[1:, 0], N[1:, 1]]
    F2 = np.c_[N[:-1, 0], N[1:, 1], N[:-1, 1]]
    F = np.vstack([F1, F2])
    cp = CreasePatternState(X=X,
                            L=L,
                            F=F
                            )
  
    cp_factory = CustomCPFactory(formed_object=cp)
    return cp_factory


if __name__ == '__main__':
    cp_factory = create_cp_factory()
    cp = cp_factory.formed_object
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cp.plot_mpl(ax, facets=True)
    plt.tight_layout()
    plt.show()
