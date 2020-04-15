'''
Created on Apr 26, 2016

@author: rch
'''

r'''
This is used to show the evaluation steps within the 
deformation energy.
'''
from custom_factory_mpl import create_cp_factory
import numpy as np

if __name__ == '__main__':
    cp_factory = create_cp_factory()
    cp = cp_factory.formed_object
    print('x', cp.x)

    print('normed normal vectors of the facets\n', cp.norm_F_normals)
    print('normed normal facet vectors adjacent to the lines\n', cp.norm_iL_F_normals)

    cp.u[3, 2] = 0.5
    cp.u = cp.u
    print('iL_psi - u = 1.0', cp.iL_psi, cp.iL_psi_0)
    phi_iL = cp.iL_psi - cp.iL_psi_0
    print('iL_phi', phi_iL)

    print('normed normal vectors of the facets\n', cp.norm_F_normals)
    print('normed normal facet vectors adjacent to the lines\n', cp.norm_iL_F_normals)

    # change the position
    cp.u[3, 2] = -0.5
    cp.u = cp.u
    print('iL_psi - u = -1.0', cp.iL_psi, cp.iL_psi_0)
    phi_iL = cp.iL_psi - cp.iL_psi_0
    print('iL_phi', phi_iL)

    print('normed normal vectors of the facets\n', cp.norm_F_normals)
    print('normed normal facet vectors adjacent to the lines\n', cp.norm_iL_F_normals)

    import sympy as sp
    psi_, psi0_ = sp.symbols('psi,psi0')

    print(sp.simplify(sp.asin(psi_) - sp.asin(psi0_)))
    print(sp.diff(sp.asin(psi_), psi_))
