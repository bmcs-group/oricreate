'''
Created on Apr 26, 2016

@author: rch
'''

r'''
This is used to show the evaluation steps within the 
deformation energy.
'''
from custom_factory_mpl import create_cp_factory


def create_fu():
    cp_factory = create_cp_factory()
    # begin
    from oricreate.fu import FuBendingEnergy
    # Link the pattern factory with the goal function client.
    fu_benergy = FuBendingEnergy(forming_task=cp_factory)

    # Change the vertical coordinate to get
    # a non-zero value of potential energy
    cp = cp_factory.formed_object
    # Note: the above assignments to array elements are not
    # registered by the change notification system.
    # In order to trigger the dependency chain notifying
    # the fu_poteng instance that something has changed
    # an assignment to an array as a whole is necessary:

    print 'fu:', fu_benergy.get_f(cp.u)
    print 'f_du:\n', fu_benergy.get_f_du(cp.u)
    # end
    return cp_factory

if __name__ == '__main__':
    import numpy as np
    cp_factory = create_fu()
    cp = cp_factory.formed_object
    print 'x', cp.x
    cp.u[3, 2] = 1.0
    cp.u = cp.u
    print 'iL_psi - u = 1.0', cp.iL_psi, cp.iL_psi2, cp.iL_psi_0
    phi_iL = cp.iL_psi2 - cp.iL_psi_0
    print 'iL_phi', phi_iL

    # change the position
    cp.u[3, 2] = -1.0
    cp.u = cp.u
    print 'iL_psi - u = -1.0', cp.iL_psi, cp.iL_psi2, cp.iL_psi_0
    phi_iL = cp.iL_psi2 - cp.iL_psi_0
    print 'iL_phi', phi_iL

    import sympy as sp
    psi_, psi0_ = sp.symbols('psi,psi0')

    print sp.simplify(sp.asin(psi_) - sp.asin(psi0_))
    print sp.diff(sp.asin(psi_), psi_)
