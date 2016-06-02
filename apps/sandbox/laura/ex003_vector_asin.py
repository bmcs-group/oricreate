r'''

Calculate the derivatives of a dihedral angle.

'''

import numpy as np
from oricreate.api import CreasePatternState, CustomCPFactory
from oricreate.api import GuConstantLength, GuDofConstraints, \
    SimulationConfig, SimulationTask, fix, FTV
from oricreate.fu import \
    FuTotalPotentialEnergy
    
from oricreate.util.einsum_utils import \
    DELTA, EPS


def create_cp_factory():
    cp = CreasePatternState(X=[[0, 0, 0],
                               [1, 0, 0],
                               [1, 1, 0],
                               [2, 1, 1]],
                            L=[[0, 1],
                               [1, 2],
                               [2, 0],
                               [1, 3],
                               [3, 2]],
                            F=[[0, 1, 2],
                               [1, 3, 2]]
                            )

    cp_factory = CustomCPFactory(formed_object=cp)
    return cp_factory

if __name__ == '__main__':

    # end_doc

    cp_factory = create_cp_factory()

    from oricreate.util import \
        get_cos_theta, get_cos_theta_du, get_theta, get_theta_du

    cp = cp_factory.formed_object

    a, b = np.einsum('fi...->if...', cp.iL_F_normals)
    a_du, b_du = np.einsum('fi...->if...', cp.iL_F_normals_du)

    print 'x', cp.x[cp.F].shape
    print cp.x[cp.F]
    print 'F_normals', cp.F_normals.shape
    print cp.F_normals
    print 'F_normals_du', cp.F_normals_du.shape
    print cp.F_normals_du

    print 'a', a
    print 'b', b
    print 'a_du', a_du
    print 'b_du', b_du

    cross_normals= np.einsum('...i,...j,...ijk->...k', a, b, EPS)
    print 'crossproduct normals', cross_normals
    
    cross_normals_du_1 = np.einsum('...i,...j,...ijk->...k', a_du, b, EPS)
    print 'derivation crossproduct normals part1', cross_normals_du_1
    
    cross_normals_du_2 = np.einsum('...i,...j,...ijk->...k', a, b_du, EPS)
    print 'derivation crossproduct normals part2', cross_normals_du_2
       
    cross_normals_du=np.zeros((1,3,4,3))
     
    cross_normals_du[0,:,0,:]=cross_normals_du_1[0,:,0,:]
    cross_normals_du[0,:,1,:]=cross_normals_du_1[0,:,1,:]+cross_normals_du_2[0,:,0,:]
    cross_normals_du[0,:,2,:]=cross_normals_du_1[0,:,2,:]+cross_normals_du_2[0,:,2,:]
    cross_normals_du[0,:,3,:]=cross_normals_du_2[0,:,1,:]    
       
    print 'derivation crossproduct normals', cross_normals_du

    print('gamma')
    print(get_cos_theta(a, b))
    print('theta')
    print(get_theta(a, b)) * 4.0

    print('gamma_du')
    print(get_cos_theta_du(a, a_du, b, b_du))
    print 'theta_du'
    print(get_theta_du(a, a_du, b, b_du))
