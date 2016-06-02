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

    ab = np.einsum('...i,...i->...', a, b)
    print 'ab', ab

    mag_a = np.sqrt(np.einsum('...i,...i->...', a, a))
    mag_b = np.sqrt(np.einsum('...i,...i->...', b, b))
    print 'mag_a, mag_b', mag_a, mag_b

    mag_ab = mag_a * mag_b
    gamma = ab / mag_ab

    print 'gamma', gamma

    f0_term = b - gamma * mag_b / mag_a * a
    f1_term = a - gamma * mag_a / mag_b * b

    f0_gamma_du = 1 / mag_ab * np.einsum('...i,...i->...', a_du, f0_term)
    f1_gamma_du = 1 / mag_ab * np.einsum('...i,...i->...', b_du, f1_term)

    f_term = np.concatenate(
        (f0_gamma_du[:, np.newaxis, ...], f1_gamma_du[:, np.newaxis, ...]), axis=1)
    print 'f_term', f_term.shape
    print f_term

    print cp.F_N
    iL_F_map = cp.F[cp.iL_F]
    print 'iL_F_map'
    print iL_F_map

    gamma_du = np.zeros((cp.n_iL, cp.n_N, cp.n_D), dtype='float_')
    print 'gamma_du.shape', gamma_du.shape
    gamma_du[0, iL_F_map[0, 0]] += f_term[0, 0]
    gamma_du[0, iL_F_map[0, 1]] += f_term[0, 1]
    print 'gamma_du'
    print gamma_du
