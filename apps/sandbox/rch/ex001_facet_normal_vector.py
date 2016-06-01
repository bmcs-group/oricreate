r'''

Calculate the derivatives of a dihedral angle.

'''

import numpy as np
from oricreate.api import CreasePatternState, CustomCPFactory
from oricreate.api import GuConstantLength, GuDofConstraints, \
    SimulationConfig, SimulationTask, fix, FTV
from oricreate.fu import \
    FuTotalPotentialEnergy


def create_cp_factory():
    cp = CreasePatternState(X=[[0, 0, 0],
                               [1, 0, 0],
                               [1, 2, 0],
                               ],
                            L=[[0, 1],
                               [1, 2],
                               [2, 0],
                               ],
                            F=[[0, 1, 2],
                               ]
                            )

    cp_factory = CustomCPFactory(formed_object=cp)
    return cp_factory

if __name__ == '__main__':

    # end_doc

    cp_factory = create_cp_factory()

    cp = cp_factory.formed_object

    print 'F_normals', cp.norm_F_normals.shape
    print cp.norm_F_normals
    print 'F_normals_du', cp.norm_F_normals_du.shape

    F_normals_mag = np.einsum()

    print cp.norm_F_normals_du
