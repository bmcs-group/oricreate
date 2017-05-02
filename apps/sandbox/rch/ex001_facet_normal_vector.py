r'''

Calculate the derivatives of a dihedral angle.

'''

import numpy as np
from oricreate.api import CreasePatternState, CustomCPFactory
from oricreate.api import GuConstantLength, GuDofConstraints, \
    SimulationConfig, SimulationTask, fix, FTV
from oricreate.fu import \
    FuPotEngTotal


def create_cp_factory():
    cp = CreasePatternState(X=[[0, 0, 0],
                               [1, 0, 0],
                               [.5, .5, 0],
                               ],
                            L=[[0, 1],
                               [1, 2],
                               [2, 0],
                               ],
                            F=[[0, 2, 1],
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

    print 'F_N'
    print cp.F_N

    print 'F_L_vectors', cp.F_L_vectors.shape
    print cp.F_L_vectors

    print 'F-L_vectors_dul', cp.F_L_vectors_dul.shape
    print cp.F_L_vectors_dul
