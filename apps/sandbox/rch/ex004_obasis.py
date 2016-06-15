r'''

Calculate the derivatives of a dihedral angle.

'''

import numpy as np
from oricreate.api import CreasePatternState, CustomCPFactory


from oricreate.util.einsum_utils import \
    DELTA, EPS

z_e = 0


def create_cp_factory():
    cp = CreasePatternState(X=[[0, 0, z_e],
                               [1, 0, 0],
                               [1, 1, 0],
                               [2, 1, z_e]],
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

    cp = cp_factory.formed_object

    vl = cp.iL_vectors
    nl0, nl1 = np.einsum('fi...->if...', cp.iL_F_normals)

    print 'vl', vl.shape
    print vl
    print 'nl0', nl0.shape
    print nl0
    print 'nl1', nl1.shape
    print nl1

    norm_vl = np.sqrt(np.einsum('...i,...i->...', vl, vl))
    norm_nl0 = np.sqrt(np.einsum('...i,...i->...', nl0, nl0))
    norm_nl1 = np.sqrt(np.einsum('...i,...i->...', nl1, nl1))

    unit_vl = vl / norm_vl[:, np.newaxis]
    unit_nl0 = nl0 / norm_nl0[:, np.newaxis]
    unit_nl1 = nl1 / norm_nl1[:, np.newaxis]

    print 'unit_vl', unit_vl.shape
    print unit_vl
    print 'unit_nl0', unit_nl0.shape
    print unit_nl0
    print 'unit_nl1', unit_nl1.shape
    print unit_nl1

    Tl0 = np.vstack(
        [unit_vl,
         unit_nl0,
         np.einsum('...j,...k,...ijk->...i', unit_vl, unit_nl0, EPS)]
    )

    print 'Tl0', Tl0.shape
    print Tl0
    unit_nl01 = np.einsum('...ij,...j->...i', Tl0, unit_nl1)

    print 'unit_nl01[:,2]', unit_nl01[:, 2]
    print unit_nl01[:, 2]

    psi = np.arcsin(unit_nl01[:, 2])

    print 'psi', psi

    vl_dul = cp.iL_vectors_dul
    nl0_dul0, nl1_dul1 = np.einsum('fi...->if...', cp.iL_F_normals_du)

    print cp.iL_N.shape
    print 'vl_dul', vl_dul.shape
    print vl_dul
    print 'nl0', nl0_dul0.shape
    print nl0_dul0
    print 'nl1', nl1_dul1.shape
    print nl1_dul1

    unit_nl0_dul0 = 1 / norm_nl0[:, np.newaxis, np.newaxis, np.newaxis] * (
        nl0_dul0 -
        np.einsum('...j,...i,...iNd->...jNd', unit_nl0, unit_nl0, nl0_dul0)
    )
    unit_nl1_dul1 = 1 / norm_nl1[:, np.newaxis, np.newaxis, np.newaxis] * (
        nl1_dul1 -
        np.einsum('...j,...i,...iNd->...jNd', unit_nl1, unit_nl1, nl1_dul1)
    )
    unit_vl_dul = 1 / norm_vl[:, np.newaxis, np.newaxis, np.newaxis] * (
        vl_dul -
        np.einsum('...j,...i,...iNd->...jNd', unit_vl, unit_vl, vl_dul)
    )

    print 'unit_nl0_dul0', unit_nl0_dul0.shape
    print unit_nl0_dul0
    print 'unit_nl1_dul1', unit_nl1_dul1.shape
    print unit_nl1_dul1
    print 'unit_vl_dul', unit_vl_dul.shape
    print unit_vl_dul

    Tl0_dul0 = np.vstack(
        [np.zeros_like(unit_nl0_dul0),
         unit_nl0_dul0,
         np.einsum('...j,...kNd,...ijk->...iNd', unit_vl, unit_nl0_dul0, EPS)
         ]
    )

    print 'Tl0_dul0', Tl0_dul0.shape
    print Tl0_dul0

    Tl0_dul = np.vstack(
        [unit_vl_dul,
         np.zeros_like(unit_vl_dul),
         np.einsum('...jNd,...k,...ijk->...iNd', unit_vl_dul, unit_nl0, EPS)
         ]
    )

    print 'Tl0_dul0', Tl0_dul.shape
    print Tl0_dul

    rho = 1 / np.sqrt((1 - unit_nl01[:, 2]**2))

    print 'rho', unit_nl01[:, 2]

    unit_nl01_dul = np.einsum(
        '...,...j,...ijNd->...iNd', rho, unit_nl1, Tl0_dul)
    unit_nl01_dul0 = np.einsum(
        '...,...j,...ijNd->...iNd', rho, unit_nl1, Tl0_dul0)
    unit_nl01_dul1 = np.einsum(
        '...,...jNd,...ij->...iNd', rho, unit_nl1_dul1, Tl0)

    print 'unit_nl01_dul', unit_nl01_dul.shape
    print unit_nl01_dul
    print 'unit_nl01_dul0', unit_nl01_dul0.shape
    print unit_nl01_dul0
    print 'unit_nl01_dul1', unit_nl01_dul1.shape
    print unit_nl01_dul1
