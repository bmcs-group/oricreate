r'''

Calculate the derivatives of a dihedral angle.

'''

import numpy as np
from oricreate.api import CreasePatternState, CustomCPFactory


from oricreate.util.einsum_utils import \
    DELTA, EPS

z_e = 0.0


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

    a, b = np.einsum('fi...->if...', cp.iL_F_normals)
    a_du, b_du = np.einsum('fi...->if...', cp.iL_F_normals_du)

    c = np.einsum('...i,...j,...ijk->...k', a, b, EPS)
    print('c')
    print(c)

    mag_a = np.sqrt(np.einsum('...i,...i->...', a, a))
    mag_b = np.sqrt(np.einsum('...i,...i->...', b, b))
    mag_c = np.sqrt(np.einsum('...i,...i->...', c, c))
    mag_ab = mag_a * mag_b

    mag_c[np.where(mag_c == 0)] = 1.e-19
    normed_c = c / mag_c
    gamma = mag_c / mag_ab
    print('gamma', gamma)
    theta = np.arcsin(gamma)
    print('theta', theta)

    print('psi')
    print(cp.iL_psi)
    iL_Fa_c_du = \
        np.einsum('...iIe,...j,...ijk,...k->...Ie', a_du, b, EPS, normed_c)
    print('Fa_c_du', iL_Fa_c_du.shape)
    print(iL_Fa_c_du)
    iL_Fb_c_du = \
        np.einsum('...i,...jIe,...ijk,...k->...Ie', a, b_du, EPS, normed_c)
    print('Fb_c_du', iL_Fb_c_du.shape)
    print(iL_Fb_c_du)

    # calculate the derivatives of the angle between the two normal vectors
    iL_Fa_x_du = gamma * mag_b / mag_a * \
        np.einsum('...iId,...i->...Id', a_du, a)
    print('iL_Fa_x_du', iL_Fa_x_du.shape)
    iL_Fb_x_du = gamma * mag_a / mag_b * \
        np.einsum('...iId,...i->...Id', b_du, b)
    # run the contraction along the component index of vectors a and b
    # (index i) - preserve the indexes I and d.
    iL_Fa_gamma_du = 1 / mag_ab * (iL_Fa_c_du - iL_Fa_x_du)
    iL_Fb_gamma_du = 1 / mag_ab * (iL_Fb_c_du - iL_Fb_x_du)

    # Keep the terms for left (a) and right (b) facets in separate arrays
    # this is necessary to be able to incrementally derivatives add up
    # contributions from left and right facets to shared nodes
    # of the interior line.

    # get the map of facet nodes attached to interior lines
    iL_Fa_N_map = cp.F[cp.iL_F[:, 0]].reshape(cp.n_iL, -1)
    iL_Fb_N_map = cp.F[cp.iL_F[:, 1]].reshape(cp.n_iL, -1)
    # enumerate the interior lines and broadcast it N and D into dimensions
    iL_map = np.arange(cp.n_iL)[:, np.newaxis, np.newaxis]
    # broadcast the facet node map into D dimension
    Na_map = iL_Fa_N_map[:, :, np.newaxis]
    Nb_map = iL_Fb_N_map[:, :, np.newaxis]
    # broadcast the spatial dimension map into iL and N dimensions
    D_map = np.arange(3)[np.newaxis, np.newaxis, :]
    # allocate the gamma derivatives of iL with respect to N and D dimensions
    gamma_du = np.zeros((cp.n_iL, cp.n_N, cp.n_D), dtype='float_')
    # add the contributions gamma_du from the left and right facet
    # Note: this cannot be done in a single step since the incremental
    # assembly is not possible within a single index expression.
    gamma_du[iL_map, Na_map, D_map] += iL_Fa_gamma_du
    gamma_du[iL_map, Nb_map, D_map] += iL_Fb_gamma_du
    print('Na_map')
    print(Na_map)
    print(Nb_map)
    print('iL_Fa_gamma_du')
    print(iL_Fa_gamma_du)
    print(iL_Fb_gamma_du)
    print('gamma_du')
    print(gamma_du)

    theta_du = np.einsum(
        '...,...Ie->...Ie', 1. / np.sqrt(1. - gamma ** 2), gamma_du)

    print('theta_du')
    print(theta_du)
