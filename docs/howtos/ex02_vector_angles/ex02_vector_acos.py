r'''

Calculate the derivatives of a dihedral angle.

'''

if __name__ == '__main__':

    # end_doc

    import numpy as np
    from oricreate.util import \
        get_cos_theta, get_cos_theta_du, get_theta, get_theta_du

    a = np.array([[1, 0], [1, 0], [1, 0]], dtype='f')
    b = np.array([[0, 1], [1, 1], [1, -1]], dtype='f')

    print('gamma')
    print((get_cos_theta(a, b)))
    print('theta')
    print((get_theta(a, b)))

    I = np.diag(np.ones((2,), dtype='f'))
    # dimensions of the derivatives are stored as:
    # index of a vector,
    # component of the vector,
    # index of a node,
    # component of a node.
    a_du = np.array([[[[-1, 0]], [[0, -1]]],
                     [[[-1, 0]], [[0, -1]]],
                     [[[-1, 0]], [[0, -1]]]], dtype='f')

    b_du = np.array([[[[-1, 0]], [[0, -1]]],
                     [[[-1, 0]], [[0, -1]]],
                     [[[-1, 0]], [[0, -1]]]], dtype='f')

    print('gamma_du')
    print((get_cos_theta_du(a, a_du, b, b_du)))
    print('theta_du')
    print((get_theta_du(a, a_du, b, b_du)))
