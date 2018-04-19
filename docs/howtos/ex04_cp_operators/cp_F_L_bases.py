r'''

This example demonstrates the crease pattern factory usage
for Yoshimura crease pattern.


'''


def create_cp():
    # begin
    from oricreate.api import CreasePatternState
    import numpy as np

    x = np.array([[0, 0, 0],
                  [1, 0, 0],
                  [1, 1, 0],
                  [2, 1, 0]
                  ], dtype='float_')

    L = np.array([[0, 1], [1, 2], [2, 0],
                  [1, 3], [2, 3]],
                 dtype='int_')

    F = np.array([[0, 1, 2],
                  [1, 3, 2],
                  ], dtype='int_')

    cp = CreasePatternState(X=x,
                            L=L,
                            F=F)

    print('Initial configuration')
    print('Orthonormal base vectors of a first edge\n', cp.F_L_bases[:, 0, :, :])

    return cp

    cp.u[1, 2] = 1.0
    cp.u[2, 2] = 1.0
    cp.u = cp.u

    print('Displaced configuration')
    print('Orthonormal base vectors of a first edge r\n', cp.F_L_bases[:, 0, :, :])

    # end
    return cp

if __name__ == '__main__':
    create_cp()
