r'''

This example demonstrates the crease pattern factory usage
for Yoshimura crease pattern.


'''


def create_cp():
    # begin
    from oricreate.api import YoshimuraCPFactory

    cp_factory = YoshimuraCPFactory(n_x=1, n_y=2, L_x=2, L_y=4)
    cp = cp_factory.formed_object

    print('Initial configuration')
    print('Integration point positions r within facets:\n', cp.Fa_r)
    print('Normal vectors at integration positions r\n', cp.Fa_normals)

    cp.u[5, 2] = 1.0
    cp.u[6, 2] = 1.0
    cp.u = cp.u

    print('Displaced configuration')
    print('Integration point positions r within facets:\n', cp.Fa_r)
    print('Normal vectors at integration positions r\n', cp.Fa_normals)

    # end
    return cp

if __name__ == '__main__':
    create_cp()
