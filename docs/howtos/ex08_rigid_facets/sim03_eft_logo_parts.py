r'''

Fold the twist logo of Engineered Folding Research Center
---------------------------------------------------------

This example shows quadrilateral facets with a twist fold.

'''

import numpy as np
from oricreate.gu import GuConstantLength, GuDofConstraints, fix
from oricreate.simulation_step import \
    SimulationStep, SimulationConfig


def create_cp_factory():
    # begin

    from oricreate.api import CreasePatternState, CustomCPFactory, CreasePatternViz3D

    x = np.array([[0, 0, 0],
                  [2, 0, 0],
                  [3, 0, 0],
                  [4, 0, 0],
                  [0, 1, 0],
                  [2, 1, 0],
                  [0, 2, 0],
                  [1, 2, 0],
                  [3, 2, 0],
                  [4, 2, 0],
                  [2, 3, 0],
                  [4, 3, 0],
                  [0, 4, 0],
                  [1, 4, 0],
                  [2, 4, 0],
                  [4, 4, 0],
                  ], dtype='float_')

    L = np.array([[0, 1], [1, 2], [2, 3],
                  [4, 5],
                  [6, 7],
                  [8, 9],
                  [10, 11],
                  [12, 13], [13, 14],  [14, 15],
                  [4, 6],
                  [0, 4],
                  [6, 12],
                  [7, 13],
                  [1, 5],  [10, 14],
                  [2, 8],
                  [3, 9],
                  [9, 11],  [11, 15],
                  [7, 5],
                  [5, 8],
                  [8, 10], [10, 7]
                  ],
                 dtype='int_')

    F = np.array([[0, 1, 5, 4],
                  [1, 2, 8, 5],
                  [2, 3, 9, 8],
                  [9, 11, 10, 8],
                  [15, 14, 10, 11],
                  [14, 13, 7, 10],
                  [13, 12, 6, 7],
                  [6, 4, 5, 7],
                  [7, 5, 8, 10]
                  ], dtype='int_')

    L_range = np.arange(len(L))

    x_mid = (x[F[:, 1]] + x[F[:, 3]]) / 2.0
    x_mid[:, 2] -= 0.5
    n_F = len(F)
    n_x = len(x)
    x_mid_i = np.arange(n_x, n_x + n_F)

    L_mid = np.array([[F[:, 0], x_mid_i[:]],
                      [F[:, 1], x_mid_i[:]],
                      [F[:, 2], x_mid_i[:]],
                      [F[:, 3], x_mid_i[:]]])
    L_mid = np.vstack([L_mid[0].T, L_mid[1].T, L_mid[2].T, L_mid[3].T])

    x_derived = np.vstack([x, x_mid])
    L_derived = np.vstack([L, F[:-1, (1, 3)], L_mid])
    F_derived = np.vstack([F[:, (0, 1, 3)], F[:, (1, 2, 3)]])

    cp = CreasePatternState(X=x_derived,
                            L=L_derived,
                            F=F_derived
                            )

    cp.viz3d.L_selection = L_range

    cp.u[(2, 3, 8, 9), 2] = 0.001
    cp.u[(6, 7, 12, 13), 2] = -0.0005
    cp.u[(10, 11, 14, 15), 2] = 0.0005
    print 'n_N', cp.n_N
    print 'n_L', cp.n_L
    print 'n_free', cp.n_dofs - cp.n_L

    cp_factory = CustomCPFactory(formed_object=cp)
    # end
    return cp_factory

if __name__ == '__main__':

    cp_factory = create_cp_factory()
    cp = cp_factory.formed_object

    from oricreate.api import FTV
    ftv = FTV()
    ms = cp.viz3d.register(ftv)

    # Link the crease factory it with the constraint client
    gu_constant_length = GuConstantLength()
    dof_constraints = fix(
        [0], [0, 1, 2]) + fix([1], [1, 2]) + fix([5], [2]) + fix([3], [0], -1.9599)
    gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)

    sim_config = SimulationConfig(gu={'cl': gu_constant_length,
                                      'dofs': gu_dof_constraints})
    sim_step = SimulationStep(forming_task=cp_factory,
                              config=sim_config, acc=1e-5, MAX_ITER=1000)

#     import mayavi.mlab as m
#     m.figure(bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.6, 0.6, 0.6))

    elevation = 120

#     ms = cp.plot_mlab(m, nodes=True, lines=True,
#                       L_selection=cp.viz3d.L_selection)

    m = ftv.mlab
    f = m.gcf()

    n_u = 4
    azimuth_step = 360.0 / 2 / n_u
    phi_range = np.linspace(0.1, np.pi - 0.01, n_u)
    u_range_cos = np.cos(phi_range) - 1
    u_range = np.hstack([u_range_cos, u_range_cos[::-1]])
    fname_list = []
    for i, u in enumerate(u_range):
        gu_dof_constraints.dof_constraints[-1][-1] = u
        sim_step._solve_nr(1.0)
        f.scene.camera.azimuth(-azimuth_step)
        ms.set(points=cp.x)
        fname = 'tmp/eftlogo%03d.jpg' % i
        fname_list.append(fname)
        m.savefig(fname, figure=f, magnification=2.4)

    import string
    import os
    animation_file = 'anim01.gif'
    images = string.join(fname_list, ' ')
    destination = animation_file

    import platform
    if platform.system() == 'Linux':
        os.system('convert ' + images + ' ' + destination)
        # os.system('png2swf -o%s ' % destination + images)
    else:
        raise NotImplementedError(
            'film production available only on linux')
    print 'animation saved in', destination

    m.show()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cp.plot_mpl(ax, facets=True)
    plt.tight_layout()
    plt.show()
