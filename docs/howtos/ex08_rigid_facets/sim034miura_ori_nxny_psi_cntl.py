r'''

Fold the Miura ori crease pattern using psi control
---------------------------------------------------

'''

import numpy as np
from oricreate.api import \
    SimulationTask, SimulationConfig, \
    FTV, FTA
from oricreate.gu import \
    GuConstantLength, GuDofConstraints, GuPsiConstraints, fix


def create_cp_factory():
    # begin
    from oricreate.api import MiuraOriCPFactory
    cp_factory = MiuraOriCPFactory(L_x=30,
                                   L_y=21,
                                   n_x=8,
                                   n_y=8,
                                   d_0=0.6,
                                   d_1=-0.6)
    # end
    return cp_factory


if __name__ == '__main__':

    cpf = create_cp_factory()
    cp = cpf.formed_object

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        cp.plot_mpl(ax, facets=True)
        plt.tight_layout()
        plt.show()

    # Link the crease factory it with the constraint client
    gu_constant_length = GuConstantLength()
    nx2 = 1
    ny2 = 2
    dof_constraints = fix(cpf.N_grid[nx2 - 1, ny2], [1]) \
        + fix(cpf.N_grid[nx2, ny2], [0, 1, 2]) \
        + fix(cpf.N_grid[nx2, (ny2 - 1, ny2 + 1)], [2])
    gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
    psi_max = np.pi * 0.498
    diag_psi_mask = np.ones_like(cpf.L_d_grid, dtype=np.bool_)
    diag_psi_mask[1:-1, 1:-1] = False
    diag_psi_constraints = [([(i, 1.0)], 0)
                            for i in cpf.L_d_grid[diag_psi_mask].flatten()]
    print('psi constraints', diag_psi_constraints)
    print('controlled line', cpf.L_h_grid[nx2, ny2])
    gu_psi_constraints = \
        GuPsiConstraints(forming_task=cpf,
                         psi_constraints=diag_psi_constraints +
                         [([(cpf.L_h_grid[nx2, ny2], 1.0)],
                           lambda t: psi_max * t),
                          ])

    sim_config = SimulationConfig(goal_function_type='none',
                                  gu={'cl': gu_constant_length,
                                      'dofs': gu_dof_constraints,
                                      'psi': gu_psi_constraints},
                                  acc=1e-8, MAX_ITER=200)
    sim_task = SimulationTask(previous_task=cpf,
                              config=sim_config,
                              n_steps=20)

    cp = sim_task.formed_object
    cp.u[cpf.N_grid[::2, :].flatten(), 2] = -0.1
    cp.u[cpf.N_grid[0, ::2].flatten(), 2] = -0.2
    sim_task.u_1

    import pylab as p
    cp.plot_mpl(p.axes())
    p.show()

    ftv = FTV()
#    ftv.add(sim_task.sim_history.viz3d['node_numbers'], order=5)
    ftv.add(sim_task.sim_history.viz3d['cp'])
    ftv.add(gu_dof_constraints.viz3d['default'])

    fta = FTA(ftv=ftv)
    fta.init_view(a=-45, e=65, d=60, f=(0, 0, 0), r=-50)
    fta.add_cam_move(a=-80, e=80, n=20, d=35, r=-45,
                     duration=10,
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped')

    fta.plot()
    fta.configure_traits()
