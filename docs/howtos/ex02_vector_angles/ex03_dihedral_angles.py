import numpy as np
from oricreate.api import \
    CreasePatternState, CustomCPFactory, \
    GuPsiConstraints, fix, GuDofConstraints, \
    GuConstantLength, SimulationConfig, SimulationTask, \
    FTV


def create_cp_factory():

    cp = CreasePatternState(X=[[0, 0, 0],
                               [1, 0, 0],
                               [1, 1, 0],
                               [2, 1, 0]
                               ],
                            L=[[0, 1], [1, 2], [2, 0], [1, 3],
                                [3, 2]],
                            F=[[0, 1, 2], [1, 3, 2]]
                            )

    cp_factory = CustomCPFactory(formed_object=cp)
    # end
    return cp_factory


if __name__ == '__main__':
    cpf = create_cp_factory()
    cp = cpf.formed_object
    cp.u[3, 2] = 0.0
    cp.u[3, 0] = 0.0
    print('F_normals', cp.norm_F_normals)
    print('iL_psi', cp.iL_psi)
    print('iL_psi_du', cp.iL_psi_du)

    psi_max = 3.999 * np.pi / 4.0
    psi_constr = [([(i, 1.0)], lambda t: psi_max * t)
                  for i in cp.iL]

    gu_psi_constraints = \
        GuPsiConstraints(forming_task=cpf,
                         psi_constraints=psi_constr)

    F_u_fix = cp.F_N[1]
    dof_constraints = fix([F_u_fix[0]], [0, 1, 2]) + \
        fix([F_u_fix[1]], [1, 2]) + \
        fix([F_u_fix[2]], [2])

    gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
    gu_constant_length = GuConstantLength()
    sim_config = SimulationConfig(goal_function_type='none',
                                  gu={'cl': gu_constant_length,
                                      'u': gu_dof_constraints,
                                      'psi': gu_psi_constraints},
                                  acc=1e-5, MAX_ITER=500,
                                  debug_level=0)
    st = SimulationTask(previous_task=cpf,
                        config=sim_config, n_steps=10)
    st.u_1

    ftv = FTV()

    ftv.add(st.sim_history.viz3d['cp'])
    ftv.plot()
    ftv.configure_traits()
