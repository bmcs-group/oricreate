import numpy as np
from oricreate.api import CreasePatternState, CustomCPFactory
from oricreate.api import CreasePatternState, CustomCPFactory
from oricreate.api import GuConstantLength, GuDofConstraints, \
    SimulationConfig, SimulationTask, fix, FTV
from oricreate.api import YoshimuraCPFactory
from oricreate.fu import \
    FuPotEngTotal


# def create_cp_factory(n=4):
#     x, y = np.mgrid[0:n + 1, 0:2]
#     z = np.zeros_like(x)
#     X = np.c_[x.flatten(), y.flatten(), z.flatten()]
#     N = np.arange((2*n)+2).reshape(-1, 2)
#     L1 = np.c_[N[:-1, 0], N[1:, 1]]
#     L2 = np.c_[N[0:, 0], N[0:, 1]]
#     L3 = np.c_[N[:-1, 0], N[1:, 0]]
#     L4 = np.c_[N[:-1, 1], N[1:, 1]]
#     L = np.vstack([L1, L2, L3, L4])
#     F1 = np.c_[N[:-1, 0], N[1:, 0], N[1:, 1]]
#     F2 = np.c_[N[:-1, 0], N[1:, 1], N[:-1, 1]]
#     F = np.vstack([F1, F2])
#     cp = CreasePatternState(X=X,
#                             L=L,
#                             F=F
#                             )
#
#     cp_factory = CustomCPFactory(formed_object=cp)
#     return cp_factory
def create_cp_factory():

    cp_factory = YoshimuraCPFactory(L_x=4, L_y=1, n_x=2, n_y=2)
    cp = cp_factory.formed_object

    cp_factory = CustomCPFactory(formed_object=cp)
    return cp_factory


if __name__ == '__main__':

    n = 2
    cp_factory_task = create_cp_factory()
    cp = cp_factory_task.formed_object

#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     cp.plot_mpl(ax, facets=True)
#     plt.tight_layout()
#     plt.show()

#     iL_F0 = cp.iL_F[:, 0]
#     print 'iL_F0', iL_F0.shape
#     L_of_F0_of_iL = cp.F_L[iL_F0, :]
#     print 'L_of_F0_of_iL.shape', L_of_F0_of_iL.shape
#     print 'L_of_F0_of_iL', L_of_F0_of_iL
#     iL = cp.iL
#     print 'iL', iL
#     iL_within_F0 = np.where(iL[:, np.newaxis] == L_of_F0_of_iL)
#     print 'F', cp.F.shape
#     print 'F_L_vectors', cp.F_L_vectors.shape
#     print 'iL_within_F0', iL_within_F0
#     print 'iL_vectors', cp.iL_vectors.shape
#     print 'iF_L_vectors', cp.F_L_vectors[iL_within_F0]

    # Link the crease factory it with the constraint client
    gu_constant_length = GuConstantLength()

    dof_constraints = fix([0], [0, 1, 2]) + fix([1], [1, 2]) \
        + fix([3], [2])
    gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)

    sim_config = SimulationConfig(goal_function_type='total potential energy',
                                  debug_level=0,
                                  use_f_du=False,
                                  gu={'cl': gu_constant_length,
                                      'dofs': gu_dof_constraints},
                                  acc=1e-5, MAX_ITER=1000)

    F_nodes = np.linspace(0, -10, n)
    F_ext_list = [(i + 1, 2, F_n) for i, F_n in enumerate(F_nodes)]

    fu_tot_poteng = FuPotEngTotal(kappa=1000,
                                  F_ext_list=F_ext_list)  # (2 * n, 2, -1)])
    sim_config._fu = fu_tot_poteng
    sim_task = SimulationTask(previous_task=cp_factory_task,
                              config=sim_config,
                              n_steps=1)

    cp.u[1:n + 1, 2] = np.linspace(0, -0.001, n)
    cp.u[n + 1:(2 * n) + 1, 2] = np.linspace(0, -0.0005, n)
    print 'kinematic simulation: u', sim_task.u_1

    cp = sim_task.formed_object
    iL_phi = cp.iL_psi2 - cp.iL_psi_0
    print 'phi',  iL_phi
    iL_length = np.linalg.norm(cp.iL_vectors, axis=1)
    iL_m = sim_config._fu.kappa * iL_phi * iL_length

    print 'moments', iL_m

    ftv = FTV()
    ftv.add(sim_task.formed_object.viz3d)
    ftv.add(gu_dof_constraints.viz3d)
    ftv.add(fu_tot_poteng.viz3d)
    ftv.plot()
    ftv.update(vot=1, force=True)
    ftv.show()
