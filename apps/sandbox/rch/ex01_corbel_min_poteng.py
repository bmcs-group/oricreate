import numpy as np
from oricreate.api import CreasePatternState, CustomCPFactory
from oricreate.api import GuConstantLength, GuDofConstraints, \
    SimulationConfig, SimulationTask, fix, FTV
from oricreate.fu import \
    FuPotEngTotal


def create_cp_factory02(n=4):
    x, y = np.mgrid[0:n + 1, 0:2]
    z = np.zeros_like(x)
    X = np.c_[x.flatten(), y.flatten(), z.flatten()]
    N = np.arange((n) * 2).reshape(2, n)
    L1 = np.c_[N[0], N[1]]
    L2 = np.c_[N[0, :-1], N[0, 1:]]
    L3 = np.c_[N[1, :-1], N[1, 1:]]
    L4 = np.c_[N[0, :-1], N[1, 1:]]
    L = np.vstack([L1, L2, L3, L4])
    F1 = np.c_[N[0, :-1], N[1, 1:], N[1, :-1]]
    F2 = np.c_[N[0, :-1], N[0, :-1], N[1, 1:]]
    F = np.vstack([F1, F2])
    cp = CreasePatternState(X=X,
                            L=L,
                            F=F
                            )

    cp_factory = CustomCPFactory(formed_object=cp)
    return cp_factory


def create_cp_factory(n=4, b=1):
    '''
    define quantity of facets
    number of facets=2n-1
    n = 4
    '''

    # create coordinates

    # number of coordinates
    num_coord = 2 * n + 1

    # create array
    X = np.zeros((num_coord, 3))

    X[n + 1:2 * n + 1, 1] = b

    i = 1

    while i <= n:
        X[i, 0] = i
        X[n + i, 0] = i
        i = i + 1

    print 'X', X

    # create lines

    # number of lines
    num_lines = 4 * n - 1

    # create array
    L = np.zeros((num_lines, 2))

    i = 0

    while i <= n - 1:
        j = 3 * i
        L[j, 0] = i
        L[j, 1] = i + 1
        i = i + 1

    i = 0

    while i + 1 <= n:
        j = 3 * i + 1
        L[j, 0] = i + 1
        L[j, 1] = i + n + 1
        i = i + 1

    i = 0

    while i + 2 <= n + 1:
        j = 3 * i + 2
        L[j, 0] = i + 1 + n
        L[j, 1] = i
        i = i + 1

    i = 0

    while i <= n - 2:
        j = i + n + 1
        L[3 * n + i, 0] = j
        k = i + n + 2
        L[3 * n + i, 1] = k
        i = i + 1

    print 'L', L

    # create facets

    # number of facets
    num_facet = 2 * n - 1

    # create array
    F = np.zeros((num_facet, 3))

    i = 0

    while i <= n - 1:
        F[i, 0] = i
        F[i, 1] = i + 1
        F[i, 2] = i + n + 1
        i = i + 1

    i = 1

    while i <= n - 1:
        F[n + i - 1, 0] = i
        F[n + i - 1, 1] = i + n + 1
        F[n + i - 1, 2] = i + n
        i = i + 1

    print 'F', F

    cp = CreasePatternState(X=X,
                            L=L,
                            F=F
                            )

    cp_factory = CustomCPFactory(formed_object=cp)
    return cp_factory

if __name__ == '__main__':

    n = 10
    cp_factory_task = create_cp_factory(n=n, b=10.5)
    cp = cp_factory_task.formed_object

    #cp.x_0[4, 2] = 0.3

    # Link the crease factory it with the constraint client
    gu_constant_length = GuConstantLength()

    dof_constraints = fix([0], [0, 1, 2]) + fix([1], [1, 2]) \
        + fix([n + 1], [2])  # + fix([n * 2], [1])
    gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)

    sim_config = SimulationConfig(goal_function_type='total potential energy',
                                  debug_level=0,
                                  use_f_du=True,
                                  gu={'cl': gu_constant_length,
                                      'dofs': gu_dof_constraints},
                                  acc=1e-7, MAX_ITER=1000)

    F_nodes = np.linspace(0, -10, n)
    F_ext_list = [(i + 1, 2, F_n) for i, F_n in enumerate(F_nodes)]
    #F_ext_list = [(2 * n, 2, -10)]

    fu_tot_poteng = FuPotEngTotal(kappa=np.array([5000.0]), fu_factor=1,
                                  F_ext_list=F_ext_list)  # (2 * n, 2, -1)])
    sim_config._fu = fu_tot_poteng
    sim_task = SimulationTask(previous_task=cp_factory_task,
                              config=sim_config,
                              n_steps=1)
    cp = sim_task.formed_object
    cp.x_0 = np.copy(sim_task.previous_task.formed_object.X)
    cp.u[:, :] = 0.0
    fu_tot_poteng.forming_task = sim_task

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
    ftv.add(fu_tot_poteng.viz3d_dict['node_load'])
    ftv.plot()
    ftv.update(vot=1, force=True)
    ftv.show()
