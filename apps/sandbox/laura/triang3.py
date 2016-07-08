import numpy as np
from oricreate.api import CreasePatternState, CustomCPFactory
from oricreate.api import GuConstantLength, GuDofConstraints, \
    SimulationConfig, SimulationTask, fix, FTV
from oricreate.fu import \
    FuTotalPotentialEnergy
        
def create_cp_factory(n=1):
    
    from oricreate.api import CreasePatternState, CustomCPFactory
          
    x, y = np.mgrid[0:n + 1, 0:2]
    z = np.zeros_like(x)
    X1 = np.c_[-1,0,0]
    X2 = np.c_[x.flatten(), y.flatten(), z.flatten()]
    X = np.vstack([X1,X2])
    print 'X', X
    N = np.arange(1,(2*n)+3).reshape(-1, 2)
    print 'N', N
    L00 = np.c_[0,2]
    L01 = np.c_[1,0]
    L1 = np.c_[N[:-1, 1], N[1:, 0]]
    L2 = np.c_[N[0:, 0], N[0:, 1]]
    L3 = np.c_[N[:-1, 0], N[1:, 0]]
    L4 = np.c_[N[:-1, 1], N[1:, 1]]
    L = np.vstack([L00, L01, L1, L2, L3, L4])
    print 'L', L
    F0 = np.c_[0, 1, 2]
    F1 = np.c_[N[:-1, 0], N[1:, 0], N[:-1, 1]]
    F2 = np.c_[N[1:, 0], N[1:, 1], N[:-1, 1]]
    F = np.vstack([F0, F1, F2])
    print 'F', F
    
    cp = CreasePatternState(X=X,
                            L=L,
                            F=F
                            )

    cp_factory = CustomCPFactory(formed_object=cp)
    return cp_factory

# if __name__ == '__main__':
#     cp_factory = create_cp_factory()
#     cp = cp_factory.formed_object
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     cp.plot_mpl(ax, facets=True)
#     plt.tight_layout()
#     plt.show()

if __name__ == '__main__':
 
    n = 1
    cp_factory_task = create_cp_factory(n=n)
    cp = cp_factory_task.formed_object
    cp.x_0[3, 2] = 0.3
 
    # Link the crease factory it with the constraint client
    gu_constant_length = GuConstantLength()
 
    dof_constraints = fix([0], [0,2]) + fix([2], [1, 2]) \
        + fix([4], [1])
    gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
 
    sim_config = SimulationConfig(goal_function_type='potential_energy',
                                  debug_level=0,
                                  use_f_du=False,
                                  gu={'cl': gu_constant_length,
                                      'dofs': gu_dof_constraints},
                                  acc=1e-5, MAX_ITER=1000)
 
    F_ext_list = [(3, 2, -1)]
 
    fu_tot_poteng = FuTotalPotentialEnergy(kappa=1000,
                                           F_ext_list=F_ext_list)
    sim_config._fu = fu_tot_poteng
    sim_task = SimulationTask(previous_task=cp_factory_task,
                              config=sim_config,
                              n_steps=1)
 
    cp.u[:, :] = 0.001
    print 'kinematic simulation: u', sim_task.u_1
 
    cp = sim_task.formed_object
    iL_phi = cp.iL_psi2 - cp.iL_psi_0
    print 'phi',  iL_phi
    iL_length = np.linalg.norm(cp.iL_vectors, axis=1)
    iL_m = sim_config._fu.kappa * iL_phi * iL_length
 
    print 'moments', iL_m
 
    ftv = FTV()
    ftv.add(sim_task.formed_object.viz3d)
    ftv.add(sim_task.formed_object.viz3d_dict['node_numbers'], order=5)
    ftv.add(gu_dof_constraints.viz3d)
    ftv.add(fu_tot_poteng.viz3d)
    ftv.plot()
    ftv.update(vot=1, force=True)
    ftv.show()

