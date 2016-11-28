from traits.api import \
    HasTraits, Float, Property, cached_property, Instance, \
    Int

import numpy as np
from oricreate.api import \
    CustomCPFactory,  CreasePatternState,  fix, MapToSurface,\
    GuConstantLength, GuDofConstraints, GuPsiConstraints,\
    SimulationConfig, SimulationTask, \
    FTV, FTA
from oricreate.forming_tasks.forming_task import FormingTask


class OctahederFormingProcess(HasTraits):
    '''
    Define the simulation task prescribing the boundary conditions, 
    target surfaces and configuration of the algorithm itself.
    '''

    L_green = Float(1.0, auto_set=False, enter_set=True, input=True)
    L_blue = Float(1.0, auto_set=False, enter_set=True, input=True)
    L_red = Float(1.0, auto_set=False, enter_set=True, input=True)
    n_steps = Int(10, auto_set=False, enter_set=True, input=True)
    phi_max = Float(np.pi / 3., auto_set=False, enter_set=True, input=True)

    factory_task = Property(Instance(FormingTask))
    '''Factory task generating the crease pattern.
    '''
    @cached_property
    def _get_factory_task(self):
        dx = self.L_blue
        dy = self.L_blue * np.sin(np.pi / 3)
        cp = CreasePatternState(X=[[0, 0, 0],  # 0
                                   [dx, 0, 0],  # 1
                                   [-dx / 2.0, dy, 0],  # 2
                                   [dx / 2.0, dy, 0],  # 3
                                   [3 * dx / 2.0, dy, 0],  # 4
                                   [0, 2 * dy, 0],  # 5
                                   [-dx / 2.0, -dy, 0],  # 6
                                   [dx / 2.0, -dy, 0],  # 7
                                   [3 * dx / 2.0, -dy, 0],  # 8
                                   [0, 2 * -dy, 0],  # 9
                                   ],
                                L=[[0, 1],  # 1
                                   [2, 3],  # 2
                                   [3, 4],  # 3
                                   [6, 7],  # 4
                                   [7, 8],  # 5
                                   [0, 2],
                                   [0, 3],
                                   [1, 3],
                                   [1, 4],
                                   [2, 5],
                                   [3, 5],
                                   [0, 6],
                                   [0, 7],
                                   [1, 7],
                                   [1, 8],
                                   [6, 9],
                                   [7, 9],
                                   ],
                                F=[[0, 3, 2],
                                    [0, 1, 3],
                                    [1, 4, 3],
                                    [2, 3, 5],
                                    [0, 6, 7],
                                    [0, 7, 1],
                                    [1, 7, 8],
                                    [6, 9, 7]]
                                )
        return CustomCPFactory(formed_object=cp)
    init_displ_task = Property(Instance(FormingTask))
    '''Initialization to render the desired folding branch. 
    '''
    @cached_property
    def _get_init_displ_task(self):
        cp = self.factory_task.formed_object
        return MapToSurface(previous_task=self.factory_task,
                            target_faces=[(self.ctf, cp.N)])

    fold_task = Property(Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_fold_task(self):

        psi_max = self.phi_max
        gu_psi_constraints = \
            GuPsiConstraints(forming_task=self.factory_task,
                             psi_constraints=[([(0, 1.0)],
                                               lambda t: psi_max * t),
                                              ([(1, 1.0)],
                                               lambda t: psi_max * t),
                                              ([(6, 1.0)],
                                               lambda t: psi_max * t),
                                              ([(7, 1.0)],
                                               lambda t: psi_max * t),
                                              ([(3, 1.0)],
                                               lambda t: psi_max * t),
                                              ([(12, 1.0)],
                                               lambda t: psi_max * t),
                                              ([(13, 1.0)],
                                               lambda t: psi_max * t),
                                              ])

        dof_constraints = fix([0], [0, 1, 2]) + fix([1], [1, 2]) \
            + fix([3], [2])

        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
        gu_constant_length = GuConstantLength()
        sim_config = SimulationConfig(goal_function_type='none',
                                      gu={'cl': gu_constant_length,
                                          'u': gu_dof_constraints,
                                          'psi': gu_psi_constraints},
                                      acc=1e-5, MAX_ITER=500,
                                      debug_level=0)
        return SimulationTask(previous_task=self.factory_task,
                              config=sim_config, n_steps=self.n_steps)

    fold_seq_task = Property(Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_fold_seq_task(self):

        psi_max = -self.phi_max
        dt = 1.0 / 7.0

        def fold_step(t, fold_index):

            n_steps = 7.0
            dt = 1.0 / n_steps
            start_t = fold_index * dt
            end_t = (fold_index + 1) * dt

            if t < start_t:
                return 0.0
            elif t > end_t:
                return 1.0
            else:
                return (t - start_t) / (end_t - start_t)

        gu_psi_constraints = \
            GuPsiConstraints(forming_task=self.factory_task,
                             psi_constraints=[([(0, 1.0)],
                                               lambda t: psi_max * fold_step(t, 0)),
                                              ([(1, 1.0)],
                                               lambda t: psi_max * fold_step(t, 1)),
                                              ([(6, 1.0)],
                                               lambda t: psi_max * fold_step(t, 2)),
                                              ([(7, 1.0)],
                                               lambda t: psi_max * fold_step(t, 3)),
                                              ([(3, 1.0)],
                                               lambda t: psi_max * fold_step(t, 4)),
                                              ([(12, 1.0)],
                                               lambda t: psi_max * fold_step(t, 5)),
                                              ([(13, 1.0)],
                                               lambda t: psi_max * fold_step(t, 6)),
                                              ]
                             )

        dof_constraints = fix([0], [0, 1, 2]) + fix([1], [1, 2]) \
            + fix([3], [2])

        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
        gu_constant_length = GuConstantLength()
        sim_config = SimulationConfig(goal_function_type='none',
                                      gu={'cl': gu_constant_length,
                                          'u': gu_dof_constraints,
                                          'psi': gu_psi_constraints},
                                      acc=1e-5, MAX_ITER=500,
                                      debug_level=0)
        return SimulationTask(previous_task=self.factory_task,
                              config=sim_config, n_steps=self.n_steps)


if __name__ == '__main__':
    bsf_process = OctahederFormingProcess(n_steps=35,
                                          phi_max=np.pi / 2.556)
    fact = bsf_process.factory_task
    ft = bsf_process.fold_seq_task

    ft.u_1

    ftv = FTV()

    ft.sim_history.set(anim_t_start=0, anim_t_end=10)
#    ft.config.gu['dofs'].set(anim_t_start=0, anim_t_end=5)
    ft.sim_history.viz3d['cp'].set(tube_radius=0.005)
    ftv.add(ft.sim_history.viz3d['cp'])
#     ft.config.gu['dofs'].viz3d['default'].scale_factor = 0.5
#     ftv.add(ft.config.gu['dofs'].viz3d['default'])
    ft.u_1

    print 'ft_x1', ft.x_1
    cp = ft.formed_object
    print 'lt_x0', cp.x_0
    print 'lt_u', cp.u
    #cp.u[(2, 3), 2] = -0.001
    print 'lt.u_1', ft.u_1

    cp = ft.formed_object
    iL_phi = cp.iL_psi2 - cp.iL_psi_0
    print 'phi',  iL_phi

    n_cam_move = 35
    fta = FTA(ftv=ftv)
    fta.init_view(a=45.0,
                  e=54.7356103172,
                  d=7.77,
                  f=(0.500000035416,
                     0.287922133088,
                     0.409750220151),
                  r=--120.0)
    fta.add_cam_move(a=24.8592205112,
                     e=35.3839345289,
                     d=4.41536277196,
                     f=(0.500000035416,
                        0.287922133088,
                        0.409750220151),
                     r=-93.3680789265,
                     n=n_cam_move,
                     duration=10,
                     vot_start=0.0, vot_end=1.0,
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped')

    fta.plot()
    fta.configure_traits()
