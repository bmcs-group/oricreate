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


class OctaHedronFormingProcess(HasTraits):
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

        X = np.array([[0, 0],
                      [3, 0],
                      [-0.5, 1.936491673103709],
                      [-3.5, 1.936491673103709],
                      [-1.90625, -0.6051536478449089],
                      [-1.7421875, -3.600664204677208],
                      [0.1640625, -2.995510556832299],
                      [-4, 3.872983346207417],
                      [-1, 3.872983346207417],
                      [1.625, 5.325352101035199],
                      [-7, 3.872983346207417],
                      [-6.5, 1.936491673103709],
                      [-3.6484375, -4.205817852522117],
                      [-3.8125, -1.210307295689817],
                      [-1.578125, -6.596174761509507],
                      [0.328125, -5.991021113664598],
                      [-3.5, 5.809475019311126],
                      [-5.40625, 6.414628667156034],
                      [-1.4140625, -9.591685318341806],
                      [0.4921875, -8.986531670496896],
                      [-1.906250000000001, 8.351120340259744],
                      [-4.906250000000001, 8.351120340259742]
                      ], dtype=np.float_)
        X = np.c_[X[:, 0], X[:, 1], X[:, 0] * 0]
        L = [
            [1, 2],  # ***
            [2, 3],
            [1, 3],
            [1, 4],  # ***
            [3, 4],
            [1, 5],  # ***
            [4, 5],
            [1, 6],  # ***
            [5, 6],
            [1, 7],
            [6, 7],
            [4, 8],
            [3, 8],
            [8, 9],
            [3, 9],
            [9, 10],
            [3, 10],
            [4, 11],
            [8, 11],
            [4, 12],
            [11, 12],
            [6, 13],
            [5, 13],
            [13, 14],
            [5, 14],
            [7, 15],
            [6, 15],
            [7, 16],
            [15, 16],
            [11, 17],
            [8, 17],
            [11, 18],
            [17, 18],
            [16, 19],
            [15, 19],
            [16, 20],
            [19, 20],
            [18, 21],
            [17, 21],
            [18, 22],
            [21, 22],
        ]
        L = np.array(L, dtype=np.int_) - 1

        F = [[0, 1, 2],
             [0, 2, 3],
             [0, 3, 4],
             [0, 4, 5],
             [0, 5, 6],
             [2, 9, 8],
             [2, 8, 7],
             [2, 7, 3],
             [3, 7, 10],
             [3, 10, 11],
             [16, 10, 7],
             [16, 17, 10],
             [16, 20, 17],
             [17, 20, 21],
             [14, 6, 5],
             [14, 15, 6],
             [14, 18, 15],
             [15, 18, 19],
             [4, 13, 12],
             [4, 12, 5]
             ]

        F = np.array(F, dtype=np.int_)
        cp = CreasePatternState(X=X,
                                L=L,
                                F=F)
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

        psi_max = (np.pi - 138.19 / 180.0 * np.pi) * 1.0
        print psi_max

        cp = self.factory_task.formed_object
        inner_lines = cp.iL
        print 'inner_lines'

        def fold_step(t, fold_index):

            n_steps = len(inner_lines)
            dt = 1.0 / float(n_steps)
            start_t = fold_index * dt
            end_t = (fold_index + 1) * dt

            print 't', t, start_t, end_t
            if t < start_t:
                return 0.0
            elif t > end_t:
                return 1.0
            else:
                return (t - start_t) / (end_t - start_t)

        np.random.shuffle(inner_lines)

        psi_constr = [([(i, 1.0)], lambda t: psi_max * t)  # fold_step(t, i))
                      for i in inner_lines]

        gu_psi_constraints = \
            GuPsiConstraints(forming_task=self.factory_task,
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
    bsf_process = OctaHedronFormingProcess(n_steps=20,
                                           phi_max=np.pi / 2.556)

    cp = bsf_process.factory_task.formed_object

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cp.plot_mpl(ax, facets=True)
    plt.tight_layout()
    plt.show()

    fact = bsf_process.factory_task
#     ft = bsf_process.fold_seq_task
    fts = bsf_process.fold_task

#    ft.u_1
    fts.u_1

    ftv = FTV()

#     ft.sim_history.set(anim_t_start=0, anim_t_end=10)
#     ft.sim_history.viz3d['cp'].set(tube_radius=0.005)
#     ftv.add(ft.sim_history.viz3d['cp'])
    fts.sim_history.set(anim_t_start=0, anim_t_end=20)
    fts.sim_history.viz3d['cp'].set(tube_radius=0.005)
    ftv.add(fts.sim_history.viz3d['cp'])

    ftv.plot()
    ftv.configure_traits()

    n_cam_move = 20
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
    fta.add_cam_move(a=24.8592205112,
                     e=35.3839345289,
                     d=4.41536277196,
                     f=(0.500000035416,
                        0.287922133088,
                        0.409750220151),
                     r=-93.3680789265,
                     n=n_cam_move,
                     duration=10,
                     vot_start=1.0, vot_end=0.0,
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped')

    fta.plot()
    fta.configure_traits()
