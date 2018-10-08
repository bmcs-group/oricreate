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
        x1 = np.linspace(0, 4 * dx, 5)
        x2 = np.linspace(0, 5 * dx, 6)
        row0 = np.c_[dx + x1, np.zeros_like(x1), np.zeros_like(x1)]
        row1 = np.c_[-dx / 2 + x2, dy + np.zeros_like(x2), np.zeros_like(x2)]
        row2 = np.c_[x2, 2 * dy + np.zeros_like(x2), np.zeros_like(x2)]
        row3 = np.c_[-dx / 2 + x1, 3 * dy +
                     np.zeros_like(x1), np.zeros_like(x1)]
        X = np.vstack([row0, row1, row2, row3])
        L = [[0, 6],  # 0
             [2, 8],  # 1 ***
             [3, 9],
             [4, 10],
             [0, 7],
             [1, 8],  # 5
             [2, 9],
             [3, 10],  # 7 ***
             [1, 2],
             [3, 4],
             [6, 11],  # 10 ***
             [7, 12],  # 11 ***
             [8, 13],  # 12 ***
             [9, 14],  # 13 ***
             [10, 15],  # 14 ***
             [5, 11],  # 15
             [6, 12],  # 16 ***
             [7, 13],  # 17 ***
             [8, 14],  # 18 ***
             [9, 15],  # 19 ***
             [10, 16],  # 20
             [5, 6],
             [6, 7],  # 22 ***
             [7, 8],
             [8, 9],  # 24 ***
             [9, 10],  # 25 ***
             [11, 18],  # 26 ***
             [12, 19],
             [13, 20],
             [14, 21],
             [11, 17],  # 30
             [12, 18],  # 31
             [13, 19],  # 32 ***
             [15, 21],
             [11, 12],  # 34 ***
             [12, 13],  # 35 ***
             [13, 14],  # 36
             [14, 15],  # 37 ***
             [15, 16],  # 38
             [17, 18],
             [19, 20]  # 40
             ]
        F = [[0, 7, 6],
             [1, 2, 8],
             [2, 9, 8],
             [3, 10, 9],
             [3, 4, 10],
             [5, 6, 11],
             [6, 12, 11],
             [6, 7, 12],
             [7, 13, 12],
             [7, 8, 13],
             [8, 14, 13],
             [8, 9, 14],
             [9, 15, 14],
             [9, 10, 15],
             [10, 16, 15],
             [11, 18, 17],
             [11, 12, 18],
             [12, 13, 19],
             [13, 20, 19],
             [14, 15, 21]
             ]
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

        psi_max = np.pi - 138.19 / 180.0 * np.pi
        inner_lines = self.factory_task.formed_object.iL

        def fold_step(t, fold_index):

            n_steps = len(inner_lines)
            print 'n_steps', n_steps, fold_index
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

        FN = lambda i: lambda t: psi_max * fold_step(t, i)

        psi_constr = [([(L_idx, 1.0)], FN(i))
                      for i, L_idx in enumerate(inner_lines)]

        gu_psi_constraints = \
            GuPsiConstraints(forming_task=self.factory_task,
                             psi_constraints=psi_constr)

        dof_constraints = fix([6], [0, 1, 2]) + \
            fix([7], [1, 2]) + \
            fix([13], [2])

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
    bsf_process = OctahederFormingProcess(n_steps=40,
                                          phi_max=np.pi / 2.556)

    cp = bsf_process.factory_task.formed_object

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
