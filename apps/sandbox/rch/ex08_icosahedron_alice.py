from oricreate.api import \
    CustomCPFactory,  CreasePatternState,  fix, link,\
    FuNodeDist, \
    GuConstantLength, GuDofConstraints, GuPsiConstraints,\
    SimulationConfig, SimulationTask, \
    FTV, FTA
from oricreate.forming_tasks.forming_task import FormingTask
from oricreate.view import FormingTaskTree
from traits.api import \
    HasTraits, Float, Property, cached_property, Instance, \
    Int

import numpy as np


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

    fold_task = Property(Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_fold_task(self):

        psi_max = self.fix_psi

        cp = self.factory_task.formed_object
        inner_lines = cp.iL

        psi_constr = [([(i, 1.0)], lambda t: psi_max * t)
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

    fix_psi = Float((np.pi - 138.19 / 180.0 * np.pi) * 0.5)

    def get_single_step_fold_task(self, seq_fold_task):

        ft = self.factory_task
        seq_cp = seq_fold_task.formed_object
        iL = seq_cp.iL
        iL_psi = seq_cp.iL_psi

        def fold_step(t, start_t=0.0, end_t=1.0):
            if t < start_t:
                return 0.0
            elif t > end_t:
                return 1.0
            else:
                return (t - start_t) / (end_t - start_t)

        FN = lambda psi, start_t, end_t: lambda t: psi * \
            fold_step(t, start_t, end_t)

        print('iL', iL)

        trange = np.zeros((len(iL), 2), dtype=np.float_)
        trange[:, 0] = 0.0
        trange[:, 1] = 0.4
        trange[2, :] = [0.3, 0.6]
        trange[7, :] = [0.5, 0.8]
        trange[10, :] = [0.5, 0.8]
        trange[11, :] = [0.6, 0.8]
        trange[15, :] = [0.6, 0.8]
        trange[16, :] = [0.8, 1.0]
        trange[18, :] = [0.6, 0.8]
        trange[14, :] = [0.8, 1.0]

        psi_constraints = [([(i, 1.0)], FN(i_psi, i_start, i_end))
                           for i, i_psi, i_start, i_end
                           in zip(iL, iL_psi, trange[:, 0], trange[:, 1])]

        gu_psi_constraints = \
            GuPsiConstraints(forming_task=ft,
                             psi_constraints=psi_constraints)

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
        return SimulationTask(previous_task=ft,
                              config=sim_config, n_steps=50)

    fix_psi = Float(-(np.pi - 138.19 / 180.0 * np.pi) * 0.5)

    def get_merge_nodes_task(self, prev_ft, merge_nodes=[[]],
                             link_nodes=[], fix_node=[],
                             stiff_lines=[], init_nodes=(), init_val=0.0):

        cp = self.factory_task.formed_object

        psi_constr = [([(i, 1.0)], lambda t: self.fix_psi * t)  # fold_step(t, i))
                      for i in stiff_lines]

        gu_psi_constraints = \
            GuPsiConstraints(forming_task=prev_ft,
                             psi_constraints=psi_constr)

        F_u_fix = cp.F_N[1]
        dof_constraints = fix([F_u_fix[0]], [0, 1, 2]) + \
            fix([F_u_fix[1]], [1, 2]) + \
            fix([F_u_fix[2]], [2])

        if len(link_nodes) > 0:
            ln = np.array(link_nodes, dtype=np.int_)
            link_nodes1, link_nodes2 = ln.T
            ldofs = link(
                link_nodes1, [0, 1, 2], 1.0,
                link_nodes2, [0, 1, 2], -1.0)
            dof_constraints += ldofs

        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
        gu_constant_length = GuConstantLength()

        sim_config = SimulationConfig(goal_function_type='total potential energy',
                                      gu={'cl': gu_constant_length,
                                          'gu': gu_dof_constraints,
                                          'psi': gu_psi_constraints},
                                      acc=1e-5, MAX_ITER=500)

        fu_node_dist = \
            FuNodeDist(forming_task=prev_ft,
                       L=merge_nodes,
                       )

        sim_config._fu = fu_node_dist

        st = SimulationTask(previous_task=prev_ft,
                            config=sim_config,
                            n_steps=1)

        fu_node_dist.forming_task = st

        cp = st.formed_object
        return st

if __name__ == '__main__':
    bsf_process = OctaHedronFormingProcess(n_steps=1,
                                           phi_max=np.pi / 2.556)

    cp = bsf_process.factory_task.formed_object

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        cp.plot_mpl(ax, facets=True)
        plt.tight_layout()
        plt.show()

    fact = bsf_process.factory_task

    fold_task = bsf_process.fold_task

    fold_task.u_1

    fts1 = bsf_process.get_merge_nodes_task(
        prev_ft=fold_task,
        merge_nodes=[[1, 9],
                     [6, 1]],
        stiff_lines=[10, 25,
                     28, 33, 8, 22,
                     11, 17, 18, 29, 32, 37],
    )
    fts1.u_1

    fts2 = bsf_process.get_merge_nodes_task(
        prev_ft=fold_task,
        merge_nodes=[[11, 4],
                     [13, 10],
                     [15, 8],
                     [1, 9],
                     [6, 1]],
        stiff_lines=[28, 33,
                     18, 29, 32, 37],
    )

    fts2.u_1

    fts3 = bsf_process.get_merge_nodes_task(
        prev_ft=fts2,
        merge_nodes=[[20, 14],
                     [16, 18],
                     [19, 7],
                     [11, 4],
                     [13, 10],
                     [15, 8],
                     [1, 9],
                     [6, 1]],
        stiff_lines=[],
    )

    fts3.u_1

    fts4 = bsf_process.get_merge_nodes_task(
        prev_ft=fts3,
        merge_nodes=[[21, 5],
                     [17, 12],
                     [20, 14],
                     [16, 18],
                     [19, 7],
                     [11, 4],
                     [13, 10],
                     [15, 8],
                     [1, 9],
                     [6, 1]],
        stiff_lines=[],
    )

    fts4.u_1

    single_step_fold_task = bsf_process.get_single_step_fold_task(fts4)

    single_step_fold_task.u_1

    sscp = single_step_fold_task.formed_object
    print('phi', sscp.iL_psi)

    fts = fts4
    fts = single_step_fold_task

    #ftt = FormingTaskTree(root=bsf_process.factory_task)
    # ftt.configure_traits()

    ftv = FTV()

#     ft.sim_history.set(anim_t_start=0, anim_t_end=10)
#     ft.sim_history.viz3d['cp'].set(tube_radius=0.005)
#     ftv.add(ft.sim_history.viz3d['cp'])
    fts.sim_history.set(anim_t_start=0, anim_t_end=100)
    fts.sim_history.viz3d['cp'].set(tube_radius=0.01)
    ftv.add(fts.sim_history.viz3d['cp'])

    ftv.plot()
    ftv.configure_traits()

    if True:
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

        fta.plot()
        fta.configure_traits()
