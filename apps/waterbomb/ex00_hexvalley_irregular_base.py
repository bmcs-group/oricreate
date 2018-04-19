'''
Created on Jan 20, 2016

@author: rch
'''

from traits.api import \
    Float, HasTraits, Property, cached_property, Int, \
    Instance, Array, Bool, List

import numpy as np
from oricreate.api import CustomCPFactory, \
    fix, link, r_, s_, t_, MapToSurface,\
    GuConstantLength, GuDofConstraints, \
    GuPsiConstraints, SimulationConfig, SimulationTask, \
    FTV, FTA
from oricreate.api import MappingTask
from oricreate.crease_pattern.crease_pattern_state import CreasePatternState
from oricreate.export import \
    InfoCadMeshExporter, ScaffoldingExporter
from oricreate.forming_tasks.forming_task import FormingTask
from oricreate.fu import \
    FuPotEngTotal
from oricreate.mapping_tasks.mask_task import MaskTask
from oricreate.simulation_tasks.simulation_history import \
    SimulationHistory
import sympy as sp


class HexYoshiFormingProcess(HasTraits):
    '''
    Define the simulation task prescribing the boundary conditions, 
    target surfaces and configuration of the algorithm itself.
    '''

    u_max = Float(0.1, auto_set=False, enter_set=True, input=True)
    n_fold_steps = Int(30, auto_set=False, enter_set=True, input=True)
    n_load_steps = Int(30, auto_set=False, enter_set=True, input=True)

    factory_task = Property(Instance(FormingTask))
    '''Factory task generating the crease pattern.
    '''
    @cached_property
    def _get_factory_task(self):
        h_long = 2.0
        h_short = 0.9
        v = 4.0
        X_base = [[0, 0, 0],
                  [-h_short, 0, 0],
                  [-h_long / 2.0, -v, 0],
                  [h_long / 2.0, -v, 0],
                  [h_short, 0, 0],
                  [h_long / 2.0, v, 0],
                  [-h_long / 2.0, v, 0],
                  ]
        X_add_left = [
            [-h_long - h_short, 0, 0],
            [-h_short - h_long / 2.0, -v, 0],
            [-h_short - h_long / 2.0, v, 0],
            [-2 * h_short - h_long / 2.0, -v, 0],
            [-2 * h_short - h_long / 2.0, v, 0],
            [-2.0 * h_short - h_long, 0, 0],
            [-2 * h_short - 3. / 2. * h_long, -v, 0],
            [-2 * h_short - 3. / 2. * h_long, v, 0],
        ]
        L_base = [[0, 1],
                  [0, 2],
                  [0, 3],
                  [0, 4],
                  [0, 5],
                  [0, 6],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 1]
                  ]
        L_add_left = [[1, 7],
                      [1, 8],
                      [1, 9],
                      [7, 8],
                      [7, 9],
                      [2, 8],
                      [6, 9],
                      [7, 10],
                      [7, 11],
                      [8, 10],
                      [9, 11],
                      [10, 12],
                      [11, 12],
                      [7, 12],
                      [12, 13],
                      [12, 14],
                      [10, 13],
                      [11, 14]
                      ]
        F_base = [[0, 1, 2],
                  [0, 2, 3],
                  [0, 3, 4],
                  [0, 4, 5],
                  [0, 5, 6],
                  [0, 6, 1]
                  ]
        F_add_left = [[1, 7, 8],
                      [1, 7, 9],
                      [1, 8, 2],
                      [1, 9, 6],
                      [7, 8, 10],
                      [7, 9, 11],
                      [7, 12, 10],
                      [7, 12, 11],
                      [12, 13, 10],
                      [12, 14, 11]
                      ]
        cp = CreasePatternState(X=X_base + X_add_left,
                                L=L_base + L_add_left,
                                F=F_base + F_add_left)
        return CustomCPFactory(formed_object=cp)

    psi_lines = List([1, 2, 4, 13, 14, 15, 16, 23, 24])

    psi_max = Float(-np.pi * 0.54)

    fold_angle_cntl = Property(Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_fold_angle_cntl(self):
        fixed_nodes_x = fix(
            [2], (0))
        fixed_nodes_y = fix(
            [2, 3], (1))
        fixed_nodes_z = fix(
            [2, 3, 6], (2))

        dof_constraints = fixed_nodes_x + fixed_nodes_z + fixed_nodes_y

        def FN(psi): return lambda t: psi * t
        psi_constr = [([(i, 1.0)], FN(self.psi_max))
                      for i in self.psi_lines]

        gu_psi_constraints = \
            GuPsiConstraints(forming_task=self.factory_task,
                             psi_constraints=psi_constr)

        gu_dof_constraints = GuDofConstraints(
            dof_constraints=dof_constraints)

        gu_constant_length = GuConstantLength()
        sim_config = SimulationConfig(goal_function_type='none',
                                      gu={'cl': gu_constant_length,
                                          'u': gu_dof_constraints,
                                          'psi': gu_psi_constraints
                                          },
                                      acc=1e-5, MAX_ITER=500,
                                      debug_level=0)

        st = SimulationTask(previous_task=self.factory_task,
                            config=sim_config, n_steps=self.n_fold_steps)

        cp = st.formed_object
        cp.u[(0, 12), 2] -= 0.4
        cp.u[(7, 1, 4), 2] += 0.4
        cp.u[(8, 9), 2] -= 0.4

        return st


class HexYoshiFormingProcessFTV(FTV):

    model = Instance(HexYoshiFormingProcess)


if __name__ == '__main__':
    bsf_process = HexYoshiFormingProcess(u_max=3.999,
                                         n_fold_steps=10,
                                         n_load_steps=10)

    ftv = HexYoshiFormingProcessFTV(model=bsf_process)

    fa = bsf_process.factory_task

    if True:
        import pylab as p
        ax = p.axes()
        fa.formed_object.plot_mpl(ax)
        p.show()

    show_fold_angle_cntl = True

    fta = FTA(ftv=ftv)
    fta.init_view(a=33.4389721223,
                  e=61.453898329,
                  d=5.0,
                  f=(1.58015494765,
                     1.12671403563,
                     -0.111520325399),
                  r=-105.783218753)

    if show_fold_angle_cntl:
        ft = bsf_process.fold_angle_cntl
        print 'NDOFS', ft.formed_object.n_dofs
        print ft.sim_step
        ft.sim_history.set(anim_t_start=0, anim_t_end=10)
        ft.config.gu['u'].set(anim_t_start=0, anim_t_end=5)
        ft.sim_history.viz3d['cp'].set(tube_radius=0.002)
        ftv.add(ft.sim_history.viz3d['cp'])
#        ftv.add(ft.sim_history.viz3d['node_numbers'])
        ft.config.gu['u'].viz3d['default'].scale_factor = 0.5
        ftv.add(ft.config.gu['u'].viz3d['default'])
        ft.u_1
        fta.add_cam_move(duration=10, n=20)

    fta.plot()
    fta.configure_traits()
