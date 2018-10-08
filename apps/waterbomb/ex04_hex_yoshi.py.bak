'''
Created on Jan 20, 2016

@author: rch
'''

from traits.api import \
    Float, HasTraits, Property, cached_property, Int, \
    Instance, Array, Bool

import numpy as np
from oricreate.api import MappingTask
from oricreate.api import YoshimuraCPFactory, \
    fix, link, r_, s_, t_, MapToSurface,\
    GuConstantLength, GuDofConstraints, SimulationConfig, SimulationTask, \
    FTV, FTA
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

    L_x = Float(3.0, auto_set=False, enter_set=True, input=True)
    L_y = Float(2.2, auto_set=False, enter_set=True, input=True)
    u_max = Float(0.1, auto_set=False, enter_set=True, input=True)
    n_fold_steps = Int(30, auto_set=False, enter_set=True, input=True)
    n_load_steps = Int(30, auto_set=False, enter_set=True, input=True)

    stiffening_boundary = Bool(False)

    ctf = Property(depends_on='+input')
    '''control target surface'''
    @cached_property
    def _get_ctf(self):
        h = 0.05
        return [r_, s_, h * t_ * s_ * (1 - s_ / self.L_y) - .01 * h * t_]

    factory_task = Property(Instance(FormingTask))
    '''Factory task generating the crease pattern.
    '''
    @cached_property
    def _get_factory_task(self):
        return YoshimuraCPFactory(L_x=self.L_x, L_y=self.L_y,
                                  n_x=self.n_x, n_y=self.n_y)

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
        self.init_displ_task.x_1
        u_max = self.u_max
        fa = self.factory_task
        N_h = fa.N_h
        N_i = fa.N_i
        N_v = fa.N_v
        n_x, n_y = N_h.shape
        mid_n_x = n_x / 2
        i_fixed_x = N_h[mid_n_x, (0, -1)]
        fixed_nodes_x = fix(
            i_fixed_x, (0))

        n_x, n_y = N_i.shape
        mid_n_y = n_y / 2
        i_fixed_z = np.hstack([N_i[:-1:3, mid_n_y],
                               N_i[1::3, mid_n_y]])
        i_fixed_z = np.hstack([N_i[:-1:6, mid_n_y],
                               N_i[1::6, mid_n_y]])
        fixed_nodes_z = fix(
            i_fixed_z, (2))

        i_fixed_y = N_i[(1, -2), mid_n_y]
        fixed_nodes_y = fix(
            i_fixed_y, (1))

        i_link_xh_r = N_h[2::3, :]
        i_link_xh_l = N_h[0:-2:3, :]
        i_link_xi_r = N_i[3::3, :]
        i_link_xi_l = N_i[1:-2:3, :]
        link_mid = link(
            np.hstack([i_link_xh_r.flatten(), i_link_xi_r.flatten()]),
            (0), 1.0,
            np.hstack([i_link_xh_l.flatten(), i_link_xi_l.flatten()]),
            (0), -1.0,
            lambda t: -t * u_max
        )
        sym_y = link(
            N_h[(0, -1), 0], [2], 1.0,
            N_h[(0, -1), -1], [2], -1.0
        )

        dof_constraints = fixed_nodes_x + fixed_nodes_z + fixed_nodes_y + \
            link_mid + sym_y
        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
        gu_constant_length = GuConstantLength()
        sim_config = SimulationConfig(goal_function_type='gravity potential energy',
                                      gu={'cl': gu_constant_length,
                                          'dofs': gu_dof_constraints},
                                      acc=1e-5, MAX_ITER=500,
                                      debug_level=0)

        st = SimulationTask(previous_task=self.init_displ_task,
                            config=sim_config, n_steps=self.n_fold_steps)

        cp = st.formed_object

        d_u = 0.02
        i_down_h = N_h[1::3, :]
        i_down_i = N_i[2::3, :]
        i_left = N_v[0, :]
        i_right = N_v[-1, :]
        cp.u[np.hstack([i_down_h.flatten(), i_down_i.flatten()]), 2] -= d_u
        cp.u[i_left, 0] -= d_u
        cp.u[i_right, 0] += d_u
#         cp.u[(
#             36, 48, 57, 39,
#             37, 49, 58, 40,
#             38, 50, 59, 41,
#             4, 16, 28,
#             5, 17, 29,
#             6, 18, 30,
#             7, 19, 31,
#         ), 2] -= 0.2

        return st


class HexYoshiFormingProcessFTV(FTV):

    model = Instance(HexYoshiFormingProcess)


if __name__ == '__main__':
    bsf_process = HexYoshiFormingProcess(L_x=4, L_y=8, n_x=8,
                                         n_y=6, u_max=0.99,
                                         n_fold_steps=150,
                                         n_load_steps=1)

    ftv = HexYoshiFormingProcessFTV(model=bsf_process)

    fa = bsf_process.factory_task

    it = bsf_process.init_displ_task
    ft = bsf_process.fold_task

    cp = ft.formed_object
    print 'n_dofs', cp.n_dofs
    # print ft.sim_step

    if True:
        import pylab as p
        ax = p.axes()
        fa.formed_object.plot_mpl(ax, nodes=False, lines=False, facets=False)
        p.show()

    show_init_task = False
    show_fold_task = True

    fta = FTA(ftv=ftv)
    fta.init_view(a=33.4389721223,
                  e=61.453898329,
                  d=5.0,
                  f=(1.58015494765,
                     1.12671403563,
                     -0.111520325399),
                  r=-105.783218753)

    if show_init_task:
        ftv.add(it.target_faces[0].viz3d['default'])
        it.formed_object.viz3d['cp'].set(tube_radius=0.002)
        ftv.add(it.formed_object.viz3d['cp'])
        #ftv.add(it.formed_object.viz3d['node_numbers'], order=5)
        it.u_1

    if show_fold_task:
        ft.sim_history.set(anim_t_start=0, anim_t_end=10)
#        ft.config.gu['dofs'].set(anim_t_start=0, anim_t_end=5)
        ft.sim_history.viz3d['cp'].set(tube_radius=0.02)
        ftv.add(ft.sim_history.viz3d['cp'])
#        ftv.add(ft.sim_history.viz3d['node_numbers'])
#        ft.config.gu['dofs'].viz3d['default'].scale_factor = 0.5
#        ftv.add(ft.config.gu['dofs'].viz3d['default'])
        ft.u_1

        fta.add_cam_move(duration=10, n=50)

    fta.plot()
    fta.configure_traits()
