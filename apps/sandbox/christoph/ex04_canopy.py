'''
Created on Jan 20, 2016

@author: rch
'''

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
from traits.api import \
    Float, HasTraits, Property, cached_property, Int, \
    Instance, Array, Bool

import numpy as np
import sympy as sp


a_, b_ = sp.symbols('a,b')


def get_fr(var_, L, H):
    fx = a_ * (var_ / L)**2 + b_ * (var_ / L)
    eqns = [fx.subs(var_, L), fx.subs(var_, L / 2) - H]
    ab_subs = sp.solve(eqns, [a_, b_])
    fx = fx.subs(ab_subs)
    return fx


class AddBoundaryTask(MappingTask):
    '''
    '''

    def _add_boundary_facet(self, N1, N2, dir_=-1, delta=0.1, N_start_idx=0):
        cp = self.previous_task.formed_object
        x1, x2 = cp.x_0[N1, :], cp.x_0[N2, :]
        dx = x1[:, 0] - x2[:, 0]
        dy = x1[:, 1] - x2[:, 1]
        dz = np.zeros_like(dy)
        dirvec = np.c_[dx, dy, dz]

        x4 = x2[:, :]
        x4[:, 1] += dir_ * delta
        x3 = np.copy(x4)
        x3[:, :] += dirvec * 0.82

        x_add = np.vstack([x3, x4])
        N3 = N_start_idx + np.arange(len(x3))
        N4 = N_start_idx + len(x3) + np.arange(len(x4))

        L_add = np.vstack([
            np.array([N1, N3]).T,
            np.array([N2, N3]).T,
            np.array([N3, N4]).T,
            np.array([N2, N4]).T
        ])

        F_add = np.vstack([
            np.array([N1, N3, N2]).T,
            np.array([N3, N4, N2]).T
        ])

        return x_add, L_add, F_add

    def _get_formed_object(self):
        '''attach additional facets at the obundary
        '''
        cp = self.previous_task.formed_object
        x_0, L, F = cp.x_0, cp.L, cp.F
        n_N = len(x_0)
        n_N_add = 8
        x_br, L_br, F_br = self._add_boundary_facet(
            [8, 37, 15, 43], [37, 15, 43, 20], -1, 0.1, n_N)
        x_bl, L_bl, F_bl = self._add_boundary_facet(
            [8, 31, 3, 27], [31, 3, 27, 0], -1, 0.1, n_N + n_N_add)
        x_tr, L_tr, F_tr = self._add_boundary_facet(
            [14, 42, 19, 46], [42, 19, 46, 22], 1, 0.1, n_N + 2 * n_N_add)
        x_tl, L_tl, F_tl = self._add_boundary_facet(
            [14, 36, 7, 30], [36, 7, 30, 2], 1, 0.1, n_N + 3 * n_N_add)
        x_0 = np.vstack([x_0, x_br, x_bl, x_tr, x_tl])
        L = np.vstack([L, L_br, L_bl, L_tr, L_tl])
        F = np.vstack([F, F_br, F_bl, F_tr, F_tl])
        return CreasePatternState(x_0=x_0,
                                  L=L,
                                  F=F)


class DoublyCurvedYoshiFormingProcess(HasTraits):
    '''
    Define the simulation task prescribing the boundary conditions, 
    target surfaces and configuration of the algorithm itself.
    '''

    L_x = Float(3.0, auto_set=False, enter_set=True, input=True)
    L_y = Float(2.2, auto_set=False, enter_set=True, input=True)
    u_x = Float(0.1, auto_set=False, enter_set=True, input=True)
    n_fold_steps = Int(30, auto_set=False, enter_set=True, input=True)
    n_load_steps = Int(30, auto_set=False, enter_set=True, input=True)

    stiffening_boundary = Bool(False)

    ctf = Property(depends_on='+input')
    '''control target surface'''
    @cached_property
    def _get_ctf(self):
        return [r_, s_, - 0.2 * t_ * r_ * (1 - r_ / self.L_x) - 0.0000015]

    factory_task = Property(Instance(FormingTask))
    '''Factory task generating the crease pattern.
    '''
    @cached_property
    def _get_factory_task(self):
        return YoshimuraCPFactory(L_x=self.L_x, L_y=self.L_y,
                                  n_x=4, n_y=12)

    mask_task = Property(Instance(MaskTask))
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_mask_task(self):
        return MaskTask(previous_task=self.factory_task,
                        F_mask=[0, 6, 12, 18, 12, 24, 36, 48,
                                96, 78, 90, 102, 54, 42,
                                72, 1, 12, 43, 49, 97, 103, 19,
                                59, 65, 71, 77, 101, 83, 96, 107,
                                47, 29, 41, 53,
                                5, 23, 95,
                                58, 76,
                                100, 106,
                                46, 52],
                        L_mask=[0, 7, 14, 21, 148, 160, 172, 154, 1, 22, 149, 155,
                                152, 158, 5, 26, 153, 165, 177, 159, 6, 13, 20, 27,
                                28, 40, 29, 41, 32, 44, 33, 45, 34, 46, 35, 47, 38, 50, 39, 51,
                                58, 52, 76, 53, 57, 81, 70, 94, 98, 75, 99, 93,
                                124, 100, 128, 129, 105, 135, 112, 142, 118, 119,
                                147, 123],
                        N_mask=[0, 7, 21, 28, 35, 47, 65, 41, 1, 29, 36, 42, 39,
                                45, 5, 33, 40, 52, 70, 46, 6, 13, 27, 34])

    add_boundary_task = Property(Instance(FormingTask))
    '''Initialization to render the desired folding branch. 
    '''
    @cached_property
    def _get_add_boundary_task(self):
        if self.stiffening_boundary:
            return AddBoundaryTask(previous_task=self.mask_task)
        else:
            return self.mask_task

    init_displ_task = Property(Instance(FormingTask))
    '''Initialization to render the desired folding branch. 
    '''
    @cached_property
    def _get_init_displ_task(self):
        cp = self.mask_task.formed_object
        return MapToSurface(previous_task=self.add_boundary_task,
                            target_faces=[(self.ctf, cp.N)])

    fold_task = Property(Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_fold_task(self):
        self.init_displ_task.x_1
#        cp = self.init_displ_task.formed_object

#        print 'nodes', x_1[(0, 1, 2, 20, 21, 22), 2]

#        cp.u[(26, 25, 24, 23), 2] = -0.01
#        cp.x[(0, 1, 2, 20, 21, 22), 2] = 0.0
        u_max = self.u_x
        fixed_nodes_z = fix(
            [0, 1, 2, 20, 21, 22], (2))
#         fixed_nodes_x = fix(
#             [8, 9, 10, 11, 12, 13, 14], (0))
        fixed_nodes_y = fix(
            [1, 21], (1))  # 5, 11, 17,
        control_left = fix(
            [0, 1, 2], (0),
            lambda t: t * u_max)
        control_right = fix(
            [20, 21, 22], (0),
            lambda t: -t * u_max)
        front_node = fix(
            [8], (1), lambda t: t * 0.03)
        back_node = fix(
            [14], (1), lambda t: -t * 0.03)

        dof_constraints = fixed_nodes_z + fixed_nodes_y + \
            control_left + control_right + front_node + back_node
        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
        gu_constant_length = GuConstantLength()
        sim_config = SimulationConfig(goal_function_type='gravity potential energy',
                                      gu={'cl': gu_constant_length,
                                          'dofs': gu_dof_constraints},
                                      acc=1e-5, MAX_ITER=500,
                                      debug_level=0)
        return SimulationTask(previous_task=self.init_displ_task,
                              config=sim_config, n_steps=self.n_fold_steps)

    turn_task = Property(Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_turn_task(self):
        self.fold_task.x_1
        fixed_nodes_z = fix(
            [0, 1, 2, 20, 21, 22], (0, 2))
        fixed_nodes_y = fix(
            [1, 21], (1))
        front_nodes = fix(
            [8, 14], (0, 1, 2))

        dof_constraints = fixed_nodes_z + fixed_nodes_y + \
            front_nodes
        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
        gu_constant_length = GuConstantLength()
        sim_config = SimulationConfig(goal_function_type='gravity potential energy',
                                      gu={'cl': gu_constant_length,
                                          'dofs': gu_dof_constraints},
                                      acc=1e-5, MAX_ITER=500,
                                      debug_level=0)
        st = SimulationTask(previous_task=self.fold_task,
                            config=sim_config, n_steps=2)
        cp = st.formed_object
        cp.x_0 = self.fold_task.x_1
        cp.x_0[:, 2] *= -1
        cp.u[:, :] = 0.0

        if self.stiffening_boundary:
            cp.u[tuple(np.arange(47, 47 + 32)), 2] = -0.2

        return st

    turn_task2 = Property(Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_turn_task2(self):

        self.fold_task.x_1

        u_z = 0.1
        fixed_nodes_xzy = fix([7, 19], (0, 1, 2))
        lift_nodes_z = fix([3, 15], (2), lambda t: t * u_z)

        dof_constraints = fixed_nodes_xzy + lift_nodes_z
        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
        gu_constant_length = GuConstantLength()
        sim_config = SimulationConfig(goal_function_type='total potential energy',
                                      gu={'cl': gu_constant_length,
                                          'dofs': gu_dof_constraints},
                                      acc=1e-5, MAX_ITER=1000,
                                      debug_level=0)
        load_nodes = []
        FN = lambda F: lambda t: t * F
        F_ext_list = [(n, 2, FN(-10)) for n in load_nodes]
        fu_tot_poteng = FuPotEngTotal(kappa=np.array([1000]),
                                      F_ext_list=F_ext_list)
        sim_config._fu = fu_tot_poteng
        st = SimulationTask(previous_task=self.fold_task,
                            config=sim_config, n_steps=1)
        fu_tot_poteng.forming_task = st
        cp = st.formed_object
        cp.u[(3, 15), 2] = u_z
        return st

    load_factor = Float(1.0, input=True, enter_set=True, auto_set=False)

    load_task = Property(Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_load_task(self):
        self.turn_task.x_1

        fixed_nodes_yz = fix([0, 2, 20,  22], (1, 2))  # + \
        fixed_nodes_x = fix([0, 2, 20, 22], (0))  # + \
        #    fix([1, 21], [0, 2])
        link_bnd = []
        if self.stiffening_boundary:
            link_bnd = link([48, 49, 50, 56, 57, 58, 64, 65, 66, 72, 73, 74],
                            [0, 1, 2], 1.0,
                            [51, 52, 53, 59, 60, 61, 67, 68, 69, 75, 76, 77],
                            [0, 1, 2], -1.0)

        dof_constraints = fixed_nodes_x + fixed_nodes_yz + link_bnd
        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
        gu_constant_length = GuConstantLength()
        sim_config = SimulationConfig(goal_function_type='total potential energy',
                                      gu={'cl': gu_constant_length,
                                          'dofs': gu_dof_constraints},
                                      acc=1e-5, MAX_ITER=1000,
                                      debug_level=0)

        FN = lambda F: lambda t: t * F

        H = 0
        P = 3.5 * self.load_factor
        F_ext_list = [(33, 2, FN(-P)), (34, 2, FN(-P)), (11, 2, FN(-P)), (39, 2, FN(-P)), (40, 2, FN(-P)), (4, 0, FN(0.1609 * H)), (4, 2, FN(-0.2385 * H)), (10, 2, FN(-0.3975 * H)), (16, 0, FN(-0.1609 * H)), (16, 2, FN(-0.2385 * H)),
                      (6, 0, FN(0.1609 * H)), (6, 2, FN(-0.2385 * H)), (12, 2, FN(-0.3975 * H)), (18, 0, FN(-0.1609 * H)), (18, 2, FN(-0.2385 * H))]

        fu_tot_poteng = FuPotEngTotal(kappa=np.array([5.28]),
                                      F_ext_list=F_ext_list)


#         load_nodes = [10, 11, 12]
#         FN = lambda F: lambda t: t * F
#         F_ext_list = [(n, 2, FN(-10)) for n in load_nodes]
#         fu_tot_poteng = FuPotEngTotal(kappa=np.array([10]),
# F_ext_list=F_ext_list)  # (2 * n, 2, -1)])
        sim_config._fu = fu_tot_poteng
        st = SimulationTask(previous_task=self.turn_task,
                            config=sim_config, n_steps=self.n_load_steps)
        fu_tot_poteng.forming_task = st
        cp = st.formed_object
        cp.x_0 = self.turn_task.x_1
        cp.u[:, :] = 0.0
        return st

    measure_task = Property(Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_measure_task(self):
        mt = MappingTask(previous_task=self.turn_task)
        mt.formed_object.reset_state()
        return mt


class DoublyCurvedYoshiFormingProcessFTV(FTV):

    model = Instance(DoublyCurvedYoshiFormingProcess)
