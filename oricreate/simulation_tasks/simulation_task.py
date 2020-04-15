<<<<<<< master
# -------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Jan 29, 2013 by: rch

import platform
import time
<<<<<<< HEAD

from traits.api import Instance, \
    Property, cached_property, Str, \
    Array, Dict, provides, Int, Bool, DelegatesTo
from traitsui.api import \
    View, UItem, Item, Group

=======
from traits.api import Instance, \
    Property, cached_property, Str, \
    Array, Dict, implements, Int, Bool, DelegatesTo
from traitsui.api import \
    View, UItem, Item, Group
from .i_simulation_task import \
    ISimulationTask
>>>>>>> 2to3
import numpy as np
from oricreate.crease_pattern import \
    CreasePattern
from oricreate.forming_tasks import \
    FormingTask
from oricreate.gu import \
    GuConstantLength, \
    GuDevelopability, \
    GuFlatFoldability, \
    GuPointsOnSurface, \
    GuGrabPoints, \
    GuPointsOnLine, GuDofConstraints
from oricreate.mapping_tasks import \
    MapToSurface
from oricreate.opt import \
    IGu
from oricreate.simulation_step import \
    SimulationStep, SimulationConfig
from oricreate.simulation_tasks.simulation_history import SimulationHistory

<<<<<<< HEAD
from .i_simulation_task import \
    ISimulationTask


=======
>>>>>>> 2to3
if platform.system() == 'Linux':
    sysclock = time.time
elif platform.system() == 'Windows':
    sysclock = time.clock


<<<<<<< HEAD
@provides(ISimulationTask)
class SimulationTaskBase(FormingTask):
=======
class SimulationTask(FormingTask):

    '''SimulationTask tasks use the crease pattern state
    attained within the previous FormingTask task
    and bring it to the next stage.

    It is realized as a specialized configurations
    of the FormingStep with the goal to
    represent the.
    '''
    implements(ISimulationTask)

    config = Instance(SimulationConfig)
    '''Configuration of the simulation
    '''

    def _config_default(self):
        return SimulationConfig()

    n_steps = Int(1, auto_set=False, enter_set=True)
    '''Number of simulation steps.
    '''

    sim_step = Property(depends_on='config')
    '''Simulation step resim_step =alizing the transition
    from initial to final state for the increment
    of the time-dependent constraints.
    '''
    @cached_property
    def _get_sim_step(self):
        return SimulationStep(forming_task=self,
                              config=self.config)
>>>>>>> 2to3

    # =========================================================================
    # Geometric data
    # =========================================================================

    u_0 = Property(Array(float))
    '''Attribute storing the optional user-supplied initial array.
    It is used as the trial vector of unknown displacements
    for the current FormingTask simulation.
    '''

    def _get_u_0(self):
        return np.copy(self.source_task.formed_object.u)

    u_1 = Property(Array(float))
    '''Final state of the FormingTask process that can be used
    by further FormingTask controller.
    '''

    def _get_u_1(self):
        return self.u_t[-1]

<<<<<<< HEAD
=======
    # ==========================================================================
    # Solver parameters
    # ==========================================================================

>>>>>>> 2to3
    n_steps = Int(1, auto_set=False, enter_set=True)
    '''Number of time steps.
    '''

    time_arr = Array(float, auto_set=False, enter_set=True)
    '''User specified time array overriding the default one.
    '''

<<<<<<< HEAD
=======
    unfold = Bool(False)
    '''Reverse the time array. So it's possible to unfold
    a structure. If you optimize a pattern with FindFormForGeometry
    you can unfold it at least with FoldRigidly to it's flatten
    shape.
    '''

>>>>>>> 2to3
    t_arr = Property(Array(float), depends_on='unfold, n_steps, time_array')
    '''Generated time array.
    '''
    @cached_property
    def _get_t_arr(self):
        if len(self.time_arr) > 0:
            return self.time_arr
        t_arr = np.linspace(0, 1., self.n_steps + 1)
        if(self.unfold):
            # time array will be reversed if unfold is true
            t_arr = t_arr[::-1]
        return t_arr

    sim_history = Property(Instance(SimulationHistory))
    '''History of calculated displacements.
    '''
    @cached_property
    def _get_sim_history(self):
        cp = self.cp
        return SimulationHistory(
<<<<<<< HEAD
            x_0=cp.x_0, L=cp.L, F=cp.F, u_t=self.u_t,
            t_record=self.t_arr
        )


class SimulationTask(SimulationTaskBase):

    '''SimulationTask tasks use the crease pattern state
    attained within the previous FormingTask task
    and bring it to the next stage.

    It is realized as a specialized configurations
    of the FormingStep with the goal to
    represent the.
    '''

    config = Instance(SimulationConfig)
    '''Configuration of the simulation
    '''

    def _config_default(self):
        return SimulationConfig()

    sim_step = Property(depends_on='config')
    '''Simulation step resim_step =alizing the transition
    from initial to final state for the increment
    of the time-dependent constraints.
    '''
    @cached_property
    def _get_sim_step(self):
        return SimulationStep(forming_task=self,
                              config=self.config)

    # ==========================================================================
    # Solver parameters
    # ==========================================================================

    unfold = Bool(False)
    '''Reverse the time array. So it's possible to unfold
    a structure. If you optimize a pattern with FindFormForGeometry
    you can unfold it at least with FoldRigidly to it's flatten
    shape.
    '''
=======
            x_0=cp.x_0, L=cp.L, F=cp.F, u_t=self.u_t, t_record=self.t_arr
        )

>>>>>>> 2to3
    record_iter = DelegatesTo('sim_step')

    u_t = Property(depends_on='source_config_changed, unfold')
    '''Displacement history for the current FoldRigidly process.
    '''
    @cached_property
    def _get_u_t(self):
        '''Solve the problem with the appropriate solver
        '''
        u_t_list = [self.cp.u]
        time_start = sysclock()
        for t in self.t_arr[1:]:
            print('time: %g' % t)
            self.sim_step.t = t
            #U = self.sim_step.U_t
            try:
                U = self.sim_step.U_t
            except Exception as inst:
                print(inst)
                break

            if self.sim_step.record_iter:
                u_t_list += self.sim_step.u_it_list
                self.sim_step.clear_iter()

            u_t_list.append(U.reshape(-1, 3))

        time_end = sysclock()
        print('==== solved in ', time_end - time_start, '=====')
        return np.array(u_t_list)

    traits_view = View(
        Group(
            UItem('config@'),
        ),
        Item('n_steps'),
    )


class FindFormForGeometry(SimulationTask):

    '''FindFormForGeometry forms the crease pattern, so flat foldability
    conditions are fulfilled

    The crease pattern is incrementally  deformed, till every inner
    node fulfills the condition, that the sum of the angels between
    the connecting crease lines is at least 2*pi. Every other constraints
    are deactivated.

    For this condition the connectivity of all inner nodes must be
    put in the object.
    '''

    name = Str('form finding')

    Gu = Dict(Str, IGu)
    '''Equality constraints.
    '''

    def _Gu_default(self):
        return {'ff': GuFlatFoldability(FormingTask=self),
                'uf': GuDevelopability(FormingTask=self),
                'ps': GuPointsOnSurface(FormingTask=self),
                'dc': GuDofConstraints(FormingTask=self)
                }

    U_1 = Property(depends_on='source_config_changed, _U_0')
    '''Initial displacement for the next step after form finding.
    The target configuration has no perturbation at the end.
    '''
    @cached_property
    def _get_U_1(self):
        return np.zeros_like(self.U_t[-1])


class FoldRigidly(SimulationTask):

    '''FoldRigidly folds a crease pattern while
    using the classic constraints like
    constant length, DOF constraints and surface constraints.

    This class serves for the analysis of the FoldRigidly process
    of a crease pattern. All classic constraints can be used.
    Only special elements, like GP and LP
    are not included. But sliding faces and target faces are supported.
    '''

    name = Str('fold to target surfaces')

    Gu = Dict(Str, IGu)
    '''Equality constraints.
    '''

    def _Gu_default(self):
        return {'cl': GuConstantLength(FormingTask=self),
                'ps': GuPointsOnSurface(FormingTask=self),
                'dc': GuDofConstraints(FormingTask=self)
                }


class Lift(SimulationTask):

    ''' Lifting class is for lifting a crease pattern with a crane.

    Lifting takes all equality constraints and is used to simulate
    the lifting act with a crane structure.
    To be able to lift the structure, you need to have a pre-deformation u_0.
    In Lifting you can set an target face to init_tf_lst and with this
    target face, Lifting will initialize a pre-deformation fully automatically.
    Instead of this you can although put in your own pre-deformation.
    '''

    name = Str('fold to displacement')

    goal_function_type = 'none'

    Gu = Dict(Str, IGu)
    '''Equality constraints.
    '''

    def _Gu_default(self):
        return {'cl': GuConstantLength(FormingTask=self),
                'gp': GuGrabPoints(FormingTask=self),
                'pl': GuPointsOnLine(FormingTask=self),
                'ps': GuPointsOnSurface(FormingTask=self),
                'dc': GuDofConstraints(FormingTask=self)
                }


if __name__ == '__main__':

    from oricreate.util import t_, x_, z_

    cp = CreasePattern(X=[[0, 0, 0],
                          [1, 0, 0],
                          [1, 1, 0],
                          [0, 1, 0],
                          [0.2, 0.2, 0],
                          [0.5, 0.5, 0.0],
                          [0.6, 0.4, 0.0]],
                       L=[[0, 1],
                          [1, 2],
                          [2, 3],
                          [3, 0],
                          [1, 3]],
                       F=[[0, 1, 3],
                          [1, 2, 3]])

    init = MapToSurface(cp=cp)
    init.U_0[5] = 0.05

    lift = Lift(source=init, n_steps=10)
    print('initial vector', lift.U_0)

#    lift.TS = [[r_ , s_, 0.01 + t_ * (0.5)]]
    lift.CS = [[z_ - 4 * 0.4 * t_ * x_ * (1 - x_ / 3)]]
    lift.GP = [[4, 0]]
    lift.LP = [[5, 4],
               [6, 4]]

    lift.cnstr_lhs = [[(0, 0, 1.0)],
                      [(0, 1, 1.0)],
                      [(0, 2, 1.0)],
                      [(3, 0, 1.0)],
                      [(3, 2, 1.0)],
                      [(2, 2, 1.0)],
                      [(5, 0, 1.0)],
                      [(6, 0, 1.0)]]
    lift.cnstr_rhs[0] = 0.9
    print(lift.U_1)
#
=======
# -------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Jan 29, 2013 by: rch

import platform
import time
from traits.api import Instance, \
    Property, cached_property, Str, \
    Array, Dict, implements, Int, Bool, DelegatesTo
from traitsui.api import \
    View, UItem, Item, Group
from .i_simulation_task import \
    ISimulationTask
import numpy as np
from oricreate.crease_pattern import \
    CreasePattern
from oricreate.forming_tasks import \
    FormingTask
from oricreate.gu import \
    GuConstantLength, \
    GuDevelopability, \
    GuFlatFoldability, \
    GuPointsOnSurface, \
    GuGrabPoints, \
    GuPointsOnLine, GuDofConstraints
from oricreate.mapping_tasks import \
    MapToSurface
from oricreate.opt import \
    IGu
from oricreate.simulation_step import \
    SimulationStep, SimulationConfig
from oricreate.simulation_tasks.simulation_history import SimulationHistory

if platform.system() == 'Linux':
    sysclock = time.time
elif platform.system() == 'Windows':
    sysclock = time.clock


class SimulationTask(FormingTask):

    '''SimulationTask tasks use the crease pattern state
    attained within the previous FormingTask task
    and bring it to the next stage.

    It is realized as a specialized configurations
    of the FormingStep with the goal to
    represent the.
    '''
    implements(ISimulationTask)

    config = Instance(SimulationConfig)
    '''Configuration of the simulation
    '''

    def _config_default(self):
        return SimulationConfig()

    n_steps = Int(1, auto_set=False, enter_set=True)
    '''Number of simulation steps.
    '''

    sim_step = Property(depends_on='config')
    '''Simulation step resim_step =alizing the transition
    from initial to final state for the increment
    of the time-dependent constraints.
    '''
    @cached_property
    def _get_sim_step(self):
        return SimulationStep(forming_task=self,
                              config=self.config)

    # =========================================================================
    # Geometric data
    # =========================================================================

    u_0 = Property(Array(float))
    '''Attribute storing the optional user-supplied initial array.
    It is used as the trial vector of unknown displacements
    for the current FormingTask simulation.
    '''

    def _get_u_0(self):
        return np.copy(self.source_task.formed_object.u)

    u_1 = Property(Array(float))
    '''Final state of the FormingTask process that can be used
    by further FormingTask controller.
    '''

    def _get_u_1(self):
        return self.u_t[-1]

    # ==========================================================================
    # Solver parameters
    # ==========================================================================

    n_steps = Int(1, auto_set=False, enter_set=True)
    '''Number of time steps.
    '''

    time_arr = Array(float, auto_set=False, enter_set=True)
    '''User specified time array overriding the default one.
    '''

    unfold = Bool(False)
    '''Reverse the time array. So it's possible to unfold
    a structure. If you optimize a pattern with FindFormForGeometry
    you can unfold it at least with FoldRigidly to it's flatten
    shape.
    '''

    t_arr = Property(Array(float), depends_on='unfold, n_steps, time_array')
    '''Generated time array.
    '''
    @cached_property
    def _get_t_arr(self):
        if len(self.time_arr) > 0:
            return self.time_arr
        t_arr = np.linspace(0, 1., self.n_steps + 1)
        if(self.unfold):
            # time array will be reversed if unfold is true
            t_arr = t_arr[::-1]
        return t_arr

    sim_history = Property(Instance(SimulationHistory))
    '''History of calculated displacements.
    '''
    @cached_property
    def _get_sim_history(self):
        cp = self.cp
        return SimulationHistory(
            x_0=cp.x_0, L=cp.L, F=cp.F, u_t=self.u_t, t_record=self.t_arr
        )

    record_iter = DelegatesTo('sim_step')

    u_t = Property(depends_on='source_config_changed, unfold')
    '''Displacement history for the current FoldRigidly process.
    '''
    @cached_property
    def _get_u_t(self):
        '''Solve the problem with the appropriate solver
        '''
        u_t_list = [self.cp.u]
        time_start = sysclock()
        for t in self.t_arr[1:]:
            print('time: %g' % t)
            self.sim_step.t = t
            #U = self.sim_step.U_t
            try:
                U = self.sim_step.U_t
            except Exception as inst:
                print(inst)
                break

            if self.sim_step.record_iter:
                u_t_list += self.sim_step.u_it_list
                self.sim_step.clear_iter()

            u_t_list.append(U.reshape(-1, 3))

        time_end = sysclock()
        print('==== solved in ', time_end - time_start, '=====')
        return np.array(u_t_list)

    traits_view = View(
        Group(
            UItem('config@'),
        ),
        Item('n_steps'),
    )


class FindFormForGeometry(SimulationTask):

    '''FindFormForGeometry forms the crease pattern, so flat foldability
    conditions are fulfilled

    The crease pattern is incrementally  deformed, till every inner
    node fulfills the condition, that the sum of the angels between
    the connecting crease lines is at least 2*pi. Every other constraints
    are deactivated.

    For this condition the connectivity of all inner nodes must be
    put in the object.
    '''

    name = Str('form finding')

    Gu = Dict(Str, IGu)
    '''Equality constraints.
    '''

    def _Gu_default(self):
        return {'ff': GuFlatFoldability(FormingTask=self),
                'uf': GuDevelopability(FormingTask=self),
                'ps': GuPointsOnSurface(FormingTask=self),
                'dc': GuDofConstraints(FormingTask=self)
                }

    U_1 = Property(depends_on='source_config_changed, _U_0')
    '''Initial displacement for the next step after form finding.
    The target configuration has no perturbation at the end.
    '''
    @cached_property
    def _get_U_1(self):
        return np.zeros_like(self.U_t[-1])


class FoldRigidly(SimulationTask):

    '''FoldRigidly folds a crease pattern while
    using the classic constraints like
    constant length, DOF constraints and surface constraints.

    This class serves for the analysis of the FoldRigidly process
    of a crease pattern. All classic constraints can be used.
    Only special elements, like GP and LP
    are not included. But sliding faces and target faces are supported.
    '''

    name = Str('fold to target surfaces')

    Gu = Dict(Str, IGu)
    '''Equality constraints.
    '''

    def _Gu_default(self):
        return {'cl': GuConstantLength(FormingTask=self),
                'ps': GuPointsOnSurface(FormingTask=self),
                'dc': GuDofConstraints(FormingTask=self)
                }


class Lift(SimulationTask):

    ''' Lifting class is for lifting a crease pattern with a crane.

    Lifting takes all equality constraints and is used to simulate
    the lifting act with a crane structure.
    To be able to lift the structure, you need to have a pre-deformation u_0.
    In Lifting you can set an target face to init_tf_lst and with this
    target face, Lifting will initialize a pre-deformation fully automatically.
    Instead of this you can although put in your own pre-deformation.
    '''

    name = Str('fold to displacement')

    goal_function_type = 'none'

    Gu = Dict(Str, IGu)
    '''Equality constraints.
    '''

    def _Gu_default(self):
        return {'cl': GuConstantLength(FormingTask=self),
                'gp': GuGrabPoints(FormingTask=self),
                'pl': GuPointsOnLine(FormingTask=self),
                'ps': GuPointsOnSurface(FormingTask=self),
                'dc': GuDofConstraints(FormingTask=self)
                }


if __name__ == '__main__':

    from oricreate.util import t_, x_, z_

    cp = CreasePattern(X=[[0, 0, 0],
                          [1, 0, 0],
                          [1, 1, 0],
                          [0, 1, 0],
                          [0.2, 0.2, 0],
                          [0.5, 0.5, 0.0],
                          [0.6, 0.4, 0.0]],
                       L=[[0, 1],
                          [1, 2],
                          [2, 3],
                          [3, 0],
                          [1, 3]],
                       F=[[0, 1, 3],
                          [1, 2, 3]])

    init = MapToSurface(cp=cp)
    init.U_0[5] = 0.05

    lift = Lift(source=init, n_steps=10)
    print('initial vector', lift.U_0)

#    lift.TS = [[r_ , s_, 0.01 + t_ * (0.5)]]
    lift.CS = [[z_ - 4 * 0.4 * t_ * x_ * (1 - x_ / 3)]]
    lift.GP = [[4, 0]]
    lift.LP = [[5, 4],
               [6, 4]]

    lift.cnstr_lhs = [[(0, 0, 1.0)],
                      [(0, 1, 1.0)],
                      [(0, 2, 1.0)],
                      [(3, 0, 1.0)],
                      [(3, 2, 1.0)],
                      [(2, 2, 1.0)],
                      [(5, 0, 1.0)],
                      [(6, 0, 1.0)]]
    lift.cnstr_rhs[0] = 0.9
    print(lift.U_1)
#
>>>>>>> interim stage 1
