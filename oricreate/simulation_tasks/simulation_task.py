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
# Created on Jan 29, 2013 by: matthias

from traits.api import Instance, \
    Property, cached_property, Str, \
    Array, Dict, implements, Int, Bool

from eq_cons import \
    EqConsConstantLength
from eq_cons import \
    EqConsDevelopability, \
    EqConsFlatFoldability
from eq_cons import \
    EqConsPointsOnSurface
from eq_cons import \
    IEqCons, GrabPoints, \
    PointsOnLine, DofConstraints
from oricreate import \
    CreasePattern, \
    FormingTask, IFormingTask, \
    ISimulationTask, SimulationStep, \
    MapToSurface

import numpy as np
import time
import platform

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

    source = Instance(IFormingTask)

    n_steps = Int(1, auto_set=False, enter_set=True)
    '''Number of simulation steps.
    '''

    sim_step = Instance(SimulationStep)
    '''Simulation step realizing the transition
    from initial to final state for the increment
    of the time-dependent constraints.
    '''

    cp = Property(depends_on='source')
    '''Instance of a crease pattern.
    '''
    @cached_property
    def _get_cp(self):
        return self.source.cp

    def _set_cp(self, value):
        if self.source:
            msg = 'crease pattern already available in the source object'
            raise ValueError(msg)
        print 'No source: an initialization ',
        print 'with the supplied crease pattern will be added.'
        self.source = MapToSurface(cp=value)

    X_0 = Property(depends_on='source')
    '''Initial configuration given as the last
    configuration of the previous FormingTask step.
    '''
    @cached_property
    def _get_X_0(self):
        return self.source.X_1

    # =========================================================================
    # Geometric data
    # =========================================================================

    L = Property()
    '''Array of crease_lines defined by pairs of node numbers.
    '''

    def _get_L(self):
        return self.source.L

    F = Property()
    '''Array of crease facets defined by list of node numbers.
    '''

    def _get_F(self):
        return self.source.F

    U_0 = Property(Array(float))
    '''Attribute storing the optional user-supplied initial array.
    It is used as the trial vector of unknown displacements
    for the current FormingTask simulation.
    '''

    def _get_U_0(self):
        return self.source.U_1

    X_1 = Property(Array(float))
    '''Final state of the FormingTask process that can be used
    by further FormingTask controller.
    '''

    def _get_X_1(self):
        return self.X_0 + self.U_t[-1]

    U_1 = Property(Array(float))
    '''Final state of the FormingTask process that can be used
    by further FormingTask controller.
    '''

    def _get_U_1(self):
        return np.zeros_like(self.X_t[-1])

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

    U_t = Property(depends_on='source_config_changed, unfold')
    '''Displacement history for the current FoldRigidly process.
    '''
    @cached_property
    def _get_U_t(self):
        '''Solve the problem with the appropriate solver
        '''
        time_start = sysclock()

        if self.goal_function_type_ is not None:
            U_t = self._solve_fmin(self.U_0, self.acc)
        else:
            U_t = self._solve_nr(self.U_0, self.acc)

        time_end = sysclock()
        print '==== solved in ', time_end - time_start, '====='

        return U_t

    def _solve_nr(self, U_0, acc=1e-4):
        '''Find the solution using the Newton - Raphson procedure.
        '''
        print '==== solving using Newton-Raphson ===='
        U_t = [np.copy(self.U_0)]
        # time loop without the initial time step
        for t in self.t_arr[1:]:
            print 'step', t,
            U_t.append(np.copy(self.sim_step._solve_nr(t)))
        return np.array(U_t, dtype='f')

    def _solve_fmin(self, U_0, acc=1e-4):
        '''Solve the problem using the
        Sequential Least Square Quadratic Programming method.
        '''
        print '==== solving with SLSQP optimization ===='
        U_t = [np.copy(self.U_0)]
        # time loop without the initial time step
        for t in self.t_arr[1:]:
            print 'step', t,
            U_t.append(np.copy(self.sim_step._solve_fmin(t)))
        return np.array(U_t, dtype='f')

    # =========================================================================
    # Auxiliary elements that can be used interim steps of computation.
    # They are not included in the crease pattern representation.
    # =========================================================================
    x_aux = Array(dtype=float, value=[])
    '''Auxiliary nodes used for visualization.
    '''

    L_aux = Array(dtype=int, value=[])
    '''Auxiliary lines used for visualization.
    '''

    F_aux = Array(dtype=int, value=[])
    '''Auxiliary facets used for visualization.
    '''


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

    eqcons = Dict(Str, IEqCons)
    '''Equality constraints.
    '''

    def _eqcons_default(self):
        return {'ff': EqConsFlatFoldability(FormingTask=self),
                'uf': EqConsDevelopability(FormingTask=self),
                'ps': EqConsPointsOnSurface(FormingTask=self),
                'dc': DofConstraints(FormingTask=self)
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

    eqcons = Dict(Str, IEqCons)
    '''Equality constraints.
    '''

    def _eqcons_default(self):
        return {'cl': EqConsConstantLength(FormingTask=self),
                'ps': EqConsPointsOnSurface(FormingTask=self),
                'dc': DofConstraints(FormingTask=self)
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

    eqcons = Dict(Str, IEqCons)
    '''Equality constraints.
    '''

    def _eqcons_default(self):
        return {'cl': EqConsConstantLength(FormingTask=self),
                'gp': GrabPoints(FormingTask=self),
                'pl': PointsOnLine(FormingTask=self),
                'ps': EqConsPointsOnSurface(FormingTask=self),
                'dc': DofConstraints(FormingTask=self)
                }

if __name__ == '__main__':

    from view import FormingView
    from util import t_, x_, z_
    from eq_cons import CF

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
    print 'initial vector', lift.U_0

#    lift.TS = [[r_ , s_, 0.01 + t_ * (0.5)]]
    lift.CS = [[z_ - 4 * 0.4 * t_ * x_ * (1 - x_ / 3)]]
    lift.GP = [[4, 0]]
    lift.LP = [[5, 4],
               [6, 4]]
    lift.cf_lst = [(CF(Rf=lift.CS[0][0]), [1])]

    lift.cnstr_lhs = [[(0, 0, 1.0)],
                      [(0, 1, 1.0)],
                      [(0, 2, 1.0)],
                      [(3, 0, 1.0)],
                      [(3, 2, 1.0)],
                      [(2, 2, 1.0)],
                      [(5, 0, 1.0)],
                      [(6, 0, 1.0)]]
    lift.cnstr_rhs[0] = 0.9
    print lift.U_1
#
    v = FormingView(root=init)
    v.configure_traits()
