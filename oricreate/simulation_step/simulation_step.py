# -------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in oricreate/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using oricreate open source!
#
# Created on Jan 29, 2013 by: rch

import platform
from scipy.optimize import \
    fmin_slsqp
import time
from traits.api import \
    HasTraits, Event, Property, cached_property, \
    Int, Float, Bool, DelegatesTo, \
    Instance, WeakRef, Array

import numpy as np
from oricreate.crease_pattern import \
    CreasePatternState
from oricreate.forming_tasks import \
    FormingTask
from simulation_config import \
    SimulationConfig


if platform.system() == 'Linux':
    sysclock = time.time
elif platform.system() == 'Windows':
    sysclock = time.clock


class SimulationStep(HasTraits):

    r"""Class implementing the transition of the crease pattern state
    to the target time :math:`t`.
    """

    forming_task = WeakRef(FormingTask)
    r'''Backward link to the client forming tasks.
    This may be an incremental time stepping SimulationTask
    of a MappingTaks performed in a single iterative step.
    '''

    source_config_changed = Event
    r'''Notification event for changes in the configuration
    of the optimization problem.
    '''

    config = WeakRef(SimulationConfig)
    r'''Configuration of the optimization problem.
    '''

    # =====================================================================
    # Cached properties derived from configuration and position in the pipe
    # =====================================================================
    cp_state = Property(Instance(CreasePatternState),
                        depends_on='source_config_changed')
    r'''Crease pattern state.
    '''
    @cached_property
    def _get_cp_state(self):
        return self.forming_task.formed_object

    U = Property(Array(float))
    r'''Displacement vector of the cp_state
    '''

    def _get_U(self):
        return self.cp_state.U

    fu = Property(depends_on='source_config_changed')
    r'''Goal function object.
    '''
    @cached_property
    def _get_fu(self):
        self.config.fu.forming_task = self.forming_task
        return self.config.fu

    gu_lst = Property(depends_on='source_config_changed')
    r'''Equality constraint object.
    '''
    @cached_property
    def _get_gu_lst(self):
        for gu in self.config.gu_lst:
            gu.forming_task = self.forming_task
        return self.config.gu_lst

    hu_lst = Property(depends_on='source_config_changed')
    r'''Inequality constraint object.
    '''
    @cached_property
    def _get_hu_lst(self):
        for hu in self.config.gu_lst:
            hu.forming_task = self.forming_task
        return self.config.hu_lst

    debug_level = DelegatesTo("config")
    # ===========================================================================
    # Configuration parameters for the iterative solver
    # ===========================================================================
    t = Float(0.0, auto_set=False, enter_set=True)
    r'''Current time within the step in the range (0,1).
    '''

    show_iter = Bool(False, auto_set=False, enter_set=True)
    r'''Saves the first 10 iteration steps, so they can be analyzed
    '''

    MAX_ITER = Int(100, auto_set=False, enter_set=True)
    r'''Maximum number of iterations.
    '''

    acc = Float(1e-4, auto_set=False, enter_set=True)
    r'''Required accuracy.
    '''

    use_G_du = Bool(True, auto_set=False, enter_set=True)
    r'''Switch the use of constraint derivatives on.
    '''

    # ===========================================================================
    # Iterative solvers
    # ===========================================================================
    def _solve_nr(self, t):
        '''Find the solution using the Newton-Raphson procedure.
        '''
        i = 0

        while i <= self.MAX_ITER:
            dR = self.get_G_du(t)
            R = self.get_G(t)
            nR = np.linalg.norm(R)
            if nR < self.acc:
                print '==== converged in ', i, 'iterations ===='
                break
            try:
                d_U = np.linalg.solve(dR, -R)
                self.U += d_U
                i += 1
            except Exception as inst:
                print '=== Problems solving iteration step %d  ====' % i
                print '=== Exception message: ', inst
        else:
            print '==== did not converge in %d iterations ====' % i

        return self.U

    def _solve_fmin(self):
        '''Solve the problem using the
        Sequential Least Square Quadratic Programming method.
        '''
        print '==== solving with SLSQP optimization ===='
        d0 = self.get_f_t(self.U)
        eps = d0 * 1e-4
        get_G_du_t = None

        if self.use_G_du:
            get_G_du_t = self.get_G_du_t

        info = fmin_slsqp(self.get_f_t,
                          self.U,
                          fprime=self.get_f_du_t,
                          f_eqcons=self.get_G_t,
                          fprime_eqcons=get_G_du_t,
                          acc=self.acc, iter=self.MAX_ITER,
                          iprint=0,
                          full_output=True,
                          epsilon=eps)
        U, f, n_iter, imode, smode = info
        if imode == 0:
            print '(time: %g, iter: %d, f: %g)' % (self.t, n_iter, f)
        else:
            print '(time: %g, iter: %d, f: %g, err: %d, %s)' % \
                (time, n_iter, f, imode, smode)
        return U

    # ==========================================================================
    # Goal function
    # ==========================================================================
    def get_f_t(self, U):
        '''Get the goal function value.
        '''
        self.cp_state.U = U
        f = self.get_f()
        if self.debug_level > 0:
            print 'f:\n', f
        return f

    def get_f(self):
        return self.fu.get_f(self.t)

    def get_f_du_t(self, U):
        '''Get the goal function derivatives.
        '''
        self.cp_state.U = U
        f_du = self.get_f_du()
        if self.debug_level > 1:
            print 'f_du.shape:\n', f_du.shape
            print 'f_du:\n', f_du
        return f_du

    def get_f_du(self):
        return self.fu.get_f_du(self.t)

    # ==========================================================================
    # Equality constraints
    # ==========================================================================
    def get_G_t(self, U):
        self.cp_state.U = U
        g = self.get_G(self.t)
        if self.debug_level > 0:
            print 'G:\n', [g]
        return g

    def get_G(self, t=0):
        g_lst = [gu.get_G(t) for gu in self.gu_lst]
        if(g_lst == []):
            return []
        return np.hstack(g_lst)

    def get_G_du_t(self, U):
        self.cp_state.U = U
        self.cp_state.U[:] = U[:]
        return self.get_G_du(self.t)

    def get_G_du(self, t=0):
        g_du_lst = [gu.get_G_du(t) for gu in self.gu_lst]
        if(g_du_lst == []):
            return []
        g_du = np.vstack(g_du_lst)
        if self.debug_level > 1:
            print 'G_du.shape:\n', g_du.shape
            print 'G_du:\n', [g_du]
        return g_du

    # =========================================================================
    # Output data
    # =========================================================================

    x_0 = Property
    '''Initial position of all nodes.
    '''

    def _get_x_0(self):
        return self.X_0.reshape(-1, self.n_D)

    X_t = Property()
    '''History of nodal positions [time, node*dim]).
    '''

    def _get_X_t(self):
        return self.X_0[np.newaxis, :] + self.U_t

    x_t = Property()
    '''History of nodal positions [time, node, dim].
    '''

    def _get_x_t(self):
        n_t = self.X_t.shape[0]
        return self.X_t.reshape(n_t, -1, self.n_D)

    x_1 = Property
    '''Final position of all nodes.
    '''

    def _get_x_1(self):
        return self.x_t[-1]

    u_t = Property()
    '''History of nodal positions [time, node, dim].
    '''

    def _get_u_t(self):
        n_t = self.U_t.shape[0]
        return self.U_t.reshape(n_t, -1, self.n_D)

    u_1 = Property()
    '''Final nodal positions [node, dim].
    '''

    def _get_u_1(self):
        return self.u_t[-1]

if __name__ == '__main__':
    fs = SimulationStep()
    fs.configure_traits()
