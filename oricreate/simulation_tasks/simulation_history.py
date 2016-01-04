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
# Created on Sep 7, 2011 by: rch

from traits.api import \
    Property, cached_property, \
    Array, List, Int, on_trait_change

import numpy as np
from oricreate.crease_pattern import \
    CreasePattern


INPUT = '+cp_input'


class SimulationHistory(CreasePattern):

    r'''
    This class is used to record the motion history
    of a crease pattern during a SimulationTask task.

    It maintains an array :math:`\bm{u}_t` of displacement vectors
    corresponding to each time step.

    '''

    u_t_list = List(Array(value=[], dtype='float_'), cp_input=True)
    r'''Displacement array with ``(n_t,n_N,n_D)`` values.
    '''

    u_t = Property(Array(dtype='float_'), depends_on='+INPUT')

    @cached_property
    def _get_u_t(self):
        return np.array(self.u_t_list, dtype='float_')

    x_t = Property(depends_on=INPUT)
    r'''Interim coordinates of the crease pattern
    '''
    @cached_property
    def _get_x_t(self):
        return self.x_0[np.newaxis, ...] + self.u_t

    @on_trait_change('vot')
    def _set_time_step(self):
        n_t = len(self.x_t) - 1
        i_t = int(self.vot * n_t)
        self.time_step = i_t

    time_step = Int(0, cp_input=True)
    r'''Current time step
    '''
    x = Property
    '''Current position of the coordinates.
    '''

    def _get_x(self):
        return self.x_t[self.time_step]

if __name__ == '__main__':

    # trivial example with a single triangle positioned

    cp = CreasePatternState(x_0=[[0, 0, 0],
                                 [1, 0, 0],
                                 [1, 1, 0],
                                 [0.667, 0.333, 0],
                                 [0.1, 0.05, 0]],
                            L=[[0, 1],
                               [1, 2],
                               [2, 0]],
                            F=[[0, 1, 2]]
                            )

    print 'vectors\n', cp.L_vectors
    print 'lengths\n', cp.L_lengths

    cp.u = np.zeros_like(cp.x_0)
    cp.u[:, 2] = 1.0

    print 'x\n', cp.x
