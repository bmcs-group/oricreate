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
    Array, List

from oricreate.crease_pattern import \
    CreasePatternState

import numpy as np


INPUT = '+cp_input'


class FormingHistory(CreasePatternState):

    r'''
    This class is used to record the motion history
    of a crease pattern during a FormingTask task.

    It maintains an array :math:`\bm{u}_t` of displacement vectors
    corresponding to each time step.

    '''

    u_t = List(Array(value=[], dtype='float_'), cp_input=True)
    r'''Displacement array with ``(n_N,n_D)`` values.
    '''

    def u_t_default(self):
        return np.zeros_like(self.x_0)

    x_t = Property(depends_on=INPUT)
    r'''Interim coordinates of the crease pattern
    '''
    @cached_property
    def _get_x(self):
        if len(self.x_0) == len(self.u):
            return self.x_0 + self.u
        else:
            return self.x_0

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
