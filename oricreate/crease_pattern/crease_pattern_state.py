# -------------------------------------------------------------------------------
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
    Array, on_trait_change, NO_COMPARE

from crease_pattern import \
    CreasePattern

from crease_pattern_operators import \
    CreaseNodeOperators, CreaseLineOperators, CreaseFacetOperators, \
    CreaseCummulativeOperators

import numpy as np

INPUT = '+cp_input'


class CreasePatternState(CreasePattern,
                         CreaseNodeOperators,
                         CreaseLineOperators,
                         CreaseFacetOperators,
                         CreaseCummulativeOperators,
                         ):

    r'''
    This class is used by the FormingTask tasks during the
    iteration to realize the predictor-corrector steps.
    It inherits from ``CreasePattern`` and possesses
    its own copy of the displacement
    vector :math:`\bm{u}` and provides this vector in a flattened
    form ``cp_state.U`` as property trait so that it can enter
    be directly transmitted to the
    the linear algebra operators. At the same time, the result
    of the calculation step obtained in a flattened form can be
    updated within this class using the property setter method
    ``cp_state.U = U``
    '''

    u = Array(value=[], dtype='float_', cp_input=True,
              comparison_mode=NO_COMPARE)
    r'''Displacement array with ``(n_N,n_D)`` values.
    '''

    def u_default(self):
        return np.zeros_like(self.x_0)

    @on_trait_change('X')
    def _reset_u(self):
        self.u = np.zeros_like(self.x_0)

    U = Property
    r'''Array of initial coordinates ``(n_N,n_D)`` as ``[x1,x2,x3]``.
    '''

    def _get_U(self):
        return self.u.flatten()

    def _set_U(self, U):
        self.u = U.reshape(self.n_N, self.n_D)

    x = Property(depends_on=INPUT)
    r'''Interim coordinates of the crease pattern

    ..  math::
        \bm{x} = \bm{x}_0 + \bm{u}

    '''
    @cached_property
    def _get_x(self):
        if len(self.x_0) == len(self.u):
            return self.x_0 + self.u
        else:
            return self.x_0

    n_dofs = Property(depends_on=INPUT)
    '''Total number of displacement degrees of freedom.
    '''
    @cached_property
    def _get_n_dofs(self):
        return self.n_N * self.n_D

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
