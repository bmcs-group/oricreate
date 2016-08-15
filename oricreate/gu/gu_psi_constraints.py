'''
Created on Feb 19, 2015

@author: rch
'''

from traits.api import \
    implements, \
    Array, DelegatesTo, cached_property, Property
import types

from gu import Gu
from gu_psi_constraints_viz3d2 import \
    GuPsiConstraintsViz3D
import numpy as np
from oricreate.opt import \
    IGu
from oricreate.viz3d import \
    Visual3D


class GuPsiConstraints(Gu, Visual3D):

    '''Explicit constraints for selected of freedom.
    '''
    implements(IGu)

    psi_constraints = Array
    '''Specification of explicit constraint for particular dihedral psis.

    psi constraints are specified as a list of equations with values
    to be inserted on the left- and the right-hand-side of the equation system.
    The dof is identified with the number of the node and the direction (0,1,2)
    for x,y,z values::

        [([(line1, coefficient1), ... ], value1 ),
         ([(line2, coefficient2), ... ], value2 )
         ... ]

    Convenience constructors for containers of (node, direction pairs)
    are provided for the most usual cases:
    :func:`oricrete.fix_psis` and :func:`oricrete.link_psis`.
    '''

    def validate_input(self):
        cp = self.formed_object
        for i, psi_cnstr in enumerate(self.psi_constraints):  # @UnusedVariable
            lhs, rhs = psi_cnstr  # @UnusedVariable
            for l, c in lhs:  # @UnusedVariable
                if cp.L_iL[l] < 0:
                    raise IndexError, \
                        'GuPsiConstraint: line index %d does ' \
                        'not refer to an interior line: '\
                        'must be one of %s' % (l, cp.iL)

    def get_G(self, t=0):
        ''' Calculate the residue for given constraint equations
        '''
        cp = self.formed_object
        iL_psi = cp.iL_psi
        G = np.zeros((len(self.psi_constraints)), dtype='float_')
        for i, psi_cnstr in enumerate(self.psi_constraints):
            lhs, rhs = psi_cnstr
            for l, c in lhs:  # @UnusedVariable
                il = cp.L_iL[l]
                G[i] += c * iL_psi[il]
            G[i] -= rhs(t) if isinstance(rhs, types.FunctionType) else rhs
        return G

    def get_G_du(self, t=0.0):
        ''' Calculate the residue for given constraint equations
        '''
        cp = self.formed_object
        iL_psi_du = cp.iL_psi_du
        G_du = np.zeros((len(self.psi_constraints), cp.n_dofs),
                        dtype='float_')
        for i, psi_cnstr in enumerate(self.psi_constraints):
            lhs, rhs = psi_cnstr  # @UnusedVariable
            for l, c in lhs:  # @UnusedVariable
                il = cp.L_iL[l]
                G_du[i, :] += c * iL_psi_du[il, :].flatten()

        return G_du

    viz3d_dict = Property

    @cached_property
    def _get_viz3d_dict(self):
        return dict(default=GuPsiConstraintsViz3D(vis3d=self))
