'''
Created on Feb 19, 2015

@author: rch
'''

from traits.api import \
    implements, \
    Array, cached_property, Property, Dict, Float


from .hu import Hu
from .hu_psi_constraints_viz3d2 import \
    HuPsiConstraintsViz3D
import numpy as np
from oricreate.opt import \
    IHu
from oricreate.viz3d import \
    Visual3D


class HuPsiConstraints(Hu, Visual3D):

    '''Explicit constraints for selected of freedom.
    '''
    implements(IHu)

    psi_constraints = Array
    '''Specification of explicit constraint for particular dihedral psis.

    psi constraints are specified as a list of equations with values
    to be inserted on the left- and the right-hand-side of the equation system.
    The dof is identified with the number of the node and the direction (0,1,2)
    for x,y,z values::

        [([(line1, True ),
         ([(line2, False )
         ... ]

    Convenience constructors for containers of (node, direction pairs)
    are provided for the most usual cases:
    :func:`oricrete.fix_psis` and :func:`oricrete.link_psis`.
    '''

    sign = Dict({True: 1, False: -1})

    threshold = Float(np.pi * 0.0)

    def validate_input(self):
        cp = self.formed_object
        for i, psi_cnstr in enumerate(self.psi_constraints):  # @UnusedVariable
            l, sign = psi_cnstr  # @UnusedVariable
            if cp.L_iL[l] < 0:
                raise IndexError('GuPsiConstraint: line index %d does ' \
                    'not refer to an interior line: '\
                    'must be one of %s' % (l, cp.iL))

    def get_H(self, t=0):
        ''' Calculate the residue for given constraint equations
        '''
        cp = self.formed_object
        iL_psi = cp.iL_psi
        H = np.zeros((len(self.psi_constraints)), dtype='float_')
        for i, psi_cnstr in enumerate(self.psi_constraints):
            l, is_mountain = psi_cnstr
            sign = self.sign[is_mountain]
            il = cp.L_iL[l]
            H[i] = sign * iL_psi[il] + self.threshold
        return H

    def get_H_du(self, t=0.0):
        ''' Calculate the residue for given constraint equations
        '''
        cp = self.formed_object
        iL_psi_du = cp.iL_psi_du
        H_du = np.zeros((len(self.psi_constraints), cp.n_dofs),
                        dtype='float_')
        for i, psi_cnstr in enumerate(self.psi_constraints):
            l, is_mountain = psi_cnstr
            sign = self.sign[is_mountain]
            il = cp.L_iL[l]
            H_du[i, :] = sign * iL_psi_du[il, :].flatten()

        return H_du

    viz3d_dict = Property

    @cached_property
    def _get_viz3d_dict(self):
        return dict(default=HuPsiConstraintsViz3D(vis3d=self))
