'''
Created on Feb 19, 2015

@author: rch
'''

import types

from traits.api import \
    provides, Tuple, \
    cached_property, Property

<<<<<<< master
<<<<<<< HEAD
=======
=======
>>>>>>> interim stage 1
from .fu import Fu
from .fu_target_psi_value_viz3d2 import \
    FuTargetPsiValueViz3D
>>>>>>> 2to3
from oricreate.opt import \
    IFu
from oricreate.viz3d import \
    Visual3D

from .fu import Fu
from .fu_target_psi_value_viz3d2 import \
    FuTargetPsiValueViz3D


@provides(IFu)
class FuTargetPsiValue(Fu, Visual3D):

    '''Explicit constraints for selected of freedom.
    '''

    psi_value = Tuple
    '''Specification of explicit constraint for particular dihedral psis.

    psi constraints are specified as a list of equations with values
    to be inserted on the left- and the right-hand-side of the equation system.
    The dof is identified with the number of the node and the direction (0,1,2)
    for x,y,z values::

        (line1, value)

    Convenience constructors for containers of (node, direction pairs)
    are provided for the most usual cases:
    :func:`oricrete.fix_psis` and :func:`oricrete.link_psis`.
    '''

    def validate_input(self):
        cp = self.formed_object
        l, value = self.psi_value  # @UnusedVariable
        if cp.L_iL[l] < 0:
<<<<<<< master
<<<<<<< HEAD
            raise IndexError('GuPsiConstraint: line index %d does '
                             'not refer to an interior line: '
                             'must be one of %s' % (l, cp.iL))
=======
            raise IndexError('GuPsiConstraint: line index %d does ' \
                'not refer to an interior line: '\
                'must be one of %s' % (l, cp.iL))
>>>>>>> 2to3
=======
            raise IndexError('GuPsiConstraint: line index %d does ' \
                'not refer to an interior line: '\
                'must be one of %s' % (l, cp.iL))
>>>>>>> interim stage 1

    def get_f(self, t=0):
        ''' Calculate the residue for given constraint equations
        '''
        cp = self.formed_object
        iL_psi = cp.iL_psi
        l, v = self.psi_value
        value = v(t) if isinstance(v, types.FunctionType) else v
        il = cp.L_iL[l]
        return 0.5 * (iL_psi[il] - value) ** 2

    def get_f_du(self, t=0.0):
        ''' Calculate the residue for given constraint equations
        '''
        cp = self.formed_object
        iL_psi = cp.iL_psi
        iL_psi_du = cp.iL_psi_du
        l, v = self.psi_value  # @UnusedVariable
        value = v(t) if isinstance(v, types.FunctionType) else v
        il = cp.L_iL[l]
        return (iL_psi[il] - value) * iL_psi_du[il]

    viz3d_dict = Property

    @cached_property
    def _get_viz3d_dict(self):
        return dict(default=FuTargetPsiValueViz3D(vis3d=self))
