#-------------------------------------------------------------------------
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
# Created on Nov 18, 2011 by: matthias

from traits.api import \
    provides

<<<<<<< master
<<<<<<< HEAD
=======
=======
>>>>>>> interim stage 1
from .fu import \
    Fu
>>>>>>> 2to3
from oricreate.opt import \
    IFu

from .fu import \
    Fu


@provides(IFu)
class FuPotEngBending(Fu):

    '''Optimization criteria based on minimum Bending energy of gravity.

    This plug-in class lets the crease pattern operators evaluate the
    integral over the spatial domain in an instantaneous configuration
    '''

    def get_f(self, t=0):
        '''Get the bending energy of gravity.
        '''
        return self.forming_task.formed_object.V

    def get_f_du(self, t=0):
        '''Get the derivatives with respect to individual displacements.
        '''
        return self.forming_task.formed_object.V_du
