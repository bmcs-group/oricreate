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
    implements

import numpy as np

from i_opt_crit import \
    IOptCrit
from opt_crit import \
    OptCrit


class OptCritPotentialEnergy(OptCrit):

    '''Optimization criteria based on minimum potential energy of gravity.

    This plug-in class lets the crease pattern operators evaluate the
    integral over the spatial domain in an instantaneous configuration
    '''

    implements(IOptCrit)

    def get_f(self, u, t=0):
        '''Get the potential energy of gravity.
        '''
        return self.simulation_task.formed_object.get_V(u)

    def get_f_du(self, u, t=0):
        '''Get the derivatives with respect to individual displacements.
        '''
        return self.simulation_task.formed_object.get_V_du(u)
