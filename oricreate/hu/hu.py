# -------------------------------------------------------------------------
#
# Copyright (c) 2009-2013, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Jan 3, 2013 by: rch, schmerl

from traits.api import \
    DelegatesTo, \
    Bool

from oricreate.opt import \
    OptComponent, IGu


class Hu(OptComponent):

    '''Base class of equality constraints.
    '''

    x_0 = DelegatesTo('forming_task')
    '''Nodal coordinates
    '''

    has_H_du = Bool(True)
    '''Indicates the derivatives are unavailable for a given
    type of constraint.
    '''
