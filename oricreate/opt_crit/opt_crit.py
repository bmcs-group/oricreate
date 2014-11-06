#-------------------------------------------------------------------------
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
    HasStrictTraits, WeakRef, \
    DelegatesTo, Bool


class OptCrit(HasStrictTraits):

    simulation_task = WeakRef
    '''Link to the FormingTask tool.
    '''

    x_0 = DelegatesTo('FormingTask')
    '''Nodal coordinates
    '''

    has_f_du = Bool(True)
    '''Indicates the derivatives are unavailable for a given
    type of constraint.
    '''

    def __init__(self, FormingTask, *args, **kw):
        '''Initialization requiring the FormingTask tool.
        '''
        self.FormingTask = FormingTask
        super(HasStrictTraits, self).__init__(*args, **kw)
