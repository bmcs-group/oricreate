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
# Created on Jan 29, 2013 by: rch

from traits.api import \
    Interface, Property, Self, Instance

from i_formed_object import IFormedObject


class IFormingTask(Interface):

    r'''IFormingTask(s) constitute a FormingProcess.
    They can be chained. Each FormingTask has a
    previous forming task.
    '''
    previous_task = Instance(Self)

    source_task = Instance(Self)

    formed_object = Instance(IFormedObject)
