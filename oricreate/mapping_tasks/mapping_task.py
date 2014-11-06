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
    Instance

from crease_pattern import \
    CreasePattern

from forming_tasks import \
    FormingTask  # @UnresolvedImport


class MappingTask(FormingTask):

    r'''Task mapping the initial state of the formed object
    to the final state in a single step using a mapping function.
    The topology and size of the formed object can change.
    There is no forming process involved.
    '''

    def _get_formed_object(self):
        return CreasePattern(X=self.previous_task.formed_object.X,
                             L=self.previous_task.formed_object.L,
                             F=self.previous_task.formed_object.F)


if __name__ == '__main__':
    cp = CreasePattern(X=[[1, 2, 3]])
    it = CreasePatternGenerator(formed_object=cp)
    ft = MappingTask(previous_task=it)
    print ft.formed_object
