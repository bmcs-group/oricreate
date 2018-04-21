# -------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in Simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Jan 29, 2013 by: rch

import copy
from traits.api import \
    HasStrictTraits, Event, Property, Str, \
    List, Instance, implements, \
    cached_property, Enum, Array
from traitsui.api import \
    View, Item

from .i_formed_object import \
    IFormedObject
from .i_forming_task import \
    IFormingTask


class FormingTask(HasStrictTraits):

    r"""Control node in the design process of
    a crease pattern element.
    """
    source_config_changed = Event

    # implements(IFormingTask)

    node = Str
    '''Name of the tree node
    '''

    def _node_default(self):
        return self.__class__.__name__

    t = Enum([0, 1], enter_set=True, auto_set=False)
    '''Time scale ranging between 0 and 1 by default.
    The editing forming tasks have just initial and the final 
    state. For simulation tasks float variable time is introduced.'''

    previous_task = Instance(IFormingTask)
    r'''Previous FormingTask simulation providing
    the source for the current one.
    '''

    def _previous_task_changed(self):
        self.previous_task.following_tasks.append(self)

    following_tasks = List()
    '''Tasks using referring to this task as a source.
    '''

    formed_object = Property(Instance(IFormedObject),
                             depends_on='source_config_changed')
    r'''Subject of forming.
    '''
    @cached_property
    def _get_formed_object(self):
        if self.previous_task:
            prev_fo = self.previous_task.formed_object
            fo = copy.deepcopy(prev_fo)
            return fo
        else:
            raise NotImplementedError('No pre-fabrication method'
                                      'for the object to form')

    cp = Property
    '''Instance of a crease pattern.
    '''

    def _get_cp(self):
        return self.formed_object

    source_task = Property
    r'''The task without ``previous_task`` is the initial task.
    Recursive search.
    '''

    def _get_source_task(self):
        if self.previous_task:
            return self.source_task
        else:
            return self

    x_0 = Property(Array(float))
    '''Initial position of the forming task.
    '''

    def _get_x_0(self):
        return self.formed_object.x_0

    x_1 = Property(Array(float))
    '''Final state of the FormingTask process that can be used
    by further FormingTask controller.
    '''

    def _get_x_1(self):
        return self.x_0 + self.u_1

    traits_view = View(Item('node'))


if __name__ == '__main__':
    ft = FormingTask()
    ft.configure_traits()
