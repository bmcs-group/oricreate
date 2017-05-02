'''
Created on Dec 3, 2015

@author: rch
'''

from traits.api import \
    HasTraits, WeakRef, Str, Event, Property

from oricreate.forming_tasks import \
    FormingTask


class Exporter(HasTraits):
    '''Base class for visualization objects.
    Each state and operator objects like crease pattern
    or constraint can define provide tailored visualizations
    transferring the information into a view objects shared
    within a particular forming task or a whole forming process.
    '''

    label = Str('default')
    '''Label of the visualization object.
    '''

    forming_task = WeakRef(FormingTask)
    '''Link to the visual object to transform into the 
    forming_task_view3d.
    '''

    forming_task_changed = Event
    '''Event registering changes in the source object.
    '''
