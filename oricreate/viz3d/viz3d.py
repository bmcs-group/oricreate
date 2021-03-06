'''
Created on Dec 3, 2015

@author: rch
'''

from traits.api import \
    HasStrictTraits, WeakRef, Str, Event, Property, Int, Bool, \
    Dict, PrototypedFrom
import traitsui.api as tui
from .visual3d import \
    Visual3D


class Viz3D(HasStrictTraits):
    '''Base class for visualization objects.
    Each state and operator objects like crease pattern
    or constraint can define provide tailored visualizations
    transferring the information into a view objects shared
    within a particular forming task or a whole forming process.
    '''

    label = Str('default')
    '''Label of the visualization object.
    '''

    anim_t_start = PrototypedFrom('vis3d')
    anim_t_end = PrototypedFrom('vis3d')

    order = Int(1, auto_set=False, enter_set=True)
    '''Deprecated -- is only here to have a control parameter
    that avoids text visualization at the beginning of the time line
    because then mlab fails. 
    '''

    vis3d = WeakRef(Visual3D)
    '''Link to the visual object to transform into the 
    forming_task_view3d.
    '''

    vis3d_changed = Event
    '''Event registering changes in the source object.
    '''

    ftv = WeakRef
    '''Folding task view3d object. 
    '''

    pipes = Dict()

    def register(self, ftv):
        '''Construct the visualization within the forming task view3d object.
        '''
        ftv.viz3d_dict[self.label] = self
        return

    def plot(self):
        '''Plot the object within ftv
        '''
        return

    hidden = Bool(False)

    def _show(self):
        if self.hidden == True:
            self.show()
            self.hidden = False

    def _hide(self):
        if self.hidden == False:
            self.hide()
            self.hidden = True

    def hide(self):
        for pipe in list(self.pipes.values()):
            pipe.visible = False

    def show(self):
        for pipe in list(self.pipes.values()):
            pipe.visible = True

    def update_t(self, anim_t=0.0, vot=0.0):
        '''Update with regard to the global time line.
        '''
        if anim_t >= self.anim_t_start and anim_t <= self.anim_t_end \
                or self.anim_t_end < 0.0:
            self._show()
            self.update(vot)
        else:
            self._hide()

    def update(self, vot=0.0):
        '''Update the visualization within the view3d object.
        '''
        return

    min_max = Property
    '''Bounding box limits set to none by default. 
    '''

    def _get_min_max(self):
        return None, None

    viz3d_view = tui.Group(tui.Item('hode', resizable=True))
