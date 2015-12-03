'''
Created on Dec 3, 2015

@author: rch
'''

from traits.api import \
    HasTraits, WeakRef, Str

from visual3d import \
    Visual3D


class Viz3D(HasTraits):
    '''Base class for visualization objects.
    Each state and operator objects like crease pattern
    or constraint can define provide tailored visualizations
    transferring the information into a view objects shared
    within a particular forming task or a whole forming process.
    '''

    label = Str('default')
    '''Label of the visualization object.
    '''

    vis3d = WeakRef(Visual3D)
    '''Link to the visual object to transform into the 
    forming_task_view3d.
    '''

    ftv = WeakRef
    '''Folding task view3d object. 
    '''

    def register(self, ftv):
        '''Construct the visualization within the forming task view3d object.
        '''
        ftv.viz3d_dict[self.label] = self
        return

    def plot(self):
        '''Plot the object within ftv
        '''
        return

    def update(self):
        '''Update the visualization within the view3d object.
        '''
        return
