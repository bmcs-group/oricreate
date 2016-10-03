'''
Created on Dec 3, 2015

@author: rch
'''

from traits.api import \
    HasStrictTraits, Dict, Property, Float


class Visual3D(HasStrictTraits):
    '''Each state and operator object can be associated with 
    several visualization objects with a shortened class name Viz3D. 
    In order to introduce a n independent class subsystem into 
    the class structure, objects supporting visualization inherit 
    from Visual3D which introduces a dictionary viz3d objects.
    '''

    vot = Float(0.0, time_change=True)
    '''Visual object time
    '''
    viz3d = Dict({})
    '''Dictionary of visualization objects'''

    viz3d_classes = Dict
    '''Visualization classes applicable to this object. 
    '''

    def get_viz3d(self, key):
        '''Get a vizualization object given the key
        of the vizualization class. Construct it on demand
        and register in the viz3d_dict.
        '''
        viz3d = self.viz3d.get(key, None)
        if viz3d == None:
            viz3d_class = self.viz3d_classes.get(key, None)
            if viz3d_class == None:
                raise KeyError, 'No vizualization class with key %s' % key
            viz3d = viz3d_class(vis3d=self)
            self.viz3d[key] = viz3d
        return viz3d

    def viz3d_notify_change(self):
        for viz3d in self.viz3d_dict.values():
            viz3d.vis3d_changed = True
