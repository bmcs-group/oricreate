'''
Created on Dec 3, 2015

@author: rch
'''

from traits.api import \
    HasStrictTraits, Dict, Property, Float, \
    WeakRef, DelegatesTo, cached_property


class Viz3DDict(HasStrictTraits):
    '''On demand constructor of viz3d object, 
    Objects are constructed upon access using the key within  
    the viz3d_classes dictionary.
    '''

    vis3d = WeakRef

    viz3d_classes = DelegatesTo('vis3d')

    _viz3d_objects = Dict

    def __getitem__(self, key):
        viz3d = self._viz3d_objects.get(key, None)
        if viz3d == None:
            viz3d_class = self.viz3d_classes.get(key, None)
            if viz3d_class == None:
                raise KeyError, 'No vizualization class with key %s' % key
            viz3d = viz3d_class(label=key, vis3d=self.vis3d)
            self._viz3d_objects[key] = viz3d
        return viz3d


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

    anim_t_start = Float(0.0, enter_set=True, auto_set=False, input=True)
    anim_t_end = Float(-1.0, enter_set=True, auto_set=False, input=True)

    viz3d_classes = Dict
    '''Visualization classes applicable to this object. 
    '''

    viz3d = Property(Dict)
    '''Dictionary of visualization objects'''
    @cached_property
    def _get_viz3d(self):
        '''Get a vizualization object given the key
        of the vizualization class. Construct it on demand
        and register in the viz3d_dict.
        '''
        return Viz3DDict(vis3d=self)

    def viz3d_notify_change(self):
        for viz3d in self.viz3d.values():
            viz3d.vis3d_changed = True

Vis3D = Visual3D