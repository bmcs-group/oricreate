'''
Created on Dec 3, 2015

@author: rch
'''

from traits.api import \
    HasStrictTraits, Dict, Property


class Visual3D(HasStrictTraits):
    '''Each state and operator object can be associated with 
    several visualization objects with a shortened class name Viz3D. 
    In order to introduce a n independent class subsystem into 
    the class structure, objects supporting visualization inherit 
    from Visual3D which introduces a dictionary viz3d objects.
    '''

    viz3d_dict = Dict({})
    '''Dictionary of visualization objects'''

    viz3d = Property
    '''Default visualization of an object'''

    def _get_viz3d(self):
        if len(self.viz3d_dict) == 1:
            return self.viz3d_dict.values()[0]
        elif self.viz3d_dict.has_key('default'):
            return self.viz3d_dict['default']
        else:
            raise NotImplementedError, 'no default visualization object for %s' % self.__class__
