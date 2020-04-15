'''
Created on Nov 6, 2014

@author: rch
'''

from traits.api import \
    provides, on_trait_change, Property, Instance, cached_property

from .forming_task import \
    FormingTask
from .i_formed_object import \
    IFormedObject
from .i_forming_task import \
    IFormingTask


@provides(IFormingTask)
class FactoryTask(FormingTask):

    r'''Factory task that generates a a formed_object as a pre-form.

    The most simple example of the ForminingTask. It has no previous task.
    It constructs the ``formed_object`` using the method ``deliver``.
    that must be implemented by subclasses.
    '''

    @on_trait_change('+geometry')
    def notify_geometry_change(self):
        self.source_config_changed = True

    formed_object = Property(Instance(IFormedObject),
                             depends_on='source_config_changed')
    r'''Subject of forming.
    '''
    @cached_property
    def _get_formed_object(self):
        return self.deliver()

    def deliver(self):
        raise NotImplementedError('no factory function implemented for %s',
                                  self)

    previous_task = None


if __name__ == '__main__':
    ft = FactoryTask()
    print(ft.source_task)
    # print ft.formed_object
    ft.configure_traits()
