'''
Created on Nov 6, 2014

@author: rch
'''

from traits.api import \
    implements

from forming_task import \
    FormingTask

from i_forming_task import \
    IFormingTask


class FactoryTask(FormingTask):

    r'''Factory task that generates a a formed_object as a pre-form.

    The most simple example of the ForminingTask. It has no previous task.
    It constructs the ``formed_object`` using the method ``deliver``.
    that must be implemented by subclasses.
    '''
    implements(IFormingTask)

    def _get_formed_object(self):
        return self.deliver()

    def deliver(self):
        raise NotImplementedError('no factory function implemented for %s',
                                  self)

    previous_task = None

if __name__ == '__main__':
    ft = FactoryTask()
    print ft.source_task
    print ft.formed_object
