'''
Created on 14. 4. 2014

@author: Vancikv
'''

import pickle

from pyface.api import ImageResource
from traits.api import \
    Button, Instance, WeakRef, Str
from traitsui.api import \
    View, Item, HGroup, Handler, \
    UIInfo, spring, VGroup, Label
from traitsui.file_dialog import \
    open_file, save_file
from traitsui.key_bindings import \
    KeyBinding, KeyBindings
from traitsui.menu import \
    Action


plot_self = Action(name='Plot', action='plot_node')
'''Context menu action for plotting tree nodes
'''
menu_exit = Action(name='Exit', action='menu_exit')
'''Menubar action for terminating the view
'''

key_bindings = KeyBindings(
    KeyBinding(binding1='Ctrl-r',
               description='Run simulation',
               method_name='run_action'),
    KeyBinding(binding1='Ctrl-p',
               description='Pause calculation',
               method_name='pause_action'),
    KeyBinding(binding1='Ctrl-s',
               description='Stop calculation',
               method_name='stop_action')
)


class TreeViewHandler(Handler):

    '''Handler for TreeView class
    '''
    # The UIInfo object associated with the view:
    info = Instance(UIInfo)
    node = WeakRef

    ok = Button('OK')
    cancel = Button('Cancel')
    delete = Button('OK')

    exit_view = View(
        VGroup(
            Label('Do you really wish to end '
                  'the session? Any unsaved data '
                  'will be lost.'),
            HGroup(Item('ok', show_label=False, springy=True),
                   Item('cancel', show_label=False, springy=True)
                   ),
        ),
        title='Exit dialog',
        kind='live'
    )

    del_view = View(
        HGroup(
            spring,
            Item(name='', label='Really delete?'),
            Item('delete', show_label=False),
            Item('cancel', show_label=False),
        ),
        kind='popup'
    )

    def plot_node(self, info, node):
        '''Handles context menu action Plot for tree nodes
        '''
        info.object.figure.clear()
        node.plot(info.object.figure)
        info.object.data_changed = True

    def menu_exit(self, info):
        if info.initialized:
            self.info = info
            self._ui = self.edit_traits(view='exit_view')

    def _delete_fired(self):
        del self.node.db[self.node.key]
        self._ui.dispose()

    def _ok_fired(self):
        self._ui.dispose()
        self.info.ui.dispose()

    def _cancel_fired(self):
        self._ui.dispose()

    #=========================================================================
    # Toolbar actions
    #=========================================================================

    def run_action(self, info):
        print('Running action')
        info.object.run()

    def pause_action(self, info):
        print('Running pause')
        info.object.pause()

    def stop_action(self, info):
        print('Running stop')
        info.object.stop()

    def replot(self, info):
        print('Running continue')
        info.object.replot()

    def clear(self, info):
        info.object.clear()

    def anim(self, info):
        info.object.anim()

    def render(self, info):
        info.object.render()


action_strings = \
    [('Plot', 'replot', 'Replot current diagrams'),
     ('Clear', 'clear', 'Clear current diagrams'),
     ('Save', 'save', 'Save session'),
     ('Load', 'load', 'Load session'),
     ('Animate', 'anim', 'Animate current session'),
     ('Render', 'render', 'Render current session')]

toolbar_actions = [Action(name="Run",
                          tooltip='Start computation',
                          enabled_when='enable_run',
                          image=ImageResource('kt-start'),
                          action="run_action"),
                   Action(name="Pause",
                          tooltip='Pause computation',
                          enabled_when='enable_pause',
                          image=ImageResource('kt-pause'),
                          action="pause_action"),
                   Action(name="Stop",
                          tooltip='Stop computation',
                          enabled_when='enable_stop',
                          image=ImageResource('kt-stop'),
                          action="stop_action"),
                   ]
