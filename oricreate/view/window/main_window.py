'''

@author: rch
'''

from traitsui.menu import \
    Menu, MenuBar, Separator
from .forming_process_view import FormingProcessView
from .forming_task_view3d import FormingTaskView3D
import traits.api as tr
import traitsui.api as tu
from .tree_view_handler import \
    TreeViewHandler, \
    menu_exit, \
    toolbar_actions, key_bindings


class MainWindow(tr.HasStrictTraits):

    forming_process_view = tr.Instance(FormingProcessView, ())
    forming_task_scene = tr.Instance(FormingTaskView3D, ())

    forming_process = tr.Property

    def _get_forming_process(self):
        return self.forming_process_view.forming_process

    def _set_forming_process(self, fp):
        self.forming_process_view.forming_process = fp

    def _selected_node_changed(self):
        self.selected_node.ui = self

    def get_vot_range(self):
        return self.forming_task_scene.get_vot_range()

    vot = tr.DelegatesTo('forming_task_scene')

    data_changed = tr.Event

    replot = tr.Button

    def _replot_fired(self):
        self.figure.clear()
        self.selected_node.plot(self.figure)
        self.data_changed = True

    clear = tr.Button()

    def _clear_fired(self):
        self.figure.clear()
        self.data_changed = True

    view = tu.View(
        tu.HSplit(
            tu.VGroup(
                tu.Item('forming_process_view@',
                        id='oricreate.hsplit.left.tree.id',
                        resizable=True,
                        show_label=False,
                        width=300,
                        ),
                id='oricreate.hsplit.left.id',
            ),
            tu.VGroup(
                tu.Item('forming_task_scene@',
                        show_label=False,
                        resizable=True,
                        id='oricreate.hsplit.viz3d.notebook.id',
                        ),
                id='oricreate.hsplit.viz3d.id',
                label='viz sheet',
            ),
            id='oricreate.hsplit.id',
        ),
        id='oricreate.id',
        width=1.0,
        height=1.0,
        title='OriCreate',
        resizable=True,
        handler=TreeViewHandler(),
        key_bindings=key_bindings,
        toolbar=tu.ToolBar(*toolbar_actions,
                           image_size=(32, 32),
                           show_tool_names=False,
                           show_divider=True,
                           name='view_toolbar'),
        menubar=tu.MenuBar(Menu(menu_exit, Separator(),
                                name='File'),
                           )
    )


if __name__ == '__main__':
    from oricreate.forming_tasks.forming_process import FormingProcess
    from oricreate.factories.miura_ori_cp_factory import MiuraOriCPFactory
    from .point_cloud_viz3d import PointCloud
    pc = PointCloud()
    fp = FormingProcess(factory_task=MiuraOriCPFactory())
    tv = MainWindow(forming_process=fp)
    tv.forming_task_scene.add(fp.factory_task.cp.viz3d['cp'])
    tv.forming_task_scene.add(pc.viz3d['default'])
    tv.forming_task_scene.plot()
    tv.configure_traits()
