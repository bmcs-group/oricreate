# -------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Sep 8, 2011 by: matthias

from traitsui.api import \
    View, Item, UItem, Group, \
    TreeEditor, VSplit, \
    TreeNode

from oricreate.forming_tasks import FormingTask
from oricreate.forming_tasks.forming_process import FormingProcess
from oricreate.forming_tasks.i_forming_task import \
    IFormingTask
import traits.api as tr


FormingTask_tree_editor = TreeEditor(
    nodes=[
        TreeNode(node_for=[FormingTask],
                 auto_open=True,
                 label='node',
                 children='following_tasks',
                 ),
    ],
    hide_root=False,
    selected='selected',
    editable=False,
)


class FormingProcessView(tr.HasStrictTraits):
    '''Forming process viewer with task tree editor
    '''

    forming_process = tr.Instance(FormingProcess)
    root = tr.Property(tr.Instance(FormingTask))
    '''All FormingTask steps.
    '''

    def _get_root(self):
        return self.forming_process.factory_task

    selected = tr.Instance(IFormingTask)

    view1 = View(
        VSplit(
            Group(Item('root',
                       editor=FormingTask_tree_editor,
                       resizable=True,
                       show_label=False),
                  label='Design tree',
                  scrollable=True,
                  ),
            Group(UItem('selected@')),
            dock='tab'
        ),
        dock='tab',
        resizable=True,
        title='Forming Process View',
        width=1.0,
        height=1.0
    )


FPV = FormingProcessView

# =========================================================================
# Test Pattern
# =========================================================================

if __name__ == '__main__':

    ft1 = FormingTask(node='initial task')
    ft2 = FormingTask(node='fold_task #1', previous_task=ft1)
    ft3 = FormingTask(node='fold_task #2', previous_task=ft1)
    ft4 = FormingTask(node='copy_task #1', previous_task=ft2)
    ft5 = FormingTask(node='turn_task #2', previous_task=ft4)

    fp = FormingProcess(factory_task=ft1)
    view = FormingProcessView(forming_process=fp)
    view.configure_traits()
