'''
Created on Nov 7, 2014

@author: rch
'''

from oricreate.api import FTT, FTV, CustomCPFactory
from traits.api import HasTraits, Instance
from traitsui.api import View, UItem, HSplit


class MainWindow(HasTraits):

    ftt = Instance(FTT)

    def _ftt_default(self):
        return FTT(root=CustomCPFactory())

    ftv = Instance(FTV)

    def _ftv_default(self):
        return FTV()

    traits_view = View(HSplit(UItem('ftt@', width=300),
                              UItem('ftv@')),
                       width=1.0,
                       height=1.0,
                       resizable=True,
                       title='oricreate')

if __name__ == '__main__':
    mv = MainWindow()
    mv.configure_traits()
