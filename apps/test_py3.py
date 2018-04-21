'''
Created on Apr 20, 2018

@author: rch
'''

import sys

print(sys.path)
import oricreate.api

import traits.api as tr


class IZ(tr.Interface):
    def _get_it(self):
        return


class X(tr.HasStrictTraits):

    i = tr.Int()


x = X()
x.configure_traits()


def myprint():
    print ('printing')


if __name__ == '__main__':
    myprint()
