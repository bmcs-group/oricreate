#-------------------------------------------------------------------------
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
# Created on Sep 7, 2011 by: rch

from traits.api import \
    Instance

from oricreate.crease_pattern import \
    CreasePattern

from oricreate.forming_tasks import \
    FactoryTask


class CustomCPFactory(FactoryTask):

    '''Seller of the crease pattern.
    '''

    formed_object = Instance(CreasePattern)

    def _formed_object_default(self):
        return CreasePattern(X=[0, 0, 0])


if __name__ == '__main__':

    cp = CreasePattern(X=[[0, 0, 0],
                          [1, 1, 0]],
                       L=[[0, 1]])

    yf = CustomCPFactory(formed_object=cp)

    cp = yf.formed_object

    import pylab as p
    cp.plot_mpl(p.axes())
    p.show()
