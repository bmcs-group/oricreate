'''
Created on Aug 27, 2014

Given are four nodes

    .. math::
        u_1 = \left[ 0, 0 \right]

        u_2 = \left[ 1, 0 \right]

        u_3 = \left[ 0, 1 \right]

        u_4 = \left[ 1, 1 \right]

Two vector pairs

    .. math::
        a_1 = u_2 - u_1, \;\; b_1 = u_3 - u_1

        a_2 = u_2 - u_1, \;\; b_2 = u_4 - u_1

The derivatives of the vectors with respect to the node :math:`u_1` are

    .. math::

        \pard{a_1}{u_{1x}} = -1
'''

from oricreate.crease_pattern import \
    get_gamma, get_gamma_du, get_theta, get_theta_du3, get_theta_du2

import numpy as np

a = np.array([[1, 0], [1, 0]], dtype='f')
b = np.array([[0, 1], [1, 1]], dtype='f')

print get_gamma(a, b)
print get_theta(a, b)

a_du = np.array([[
                  [[-1, 0]],
                  [[0, -1]]
                  ],
                 [
                  [[-1, 0]],
                  [[0, -1]]
                  ]],
                dtype='f')

b_du = np.array([[
                  [[-1, 0]],
                  [[0, -1]]
                  ],
                 [
                  [[-1, 0]],
                  [[0, -1]]
                  ]],
                dtype='f')

print 'gamma_du3'
print get_gamma_du(a, a_du, b, b_du)
print 'theta_du2'
print get_theta_du2(a, a_du, b, b_du)
