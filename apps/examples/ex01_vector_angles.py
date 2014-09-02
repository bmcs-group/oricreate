r'''

Given are four nodes and two pairs of vectors

.. image:: figs/ex01_vector_angles.png

.. math::
    \bm{x}_1 = \left[ 0, 0 \right]

    \bm{x}_2 = \left[ 1, 0 \right]

    \bm{x}_3 = \left[ 0, 1 \right]

    \bm{x}_4 = \left[ 1, 1 \right]

This example shows the evaluation of the angle cosines :math:`\gamma`
and angles :math:`\theta` between the two pairs of vectors.
Further, the derivatives of the cosines :math:`\partial \gamma / \partial \bm{u}_1`
and angles :math:`\partial \theta / \partial \bm{u}_1`
with respect to the displacement of the node :math:`\bm{x}_1`
are evaluated.

Two vector pairs displayed in the Figure are defined as:

.. math::
    \bm{a}_1 = \bm{x}_2 - \bm{x}_1, \;\; \bm{b}_1 = \bm{x}_3 - \bm{x}_1

    \bm{a}_2 = \bm{x}_2 - \bm{x}_1, \;\; \bm{b}_2 = \bm{x}_4 - \bm{x}_1

Representing the current node positions :math:`\bm{x}_i` as a sum of the
initial position  :math:`\bm{x}^0_i` and the displacement vector :math:`\bm{u}_i`

.. math::
    \bm{x}_i = \bm{x}^0_i + \bm{u}_i

the derivatives of vectors :math:`\bm{a}` and :math:`\bm{b}`
with respect to the node :math:`\bm{u}_1` are then

.. math::

    \pard{ \bm{a}_1}{\bm{u}_{1}}
    =
    -\bm{I},
    \;\;
    \pard{ \bm{a}_2}{\bm{u}_{1}}
    =
    -\bm{I},
    \;\;
    \pard{ \bm{b}_1}{\bm{u}_{1}}
    =
    -\bm{I},
    \;\;
    \pard{ \bm{b}_2}{\bm{u}_{1}}
    =
    -\bm{I}.

where :math:`\bm{I}` represents the unit matrix.

Given the vectors above as ``numpy`` arrays::

    In [1]: a = np.array([[1, 0], [1, 0]], dtype='f')
    In [2]: b = np.array([[0, 1], [1, 1]], dtype='f')

the cosines :math:`\gamma` between the vector pairs obtained using the methods defined above deliver the following values::

    In [3]: print get_gamma(a, b)
    [ 0.          0.70710677]

The values of angles :math:`\theta` between the vector pairs are::

    In [4]: print get_theta(a, b)
    [ 1.57079637  0.78539819]

The derivatives of the cosines :math:`\partial \gamma / \partial \bm{u}_1` are::

    In [5]: print get_gamma_du(a, a_du, b, b_du)
    [[[-1.         -1.        ]]
     [[-0.35355338 -0.35355338]]]

and the derivatives of the angle values :math:`\partial \theta / \partial \bm{u}_1`  are::

    In [6]: print get_theta_du(a, a_du, b, b_du)
    [[[ 1.          1.        ]]
     [[ 0.49999997  0.49999997]]]

'''

if __name__ == '__main__':

    import numpy as np
    from oricreate.util import \
        get_gamma, get_gamma_du, get_theta, get_theta_du, get_gamma_du2


    a = np.array([[1, 0], [1, 0]], dtype='f')
    b = np.array([[0, 1], [1, 1]], dtype='f')

    print('gamma')
    print(get_gamma(a, b))
    print('theta')
    print(get_theta(a, b))

    I = np.diag(np.ones((2,), dtype='f'))
    # dimensions of the derivatives are stored as:
    # index of a vector,
    # component of the vector,
    # index of a node,
    # component of a node.
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

    print('gamma_du')
    print(get_gamma_du(a, a_du, b, b_du))

    print 'theta_du'
    print(get_theta_du(a, a_du, b, b_du))
