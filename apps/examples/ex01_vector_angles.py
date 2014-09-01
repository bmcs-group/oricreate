r'''

Given are four nodes and two pairs of vectors

.. image:: figs/ex01_vector_angles.png

.. math::
    \bm{x}_1 = \left[ 0, 0 \right]

    \bm{x}_2 = \left[ 1, 0 \right]

    \bm{x}_3 = \left[ 0, 1 \right]

    \bm{x}_4 = \left[ 1, 1 \right]

Two vector pairs displayed in the Figure are defined as:

.. math::
    \bm{a}_1 = \bm{x}_2 - \bm{x}_1, \;\; \bm{b}_1 = \bm{x}_3 - \bm{x}_1

    \bm{a}_2 = \bm{x}_2 - \bm{x}_1, \;\; \bm{b}_2 = \bm{x}_4 - \bm{x}_1

Considering current position :math:`\bm{x}_i` as a displaced configuration
of an initial vector :math:`\bm{x}^0_i` by a displacement vector :math:`\bm{u}_i`

.. math::
    \bm{x}_i = \bm{x}^0_i + \bm{u}_i

The derivatives of the vectors with respect to the node :math:`\bm{u}_1` are then

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

Given the vectors above as::

    In [1]: a = np.array([[1, 0], [1, 0]], dtype='f')
    In [2]: b = np.array([[0, 1], [1, 1]], dtype='f')

the cosines between the vector pairs obtained using the methods defined above deliver the following values::

    In [3]: print get_gamma(a, b)
    [ 0.          0.70710677]

The values of angles between the vector pairs are::

    In [4]: print get_theta(a, b)
    [ 1.57079637  0.78539819]

The derivatives of the cosines are::

    In [5]: print get_gamma_du(a, a_du, b, b_du)
    [[[-1.         -1.        ]]
     [[-0.35355338 -0.35355338]]]

and the derivatives of the angle values are::

    In [5]: print get_theta_du(a, a_du, b, b_du)
    [[[ 1.          1.        ]]
     [[ 0.49999997  0.49999997]]]

'''

if __name__ == '__main__':

    import numpy as np
    from oricreate.util import \
        get_gamma, get_gamma_du, get_theta, get_theta_du, get_gamma_du2


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

    print 'gamma_du'
    print get_gamma_du(a, a_du, b, b_du)
    print 'gamma_du2'
    print get_gamma_du2(a, a_du, b, b_du)

    print(get_theta_du(a, a_du, b, b_du))
