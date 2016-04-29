r'''

Given are five nodes and two pairs of vectors

.. image:: figs/ex01_vector_acos.png

.. math::
    \bm{x}_1 = \left[ 0, 0 \right]

    \bm{x}_2 = \left[ 1, 0 \right]

    \bm{x}_3 = \left[ 0, 1 \right]

    \bm{x}_4 = \left[ 1, 1 \right]

    \bm{x}_5 = \left[ 1, -1 \right]

This example shows the evaluation of the angle sines :math:`\gamma`
and angles :math:`\theta` between two pairs of vectors.
Further, the derivatives of the sines :math:`\partial \gamma / \partial \bm{u}_1`
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

The script demonstrates the evaluation of vectors on the example displayed above.
First, the cosines :math:`\gamma` between the vector pairs and
of the angles :math:`\theta` between them are printed.
Then, the derivatives of the cosines :math:`\partial \gamma / \partial \bm{u}_1` and
of the angle values :math:`\partial \theta / \partial \bm{u}_1`  are calculated.

'''

if __name__ == '__main__':

    # end_doc

    import numpy as np
    from oricreate.util import \
        get_sin_theta

    a = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype='f')
    b = np.array([[0, 1, 0], [1, 1, 0], [1, -1, 0]], dtype='f')

    print 'gamma', get_sin_theta(a, b)
