# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Module implementing Euler angle rotations and their conversions

See:

* http://en.wikipedia.org/wiki/Rotation_matrix
* http://en.wikipedia.org/wiki/Euler_angles
* http://mathworld.wolfram.com/EulerAngles.html

See also: *Representing Attitude with Euler Angles and Quaternions: A
Reference* (2006) by James Diebel. A cached PDF link last found here:

http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.110.5134

Euler's rotation theorem tells us that any rotation in 3D can be
described by 3 angles.  Let's call the 3 angles the *Euler angle vector*
and call the angles in the vector :math:`alpha`, :math:`beta` and
:math:`gamma`.  The vector is [ :math:`alpha`,
:math:`beta`. :math:`gamma` ] and, in this description, the order of the
parameters specifies the order in which the rotations occur (so the
rotation corresponding to :math:`alpha` is applied first).

In order to specify the meaning of an *Euler angle vector* we need to
specify the axes around which each of the rotations corresponding to
:math:`alpha`, :math:`beta` and :math:`gamma` will occur.

There are therefore three axes for the rotations :math:`alpha`,
:math:`beta` and :math:`gamma`; let's call them :math:`i` :math:`j`,
:math:`k`.

Let us express the rotation :math:`alpha` around axis `i` as a 3 by 3
rotation matrix `A`.  Similarly :math:`beta` around `j` becomes 3 x 3
matrix `B` and :math:`gamma` around `k` becomes matrix `G`.  Then the
whole rotation expressed by the Euler angle vector [ :math:`alpha`,
:math:`beta`. :math:`gamma` ], `R` is given by::

   R = np.dot(G, np.dot(B, A))

See http://mathworld.wolfram.com/EulerAngles.html

The order :math:`G B A` expresses the fact that the rotations are
performed in the order of the vector (:math:`alpha` around axis `i` =
`A` first).

To convert a given Euler angle vector to a meaningful rotation, and a
rotation matrix, we need to define:

* the axes `i`, `j`, `k`
* whether a rotation matrix should be applied on the left of a vector to
  be transformed (vectors are column vectors) or on the right (vectors
  are row vectors).
* whether the rotations move the axes as they are applied (intrinsic
  rotations) - compared the situation where the axes stay fixed and the
  vectors move within the axis frame (extrinsic)
* the handedness of the coordinate system

See: http://en.wikipedia.org/wiki/Rotation_matrix#Ambiguities

We are using the following conventions:

* axes `i`, `j`, `k` are the `z`, `y`, and `x` axes respectively.  Thus
  an Euler angle vector [ :math:`alpha`, :math:`beta`. :math:`gamma` ]
  in our convention implies a :math:`alpha` radian rotation around the
  `z` axis, followed by a :math:`beta` rotation around the `y` axis,
  followed by a :math:`gamma` rotation around the `x` axis.
* the rotation matrix applies on the left, to column vectors on the
  right, so if `R` is the rotation matrix, and `v` is a 3 x N matrix
  with N column vectors, the transformed vector set `vdash` is given by
  ``vdash = np.dot(R, v)``.
* extrinsic rotations - the axes are fixed, and do not move with the
  rotations.
* a right-handed coordinate system

The convention of rotation around ``z``, followed by rotation around
``y``, followed by rotation around ``x``, is known (confusingly) as
"xyz", pitch-roll-yaw, Cardan angles, or Tait-Bryan angles.
'''

import math

import numpy as np
from functools import reduce


_FLOAT_EPS_4 = np.finfo(float).eps * 4.0


def euler2mat(z=0, y=0, x=0):
    ''' Return matrix for rotations around z, y and x axes

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    M : array shape (3,3)
       Rotation matrix giving same rotation as for given angles

    Examples
    --------
    >>> zrot = 1.3 # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True

    The output rotation matrix is equal to the composition of the
    individual rotations

    >>> M1 = euler2mat(zrot)
    >>> M2 = euler2mat(0, yrot)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True

    You can specify rotations by named arguments

    >>> np.all(M3 == euler2mat(x=xrot))
    True

    When applying M to a vector, the vector should column vector to the
    right of M.  If the right hand side is a 2D array rather than a
    vector, then each column of the 2D array represents a vector.

    >>> vec = np.array([1, 0, 0]).reshape((3,1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
    >>> vecs2 = np.dot(M, vecs)

    Rotations are counter-clockwise.

    >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
    True

    Notes
    -----
    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
    '''
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
            [[cosz, -sinz, 0],
             [sinz, cosz, 0],
             [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
            [[cosy, 0, siny],
             [0, 1, 0],
             [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


def mat2euler(M, cy_thresh=None):
    ''' Discover Euler angle vector from 3x3 matrix

    Uses the conventions above.

    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
       threshold below which to give up on straightforward arctan for
       estimating x rotation.  If None (default), estimate from
       precision of input.

    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively

    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::

      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]

    with the obvious derivations for z, y, and x

       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)

    Problems arise when cos(y) is close to zero, because both of::

       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))

    will be close to atan2(0, 0), and highly unstable.

    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:

    See: http://www.graphicsgems.org/

    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33 * r33 + r23 * r23)
    if cy > cy_thresh:  # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy)  # atan2(sin(y), cy)
        x = math.atan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else:  # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy)  # atan2(sin(y), cy)
        x = 0.0
    return z, y, x


def euler2quat(z=0, y=0, x=0):
    ''' Return quaternion corresponding to these Euler angles

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format

    Notes
    -----
    We can derive this formula in Sympy using:

    1. Formula giving quaternion corresponding to rotation of theta radians
       about arbitrary axis:
       http://mathworld.wolfram.com/EulerParameters.html
    2. Generated formulae from 1.) for quaternions corresponding to
       theta radians rotations about ``x, y, z`` axes
    3. Apply quaternion multiplication formula -
       http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
       formulae from 2.) to give formula for combined rotations.
    '''
    z = z / 2.0
    y = y / 2.0
    x = x / 2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return np.array([
        cx * cy * cz - sx * sy * sz,
        cx * sy * sz + cy * cz * sx,
        cx * cz * sy - sx * cy * sz,
        cx * cy * sz + sx * cz * sy])


def quat2euler(q):
    ''' Return Euler angles corresponding to quaternion `q`

    Parameters
    ----------
    q : 4 element sequence
       w, x, y, z of quaternion

    Returns
    -------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Notes
    -----
    It's possible to reduce the amount of calculation a little, by
    combining parts of the ``quat2mat`` and ``mat2euler`` functions, but
    the reduction in computation is small, and the code repetition is
    large.
    '''
    # delayed import to avoid cyclic dependencies
    import nibabel.quaternions as nq
    return mat2euler(nq.quat2mat(q))


def euler2angle_axis(z=0, y=0, x=0):
    ''' Return angle, axis corresponding to these Euler angles

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    theta : scalar
       angle of rotation
    vector : array shape (3,)
       axis around which rotation occurs

    Examples
    --------
    >>> theta, vec = euler2angle_axis(0, 1.5, 0)
    >>> print(theta)
    1.5
    >>> np.allclose(vec, [0, 1, 0])
    True
    '''
    # delayed import to avoid cyclic dependencies
    import nibabel.quaternions as nq
    return nq.quat2angle_axis(euler2quat(z, y, x))


def angle_axis2euler(theta, vector, is_normalized=False):
    ''' Convert angle, axis pair to Euler angles

    Parameters
    ----------
    theta : scalar
       angle of rotation
    vector : 3 element sequence
       vector specifying axis for rotation.
    is_normalized : bool, optional
       True if vector is already normalized (has norm of 1).  Default
       False

    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively

    Examples
    --------
    >>> z, y, x = angle_axis2euler(0, [1, 0, 0])
    >>> np.allclose((z, y, x), 0)
    True

    Notes
    -----
    It's possible to reduce the amount of calculation a little, by
    combining parts of the ``angle_axis2mat`` and ``mat2euler``
    functions, but the reduction in computation is small, and the code
    repetition is large.
    '''
    # delayed import to avoid cyclic dependencies
    import nibabel.quaternions as nq
    M = nq.angle_axis2mat(theta, vector, is_normalized)
    return mat2euler(M)


def rotation_to_vtk(R):
    '''
    Concert a rotation matrix into the Mayavi/Vtk rotation paramaters (pitch, roll, yaw)
    '''
    def euler_from_matrix(matrix):
        """Return Euler angles (syxz) from rotation matrix for specified axis sequence.
        :Author:
          `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_

        full library with coplete set of euler triplets (combinations of  s/r x-y-z) at
            http://www.lfd.uci.edu/~gohlke/code/transformations.py.html

        Note that many Euler angle triplets can describe one matrix.
        """
        # epsilon for testing whether a number is close to zero
        _EPS = np.finfo(float).eps * 5.0

        # axis sequences for Euler angles
        _NEXT_AXIS = [1, 2, 0, 1]
        firstaxis, parity, repetition, frame = (1, 1, 0, 0)  # ''

        i = firstaxis
        j = _NEXT_AXIS[i + parity]
        k = _NEXT_AXIS[i - parity + 1]

        M = np.array(matrix, dtype='float', copy=False)[:3, :3]
        if repetition:
            sy = np.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
            if sy > _EPS:
                ax = np.arctan2(M[i, j],  M[i, k])
                ay = np.arctan2(sy,       M[i, i])
                az = np.arctan2(M[j, i], -M[k, i])
            else:
                ax = np.arctan2(-M[j, k],  M[j, j])
                ay = np.arctan2(sy,       M[i, i])
                az = 0.0
        else:
            cy = np.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
            if cy > _EPS:
                ax = np.arctan2(M[k, j],  M[k, k])
                ay = np.arctan2(-M[k, i],  cy)
                az = np.arctan2(M[j, i],  M[i, i])
            else:
                ax = np.arctan2(-M[j, k],  M[j, j])
                ay = np.arctan2(-M[k, i],  cy)
                az = 0.0

        if parity:
            ax, ay, az = -ax, -ay, -az
        if frame:
            ax, az = az, ax
        return ax, ay, az
    r_yxz = pl.array(euler_from_matrix(R)) * 180 / np.pi
    r_xyz = r_yxz[[1, 0, 2]]
    return r_xyz

if __name__ == '__main__':

    from .einsum_utils import EPS

    def get_orthonormal_basis(m):

        v0 = M[..., 1, :] - M[..., 0, :]
        v1 = M[..., 2, :] - M[..., 0, :]
        v2 = np.einsum('...ijk,...j,...k->...i', EPS, v0, v1)
        v1 = np.einsum('...ijk,...j,...k->...i', EPS, v2, v0)
        print(v0)
        print(v1)
        basis = np.c_[v0, v1, v2].T
        return basis / np.linalg.norm(basis, axis=1)[:, None]

    M = np.array([[[0, 0, 0],
                   [0, 1, 0],
                   [-1, 0, 0]],
                  [[0, 0, 0],
                   [-1, 0, 0],
                   [0, 0, -1]]], dtype='float')

#    basis = get_orthonormal_basis(M)

    basis = np.array([[0.,   -0.70710678, 0.70710678],
                      [0.94280904, -0.23570226, -0.23570226],
                      [0.33333333,  0.66666667,  0.66666667]], dtype='float_')

    print('basis\n', basis)
    ae = mat2euler(basis)
    print('euler\n', ae)
    print('basis\n', euler2mat(*ae))

    # print 'newone', rotation_to_vtk(basis[0])
