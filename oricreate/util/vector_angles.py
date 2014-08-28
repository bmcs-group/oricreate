'''
Created on Aug 28, 2014

@author: rch
'''

import numpy as np

def get_gamma(a, b):
    '''Get the cosine of an angle between two vectors :math:`a` and :math:`b`
    Given two compatible arrays (n-dimensional) :math:`a`
    and :math:`b` of vectors with last index :math:`I`
    labeling the vector component the array :math:`\\gamma`
    of angle cosines between each pair of vectors in :math:`a`, :math:`b`.
    gets calculated as

    .. math::
        \gamma = \\cos{(\\theta)}
                = \\frac{a \cdot b}{ \left\| a \\right\| \left\| b \\right\| }
                = \\frac{a_d b_d}{ \\sqrt{a_d a_d} \\sqrt{b_d b_d} }

    corresponding to the code::

        ab = np.einsum('...d,...d->...', a, b)
        sqrt_aa = np.sqrt(np.einsum('...d,...d->...', a, a))
        sqrt_bb = np.sqrt(np.einsum('...d,...d->...', b, b))
        sqrt_aa_x_sqrt_bb = sqrt_aa * sqrt_bb
        gamma = ab / sqrt_aa_x_sqrt_bb)

    '''
    ab = np.einsum('...d,...d->...', a, b)
    sqrt_aa = np.sqrt(np.einsum('...d,...d->...', a, a))
    sqrt_bb = np.sqrt(np.einsum('...d,...d->...', b, b))
    c_ab = ab / (sqrt_aa * sqrt_bb)
    return c_ab

def get_gamma_du(a, a_du, b, b_du):
    '''Given two arrays (n-dimensional) of vectors :math:`a, b` and their
    derivatives a_du and b_du with respect to the nodal displacments du
    return the derivatives of the cosine between mutual angles theta between
    a and b with respect to the node displacement du.

    .. math::
        \\frac{ \partial \gamma }{ \partial u_{Ie}} =
        \\frac{1}{ \left(\left\| a \\right\| \left\| b \\right\| \\right)^2 }
        \\left[
        \\frac{\partial \left(a_d  b_d \\right)}{\partial u_{Ie}}
        \left(\left\| a \\right\| \left\| b \\right\| \\right)
        -
        \\frac{\partial \left( \left\| a \\right\| \left\| b \\right\| \\right) }{ \partial u_{Ie} }
        \;
        \left( a_d b_d \\right)
        \\right]

    By factoring out the term :math:`\left(\left\| a \\right\| \left\| b \\right\| \\right)`
    the expression reduces to a form

    .. math::
        \\frac{ \partial \gamma }{ \partial u_{Ie}} =
        \\frac{1}{ \left\| a \\right\| \left\| b \\right\|  }
        \\left[
        \\frac{\partial \left(a_d  b_d \\right)}{\partial u_{Ie}}
        -
        \\frac{\partial \left( \left\| a \\right\| \left\| b \\right\| \\right) }{ \partial u_{Ie} }
        \;
        \gamma
        \\right]

    The terms with derivatives can be further elaborated to

    .. math::
        \\frac{\partial \left(a_d  b_d \\right)}{\partial u_{Ie}}
        =
        \\frac{\partial a_d }{\partial u_{Ie}} \; b_d
        +
        \\frac{\partial b_d }{\partial u_{Ie}} \; a_d
        \;\;\mathrm{and}\;\;
        \\frac{\partial \left( \left\| a \\right\| \left\| b \\right\| \\right) }{ \partial u_{Ie} }
        =
        \\frac{\partial \left\| a \\right\|  }{ \partial u_{Ie} } \; \left\| b \\right\|
        +
        \\frac{\partial \left\| b \\right\|  }{ \partial u_{Ie} } \; \left\| a \\right\|

    The derivative of a vector norm in a summation form can be rewritten as

    .. math::
        \\frac{\partial \left\| a \\right\|  }{ \partial u_{Ie} }
        =
        \\frac{\partial \\sqrt{ a_d a_d } }{ \partial u_{Ie} }
        =
        \\frac{1}{ \\sqrt{ a_d a_d } }
        \left(
        \\frac{\partial a_{d}}{\partial u_{Ie}}
        a_{d}
        \\right)

    Using these expressions and using the terms given in the function ``get_gamma``
    the code for a vectorized evaluation for the supplied multidimensional arrays of
    ``a, a_du, b, b_du`` looks as follows::

        ab_du = np.einsum('...dIe,...d->...Ie', a_du, b) + np.einsum('...dIe,...d->...Ie', b_du, a)
        sqrt_aa_du = 1 / sqrt_aa * np.einsum('...dIe,...d->...Ie', a_du, a)
        sqrt_bb_du = 1 / sqrt_bb * np.einsum('...dIe,...d->...Ie', b_du, b)
        sqrt_aa_x_sqrt_bb_du = (np.einsum(...Ie,...->...Ie', sqrt_aa_du, sqrt_bb) +
                                np.einsum(...Ie,...->...Ie', sqrt_bb_du, sqrt_aa)
        gamma_du = 1 / sqrt_aa_x_sqrt_bb * (ab_du - sqrt_aa_x_sqrt_bb_du * gamma)

    '''
    ab = np.einsum('...d,...d->...', a, b)
    sqrt_aa = np.sqrt(np.einsum('...d,...d->...', a, a))
    sqrt_bb = np.sqrt(np.einsum('...d,...d->...', b, b))
    sqrt_aa_x_sqrt_bb = sqrt_aa * sqrt_bb
    gamma = ab / sqrt_aa_x_sqrt_bb

    ab_du = (np.einsum('...dIe,...d->...Ie', a_du, b) +
             np.einsum('...dIe,...d->...Ie', b_du, a))

    sqrt_aa_du_x_sqrt_bb = np.einsum('...,...dIe,...d->...Ie', sqrt_bb / sqrt_aa, a_du, a)
    sqrt_bb_du_x_sqrt_aa = np.einsum('...,...dIe,...d->...Ie', sqrt_aa / sqrt_bb, b_du, b)
    sqrt_aa_x_sqrt_bb_du = sqrt_aa_du_x_sqrt_bb + sqrt_bb_du_x_sqrt_aa
    sqrt_aa_x_sqrt_bb_du_gamma = np.einsum('...Ie,...->...Ie', sqrt_aa_x_sqrt_bb_du, gamma)

    gamma_du = np.einsum('...,...Ie->...Ie',
                         1. / sqrt_aa_x_sqrt_bb, (ab_du - sqrt_aa_x_sqrt_bb_du_gamma))

    return gamma_du

def get_theta_du2(a, a_du, b, b_du):
    '''Given two arrays (n-dimensional) of vectors a, b and their
    derivatives a_du and b_du with respect to the nodal displacments du
    return the derivatives of the mutual angles theta between
    a and b with respect to the node displacement du.
    '''
    ab = np.einsum('...i,...i->...', a, b)
    sqrt_aa = np.sqrt(np.einsum('...i,...i->...', a, a))
    sqrt_bb = np.sqrt(np.einsum('...i,...i->...', b, b))
    sqrt_aa_x_sqrt_bb = sqrt_aa * sqrt_bb
    gamma = ab / sqrt_aa_x_sqrt_bb

    ab_du = (np.einsum('...iKj,...i->...Kj', a_du, b) +
             np.einsum('...iKj,...i->...Kj', b_du, a))

    gamma_bb__aa = gamma * sqrt_bb / sqrt_aa
    gamma_aa__bb = gamma * sqrt_aa / sqrt_bb
    gamma_aa_bb_du = (np.einsum('...,...iKj,...i->...Kj',
                               gamma_bb__aa, a_du, a) +
                     np.einsum('...,...iKj,...i->...Kj',
                               gamma_aa__bb, b_du, b))

    print 'xxx2'
    print gamma_aa_bb_du

    bb__aa = sqrt_bb / sqrt_aa
    aa__bb = sqrt_aa / sqrt_bb
    aa_bb_du = (np.einsum('...,...iKj,...i->...Kj',
                          bb__aa, a_du, a) +
                np.einsum('...,...iKj,...i->...Kj',
                          aa__bb, b_du, b))
    ygamma_aa_bb_du = np.einsum('...,...Kj->...Kj', gamma, aa_bb_du)
    print 'yyy2'
    print ygamma_aa_bb_du

    gamma_du = np.einsum('...,...Kj->...Kj', 1. / sqrt_aa_x_sqrt_bb, (ab_du - gamma_aa_bb_du))

    print 'gamma_du2'
    print gamma_du

    sqarg = 1 - gamma ** 2
    theta_du = np.einsum('...,...Kj->...Kj', -1. / np.sqrt(sqarg), gamma_du)
    return theta_du

def get_theta(a, b):
    '''
    Get the angle between two vectors :math:`a` and :math:`b`.
    Using the function ``get_c_ab(a,b)`` delivering the cosine of the angle
    this method just evaluates the expression

    .. math::

        \\theta = \\arccos{( c^{(ab)} )}

    realized by calling the vectorized numpy function::

        theta = np.arccos(c_ab)

    '''
    c_ab = get_gamma(a, b)
    theta = np.arccos(c_ab)
    return theta

def get_theta_du3(a, a_du, b, b_du):
    '''
    Get the derivative of the angle between two vectors :math:`a` and :math:`b`
    with respect to ``du`` using the supplied chain derivatives ``a_du, b_du``.
    Using the function ``get_c_ab(a,b)`` delivering the cosine of the angle
    this method evaluates the expression

    .. math::
        \\frac{\\partial \\theta}{\\partial u_{Ie}}
        =
        \\frac{\\partial \\theta}{\\partial c^{(ab)}}
        \cdot
        \\frac{\\partial c^{(ab)}}{\\partial u_{Ie}}

    where

    .. math::

        \\frac{\partial \\theta}{\partial c^{ab}}
        =
        - \\frac{1}{ \sqrt{ 1 - (c^{(ab)})^2}}

    ::

        theta_du = - 1 / np.sqrt(1 - c_ab * c_ab) * c_ab_du
    '''
    gamma = get_gamma(a, b)
    gamma_du = get_gamma_du(a, a_du, b, b_du)
    theta_du = np.einsum('...,...Ie->...Ie', -1 / np.sqrt(1 - gamma ** 2), gamma_du)
    return theta_du
