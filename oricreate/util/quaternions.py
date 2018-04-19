'''
Created on Dec 5, 2015

Quaternions conversions
=======================

'''

import numpy as np


def q_normalize(q, axis=1):
    sq = np.sqrt(np.sum(q * q, axis=axis))
    sq[np.where(sq == 0)] = 1.e-19
    return q / sq[:, np.newaxis]


def v_normalize(q, axis=1):
    sq = np.sqrt(np.sum(q * q, axis=axis))
    sq[np.where(sq == 0)] = 1.e-19
    return q / sq[:, np.newaxis]


def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z], dtype='f')


def q_conjugate(q):
    qn = q_normalize(q.T).T
    w, x, y, z = qn
    return np.array([w, -x, -y, -z], dtype='f')


def qv_mult(q1, u):
    print('shapes')
    print('q1', q1.shape, 'u', u.shape)
    zero_re = np.zeros((u.shape[0], u.shape[1]), dtype='f')
    print('zero_re', zero_re.shape)
    q2 = np.concatenate([zero_re[:, :, np.newaxis], u], axis=2)
    print('q2', q2.shape)
    q2 = np.rollaxis(q2, 2)
    print('q2', q2.shape)
    q12 = q_mult(q1[:, :, np.newaxis], q2[:, :, :])
    print('q12', q12.shape)
    q_con = q_conjugate(q1)
    print('q_con', q_con.shape)
    q = q_mult(q12, q_con[:, :, np.newaxis])
    print('q', q.shape)
    q = np.rollaxis(np.rollaxis(q, 2), 2)
    print('q', q.shape)
    return q[:, :, 1:]


def axis_angle_to_q(v, theta):
    v_ = v_normalize(v, axis=1)
    x, y, z = v_.T
    theta = theta / 2
    w = np.cos(theta)
    x = x * np.sin(theta)
    y = y * np.sin(theta)
    z = z * np.sin(theta)
    return np.array([w, x, y, z], dtype='f')


def q_to_axis_angle(q):
    w, v = q[0, :], q[1:, :]
    theta = np.arccos(w) * 2.0
    return theta, v_normalize(v)
