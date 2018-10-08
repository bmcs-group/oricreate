'''
Created on Feb 13, 2015

@author: rch
'''
from oricreate.api import YoshimuraCPFactory
import numpy as np


def test_crease_pattern_state_transitions():
    '''Test the moment and normal force calculated for a cross section.
    '''

    cp_factory = YoshimuraCPFactory(n_x=1, n_y=2, L_x=4, L_y=2)
    cp = cp_factory.formed_object

    assert np.allclose(cp.x, np.array([[0.,  0.,  0.],
                                       [0.,  2.,  0.],
                                       [4.,  0.,  0.],
                                       [4.,  2.,  0.],
                                       [0.,  1.,  0.],
                                       [4.,  1.,  0.],
                                       [2.,  1.,  0.]]))

    cp.u[2, 2] = 1.0
    cp.u = cp.u
    assert cp.x[2, 2] == 1.0

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    test_crease_pattern_state_transitions()
