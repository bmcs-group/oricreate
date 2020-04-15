'''
Created on Jan 11, 2017

@author: rch
'''

from oricreate.api import \
    CustomCPFactory,  CreasePatternState,  fix, link,\
    FuNodeDist, \
    GuConstantLength, GuDofConstraints, GuPsiConstraints,\
    SimulationConfig, SimulationTask, \
    FTV, FTA
from oricreate.forming_tasks.forming_task import FormingTask
from oricreate.view import FormingTaskTree
from traits.api import \
    HasTraits, Float, Property, cached_property, Instance, \
    Int

import numpy as np


class HPCPFormingProcess(HasTraits):
    '''
    Define the simulation task prescribing the boundary conditions, 
    target surfaces and configuration of the algorithm itself.
    '''

    L_green = Float(1.0, auto_set=False, enter_set=True, input=True)
    n_steps = Int(10, auto_set=False, enter_set=True, input=True)

    factory_task = Property(Instance(FormingTask))
    '''Factory task generating the crease pattern.
    '''
    @cached_property
    def _get_factory_task(self):

        l_length = self.L_green
        X = np.array([[0, 0],
                      [l_length / 2.0, 0],
                      [l_length, 0],
                      [0, l_length / 2.0],
                      [0, l_length],
                      [-l_length / 2.0, 0],
                      [-l_length, 0],
                      [0, -l_length / 2.0],
                      [0, -l_length],
                      [l_length / 2, l_length / 2],
                      [-l_length / 2, l_length / 2],
                      [-l_length / 2, -l_length / 2],
                      [l_length / 2, -l_length / 2],
                      ], dtype=np.float_)
        X = np.c_[X[:, 0], X[:, 1], X[:, 0] * 0]
        L = [[0, 1],
             [1, 2],
             [0, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [0, 7],
             [7, 8],
             [1, 3],
             [2, 4],
             [3, 5],
             [4, 6],
             [5, 7],
             [6, 8],
             [7, 1],
             [8, 2],
             [1, 9],
             [9, 3],
             [3, 10],
             [10, 5],
             [5, 11],
             [11, 7],
             [7, 12],
             [12, 1]
             ]
        L = np.array(L, dtype=np.int_)

        F = [[0, 1, 3],
             [0, 3, 5],
             [0, 5, 7],
             [0, 7, 1],
             [1, 9, 3],
             [3, 10, 5],
             [5, 11, 7],
             [7, 12, 1],
             [2, 9, 1],
             ]

        F = np.array(F, dtype=np.int_)
        cp = CreasePatternState(X=X,
                                L=L,
                                F=F)
        return CustomCPFactory(formed_object=cp)

if __name__ == '__main__':
    bsf_process = HPCPFormingProcess(n_steps=1)

    cp = bsf_process.factory_task.formed_object

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        cp.plot_mpl(ax, facets=True)
        plt.tight_layout()
        plt.show()

    alpha_min = np.pi / 4
    alpha_max = 3 * np.pi / 4
    N_alpha = np.array([
        [alpha_min, alpha_max, alpha_max, alpha_min],
        [alpha_min, alpha_max, alpha_max, alpha_min],
        [alpha_max, alpha_min, alpha_min, alpha_max],
        [alpha_max, alpha_min, alpha_min, alpha_max],
    ], dtype=np.float_)

    mu_1 = (np.sin(0.5 * (N_alpha[:, 0] + N_alpha[:, 1])) /
            np.sin(0.5 * (N_alpha[:, 0] - N_alpha[:, 1])))
    mu_2 = (np.cos(0.5 * (N_alpha[:, 1] + N_alpha[:, 2])) /
            np.cos(0.5 * (N_alpha[:, 1] - N_alpha[:, 2])))
    mu_3 = - mu_1
    mu_4 = 1.0 / mu_1
    print('mu_1', mu_1)
    print('mu_2', mu_2)
    print('mu_3', mu_3)
    print('mu_4', mu_4)
    polygon_mu = np.array([mu_1[2], mu_2[3], mu_3[2], mu_4[3]])
    print(polygon_mu)
    print('product', np.product(polygon_mu))

    N_alpha = np.array([
        [alpha_min, alpha_max, alpha_max, alpha_min],
        [alpha_max, alpha_max, alpha_min, alpha_min],
        [alpha_min, alpha_min, alpha_max, alpha_max],
        [alpha_max, alpha_min, alpha_min, alpha_max],
    ], dtype=np.float_)

    mu_1 = (np.sin(0.5 * (N_alpha[:, 0] + N_alpha[:, 1])) /
            np.sin(0.5 * (N_alpha[:, 0] - N_alpha[:, 1])))
    mu_2 = (np.cos(0.5 * (N_alpha[:, 1] + N_alpha[:, 2])) /
            np.cos(0.5 * (N_alpha[:, 1] - N_alpha[:, 2])))
    mu_3 = - mu_1
    mu_4 = 1.0 / mu_1
    print('mu_1', mu_1)
    print('mu_2', mu_2)
    print('mu_3', mu_3)
    print('mu_4', mu_4)
    polygon_mu = np.array([mu_1[2], mu_2[1], mu_3[2], mu_4[3]])
    print(polygon_mu)
    print('product', np.product(polygon_mu))

    # version - 3 - HP -

    print('HP')

    N_alpha = np.array([
        [alpha_min, alpha_max, alpha_max, alpha_min],
        [alpha_min, alpha_max, alpha_max, alpha_min],
        [alpha_min, alpha_max, alpha_max, alpha_min],
        [alpha_min, alpha_max, alpha_max, alpha_min],
    ], dtype=np.float_)

    mu_1 = (np.sin(0.5 * (N_alpha[:, 0] + N_alpha[:, 1])) /
            np.sin(0.5 * (N_alpha[:, 0] - N_alpha[:, 1])))
    mu_2 = (np.cos(0.5 * (N_alpha[:, 1] + N_alpha[:, 2])) /
            np.cos(0.5 * (N_alpha[:, 1] - N_alpha[:, 2])))
    mu_3 = - mu_1
    mu_4 = 1.0 / mu_1
    print('mu_1', mu_1)
    print('mu_2', mu_2)
    print('mu_3', mu_3)
    print('mu_4', mu_4)
    polygon_mu = np.array([mu_2[0], mu_3[1], mu_4[2], mu_1[3]])
    print(polygon_mu)
    print('product', np.product(polygon_mu))

    # version - 4 - Miura-Ori

    print('Miura-Ori')
    N_alpha = np.array([
        [alpha_min, alpha_max, alpha_max, alpha_min],
        [alpha_min, alpha_max, alpha_max, alpha_min],
        [alpha_min, alpha_max, alpha_max, alpha_min],
        [alpha_min, alpha_max, alpha_max, alpha_min],
    ], dtype=np.float_)

    mu_1 = (np.sin(0.5 * (N_alpha[:, 0] + N_alpha[:, 1])) /
            np.sin(0.5 * (N_alpha[:, 0] - N_alpha[:, 1])))
    mu_2 = (np.cos(0.5 * (N_alpha[:, 1] + N_alpha[:, 2])) /
            np.cos(0.5 * (N_alpha[:, 1] - N_alpha[:, 2])))
    mu_3 = - mu_1
    mu_4 = 1.0 / mu_1
    print('mu_1', mu_1)
    print('mu_2', mu_2)
    print('mu_3', mu_3)
    print('mu_4', mu_4)
    polygon_mu = np.array([mu_3[0], mu_4[1], mu_3[2], mu_4[3]])
    print(polygon_mu)
    print('product', np.product(polygon_mu))

    # version - 5 - Twist

    print('Twist')

    N_alpha = np.array([
        [70, 90, 110, 90],
        [125, 145, 55, 35],
        [55, 145, 125, 35],
    ], dtype=np.float_)

    N_alpha *= 2 * np.pi / 360

    mu_1 = (np.sin(0.5 * (N_alpha[:, 0] + N_alpha[:, 1])) /
            np.sin(0.5 * (N_alpha[:, 0] - N_alpha[:, 1])))
    mu_2 = (np.cos(0.5 * (N_alpha[:, 1] + N_alpha[:, 2])) /
            np.cos(0.5 * (N_alpha[:, 1] - N_alpha[:, 2])))
    mu_3 = - mu_1
    mu_4 = 1.0 / mu_1
    print('mu_1', mu_1)
    print('mu_2', mu_2)
    print('mu_3', mu_3)
    print('mu_4', mu_4)
    polygon_mu = np.array([mu_3[0], mu_4[1], mu_4[2]])
    print(polygon_mu)
    print('product', np.product(polygon_mu))
