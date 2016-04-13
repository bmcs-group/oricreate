'''
Created on Jan 20, 2016

@author: rch
'''

import math
from traits.api import \
    Float, HasTraits, Property, cached_property, Int, \
    Instance

import numpy as np
from oricreate.api import \
    fix, link, \
    GuConstantLength, GuDofConstraints, SimulationConfig, \
    SimulationTask, \
    FTV, FTA
from oricreate.api import r_, s_, t_
from oricreate.crease_pattern import \
    CreasePatternNormalsViz3D, CreasePatternBasesViz3D
from oricreate.fu import FuTargetFaces, FuTF


def create_cp_factory(R, H):
    # begin
    from oricreate.api import CreasePatternState, CustomCPFactory

    h2 = H / 2.0
    c60 = math.cos(math.pi / 3.0) * R
    s60 = math.sin(math.pi / 3.0) * R

    x = np.array([[R, 0, h2],
                  [c60, s60, 0],
                  [-c60, s60, h2],
                  [-R, 0, 0],
                  [-c60, -s60, h2],
                  [c60, -s60, 0],
                  [R, 0, -h2],
                  [-c60, s60, -h2],
                  [-c60, -s60, -h2],
                  [0.0 * math.cos(math.pi / 3), -0.0 *
                   math.sin(math.pi / 3), 0],
                  [0.0 * math.cos(math.pi / 3), 0.0 *
                   math.sin(math.pi / 3), 0],
                  [-0.0, 0, 0],
                  ], dtype='float_')

    L = np.array([[0, 6], [2, 7], [4, 8],
                  [0, 5], [0, 9], [0, 10], [0, 1],
                  [6, 5], [6, 9], [6, 10], [6, 1],
                  [2, 1], [2, 10], [2, 11], [2, 3],
                  [7, 1], [7, 10], [7, 11], [7, 3],
                  [4, 3], [4, 11], [4, 9], [4, 5],
                  [8, 3], [8, 11], [8, 9], [8, 5],
                  [5, 9], [1, 10], [3, 11]],
                 dtype='int_')

    F = np.array([[0, 6, 1],
                  [0, 1, 10],
                  [0, 10, 6],
                  [6, 1, 10],
                  [2, 10, 1],
                  [2, 1, 7],
                  [2, 7, 10],
                  [7, 10, 1],
                  [2, 3, 11],
                  [2, 11, 7],
                  [2, 7, 3],
                  [7, 3, 11],
                  [4, 11, 3],
                  [4, 3, 8],
                  [4, 8, 11],
                  [8, 11, 3],
                  [4, 5, 9],
                  [4, 9, 8],
                  [4, 8, 5],
                  [8, 5, 9],
                  [0, 9, 5],
                  [0, 5, 6],
                  [0, 6, 9],
                  [6, 9, 5]
                  ], dtype='int_')

    cp = CreasePatternState(X=x, L=L, F=F)

#     cp.u[(9, 10, 11), 2] = -0.01
#     cp.u[(5, 1, 3), 2] = 0.01
#     cp.u[0, 0] = -0.01
#     cp.u[3, 0] = -0.005
#     cp.u[3, 1] = -0.005
#     cp.u[5, 0] = 0.005
#     cp.u[5, 1] = 0.005

    cp_factory = CustomCPFactory(formed_object=cp)
    # end
    return cp_factory


class HexaCycle(SimulationTask):
    '''
    Define the simulation task prescribing the boundary conditions, 
    target surfaces and configuration of the algorithm itself.
    '''

    R = Float(1.0, auto_set=False, enter_set=True, input=True)
    H = Float(1.0, auto_set=False, enter_set=True, input=True)
    n_steps = Int(1, auto_set=False, enter_set=True, input=True)

    u_max = Float(0.01, auto_set=False, enter_set=True, input=True)

    def _previous_task_default(self):
        return create_cp_factory(H=self.H, R=self.R)

    config = Property(Instance(SimulationConfig))
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_config(self):
        #fixed_xyz = fix((9, 10, 11), (2), lambda t: -self.u_max * t)

        target_face_down = FuTF([r_, s_, - self.u_max * r_**2 * s_**2],
                                [9, 10, 11])
        target_face_zero = FuTF([r_, s_, r_**2 * s_**2],
                                [1, 3, 5])
        fu_target_faces = FuTargetFaces(
            tf_lst=[target_face_down,
                    target_face_zero])

        #dof_constraints = fixed_xyz
#        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
        gu_constant_length = GuConstantLength()
        return SimulationConfig(fu=fu_target_faces,
                                gu={'cl': gu_constant_length,
                                    #                                    'dofs': gu_dof_constraints
                                    },
                                acc=1e-2, MAX_ITER=500)


class HexaCycleFTV(FTV):

    model = Instance(HexaCycle)


if __name__ == '__main__':
    hc = HexaCycle(H=0.1, n_steps=1, u_max=0.1)

    print 'load_vector', hc.sim_step.gu_lst[0].get_G(0.1)
    print 'load_vector', hc.sim_step.gu_lst[0].get_G_du(0.1)

    hc.sim_step.debug_level = 0
    ftv = HexaCycleFTV(model=hc)
    hc.u_1

    ftv.add(hc.sim_history.viz3d_dict['node_numbers'], order=5)
    ftv.add(hc.sim_history.viz3d)
#    ftv.add(hc.config.gu['dofs'].viz3d)

    print hc.formed_object.x
    ftv.plot()
    ftv.update(vot=1, force=True)
    ftv.show()

#     fta = FTA(ftv=ftv)
#     fta.init_view(a=45, e=35, d=5, f=(0, 0, 0), r=0)
#     fta.add_cam_move(a=55, e=50, n=5, d=6, r=10,
#                      duration=10,
#                      vot_fn=lambda cmt: np.linspace(0, 1, 4),
#                      azimuth_move='damped',
#                      elevation_move='damped',
#                      distance_move='damped')
#
#     fta.plot()
#     fta.render()
#     fta.configure_traits()
