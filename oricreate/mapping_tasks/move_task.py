'''
Created on Jun 20, 2013

@author: rch
'''
import math

from traits.api import \
    provides, Property, Str, Int, Float, Array, cached_property

import numpy as np
from oricreate.simulation_tasks import \
    ISimulationTask
from oricreate.simulation_tasks.simulation_task import \
    SimulationTaskBase
from .rotate_copy import qv_mult, axis_angle_to_q


@provides(ISimulationTask)
class MoveTask(SimulationTaskBase):

    ''' Move the source to along a specified path and rotation function.
    '''
    name = Str('move')

    #u_target = Array(np.float_, values=[0, 0, 0])

#     translation = Array(dtype=float, value=[])
#     rotation_axis = Array(dtype=float, value=[])
#     rotation_center = Array(dtype=float, value=[])
#     rotation_angle = Array(dtype=float, value=[])

    u_target = Array(value=[2, 0, 0], dtype=np.float_)
    rotation_axis = Array(value=[0, 0, 1], dtype=np.float_)
    rotation_center = Array(value=[0, 0, 0], dtype=np.float_)
    rotation_angle = Float(np.pi / 2.0)

    t_arr = Property(Array(float), depends_on='unfold, n_steps, time_array')
    '''Generated time array.
    '''
    @cached_property
    def _get_t_arr(self):
        if len(self.time_arr) > 0:
            return self.time_arr
        t_arr = np.linspace(0, 1., self.n_steps + 1)
        return t_arr

    xu_t = Property(depends_on='source_config_changed')
    '''Displacement history for the current FoldRigidly process.
    '''
    @cached_property
    def _xget_u_t(self):
        '''Solve the problem with the appropriate solver
        '''
        x = self.previous_task.x_0
        u = self.previous_task.u_1
        x_1 = self.previous_task.x_1
        x_t = (x_1[np.newaxis, :, :] +
               self.u_target[np.newaxis, np.newaxis, :] *
               self.t_arr[:, np.newaxis, np.newaxis]
               )
        return x_t - x[np.newaxis, :, :]

    u_t = Property(depends_on='source_config_changed')
    '''Displacement history for the current FoldRigidly process.
    '''
    @cached_property
    def _get_u_t(self):
        '''Solve the problem with the appropriate solver
        '''
        x_0_Na = self.previous_task.x_0
        x_1_Na = self.previous_task.x_1
        t_arr = self.t_arr

        translation = self.u_target
        rotation_axis = self.rotation_axis[np.newaxis, ...]
        rotation_center = self.rotation_center
        rotation_angle = self.rotation_angle

        rotation_angles_t = rotation_angle * t_arr
        translations_tNa = translation[np.newaxis, np.newaxis, :] * \
            t_arr[:, np.newaxis, np.newaxis]
        q_t = axis_angle_to_q(rotation_axis, rotation_angles_t)
        x_pulled_back_Na = x_1_Na - rotation_center[np.newaxis, :]
        x_rotated_tNa = qv_mult(q_t, x_pulled_back_Na[np.newaxis, ...])
        x_pushed_forward_tNa = x_rotated_tNa + \
            rotation_center[np.newaxis, np.newaxis, :]
        x_translated_tNa = x_pushed_forward_tNa + translations_tNa
        u_translated_tNa = x_translated_tNa - x_0_Na[np.newaxis, ...]
        return u_translated_tNa

        x_0_Na = self.previous_task.x_0
        x_1_Na = self.previous_task.x_1
        print('x_1_Na', x_1_Na)
        rotation_angles_t = self.rotation_angle * \
            self.t_arr[:, np.newaxis, np.newaxis]
        print('rotation_angles_t', rotation_angles_t)
        translations_tNa = self.translation * \
            self.t_arr[:, np.newaxis, np.newaxis]
        print('translations_tNa', translations_tNa)
        q_t = axis_angle_to_q(self.rotation_axis, rotation_angles_t)
        x_pulled_back_Na = x_1_Na - self.rotation_center[np.newaxis, :]
        print('x_pulled_back_Na', x_pulled_back_Na)
        x_rotated_tNa = qv_mult(q_t, x_pulled_back_Na)
        print('x_rotated_tNa', x_rotated_tNa)
        x_pushed_forward_tNa = x_rotated_tNa + \
            self.rotation_center[np.newaxis, np.newaxis, :]
        print('x_pushed_forward_tNa', x_pushed_forward_tNa)
        x_translated_tNa = x_pushed_forward_tNa + translations_tNa
        print('x_translated_tNa', x_translated_tNa)
        return x_translated_tNa - x_0_Na[np.newaxis, :, :]


if __name__ == '__main__':

    translation = np.array([2, 0, 0], dtype=np.float_)
    rotation_axis = np.array([[0, 0, 1]], dtype=np.float_)
    rotation_center = np.array([0, 0, 0], dtype=np.float_)
    rotation_angle = np.pi / 2.0

    x_0_Na = np.array([[1, 0, 0]], dtype=np.float_)
    x_1_Na = np.array([[1, 0, 0]], dtype=np.float_)
    t_arr = np.array([0.5, 1.0], dtype=np.float_)

    print('x_1_Na', x_1_Na)
    rotation_angles_t = rotation_angle * \
        t_arr
    print('rotation_angles_t', rotation_angles_t)
    translations_tNa = translation[np.newaxis, np.newaxis, :] * \
        t_arr[:, np.newaxis, np.newaxis]
    print('translations_tNa', translations_tNa)
    q_t = axis_angle_to_q(rotation_axis, rotation_angles_t)
    print('q_t', q_t)
    x_pulled_back_Na = x_1_Na - rotation_center[np.newaxis, :]
    print('x_pulled_back_Na', x_pulled_back_Na)
    x_rotated_tNa = qv_mult(q_t, x_pulled_back_Na[np.newaxis, ...])
    print('x_rotated_tNa', x_rotated_tNa)
    x_pushed_forward_tNa = x_rotated_tNa + \
        rotation_center[np.newaxis, np.newaxis, :]
    print('x_pushed_forward_tNa', x_pushed_forward_tNa)
    x_translated_tNa = x_pushed_forward_tNa + translations_tNa
    print('x_translated_tNa', x_translated_tNa)
    u_translated_tNa = x_translated_tNa - x_0_Na[np.newaxis, ...]
    print('u_translated_tNa', u_translated_tNa)
