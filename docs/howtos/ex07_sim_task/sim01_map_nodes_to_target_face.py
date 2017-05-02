r'''

Map a crease pattern to a target surface
----------------------------------------

This example demonstrates how to define a FormingTask
on a preform FormedObject that has been previously produced
in a CreasePatternFactory. The forming process within 
this FormingTask is controlled by the RFKSimulator
that is configured using the RFKConfig.

The target face is defined as horizontal plane at the height 8
and nodes [0,1,2] are involved in the minimum distance criterion.
'''
from custom_factory_mpl import create_cp_factory
from oricreate.forming_tasks import \
    FormingTask
from oricreate.simulation_step import \
    SimulationStep, SimulationConfig


def create_sim_step():
    cp_factory = create_cp_factory()
    # begin
    from oricreate.fu import FuTargetFaces, FuTF
    from oricreate.api import r_, s_, t_
    # Link the pattern factory with the goal function client.
    do_something = FormingTask(previous_task=cp_factory)
    # configure the forming task so that it uses
    # the rigid folding kinematics optimization framework RFKOF
    target_face = FuTF([r_, s_, r_ + 2.0], [0, 1, 2])
    fu_target_faces = FuTargetFaces(tf_lst=[target_face])
    sim_config = SimulationConfig(fu=fu_target_faces, acc=1e-8)
    sim_step = SimulationStep(forming_task=do_something,
                              config=sim_config)
    sim_step.t = 0.4
    print 'goal function for t = 0.4:', sim_step.get_f()
    sim_step.t = 0.8
    print 'goal function for t = 0.8:', sim_step.get_f()
    print 'goal function derivatives'
    print sim_step.get_f_du()
    print 'constraints'
    print sim_step.get_G()
    print 'constraint derivatives'
    print sim_step.get_G_du()
    sim_step.cp_state.u = sim_step.u_t
    print 'target displacement:\n', sim_step.cp_state.u
    print 'target position:\n', sim_step.cp_state.x
    # end
    return sim_step

if __name__ == '__main__':
    create_sim_step()
