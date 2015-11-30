r'''
This example demonstrates how to define a FormingTask
on a pre-form FormedObject that has been previously produced
in a CreasePatternFactory. The forming process within 
this FormingTask is controlled by the RFKSimulator
that is configured using the RFKConfig.

The target face is defined as horizontal plane at the height 8
and nodes [0,1,2] are involved in the minimum distance criterion.
'''

from oricreate.forming_tasks import \
    FormingTask
from oricreate.simulation_step import \
    SimulationStep, SimulationConfig


def create_sim_step():
    # begin
    from oricreate.api import CreasePatternState, CustomCPFactory
    from oricreate.fu import FuTargetFaces, TF
    from oricreate.gu import GuConstantLength
    from oricreate.api import r_, s_, t_

    cp = CreasePatternState(X=[[0, 0, 0],
                               [1, 0, 0],
                               [1, 1, 0],
                               [0, 1, 0],
                               [0.5, 0.5, 0]
                               ],
                            L=[[0, 1], [1, 2], [3, 0],
                               [0, 4], [1, 4], [2, 4], [3, 4]],
                            F=[[0, 1, 4], [1, 2, 4], [4, 3, 0]]
                            )

    cp_factory = CustomCPFactory(formed_object=cp)
    # Link the pattern factory with the goal function client.
    do_something = FormingTask(previous_task=cp_factory)
    # configure the forming task so that it uses
    # the rigid folding kinematics optimization framework RFKOF
    target_face = TF(F=[r_, s_, t_])
    fu_target_faces = FuTargetFaces(tf_lst=[(target_face, [0, 1, 4])])
    # Link the crease factory it with the constraint client
    gu_constant_length = GuConstantLength()

    sim_config = SimulationConfig(fu=fu_target_faces,
                                  gu={'cl': gu_constant_length})
    sim_step = SimulationStep(forming_task=do_something,
                              config=sim_config, acc=1e-5)

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
    print sim_step.cp_state.L_lengths
    sim_step._solve_fmin()
    print 'target position:\n', sim_step.cp_state.u

    print sim_step.cp_state.L_lengths
    # end
    return sim_step

if __name__ == '__main__':
    sim_step = create_sim_step()
