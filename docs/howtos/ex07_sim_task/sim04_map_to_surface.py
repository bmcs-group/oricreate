r'''
This example demonstrates how to define a FormingTask
on a pre-form FormedObject that has been previously produced
in a CreasePatternFactory. The forming process within 
this FormingTask is controlled by the RFKSimulator
that is configured using the RFKConfig.

The target face is defined as horizontal plane at the height 8
and nodes [0,1,2] are involved in the minimum distance criterion.
'''
from oricreate.api import \
    MapToSurface


def create_sim_step():
    from oricreate.api import CreasePatternState, CustomCPFactory

    cp = CreasePatternState(X=[[0, 0, 0],
                               [1, 0, 0],
                               [0.5, 0.5, 0],
                               [1.5, 0.5, 0],
                               [2.0, 0.0, 0]
                               ],
                            L=[[0, 1],
                               [1, 2],
                               [2, 0],
                               [1, 3],
                               [2, 3],
                               [3, 4],
                               ],
                            F=[[0, 1, 2],
                               [1, 2, 3],
                               [1, 3, 4]]
                            )

    cp_factory = CustomCPFactory(formed_object=cp)

    # begin
    from oricreate.fu import TF
    from oricreate.api import r_, s_, t_
    target_face = TF(F=[r_, s_, - r_ * s_ * t_ + 1.0])
    sim_step = MapToSurface(previous_task=cp_factory,
                            tf_lst=[(target_face, [0, 1, 2, 3, 4])])
    print 'initial position\n', cp_factory.formed_object.x
    print 'target position:\n', sim_step.x_1
    # end
    return sim_step

if __name__ == '__main__':
    create_sim_step()
