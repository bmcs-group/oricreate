r'''
Map nodes to surface
--------------------

This example demonstrates how to use the `MapToSurface` class.
The specification of the task includes the list of `target faces`
and `dof_constraints`. Using these inputs, the underlying RFKConfig
class is constructed and the RFKSimulator is triggered when
accessing the output properties, i.e. the node coordinates or displacements
`x` and `u`, respectively.

In the present example, the target plane is defined quadratic in the `x` direction.
All nodes are mapped to the surface. Furthermore, they are fixed in 
the `x` and `y` direction.
'''
from oricreate.api import \
    MapToSurface


def create_sim_step():
    from oricreate.api import CreasePatternState, CustomCPFactory

    cp = CreasePatternState(X=[[0, 0, 0],
                               [1, 0, 0],
                               [0.5, 1, 0],
                               [1.5, 1, 0],
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
    target_face = TF(F=[r_, s_, - 0.5 * (r_ - 1) * (r_ - 1) * t_ + 1.0])
    # fix all nodes in x and y direction - let the z direction free
    dof_constraints = \
        [([(i, 0, 1.0)], 0.0) for i in range(0, 5)] +\
        [([(i, 1, 1.0)], 0.0) for i in range(0, 5)]
    sim_step = MapToSurface(previous_task=cp_factory,
                            tf_lst=[(target_face, [0, 1, 2, 3, 4])],
                            dof_constraints=dof_constraints
                            )
    print 'initial position\n', cp_factory.formed_object.x
    print 'target position:\n', sim_step.x_1
    # end
    return sim_step

if __name__ == '__main__':
    create_sim_step()
