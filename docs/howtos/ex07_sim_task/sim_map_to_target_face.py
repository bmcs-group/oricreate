r'''
This example demonstrates the evaluation of a goal function
on minimum distance of selected nodes of a crease pattern
to a symbolically defined target surface
in terms of an integral distance measure.

The target surface is defined as horizontal plane at the level 1.0.
In order to make the results easily interpretable only a single
node (0) is included in the goal function.

The derivative of the distance function contains only one non-zero
value for the z-displacement of the node (0) equal to -1.
'''
from custom_factory_mpl import create_cp_factory


def create_fu():
    cp_factory = create_cp_factory()
    # begin
    from oricreate.fu import FuTargetFaces, TF
    from oricreate.api import r_, s_
    # Link the pattern factory with the goal function client.
    target_face = TF(F=[r_, s_, 1.0])
    fu_target_faces = FuTargetFaces(cp_factory,
                                    tf_lst=[(target_face, [0])])
    cp = cp_factory.formed_object
    print 'fu:', fu_target_faces.get_f(cp.u)
    print 'f_du:\n', fu_target_faces.get_f_du(cp.u)
    # end
    return fu_target_faces


def create_sim_step_tf():
    fu_target_faces = create_fu()


if __name__ == '__main__':
    create_fu()
