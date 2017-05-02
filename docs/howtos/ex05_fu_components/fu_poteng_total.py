r'''
This script demonstrates the use of a factory
with a client - i.e. a goal function using
the factory product (crease pattern) for evaluation
of the potential energy.
'''
from custom_factory_mpl import create_cp_factory


def create_fu():
    cp_factory = create_cp_factory()
    # begin
    from oricreate.fu import FuPotEngTotal
    # Link the pattern factory with the goal function client.
    fu_poteng = FuPotEngTotal(forming_task=cp_factory,
                              exclude_lines=[4, 5])
    # Change the vertical coordinate to get
    # a non-zero value of potential energy
    cp = cp_factory.formed_object
    cp.u[4, 2] = -1.0
    #cp.u[3, 2] = -1.0
    # Note: the above assignments to array elements are not
    # registered by the change notification system.
    # In order to trigger the dependency chain notifying
    # the fu_poteng instance that something has changed
    # an assignment to an array as a whole is necessary:
    cp.u = cp.u
    print 'F_V_du', cp.F_V_du
    print 'fu:', fu_poteng.get_f(cp.u)
    print 'f_du:\n', fu_poteng.get_f_du(cp.u)
    # end
    return fu_poteng

if __name__ == '__main__':
    create_fu()
