r'''

This example demonstrates the crease pattern factory usage
for Yoshimura crease pattern.


'''


def create_cp():
    # begin
    from oricreate.api import RonReshCPFactory

    cp_factory = RonReshCPFactory()
    cp = cp_factory.formed_object

    print 'Nodes of facets enumerated counter clock-wise\n', cp.F_N[:10]
    print 'Lines of facets enumerated counter clock-wise\n', cp.F_L[:10]
    # end
    return cp

if __name__ == '__main__':
    create_cp()
