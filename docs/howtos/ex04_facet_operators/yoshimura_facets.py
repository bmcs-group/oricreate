r'''

This example demonstrates the crease pattern factory usage
for Yoshimura crease pattern.


'''


def create_cp():
    # begin
    from oricreate.api import YoshimuraCPFactory

    cp_factory = YoshimuraCPFactory()
    cp = cp_factory.formed_object

    # end
    return cp

if __name__ == '__main__':
    create_cp()
