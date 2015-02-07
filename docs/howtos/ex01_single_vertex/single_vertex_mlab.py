r'''

Plot the results of single vertex example.

'''

if __name__ == '__main__':
    from single_vertex import create_cp
    cp = create_cp()
    # begin
    from oricreate.view.forming_view import FormingView
    v = FormingView(cp=cp)
    v.configure_traits()
    # end
