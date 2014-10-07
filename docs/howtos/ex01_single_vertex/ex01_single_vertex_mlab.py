r'''

Plot the results of single vertex example.

'''

if __name__ == '__main__':
    from ex04_single_vertex import create_cp
    cp = create_cp()
    # begin
    from oricreate.view.crease_pattern_view import CreasePatternView
    v = CreasePatternView(cp=cp)
    v.configure_traits()
    # end
