r'''
Construct a general crease pattern configuration,
used in the examples below demonstrating the evaluation of goal functions
and constraints.
'''

from sim_map_pattern_to_target_face import create_sim_step

if __name__ == '__main__':
    import mayavi.mlab as m
    sim_step = create_sim_step()
    cp = sim_step.cp_state
    m.figure(bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.6, 0.6, 0.6))
    cp.plot_mlab(m, lines=True)

    arr = m.screenshot()
    import pylab as p
    p.imshow(arr)
    p.axis('off')
    p.show()
