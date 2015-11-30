r'''
Construct a general crease pattern configuration,
used in the examples below demonstrating the evaluation of goal functions
and constraints.
'''

from sim04_map_to_surface import create_sim_step

if __name__ == '__main__':
    import mayavi.mlab as m
    map_to_surface = create_sim_step()

    m.figure(bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.6, 0.6, 0.6))
    map_to_surface.previous_task.formed_object.plot_mlab(m, lines=True)
    map_to_surface.formed_object.plot_mlab(m, lines=True)

    m.view(azimuth=-40, elevation=76)

    arr = m.screenshot()
    import pylab as p
    p.imshow(arr)
    p.axis('off')
    p.show()
