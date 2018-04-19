import os
from os.path import join, expanduser

import mayavi.mlab as mlab
import moviepy.editor as mpy
import numpy as np


if __name__ == '__main__':

    duration = 2  # duration of the animation in seconds (it will loop)

    # MAKE A FIGURE WITH MAYAVI

    fig = mlab.figure(size=(500, 500), bgcolor=(1, 1, 1))

    u = np.linspace(0, 2 * np.pi, 100)
    xx, yy, zz = np.cos(u), np.sin(3 * u), np.sin(u)  # Points
    l = mlab.plot3d(xx, yy, zz, representation="wireframe", tube_sides=5,
                    line_width=.5, tube_radius=0.2, figure=fig)

    # ANIMATE THE FIGURE WITH MOVIEPY, WRITE AN ANIMATED GIF

    def make_frame(t):
        """ Generates and returns the frame for time t. """
        y = np.sin(3 * u) * (0.2 + 0.5 * np.cos(2 * np.pi * t / duration))
        l.mlab_source.set(y=y)  # change y-coordinates of the mesh
        mlab.view(azimuth=360 * t / duration, distance=9)  # camera angle
        return mlab.screenshot(antialiased=True)  # return a RGB image

    animation = mpy.VideoClip(make_frame, duration=duration).resize(0.5)
    # Video generation takes 10 seconds, GIF generation takes 25s

    home_dir = expanduser('~')
    target_file = join(home_dir, '.oricreate', 'wireframe.mp4')

    animation.write_videofile(target_file, fps=20)
    # animation.write_gif("wireframe.gif", fps=20)
    print('movie saved in %s' % target_file)
