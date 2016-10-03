'''
Created on Dec 3, 2015

Animator module defines animation time ``t_a``.
The time line consists of camera transitions connecting
camera stations. Each station has an animation time stamp 
- ats.

The animator is linked with the visualization objects via
ftv.viz3d_dict. There are two possibilities how to run an update.
The viz object controls the time variable.

Consider a time dependent boundary condition bc(t)
or moving target surface (t). These time variables are defined within 
a forming task they are associated with.

The scene transition defines the mapping between the animation time
and the forming time. It controls both the camera time and 
the forming task time.

Scene transition has an attached forming task view FTV object that
contains a list of viz objects. Upon traveling through the scene, the 
scene transition lets the FTV update itself for a new FTV time.
The mapping between the movie time MT and the scene transition time STT
is provided by the BST and EST - beginning scene time and ending scene time.
Within these limits, a scene nominal time SNT - (0,1) is provided 
to define the mapping to the forming task time FTT( SNT ).

In this way, the animator can also control the simulation on
demand. By sending requests to the viz3D objects attached to the simulation task
the visualization object sets the time value of an object and fetches the 
data.

Should the animator be just an observer or should it actively control the state
of the forming tasks? Should the scene transition traveling through the time
send a reset time after finishing the scene transition?

Forming task time is passed to viz objects attached to 

Forming sequences
=================

Interpolation
=============

They should not need to define their
loop on their own. The viz object declares the loop.

@author: rch
'''

import os
import tempfile
from time import sleep
from traits.api import \
    HasStrictTraits, Callable, \
    Property, Instance, List, \
    Array, Float, Int, Tuple, \
    Str, WeakRef, Trait, Event, \
    cached_property, Range, Dict
from traitsui.api import \
    View, Item, UItem, HSplit, VGroup, HGroup, \
    TableEditor, ObjectColumn, InstanceEditor, \
    Handler, Action, ToolBar

from forming_task_view3d import \
    FTV
import numpy as np
from viz3d import \
    Viz3D


class FTAHandler(Handler):

    def add_cam_move(self, info):
        info.object.add_cam_move()

    def del_cam_move(self, info):
        info.object.del_cam_move()

    def capture_cam_pos(self, info):
        info.object.capture_cam_pos()

    def anim(self, info):
        info.object.anim()

    def render(self, info):
        info.object.render()

    def save(self, info):
        info.object.save()

    def load(self, info):
        info.object.load()

action_strings = \
    [('Add', 'add_cam_move', 'Add a camera move'),
     ('Del', 'del_cam_move', 'Delete a camera move'),
     ('Capture', 'capture_cam_pos', 'Capture camera_position'),
     ('Animate', 'anim', 'Animate on screen'),
     ('Render', 'render', 'Render to file'),
     ('Save', 'save', 'Save timeline'),
     ('Load', 'load', 'Load timeline')]

actions = [Action(name=name,
                  action=action,
                  tooltip=tooltip)
           for name, action, tooltip in action_strings]

cam_move_list_editor = TableEditor(
    columns=[ObjectColumn(label='Start', name='bts', editable=False),
             ObjectColumn(label='End', name='ets', editable=False),
             ],
    selection_mode='row',
    selected='object.selected_cam_move',
)

# UItem is Unlabeled Item


class InstanceUItem(UItem):
    """Convenience class for including an Instance in a View"""
    style = Str('custom')
    editor = Instance(InstanceEditor, ())


class CamStation(HasStrictTraits):
    '''Camera stations

    Specify the positions of the camera by defining the
    azimuth, elevation, distance and focal point
    and roll angle.
    '''
    prev_move = WeakRef

    azimuth = Float(0.0)
    elevation = Float(0.0)
    distance = Float(10.0)
    focal_point = Tuple(Float, Float, Float)

    time_stemp = Property

    def _get_time_stemp(self):
        if self.prev_move:
            return self.prev_move.ets
        else:
            return 0.0

    fpoint = Property

    roll = Float(0)

    def _get_fpoint(self):
        return np.array(list(self.focal_point), dtype='float_')

    view = View(Item('azimuth'),
                Item('elevation'),
                Item('distance'),
                Item('roll'),
                Item('focal_point'),
                Item('time_stemp', style='readonly'),
                buttons=['OK', 'Cancel'])


def linear_cam_move(low, high, n):
    low = np.array(low)
    high = np.array(high)
    t_range = np.linspace(0, 1, n)
    if isinstance(low, np.ndarray):
        if len(low.shape) > 0:
            t_range = t_range[:, np.newaxis]
    return low + (high - low) * t_range


def damped_cam_move(low, high, n):
    phi_range = np.linspace(0, np.pi, n)
    u_range = (1 - np.cos(phi_range)) / 2.0
    if isinstance(low, np.ndarray):
        if len(low.shape) > 0:
            u_range = u_range[:, np.newaxis]
    return low + (high - low) * u_range


class CamMove(HasStrictTraits):
    '''Camera transitions.

    Attach functional mapping depending on time variable
    for azimuth, elevation, distance, focal point and roll angle.
    '''

    def __init__(self, *args, **kw):
        super(CamMove, self).__init__(*args, **kw)

    fta = WeakRef
    ftv = WeakRef(FTV)
    from_station = WeakRef(CamStation)
    to_station = WeakRef(CamStation)

    changed = Event

    cam_attributes = [
        'azimuth', 'elevation', 'distance', 'fpoint', 'roll']

    azimuth_move = Trait('linear', {'linear': linear_cam_move,
                                    'damped': damped_cam_move})
    elevation_move = Trait('linear', {'linear': linear_cam_move,
                                      'damped': damped_cam_move})
    distance_move = Trait('linear', {'linear': linear_cam_move,
                                     'damped': damped_cam_move})
    fpoint_move = Trait('linear', {'linear': linear_cam_move,
                                   'damped': damped_cam_move})
    roll_move = Trait('linear', {'linear': linear_cam_move,
                                 'damped': damped_cam_move})

    duration = Float(10, label='Duration')

    bts = Property(label='Start time')

    def _get_bts(self):
        return self.from_station.time_stemp

    ets = Property(label='End time')

    def _get_ets(self):
        return self.from_station.time_stemp + self.duration

    n_t = Int(10, input=True)

    cmt = Property(Array('float_'), depends_on='n_t')
    '''Relative camera move time (CMT) running from zero to one.
    '''
    @cached_property
    def _get_cmt(self):
        return np.linspace(0, 1, self.n_t)

    viz_t_move = Property(Array('float_'))
    '''Time line range during the camera move
    '''

    def _get_viz_t_move(self):
        return np.linspace(self.bts, self.ets, self.n_t)

    vot_start = Float(0.0, auto_set=False, enter_set=True, input=True)
    vot_end = Float(1.0, auto_set=False, enter_set=True, input=True)
    vot = Property(Array('float_'), depends_on='n_t,vot_start,vot_end')
    '''Visualization object time (VOT). By default it is the same as the camera time.
    It can be mapped to a different time profile using viz_time_fn
    '''
    @cached_property
    def _get_vot(self):
        return np.linspace(self.vot_start, self.vot_end, self.n_t)

    def _get_vis3d_center_t(self):
        '''Get the center of the object'''
        return self.ftv.get_center_t

    def _get_vis3d_bounding_box_t(self):
        '''Get the bounding box of the object'''
        return self.vis.get_center(self.t_range)

    transition_arr = Property(
        Array(dtype='float_'), depends_on='changed,+input')
    '''Array with azimuth values along the transition 
    '''
    @cached_property
    def _get_transition_arr(self):
        trans_arr = [getattr(self, attr + '_move_')(getattr(self.from_station, attr),
                                                    getattr(self.to_station, attr), self.n_t)
                     for attr in self.cam_attributes]
        trans_arr.append(self.vot)
        trans_arr.append(self.viz_t_move)
        return trans_arr

    def reset_cam(self, m, a, e, d, f, r):
        m.view(azimuth=a, elevation=e, distance=d, focalpoint=f)
        m.roll(r)

    def take(self, ftv):
        for a, e, d, f, r, vot, viz_t in zip(*self.transition_arr):
            ftv.update(vot, viz_t, force=True)
            self.reset_cam(ftv.mlab, a, float(e), d, f, r)
            sleep(self.fta.anim_delay)

    def render_take(self, ftv, fname_base, format_, idx_offset):
        im_files = []
        for idx, (a, e, d, f, r, vot, viz_t) \
                in enumerate(zip(*self.transition_arr)):
            ftv.update(vot, viz_t, force=True)
            # @todo: temporary focal point determination - make it optional
            self.reset_cam(ftv.mlab, a, e, d, f, r)
            fname = '%s%03d.%s' % (fname_base, idx + idx_offset, format_)
            ftv.mlab.savefig(fname, magnification=2.2)  # size=(3200, 2000))  #
            im_files.append(fname)
        return im_files

    view = View(VGroup(HGroup(InstanceUItem('from_station@', resizable=True),
                              VGroup(Item('azimuth_move'),
                                     Item('elevation_move'),
                                     Item('distance_move'),
                                     Item('roll_move'),
                                     Item('fpoint_move'),
                                     Item('n_t'),
                                     Item('duration'),
                                     ),
                              InstanceUItem('to_station@', resizable=True),
                              ),
                       VGroup(HGroup(UItem('vot_start'),
                                     UItem('vot_end'),
                                     springy=True
                                     ),
                              label='object time range'
                              ),
                       ),
                buttons=['OK', 'Cancel'])


class FormingTaskAnim3D(HasStrictTraits):
    '''Coordinate a camera and movement of visualization object along the timeline.
    '''

    ftv = Instance(FTV)
    '''Folding task view.
    '''
    init_cam_station = Instance(CamStation)

    anim_delay = Range(low=0.0, high=1.0, value=0.0,
                       label='Animation delay [s]',
                       auto_set=False, enter_set=True, input=True)

    def _init_cam_station_default(self):
        return CamStation()

    def init_view(self, a, e, d, f, r):
        self.init_cam_station.set(
            azimuth=a, elevation=e, distance=d, focal_point=f, roll=r)
        self.ftv.mlab.view(a, e, d, f)
        self.ftv.mlab.roll(r)

    def plot(self):
        self.ftv.plot()

    cam_stations = List(Instance(CamStation))

    def _cam_stations_default(self):
        return [self.init_cam_station]

    cam_moves = List(Instance(CamMove))

    selected_cam_move = Instance(CamMove, None)

    def _selected_cam_move_changed(self):
        self.set_cam_pos()

    def add_cam_move(self, **kw):
        '''Add a new visualization object.'''
        prev_cam_station = self.cam_stations[-1]

        a = kw.pop('a', prev_cam_station.azimuth)
        e = kw.pop('e', prev_cam_station.elevation)
        d = kw.pop('d', prev_cam_station.distance)
        f = kw.pop('f', prev_cam_station.focal_point)
        r = kw.pop('r', prev_cam_station.roll)

        if prev_cam_station.prev_move:
            prev_move = prev_cam_station.prev_move
            n = kw.pop('n', prev_move.n_t)
        else:
            n = kw.pop('n', 20)
        next_cam_station = CamStation(azimuth=a, elevation=e, distance=d,
                                      focal_point=f, roll=r)
        self.cam_stations.append(next_cam_station)
        cm = CamMove(fta=self, ftv=self.ftv,
                     from_station=prev_cam_station,
                     to_station=next_cam_station, n_t=n, **kw)
        next_cam_station.prev_move = cm
        self.cam_moves.append(cm)

    def del_cam_move(self):
        '''Delete currently selected cam move.
        '''
        from_station = self.selected_cam_move.from_station
        to_station = self.selected_cam_move.to_station
        move_idx = self.cam_moves.index(self.selected_cam_move)
        stat_idx = self.cam_stations.index(to_station)
        del self.cam_stations[stat_idx]
        if move_idx < len(self.cam_moves) - 1:
            next_cam_move = self.cam_moves[move_idx + 1]
            next_cam_move.from_station = from_station
        del self.cam_moves[move_idx]
        if move_idx == len(self.cam_moves):
            move_idx -= 1

        self.selected_cam_move = self.cam_moves[move_idx]

    def capture_cam_pos(self):
        '''Get the current position from the ftv and set it to the target
        station of the cam move.
        '''
        a, e, d, f = self.ftv.mlab.view()
        r = self.ftv.mlab.roll()
        to_station = self.selected_cam_move.to_station
        to_station.set(azimuth=a, elevation=e, distance=d,
                       focal_point=tuple(f), roll=r)
        self.selected_cam_move.changed = True

    def set_cam_pos(self):
        '''Get the current position from the ftv and set it to the target
        station of the cam move.
        '''
        if self.selected_cam_move == None:
            return
        to_station = self.selected_cam_move.to_station
        self.ftv.mlab.view(azimuth=to_station.azimuth,
                           elevation=to_station.elevation,
                           distance=to_station.distance,
                           focalpoint=to_station.focal_point)
        self.ftv.mlab.roll(to_station.roll)

    def anim(self):
        for cam_move in self.cam_moves:
            cam_move.take(self.ftv)

    def render(self):
        self.ftv.mlab.options.offscreen = True
        im_files = []
        fname_base = 'anim'
        tdir = tempfile.mkdtemp()
        fname_path = os.path.join(tdir, fname_base)

        idx_offset = 0
        for cam_move in self.cam_moves:
            take_im_files = cam_move.render_take(
                self.ftv, fname_path, 'jpg', idx_offset)
            idx_offset += len(take_im_files)
            im_files.append(take_im_files)

        self.ftv.mlab.options.offscreen = False

        return im_files

    def save_timeline(self):
        raise NotImplementedError

    def load_timeline(self):
        raise NotImplementedError

    trait_view = View(
        HSplit(
            VGroup(
                Item('cam_moves',
                     style='custom', editor=cam_move_list_editor,
                     show_label=False, springy=True, width=150),
                VGroup(
                    UItem('anim_delay'),
                    label='animation delay'
                ),
            ),
            Item('selected_cam_move@', show_label=False,
                 springy=True,
                 width=800, height=200),
            show_border=True,
        ),
        toolbar=ToolBar(*actions),
        resizable=True,
        height=400,
        kind='subpanel',
        title='Timeline editor',
    )


FTA = FormingTaskAnim3D

if __name__ == '__main__':

    from visual3d import Visual3D

    class PointCloudViz3D(Viz3D):
        '''Visualization object
        '''

        def plot(self):
            x, y, z, s = self.vis3d.points
            self.pipes['points'] = self.ftv.mlab.points3d(x, y, z, s)

        def update(self):
            x, y, z, s = self.vis3d.points
            points = np.c_[x, y, z]
            pipe = self.pipes['points']
            pipe.mlab_source.set(points=points, scalars=s)

    class PointCloud(Visual3D):
        '''State object
        '''
        p = Tuple
        '''Point positions
        '''

        def _p_default(self):
            x = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
            y = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
            z = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
            s = [1, 2, 3, 4, 5, 6, 7, 8]
            return x, y, z, s

        x = Property

        def _get_x(self):
            x, y, z, s = self.p
            return np.c_[x, y, z]

        points = Property(depends_on='+time_change')

        @cached_property
        def _get_points(self):
            x, y, z, s = self.p
            s = np.array(s, float)
            s[-1] *= (1.0 - 0.9 * self.vot)
            return x, y, z, s

        viz3d_classes = dict(default=PointCloudViz3D,
                             something_else=PointCloudViz3D)

    ftv = FTV()
    pc = PointCloud()
    print pc.viz3d_classes
    ftv.add(pc.get_viz3d('default'))

    pc.viz3d['default'].set(anim_t_start=10, anim_t_end=40)

    fta = FTA(ftv=ftv)
    fta.init_view(a=0, e=0, d=8, f=(0, 0, 0), r=0)
    fta.add_cam_move(e=50, n=20,
                     duration=10,
                     azimuth_move='linear',
                     elevation_move='damped',
                     distance_move='damped')
    fta.add_cam_move(e=0, n=20,
                     duration=30,
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped')
#     fta.add_cam_move(e=90, n=100,
#                      duration=30,
#                      azimuth_move='damped',
#                      elevation_move='damped',
#                      distance_move='damped')
    fta.plot()
    fta.configure_traits()
