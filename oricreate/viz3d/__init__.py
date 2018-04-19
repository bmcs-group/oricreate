'''
Module implementing the visualization infrastructure
====================================================

Each state and operator object inheriting from Visual3D class can be visualized
using several visualization objects - subclasses of the class Viz3D. 

In order to introduce a n independent class subsystem into the class structure, 
objects supporting visualization inherit from Visual which introduces 
a dictionary viz3d objects.

Classes implementing visualizations inherit from Viz3D. 
Their names also end with Viz3D, i.e. CreasePatternViz3D of DofConstraintsViz3D.

Visualization objects are assembled within a Viz3DPipeline. 
The pipeline is constructed by a FormingTask and registered within the 
VormingTaskView3D target object. When plotting or updating the view, 
the pipeline iterates through the registered Viz3D objects 
and calls their plot( view3d ) or update( view 3d ) methods.

FormingTaskView3D (shortened as FTV) object provides an mlab 
interface as an mlab attribute

'''

from .forming_task_anim3d import FormingTaskAnim3D, FTA
from oricreate.view.window.forming_task_view3d import FormingTaskView3D, FTV
from .visual3d import Visual3D, Vis3D
from .viz3d import Viz3D
