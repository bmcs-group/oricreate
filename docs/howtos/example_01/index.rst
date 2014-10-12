
Single vertex example
=====================

This example constructs a single vertex crease pattern and
shows how to obtain the basic characteristics including angles, 
facet areas, normals to the facets. Extraction of derivatives
is provided as well. 

.. automodule:: docs.howtos.example_01.single_vertex

.. include:: single_vertex.py
   :literal:
   :start-after: # begin
   :end-before: # end

The output of the script looks as follows:

.. program-output:: python howtos/example_01/single_vertex.py

The ``plot_mpl`` commands renders the crease pattern in the base
plane. 

.. plot:: howtos/example_01/single_vertex_mpl.py
