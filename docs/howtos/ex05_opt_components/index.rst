
Components of the optimization problem
======================================

This page uses a simple crease pattern to show how to construct algorithmic objects for implemented goal functions
and constraints.

.. automodule:: docs.howtos.ex05_opt_components.custom_factory_mpl

.. include:: custom_factory_mpl.py
   :literal:
   :start-after: # begin
   :end-before: # end

.. plot:: howtos/ex05_opt_components/custom_factory_mpl.py
   :width: 400px

Goal functions
--------------

Potential energy of gravity
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: docs.howtos.ex05_opt_components.opt_component_poteng_gravity

.. include:: opt_component_poteng_gravity.py
   :literal:
   :start-after: # begin
   :end-before: # end

The output of the script looks as follows:

.. program-output:: python howtos/ex05_opt_components/opt_component_poteng_gravity.py

Constraints
-----------

Constant length constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: docs.howtos.ex05_opt_components.opt_component_constant_length

.. include:: opt_component_constant_length.py
   :literal:
   :start-after: # begin
   :end-before: # end

The output of the script looks as follows:

.. program-output:: python howtos/ex05_opt_components/opt_component_constant_length.py


Developability
^^^^^^^^^^^^^^
.. automodule:: docs.howtos.ex05_opt_components.opt_component_developability

.. include:: opt_component_developability.py
   :literal:
   :start-after: # begin
   :end-before: # end

The output of the script looks as follows:

.. program-output:: python howtos/ex05_opt_components/opt_component_developability.py



Flat foldability
^^^^^^^^^^^^^^^^
.. automodule:: docs.howtos.ex05_opt_components.opt_component_flat_foldability

.. include:: opt_component_flat_foldability.py
   :literal:
   :start-after: # begin
   :end-before: # end

The output of the script looks as follows:

.. program-output:: python howtos/ex05_opt_components/opt_component_flat_foldability.py

The obtained result whos that the crease pattern configuration at hand satisfies the flat-foldability condition. 
Further, the derivatives of the condition with respect to the nodal displacement are all
zero. This indicates that the displacement of any node in any direction affects/violates 
the condition with the same rate. 

