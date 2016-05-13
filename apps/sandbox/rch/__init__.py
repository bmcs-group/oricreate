'''
The sandbox is used for testing the force flow along the crease pattern.
Questions addressed here are:

Solve the force flow explicitly as an optimization problem of the form
without implementing it as constraint directly in the first run.
Address the question of starting vector.

L

Verification
------------
 - Do this for a corbel example, and for a single field of Yoshimura shell.
 - How sensitive is the solution given different initial displacement vectors.
 
Analysis of P3-shell
--------------------
 - Use realistic data for bending stiffness - using grout / using laminated layer (top)
   and (top/bottom)
 - Define relevant load cases to be studied numerically / experimentally
    - reflecting the transport / turning of elements 
 - Test the force flow for different types of boundary conditions.
 - Test the force flow for different types of modified angle and linkage of boundary facets

Numerics
--------
 - Further steps to derive the derivatives of diheedral angles
 
R

Infrastructure
--------------
 - Improve the transitions between forming tasks in terms of 
   x0, u, x1 to cover all options occurring in the forming pipeline  
 - Save the state of foring_task (e.g. after folding) so that the next forming task can start imme 
   immediately without having to recalculate the previous steps.

Formulation
-----------
 - Classify the condition for minimum potential energy based on the solution strategy
   - as an optimization problem - consequences for boundary conditions
   - as stationary point problem solved as a zero-value search of the derivatives  

Visualization
-------------
 - Provide additional viz3d objects to show the displaced configuration in order to 
   check the plausibility of the computation
   


'''