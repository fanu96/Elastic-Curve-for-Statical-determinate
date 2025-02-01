Elastic Curve Calculator

The code calculates the elastic curve of statically determinant beam, as an input it takes beam length, supports, loads(forces), constant EI values or, material types(we added 6)(
                                         steel
                                         aluminium
                                         concrete
                                         wood
                                         plastic
                                         rubber), and 
Geometry of the beam if EI is not constant( we added the 7 most common and implemented their formula)
                              (Rectangular
                               Circular
                               Symmetric I-beam
                               Hollow rectangle
                               Hollow circular
                               Triangular
                               T-section) and EI will be calculated, We chose the Macaulay method which is pretty similar to the double integration method, but handling
discontinuities in a single expression(sp.singularityfunction).

REQUIREMENTS

to run this code the following modules are needed

sympy(pip install sympy) For symbolic math, equation setup, and analytical integration.

NumPy(pip install numpy) For numerical evaluations and handling data arrays.

Matplotlib(pip install matplotlib): For plotting and visualizing the beam's deflection curve.