import numpy as np
from lofi_geometry_lib import *
from scipy import optimize as scop


def fittingCircles(main_spline, fitting_spline, left_x = 0, right_x = 100, step_x = 1.0):
    """Return a result array

        result[:,0] = O.x : O - fitting circle centre 

        result[:,1] = O.y : O - fitting circle centre

        result[:,2] = r : r - fitting circle radius

        Parameters
        ----------
        main_spline : Spline (interpolate.BSpline) suction side spline
        
        fitting_spline : Spline (interpolate.BSpline) pressure side spline

        step_x : float 
        """
    result = []
    # first_approx = [left_x,(left_x-right_x)/2.0,x]
    # x_aprx = (left_x-right_x)/2.0
    for x in np.arange(left_x,right_x,step_x):
        x1 = x
        y1 = float(main_spline(x1))
        point1 = Vertex(x1,y1)
        n1 = getLine((-1.0/float(main_spline(x1,nu=1))),y1-(-1.0/float(main_spline(x1,nu=1)))*x1)
        def f(x):
            x2 = x
            y2 = float(fitting_spline(x2))
            point2 = Vertex(x2,y2)
            n2 = getLine((-1.0/float(fitting_spline(x2,nu=1))),y2-(-1.0/float(fitting_spline(x2,nu=1)))*x2)
            o = n1*n2
            return (o.length(point1) - o.length(point2))**2
        sol = scop.fmin(f,x-8.0*step_x)
        x2 = sol[0]
        if f(x2) < 1e-3:
            y2 = float(fitting_spline(x2))
            point2 = Vertex(x2,y2)
            n2 = getLine((-1.0/float(fitting_spline(x2,nu=1))),y2-(-1.0/float(fitting_spline(x2,nu=1)))*x2)
            o = n1*n2
            if (o.x > left_x and o.x < right_x and o.length(point2)<abs(right_x-left_x)/4.0):
                result.append((o.x,o.y,o.length(point2)))
        else: print("SOLUTION IS BULLSHIT")
    return np.array(result)
