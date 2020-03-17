# All libiares and functions
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize
import scipy
from vertex import *
def Writing(fileName,data):
        fileName = fileName+'_PARAMETRS.txt'
        file = open(fileName,'w')
        for i in data:
                # file.write(i + '\t' + str(data[i]) + '\n')
                file.write(str(data[i])+'\n')
        file.close()
def getCurvature(f,x):
    """Return a curvature value for a spline function

    return k(float) - curvature, r(float) - radius of curvature

    Parameters
    ----------
    f : Spline (interpolate.BSpline)

    x : float (x - coordinate)
    """
    df = f.derivative(nu=1)(x)
    ddf = f.derivative(nu=2)(x)
    k = ddf/pow(1-df*df,3/2)
    return k, abs(1.0/k)
def Reading(fileName):
    with open(fileName) as file:
        data = file.readlines()
        x = []
        y = []
        for i in data:
            a = i.split('\t')
            x.append(float(i.split('\t')[0]))
            y.append(float(i.split('\t')[1]))
    return [x,y]
def sorting(matrix,raw=0):
    if raw == 0:
        j = 1
    else:
            j = 1
    n = 1 
    while n < len(matrix[raw]):
        for i in range(len(matrix[raw])-n):
            if matrix[raw][i] > matrix[raw][i+1]:
                matrix[raw][i],matrix[raw][i+1] = matrix[raw][i+1],matrix[raw][i]
                matrix[j][i],matrix[j][i+1] = matrix[j][i+1],matrix[j][i]
        n += 1
def scaling(data,b=1.0,r1=0,r2=0, B=50):
        p1 = Vertex(r1,r1)
        p2 = Vertex(B-r2,r2)
        angle = math.atan2(p1.y-p2.y,p2.x-p1.x)
        vertexArray = [Vertex(data[0][i],data[1][i]) for i in range(len(data[0]))]
        vectorMove = getVectorFromPoints(Vertex(r1,r1),Vertex(0,0))
        vertexArray = [i.move(vectorMove) for i in vertexArray]
        data = np.transpose([[i.x,i.y] for i in vertexArray])

        rotationMatrix = np.array([
                [math.cos(angle),math.sin(angle)],
                [-math.sin(angle),math.cos(angle)]
        ])
        sf = 1
        scaleFactor = sf/(B-r1-r2)
        a = np.transpose(data)
        b = [i@rotationMatrix for i in a]
        c = np.transpose(b)
        return [i*scaleFactor for i in c]
def read(fileName):
        data = Reading(fileName)
        xy = np.array(data)
        return xy            
def getLineKB(vertex1, vertex2):
    k = (vertex2.y - vertex1.y)/(vertex2.x - vertex1.x)
    b = vertex1.y - vertex1.x * k
    return k,b
def plotline(k,b,x1,x2):
    x = []
    y = []
    x.append(x1)
    x.append(x2)
    y.append(x1*k + b)
    y.append(x2*k + b)
    return x,y
def getNormal(spline,x, isUp = False):
    y = spline(x)
    k = spline.derivative()(x)
    # if k>0 and isUp == False:
    #     k*=-1
    # if k<0 and isUp == True:
    #     k*=-1
    b = y - k*x
    dx = 0.01
    point1 = Vertex(x-dx,k*(x-dx)+b)
    point2 = Vertex(x+dx,k*(x+dx)+b)
    vec = getVectorFromPoints(point1,point2).normal2vector()
    if isUp == True and vec.y < 0:
        vec.reverse()
    if isUp == False and vec.y > 0:
        vec.reverse()
    tempPoint = Vertex(x,y).move(vec)
    return getLineKB(Vertex(x,y),tempPoint)
def half_divide_method(a, b, f):
    e = 1e-6
    x = (a + b) / 2
    while math.fabs(f(x)) >= e:
        x = (a + b) / 2
        a, b = (a, x) if f(a) * f(x) < 0 else (x, b)
    return (a + b) / 2    
def getSplineFromPoints(pointsArray):
        # data = sorting([pointsArray[0],pointsArray[1]]) ##сортировка не нужна
        data = pointsArray
        tck = interpolate.splrep(data[0],data[1],k=2)
        spline = interpolate.BSpline(tck[0],tck[1],tck[2])
        return spline
def FindCamberPoints(spline_suction,spline_pressure,count=20,eps=1e-5,border=0.01,leftborderX=0.0,rightborder=1.0):
        """Return camber line

        Parameters
        ----------
        spline_suction : Spline (interpolate.BSpline)
        
        spline_pressure : Spline (interpolate.BSpline)

        count : int (numbers of inscribed circles)
        """
        ds = spline_suction.derivative(nu=1)
        dds = spline_suction.derivative(nu=2)
        dp = spline_pressure.derivative(nu=1)
        ddp = spline_pressure.derivative(nu=2)
        # Профиль предварительно отмасштабирован 
        lb = leftborderX + border
        rb = rightborder - border
        step = abs(lb-rb)/(count-1)
        x = np.arange(leftborderX+border,(rightborder-border)+step,step)
        ds_set = ds(x)
        dds_set = dds(x)
        result_data = []
        for i in x:
                X = [0.5,0.2,0.5] # начальное приблежение покачто нужно контролировать в ручную
                def function(X):
                        x1 = i
                        y1 = float(spline_suction(i))
                        k1 = float(ds(i))
                        b1 = y1 - k1*x1
                        kr = -1/k1
                        br = y1-kr*x1
                        xr = X[0]
                        yr = X[1]
                        x2 = X[2]
                        y2 = float(spline_pressure(x2))
                        k2 = float(dp(x2))
                        b2 = y2-k2*x2
                        kr2 = -1/k2
                        br2 = y2 - kr2*x2
                        p1 = Vertex(x1,y1)
                        p2 = Vertex(x2,spline_pressure(x2))
                        pr = Vertex(xr,yr)
                        return [
                                X[1]-kr*X[0]-br,
                                X[1]-kr2*X[0]-br2,
                                abs(p1.length(pr)-p2.length(pr))
                        ]
                sol = scipy.optimize.root(function,X,method='hybr',tol=eps)
                print(sol.x,'   ',sol.success)
                if sol.success and sol.x[1]<float(spline_suction(i)):
                        result_data.append(sol.x)
        return result_data

                        
from scipy.spatial.distance import cdist
from scipy.optimize import fmin

def FitInletEdge(X,Y):
      

      # Choose the inital center of fit circle as the CM
      xm = X.mean()
      ym = Y.mean()

      # Choose the inital radius as the average distance to the CM
      cm = np.array([xm,ym]).reshape(1,2)
      rm = cdist(cm, np.array([X,Y]).T).mean()

      # Best fit a circle to these points
      def err(vec):
            w = vec[0]
            v = vec[1]
            r = vec[2]
            pts = [np.linalg.norm([x-w,y-v])-r for x,y in zip(X,Y)]
            return (np.array(pts)**2).sum()

      return scipy.optimize.fmin(err,[xm,ym,rm])  
    
def FindTrailingEdgePoints(spline_camber,r2,centre):
        k = -1/spline_camber.derivative(nu=1)(centre.x)
        upVec = Vector(1,k*centre.x).setLength(r2)
        downVec = Vector(-1*upVec.x,-1*upVec.y)
        p1 = centre.move(upVec)
        p2 = centre.move(downVec)
        return p1,p2
def FindPoint(spline_pressure,r1):
        xCoord = np.arange(0,r1,0.01)
        centre = Vertex(0,0)
        def f(x):
                value = abs(Vertex(x[0],float(spline_pressure(x[0]))).length(centre) - r1)
                cond = pow(x[0]-r1,2)
                return value + cond
        result = minimize(f,0.05)
        x = result.x
        return Vertex(x,float(spline_pressure(x)))

def FindBend(spline_suction,point):
        def f(x):
                return Vertex(x[0],float(spline_suction(x[0]))).length(point)
        x = 0.1
        sol = minimize(f,x)
        x = float(sol.x)
        return x,float(spline_suction.derivative(nu=1)(x))