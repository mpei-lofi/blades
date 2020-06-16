from math import sqrt,cos,sin,acos,atan
from math import degrees,radians
from numpy import dot
class Vector(object):
    #x = float(0)
    #y = float(0)
    def __init__(self,_x,_y):
        self.x = _x
        self.y = _y
    def normalize(self):
        self.increase(1/self.length())
        return self
    def increase(self, value):
        self.x *= value
        self.y *= value
        return self
    def setLength(self, value):
        self.normalize()
        self.x *= value
        self.y *= value
        return self
    def length(self):
        return (sqrt(self.x ** 2 + self.y ** 2))
    def dot(self, vec):
        #скалярное произведение векторов
        return (self.x * vec.x + self.y * vec.y)
    def getAngleBetween(self, vec):
        return acos(self.dot(vec) / (self.length() * vec.length()))
    def reverse(self):
        self.x = self.x*-1
        self.y = self.y *-1
        return self
    def normal2vector(self):
        return Vector(-self.y,self.x).normalize()
class Vertex(object):
    x = float(0)
    y = float(0)
    def __init__(self,_x,_y):
        self.x = _x
        self.y = _y
    # def __eq__(self,point):
    #     return Vertex(point.x,point.y)
    def length(self, point):
        return sqrt((point.x - self.x)**2 + (point.y - self.y)**2)
    def move(self, vector):
        self.x+=vector.x
        self.y+=vector.y
        return self
    def rotate(self, angleDeg, direction = True):
        if (direction):
            k = 1.0
        else:
            k = -1.0
        rad = radians(angleDeg)
        x = self.x*cos(rad) + (-k)*self.y*sin(rad)
        y = (k)*self.x*sin(rad) + self.y*cos(rad)
        self.x = x
        self.y = y
        return self
class Line(object):
    k = float(0)
    b = float(0)
    # def __init__(self,_k,_b):
    #     self.k = _k
    #     self.b = _b
    def __init__(self,vertex1,vertex2):
        self.k = (vertex2.y-vertex1.y)/(vertex2.x-vertex1.x)
        self.b = (vertex1.y-self.k*vertex1.x)
    def __call__(self,x):
        return self.k*x+self.b
    def getY(self,x):
        return self.k*x+self.b
    def getX(self,y):
        return (y-self.b)/self.k
    def __mul__(self, line):
        x = (line.b-self.b)/(self.k-line.k)
        y = self.k*x+self.b
        return Vertex(x,y)
    def __eq__(self, line):
        return line;
    def getAngel(self, line):
        deg1 = degrees(atan(self.k))
        deg2 = degrees(atan(line.k))
        return abs(deg1 - deg2)

def getVectorFromPoints(firstPoint,secondPoint):
    return Vector(secondPoint.x-firstPoint.x,secondPoint.y-firstPoint.y)
def getLine(k,b):
    AB = Line(Vertex(0,0),Vertex(1,1))
    AB.k = k
    AB.b = b
    return AB
       
class VertexSet(object):
    pass