from math import sqrt
from math import acos
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
    def length(self, point):
        return sqrt((point.x - self.x)**2 + (point.y - self.y)**2)
    def move(self, vector):
        return Vertex(self.x + vector.x,self.y + vector.y)
def getVectorFromPoints(firstPoint,secondPoint):
    return Vector(secondPoint.x-firstPoint.x,secondPoint.y-firstPoint.y)
    