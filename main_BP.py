import scripts
import tkinter
import operator
from tkinter import filedialog as fd
import numpy as np
import os
import matplotlib.pyplot as plt
from statistics import median
from fit_fun import fittingCircles
from scripts import *
from lofi_geometry_lib import Vector, Vertex
import scipy as sp
import math
from readingBladesDatabase import ReadBladeDB
import requests
import io
import pandas as pd
#### MAIN ####
# ps - pressure side
# ss - suction side

#### reading data block ####
# filename = fd.askopenfilename()

data_local = r'E:\Work Roman\Git\blades\AtlasData.xlsx'
blade_name = r'С7025А'
blade = ReadBladeDB(data_local,blade_name)
GetValue =  lambda key: float(blade[key].dropna().values) if blade[key].dropna().values else None
X,YSS,YPS = blade[r'X'].values,blade[r'Yсп'].values,blade[r'Yвог'].values
points_ps = np.array([[x,y] for x,y in zip(X,YPS)])
points_ss = np.array([[x,y] for x,y in zip(X,YSS)])
CHORD = GetValue(r'Хорда')
INSTALL_ANGLE = GetValue(r'Угол Установки')
PITCH = GetValue(r'Шаг')
R_INLET = GetValue(r'R вход')
R_OUTLET = GetValue(r'R выход')

print(points_ps, points_ss, CHORD, PITCH, R_INLET, R_OUTLET, INSTALL_ANGLE, sep='\n')

if not R_INLET:
    inletEdgePoints = np.vstack((points_ps[:3],points_ss[:3]))
    result_inlet = FitInletEdge(inletEdgePoints[0],inletEdgePoints[1])
    R_INLET = result_inlet[2]
scalefactor = 1.0/(CHORD - R_INLET - R_OUTLET)
chord = CHORD * scalefactor
pitch = PITCH * scalefactor
r_inlet = R_INLET * scalefactor
r_outlet = R_OUTLET * scalefactor
#### data postprocessing ####
points_ps, points_ss = transform(points_ps,points_ss,R_INLET,R_OUTLET)
spline_ps = getSplineFromPoints(points_ps)
spline_ss = getSplineFromPoints(points_ss)
leftBorderSS, rightBorderSS = min(points_ss[:,0]),max(points_ss[:,0])
# detecting r_max,xr_max and camber line
res = fittingCircles(spline_ss,spline_ps,left_x=0.0 + 0.1,right_x=1.0 - 0.1,step_x=0.01)
radius_array = res[:,2]
r_max = np.max(radius_array)
xr_max = float(np.transpose(res)[0][np.where(radius_array == r_max)])
yr_max = float(np.transpose(res)[1][np.where(radius_array == r_max)])
# print('not sorted', points_camber)
points_camber = np.vstack(([0,0], [[point[0],point[1]] for point in res], [1,0]))
spline_camber = getSplineFromPoints(points_camber)
#### detecting x_max, y_max
points_camber = np.array([[x,float(spline_camber(x))] for x in np.arange(0,1,0.01)])
y_max = np.max(points_camber[1])
x_max = float(points_camber[0][np.where(points_camber[1] == y_max)])

#### detecting omega 1
p2up,p2down = FindTrailingEdgePoints(spline_camber,r_outlet,Vertex(1.0,0.0))
W1 = FindPoint(spline_ps,r_inlet)
angle = lambda x: math.degrees(math.atan(x)) if math.atan(x)>0 else math.degrees(math.atan(x)+math.pi/2.0)

# omega1= 2 * (angle(spline_camber.derivative(nu=1)(0.0))-angle(spline_ps.derivative(nu=1)(W1.x)))

# ОГРОМНЫЙ КОСЯК В ОПРЕДЕЛЕНИИ ОМЕГА 1
k = float(spline_camber.derivative(nu=1)(0.0))
k_temp, b_temp = getLineKB(Vertex(0.0,0.0),W1)
k2 = float(-1.0/k_temp)
b2 = W1.y - k2*W1.x
omega1 = 2 * (angle(k) - angle(k2))
if omega1 < 0.0: omega1 = 0.0 # ЧТОБЫ ОМЕГА 1 НЕ БЫЛА ОТРИЦАТЕЛЬНОЙ

print(spline_ps.derivative(nu=1)(W1.x))
rorate = lambda a : [[math.cos(a),math.sin(a)],[-math.sin(a),math.cos(a)]]
rotation_angle = (90.0-omega1/2.0)*2.0
rMatrix = rorate(math.radians(rotation_angle))
w = np.dot(rMatrix,[W1.x,W1.y])
W2 = Vertex(w[0],w[1])

# detecting BEND
angle_scale = math.degrees(math.atan2(R_INLET-R_OUTLET, (CHORD-R_OUTLET)-R_INLET))
angle_new = 180 - (INSTALL_ANGLE - angle_scale)
# t_opt = 0.75#RECOMMENDET
t = PITCH
if angle_new<0:
    angle_new = angle_new + 180
step_vec = Vector(math.cos(math.radians(angle_new)),math.sin(math.radians(angle_new)))
step_vec.setLength(pitch)
step_point = p2down.move(step_vec)
result_bend = FindBend(spline_ss,step_point)
bend_point = Vertex(result_bend[0],spline_ss(result_bend[0]))
dt_bend = result_bend[1]
k_bend = -1/dt_bend
b_bend = bend_point.y - k_bend*bend_point.x
f = lambda x : k_bend*x+b_bend - float(spline_camber(x))
x0 = 0.5
xr_bend = float(sp.optimize.root(f,x0).x)
BendPoint = Vertex(xr_bend,float(spline_camber(xr_bend)))
RBend = BendPoint.length(bend_point)
dt_camber = float(spline_camber.derivative(nu=1)(BendPoint.x))
deltaBend = math.degrees(2*(abs(math.atan(dt_bend))-abs(math.atan(dt_camber))))
AngleInlet = abs(math.degrees(math.atan(float(spline_camber.derivative(nu=1)(0.0)))))
AngleOutlet = abs(math.degrees(math.atan(float(spline_camber.derivative(nu=1)(1.0)))))
omega2 = 0.0

PARAMETRS = {
    '0':x_max,
    '1':y_max,
    '2':AngleInlet,
    '3':AngleOutlet,
    '4':r_inlet,
    '5':r_outlet,
    '6':omega1,
    '7':omega2,
    '8':r_max,
    '9':xr_max,
    '10':RBend,
    '11':xr_bend,
    '12':deltaBend
}
# filename = fd.asksaveasfilename()
# Writing(filename,PARAMETRS)

plt.figure()
ax = plt.gca()
plt.axis('equal')

#### Добавление окружностей на график ####
InletEdge = plt.Circle((0.0, 0.0), r_inlet, fill=False)
OutletEdge = plt.Circle((1.0, 0.0), r_outlet, fill=False)
RMax = plt.Circle((xr_max, yr_max), r_max, fill=False)
BendCircle = plt.Circle((BendPoint.x, BendPoint.y), RBend, fill=False)

ax.add_artist(InletEdge)
ax.add_artist(OutletEdge)
ax.add_artist(RMax)
ax.add_artist(BendCircle)
for i in range(len(res)):
    ax.add_artist(plt.Circle(
        (res[i][0], res[i][1]), radius_array[i], fill=False, color='green', ls='--'))


plt.scatter(points_ps[:,0], points_ps[:,1], color='red',
            marker='+')  # Цетр входной кромки
plt.scatter(points_ss[:,0], points_ss[:,1], color='blue',
            marker='+')  # Центр выходной кромки
plt.scatter(xr_max, yr_max, color='black', marker='+')  # Центр R MAX

plt.scatter(0,0,marker='+')
plt.scatter(1,0,marker='+')
plt.scatter(x_max,y_max,color='black',marker='+',s = 200)

plt.scatter(W1.x,W1.y,marker='o',s = 100,color='red')
plt.scatter(W2.x,W2.y,marker='o',s = 100,color='black')
plt.scatter(p2up.x,p2up.y,color ='black',marker ='o')
plt.scatter(p2down.x,p2down.y,color ='black',marker ='o')
plt.plot((-0.1,0.0),(k*-0.1,k*0.0))
plt.plot((-0.1,W1.x),(k2*-0.1+b2,W1.y))
plt.scatter(points_camber[:,0],points_camber[:,1])
plt.show()
print(k,k2,b2)
print(PARAMETRS)
print('succsses')
