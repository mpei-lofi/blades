from scripts import *
#### MAIN ####
# ps - pressure side
# ss - suction side
fileName = 'c918.txt'
points_pressure = read('c9018pressure.txt')
points_suction = read('c9018suction.txt')
points_pressure = scaling(points_pressure,r1=3.35,r2=0.3,B=47.15)#SET PARAMETRS EVERYWHERE
points_suction = scaling(points_suction,r1=3.35,r2=0.3,B=47.15)
spline_suction = getSplineFromPoints(points_suction)
spline_pressure = getSplineFromPoints(points_pressure)
result_data = FindCamberPoints(spline_suction,spline_pressure,50,eps=1e-5,leftborderX=points_pressure[0][0],rightborder=points_pressure[0][-1],border=0.01)
# detecting r_max,xr_max
radius_arr = [Vertex(result_data[i][0],result_data[i][1]).length(Vertex(result_data[i][2],float(spline_pressure(result_data[i][2])))) for i in range(0,len(result_data))]
xr_array = [i[0] for i in result_data]
points_rDistrub = np.transpose([(xr_array[i],radius_arr[i]) for i in range(0,len(radius_arr))])
spline_rDisturb = getSplineFromPoints(points_rDistrub)
r_max = np.max(radius_arr)
xr_max = float(np.transpose(result_data)[0][np.where(radius_arr == r_max)])
# inletEdgePoints = np.vstack(( np.hstack((points_pressure[0][0:2],points_suction[0][0:10])) ,np.hstack((points_pressure[1][0:2],points_suction[1][0:10])) )) 
inletEdgePoints = np.vstack((points_pressure[0][0:5],points_pressure[1][0:5]))
result_inlet = FitInletEdge(inletEdgePoints[0],inletEdgePoints[1])
points_camber = [(i[0],i[1]) for i in result_data]
points_camber.insert(0,(0,0))
points_camber.append((1,0))
spline_camber = getSplineFromPoints(np.transpose(points_camber))
r1 = 3.35
r2 = 0.3
B = 47.15
sf = 1/(B-r1-r2)
p2up,p2down = FindTrailingEdgePoints(spline_camber,r2*sf,Vertex(1,0))
scaleFactor = 1/(B-r1-r2)
W = FindPoint(spline_pressure,r1*scaleFactor)
angle = lambda x: math.degrees(math.atan(x))
omega1= 2 * (angle(spline_camber.derivative(nu=1)(0.0))-angle(spline_pressure.derivative(nu=1)(W.x)))
# detecting x_max, y_max
camberPoints = np.transpose([(x,float(spline_camber(x))) for x in np.arange(0,1,0.001)])
y_max = np.max(camberPoints[1])
x_max = float(camberPoints[0][np.where(camberPoints[1]==y_max)])
# ploting
plt.figure()
rorate = lambda a : [[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]]
rotation_angle = (90-omega1/2)*2
rMatrix = rorate(math.radians(-rotation_angle))
W1 = np.dot(rMatrix,[W.x,W.y])
# detecting BEND
angle_install = 42.3
angle_scale = 2.3456854#ERROR CORE
angle_new = 180 - (angle_install - angle_scale)
t_opt = 0.75#RECOMMENDET
t = t_opt*B
if angle_new<0:
    angle_new = angle_new + 180
step_vec = Vector(math.cos(math.radians(angle_new)),math.sin(math.radians(angle_new)))
step_vec.setLength(t*scaleFactor)
step_point = p2down.move(step_vec)
result_bend = FindBend(spline_suction,step_point)
bend_point = Vertex(result_bend[0],spline_suction(result_bend[0]))
dt_bend = result_bend[1]
k_bend = -1/dt_bend
b_bend = bend_point.y - k_bend*bend_point.x
f = lambda x : k_bend*x+b_bend - float(spline_camber(x))
x0 = 0.5
xr_bend = float(scipy.optimize.root(f,x0).x)
BendPoint = Vertex(xr_bend,float(spline_camber(xr_bend)))
RBend = BendPoint.length(bend_point)
dt_camber = float(spline_camber.derivative(nu=1)(BendPoint.x))
deltaBend = math.degrees(2*(abs(math.atan(dt_bend))-abs(math.atan(dt_camber))))
# write PARAMETRS
RInlet = r1*scaleFactor
ROutlet = r2*scaleFactor
AngleInlet = math.degrees(math.atan(float(spline_camber.derivative(nu=1)(0.0))))
AngleOutlet = abs(math.degrees(math.atan(float(spline_camber.derivative(nu=1)(1.0)))))
omega2 = 0.0
fileName = fileName[:+4]+'_Parametrs.txt'
# PARAMETRS = {   'Angle_inlet':AngleInlet,
#                 'Angle_outlet':AngleOutlet,
#                 'R_inlet':r1,
#                 'R_outlet':r2,
#                 'X_Max':x_max,
#                 'Y_max':y_max,
#                 'R_Max': r_max,
#                 'XR_Max': xr_max,
#                 'omega_1': omega1,
#                 'omega_2': omega2,
#                 'Angle_Bend':deltaBend,
#                 'R_Bend':RBend,
#                 'XR_Bend':xr_bend}
PARAMETRS = {
    '0':x_max,
    '1':y_max,
    '2':AngleInlet,
    '3':AngleOutlet,
    '4':RInlet,
    '5':ROutlet,
    '6':omega1,
    '7':omega2,
    '8':r_max,
    '9':xr_max,
    '10':RBend,
    '11':xr_bend,
    '12':deltaBend
}
Writing(fileName,PARAMETRS)
# there are need a legend
plt.axis('equal')
plt.plot([0,1],[0,0],'red')
for i in range(len(result_data)):
    c = Vertex(result_data[i][0],result_data[i][1])
    x2 = result_data[i][2]
    r = Vertex(x2,spline_pressure(x2)).length(c)
    t = np.arange(0,2*np.pi,.01)
    x = [math.cos(i)*r + c.x for i in t]
    y = [math.sin(i)*r + c.y for i in t]
    plt.scatter(x2,spline_pressure(x2),marker='x',c='red')
    plt.plot(x,y,'g--')
plt.scatter(points_suction[0],points_suction[1])
plt.scatter(points_pressure[0],points_pressure[1])
x = np.arange(points_pressure[0][0],points_pressure[0][-1],0.01)
plt.plot(x, spline_pressure(x),'black',x, spline_suction(x),'black')
r = result_inlet[2]
t = np.arange(0,2*np.pi,.01)
x = [math.cos(i)*r + result_inlet[0] for i in t]
y = [math.sin(i)*r + result_inlet[1] for i in t]
scaleFactor = 1/(42.47-2.6-0.25)
print(r,2.6*scaleFactor)
r=0.25*scaleFactor
x2 = [math.cos(i)*r + 1 for i in t]
y2 = [math.sin(i)*r + 0 for i in t]
x3 = [math.cos(i)*r1*sf + 0 for i in t]
y3 = [math.sin(i)*r1*sf + 0 for i in t]
plt.plot(x,y,'r--',x2,y2,'r--',x3,y3,'b--')
k = spline_pressure.derivative(nu=1)(points_pressure[0][-1])
b = points_pressure[1][-1] - k*points_pressure[0][-1]
plt.plot([points_pressure[0][-1],1.1],[points_pressure[1][-1],k*1.1+b])
k = spline_suction.derivative(nu=1)(points_suction[0][-1])
b = points_suction[1][-1] - k*points_suction[0][-1]
plt.plot([points_suction[0][-1],1.1],[points_suction[1][-1],k*1.1+b])
x = np.arange(np.transpose(points_camber)[0][0],np.transpose(points_camber)[0][-1]+0.01,0.01)
plt.plot(x,spline_camber(x),'green','solid')
plt.scatter(p2down.x,p2down.y,marker='d',c='red')
plt.scatter(p2up.x,p2up.y,marker='d',c='red')
plt.scatter(W.x,W.y,marker='x',c='blue')
plt.scatter(float(W1[0]),float(W1[1]),marker='x',c='blue')
plt.scatter(x_max,y_max,marker='x',c='blue')
plt.scatter(xr_max,spline_camber(xr_max),marker='x',c='blue')
plt.scatter(step_point.x,step_point.y,marker='x',c='blue')
xrmax = [math.cos(i)*r_max + xr_max for i in t]
yrmax = [math.sin(i)*r_max + float(spline_camber(xr_max)) for i in t]
plt.plot(xrmax,yrmax,'b--')
plt.scatter(bend_point.x,bend_point.y,marker='x',c='blue')
plt.scatter(BendPoint.x,BendPoint.y,marker='x',c='blue')
xrbend = [math.cos(i)*RBend + BendPoint.x for i in t]
yrbend = [math.sin(i)*RBend + BendPoint.y for i in t]
plt.plot(xrbend,yrbend,'b--')
plt.show()