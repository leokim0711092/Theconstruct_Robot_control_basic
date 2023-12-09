#!/usr/bin/env python3
import numpy as np #library for scientific computing
from matplotlib import pyplot as plt #library to plot graphs
  
def D_robot(m1,m2,l1,l2,lc1,lc2,Izz1,Izz2,th):
  D=np.zeros((2,2))
  t1=th[0,0]
  t2=th[1,0]
  D[0,0]=Izz1+Izz2+lc1**2*m1+l1**2*m2+2*np.cos(t2)*l1*lc2*m2+lc2**2*m2
  D[0,1]=Izz2+np.cos(t2)*l1*lc2*m2+lc2**2*m2
  D[1,0]=Izz2+np.cos(t2)*l1*lc2*m2+lc2**2*m2
  D[1,1]=Izz2+lc2**2*m2
  return D

def C_robot(m1,m2,l1,l2,lc1,lc2,Izz1,Izz2,th,dth):
  C=np.zeros((2,2))
  t1=th[0,0]
  t2=th[1,0]

  dt1=dth[0,0]
  dt2=dth[1,0]
  
  C[0,0]=-2 * np.sin(t2) * l1 * lc2 * m2 * dt2
  C[0,1]=-np.sin(t2) * l1 * lc2 * m2 * dt2
  C[1,0]= np.sin(t2) * l1 * lc2 * m2 * dt1;
  C[1,1]= 0.0

  return C

def g_robot(m1,m2,l1,l2,lc1,lc2,Izz1,Izz2,th,grav):
  gravity=np.zeros((2,1))
  
  t1=th[0,0]
  t2=th[1,0]

  gravity[0,0]=0.0
  gravity[1,0]=0.0

  return gravity

def Jacobian(l1,l2,th):
  J=np.zeros((2,2))
  t1=th[0,0]
  t2=th[1,0]
  J[0,0]=-np.sin(t1)*l1-np.sin(t1+t2)*l2
  J[0,1]=-np.sin(t1+t2)*l2
  J[1,0]=np.cos(t1)*l1+np.cos(t1+t2)*l2
  J[1,1]=np.cos(t1+t2)*l2
  return J

dt=0.01 #step time for the simulation

q=np.array([[20.00],[-20.00]])*np.pi/180 #initial angles [rad] 
dq=np.array([[0.0],[0.0]]) #initial angular velocity [rad/s]

f1=0.1
f2=0.5
sp_q=lambda t:np.array([[0.1*np.sin(f1*t)],[0.2*np.cos(f2*t)]]) #set point [rad]
d_sp_q=lambda t:np.array([[0.1*f1*np.cos(f1*t)],[-0.2*f2*np.sin(f2*t)]])
dd_sp_q=lambda t:np.array([[-0.1*f1*f1*np.sin(f1*t)],[-0.2*f2*f2*np.cos(f2*t)]])

input=np.array([[0.0],[0.0]]) #initial control torque [Nm]

m1=1.0 #mass link 1
m2=1.0 #mass link 2

l1=1.0 #length link 1
l2=1.0 #length link 2

lc1=0.5 #distance to the center of mass link 1
lc2=0.5 #distance to the center of mass link 2

Izz1=(1.0/12.0)*(0.1*0.01+l1**2) #inertia link 1
Izz2=(1.0/12.0)*(0.1*0.01+l2**2) #inertia link 2

grav=9.81 #gravity

Jeff1=Izz1+m1*lc1**2 #effective inertia of link 1 in the motor 
Jeff2=Izz2+m2*lc2**2 #effective inertia of link 2 in the motor

K1=0.1 #voltage vs torque, motor 1
K2=0.1 #voltage vs torque, motor 2

Beff1=0.001 #angular velocity, friction gain, motor 1
Beff2=0.001 #angular velocity, friction gain, motor 2

red1=1 #reduction relation, link-motor 1 (we assume motor is connected directly to motor)
red2=1 #reduction relation, link-motor 2 (we assume motor is connected directly to motor)

#begin
omega=10
K0_control=np.diagflat(omega**2*np.ones((2,1)))
K1_control=np.diagflat(2*omega*np.ones((2,1)))
#end

th1=[] #array to graph, theta 1
th2=[] #array to graph, theta 2

th1_sp=[] #array to graph, set point theta 1
th2_sp=[] #array to graph, set point theta 2

t0=0
tf=30

time_list=np.arange(t0,tf,dt)

for t in time_list:
  th1.append(q[0,0]*180/np.pi) #array to graph, theta 1 [deg]
  th2.append(q[1,0]*180/np.pi) #array to graph, theta 2 [deg]

  th1_sp.append(sp_q(t)[0,0]*180/np.pi) #array to graph, set point theta 1 [deg]
  th2_sp.append(sp_q(t)[1,0]*180/np.pi) #array to graph, set point theta 2 [deg]

  D=D_robot(m1,m2,l1,l2,lc1,lc2,Izz1,Izz2,q)
  C=C_robot(m1,m2,l1,l2,lc1,lc2,Izz1,Izz2,q,dq)
  g=g_robot(m1,m2,l1,l2,lc1,lc2,Izz1,Izz2,q,grav)
  JM=np.diagflat(np.array([(Jeff1)/(red1**2),(Jeff2)/(red2**2)])) #matrix of inertias
  BM=np.diagflat(np.array([(Beff1)/(red1**2),(Beff2)/(red2**2)])) #matrix of friction

  Force_effector=-np.array([[5000.0],[5000.0]]) #vector force applied to end-effector [N]
  ddq=np.linalg.inv(D+JM)@(input-Jacobian(l1,l2,q).T@Force_effector-C@dq-BM@q-g) #dynamic simulation
  dq=dq+dt*ddq #update, angular velocities
  q=q+dt*dq #update, theta angles

  #begin INVERSE DYNAMICS
  h=C@dq+BM@dq+g
  v=dd_sp_q(t)-K1_control@(dq-d_sp_q(t))-K0_control@(q-sp_q(t))
  input=(D+JM)@v+h+Jacobian(l1,l2,q).T@Force_effector 
  #end INVERSE DYNAMICS

plt.figure(0)
plt.figure(figsize=(12, 4),dpi=200)
plt.plot(time_list,th1,'g') 
plt.plot(time_list,th2,'b')

plt.plot(time_list,th1_sp,'g--') 
plt.plot(time_list,th2_sp,'b--')

plt.legend(["$\Theta_1$","$\Theta_2$","set point-$\Theta_1$","set point-$\Theta_2$"])
plt.grid()

plt.title("Force control, (5000,5000)[N]")
plt.xlabel("Time [s]")
plt.ylabel("Angular position [deg]")

plt.show()