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

dt=0.01 #step time for the simulation

q=np.array([[45.00],[25.00]])*np.pi/180 #initial angles [rad] 
sp_q=np.array([[50.0],[27.0]])*np.pi/180 #set point [rad]
dq=np.array([[0.0],[0.0]]) #initial angular velocity [rad/s]
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
etta=1 #damping ratio link
omega=10 #natural frequency for control

KP1=(omega**2*Jeff1)/(K1) #proportional gain, link 1
KP2=(omega**2*Jeff2)/(K2) #proportional gain, link 2

KD1=(2*etta*omega*Jeff1-Beff1)/K1 #derivative gain, link 1
KD2=(2*etta*omega*Jeff2-Beff2)/K2 #derivative gain, link 2
#end

th1=[] #array to graph, theta 1
th2=[] #array to graph, theta 2

th1_sp=[] #array to graph, set point theta 1
th2_sp=[] #array to graph, set point theta 2

qd_f=np.array([[90.00],[20.00]])*np.pi/180
dqd_f=np.array([[0.00],[0.00]])

qd_0=np.array([[0.00],[85.00]])*np.pi/180
dqd_0=np.array([[0.00],[0.00]])

t0=0 #simulation start time
tf=20 #simulation end time


#start---------------------------------
a0=qd_0
a1=dqd_0
a2=(3*(qd_f-qd_0)-(2*dqd_0+dqd_f)*(tf-t0))/((tf-t0)**2)
a3=(2*(qd_0-qd_f)+(dqd_0+dqd_f)*(tf-t0))/((tf-t0)**3)
#end-----------------------------------

time_list=np.arange(t0,tf,dt)

for t in time_list:
  th1.append(q[0,0]*180/np.pi) #array to graph, theta 1 [deg]
  th2.append(q[1,0]*180/np.pi) #array to graph, theta 2 [deg]

  th1_sp.append(sp_q[0,0]*180/np.pi) #array to graph, set point theta 1 [deg]
  th2_sp.append(sp_q[1,0]*180/np.pi) #array to graph, set point theta 2 [deg]
  
  sp_q=a0+a1*(t-t0)+a2*(t-t0)**2+a3*(t-t0)**3 #polynomial for trajectory interpolation

  D=D_robot(m1,m2,l1,l2,lc1,lc2,Izz1,Izz2,q)
  C=C_robot(m1,m2,l1,l2,lc1,lc2,Izz1,Izz2,q,dq)
  g=g_robot(m1,m2,l1,l2,lc1,lc2,Izz1,Izz2,q,grav)
  JM=np.diagflat(np.array([(Jeff1)/(red1**2),(Jeff2)/(red2**2)])) #matrix of inertias
  BM=np.diagflat(np.array([(Beff1)/(red1**2),(Beff2)/(red2**2)])) #matrix of friction

  ddq=np.linalg.inv(D+JM)@(input-C@dq-BM@q-g) #dynamic simulation
  dq=dq+dt*ddq #update, angular velocities
  q=q+dt*dq #update, theta angles

  #PD Control
  v1=KP1*(sp_q[0,0]-q[0,0])+KD1*(0-dq[0,0])
  v2=KP2*(sp_q[1,0]-q[1,0])+KD2*(0-dq[1,0])
  #End PD Control

  input[0,0]=K1*v1 #voltage/torque conversion
  input[1,0]=K2*v2 #voltage/torque conversion

plt.figure(0)
plt.figure(figsize=(12, 4),dpi=200)
plt.plot(time_list,th1,'g') 
plt.plot(time_list,th2,'b')

plt.plot(time_list,th1_sp,'g--') 
plt.plot(time_list,th2_sp,'b--')

plt.legend(["$\Theta_1$","$\Theta_2$","set point-$\Theta_1$","set point-$\Theta_2$"])
plt.grid()

plt.title("Trajectory interpolation method")
plt.xlabel("Time [s]")
plt.ylabel("Angular position [deg]")
Considering the control law (4) and (5) where:
plt.show()