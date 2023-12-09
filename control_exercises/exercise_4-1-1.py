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

q=np.array([[20.00],[-20.00]])*np.pi/180 #initial angles [rad] 
dq=np.array([[0.0],[0.0]]) #initial angular velocity [rad/s]

f1=1.0 #frequency 1 [rad/s]
f2=0.5 #frequency 2 [rad/s]
sp_q=lambda t:np.array([[0.5*np.sin(f1*t)],[0.75*np.cos(f2*t)]]) #set point, angles [rad]

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

#begin, programm matrices KP and KD
etta=0.5
omega=2

KP=np.array([[omega**2*Jeff1/K1,0],[0,omega**2*Jeff2/K2]])
KD=np.array([[(2*etta*omega*Jeff1-Beff1)/K1,0],[0,(2*etta*omega*Jeff2-Beff2)/K2]])
#end


th1=[] #array to graph, theta 1
th2=[] #array to graph, theta 2

th1_sp=[] #array to graph, set point theta 1
th2_sp=[] #array to graph, set point theta 2

t0=0 #simulation start time 
tf=50 #simulation end time

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

  ddq=np.linalg.inv(D+JM)@(input-C@dq-BM@q-g) #dynamic simulation
  dq=dq+dt*ddq #update, angular velocities
  q=q+dt*dq #update, theta angles

  #PD Control
  input=KP@(sp_q(t)-q)-KD@(dq)
  #End PD Control

plt.figure(0)
plt.figure(figsize=(12, 4),dpi=200)
plt.plot(time_list,th1,'g') 
plt.plot(time_list,th2,'b')

plt.plot(time_list,th1_sp,'g--') 
plt.plot(time_list,th2_sp,'b--')

plt.legend(["$\Theta_1$","$\Theta_2$","set point-$\Theta_1$","set point-$\Theta_2$"])
plt.grid()

plt.title("PD control revisited")
plt.xlabel("Time [s]")
plt.ylabel("Angular position [deg]")

plt.show()
