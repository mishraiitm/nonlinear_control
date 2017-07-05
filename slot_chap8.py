import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

t=np.linspace(0,20,10000)
r=np.sin(4*t)+np.cos(t)
#r=np.tanh(t)
mr=2
y01=[0.1,0.2]
y02=[0.5,0,0.5,0,0]
l=6
l1=10
l2=25
gamma=0.5
def f1(Y,t):
    x1,x2=Y[0],Y[1]
    return[x2,-l**2*x1-2*l*x2]

def f2(Y,t):
    x1,x2,xm1,xm2,m=Y[0],Y[1],Y[2],Y[3],Y[4]
    r=np.sin(4*t)+np.cos(t)
    e=x1-xm1
    edot=x2-xm2
    s=edot+l*e
    u=m*(l2*r-l2*xm1-l1*xm2-2*l*edot-l**2*e)
    x1dot=x2
    x2dot=u/mr
    v=l2*r-l2*xm1-l1*xm2-2*l*edot-l**2*e
    xm1dot=xm2
    xm2dot=l2*r-l2*xm1-l1*xm2
    m=-gamma*s*v
    return[x1dot,x2dot,xm1dot,xm2dot,m]

sol1=odeint(f1,y01,t)
sol2=odeint(f2,y02,t)
s=sol2[:,1]-sol2[:,3]+l*(sol2[:,0]-sol2[:,2])
edot=sol2[:,1]-sol2[:,3]
e=sol2[:,0]-sol2[:,2]
u=sol2[:,4]*(l2*r-l2*sol2[:,2]-l1*sol2[:,3]-2*l*edot-l**2*e)
#sdot=u-l2*r-l2*sol2[:,2]-l1*sol2[:,3]+l*e
v=l2*r-l2*sol2[:,2]-2*l*edot-l**2*e
sdot=(sol2[:,4]*v-l*mr*s)/mr
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t,sol1[:,0])
plt.plot(t,sol1[:,1])
plt.subplot(2,1,2)
plt.plot(sol1[:,0],sol1[:,1])
plt.figure(2)
plt.subplot(2,2,1)
plt.plot(t,sol2[:,0])
plt.plot(t,sol2[:,2],'g--')
plt.subplot(2,2,2)
plt.plot(s,sdot)
plt.plot(s[0],sdot[0],'ro')
plt.plot(s[-1],sdot[-1],'bo')
plt.subplot(2,2,3)
plt.plot(t,sol2[:,4])
plt.subplot(2,2,4)
plt.plot(t,u)
plt.show()
