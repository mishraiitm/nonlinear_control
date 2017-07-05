import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
eta=0.2
t=np.linspace(0,100,1000)

def f1(Y,t):
    x1,x2=Y[0],Y[1]
    u=x1**2-x1-x2**3
    x1dot=x2
    x2dot=u-x1**2+x2**3
    return[x1dot,x2dot]

def f2(Y,t):
    x1,x2=Y[0],Y[1]
    u=x1**2-x1-x2**3-eta*np.tanh(x2)
    x1dot=x2
    x2dot=u-x1**2+x2**3
    return[x1dot,x2dot]

def f3(Y,t):
    x1,x2=Y[0],Y[1]
    u=x1**2*np.tanh(x1)-x1*np.tanh(x1)-x2**3*np.tanh(x2)
    x1dot=x2
    x2dot=u-x1**2+x2**3
    return[x1dot,x2dot]

y0=[0.1,0.2]
sol1=odeint(f1,y0,t)
sol2=odeint(f2,y0,t)
sol3=odeint(f3,y0,t)
X,Y=np.linspace(-5,5,1000),np.linspace(-5,5,1000)
X1,Y1=np.meshgrid(X,Y)
u1=(sol1[:,0])**2-sol1[:,0]-(sol1[:,1])**3
u2=(sol2[:,0])**2-sol2[:,0]-(sol2[:,1])**3-eta*np.tanh(sol2[:,1])
u3=-2*sol3[:,0]**2-2*sol3[:,1]**3
v1=0.5*sol1[:,0]**2+0.5*sol1[:,1]**2
v2=0.5*sol2[:,0]**2+0.5*sol2[:,1]**2
v3=0.5*sol3[:,0]**2+0.5*sol3[:,1]**2
plt.figure(1)
plt.subplot(3,1,1)
plt.plot(t,sol1[:,0])
plt.subplot(3,1,2)
plt.plot(sol1[:,0],sol1[:,1])
plt.plot(sol1[0,0],sol1[0,1],'o')
plt.plot(sol1[-1,0],sol1[-1,1],'s')
plt.subplot(3,1,3)
plt.plot(t,u1)
plt.figure(2)
plt.subplot(3,1,1)
plt.plot(t,sol2[:,0])
plt.subplot(3,1,2)
plt.plot(sol2[:,0],sol2[:,1])
plt.plot(sol2[0,0],sol2[0,1],'o')
plt.plot(sol2[-1,0],sol2[-1,1],'s')
plt.subplot(3,1,3)
plt.plot(t,u2)
plt.figure(3)
plt.subplot(3,1,1)
plt.plot(t,sol3[:,0])
plt.subplot(3,1,2)
plt.plot(sol3[:,0],sol3[:,1])
plt.plot(sol3[0,0],sol3[0,1],'o')
plt.plot(sol3[-1,0],sol3[-1,1],'s')
plt.subplot(3,1,3)
plt.plot(t,u3)


fig1=plt.figure(4)
ax=fig1.add_subplot(111,projection='3d')
ax.scatter(sol1[:,0],sol1[:,1],v1)
fig2=plt.figure(5)
ax=fig2.add_subplot(111,projection='3d')
#Axes3D.plot_surface(X2,Y2,v2)
#sol2[:,0],sol2[:,1],v2=Axes3D.get_test_data(0.05)
ax.scatter(sol2[:,0],sol2[:,1],v2)

fig3=plt.figure(6)
ax=fig3.add_subplot(111,projection='3d')
ax.scatter(sol3[:,0],sol3[:,1],v3)
plt.show()
