import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
k=10
eta=1.5
c=2.5
p=1
gamma=12
t=np.linspace(0,150,10000)
def adapt(Y,t):
    x=Y[0]
    theta=Y[1]
    thetadot=gamma*np.sin(x)
    u=-p*x-theta*np.sin(x)
    return[u+1.5*np.sin(x),thetadot]


def adapt_slide(Y,t):
    x=Y[0]
    theta=Y[1]
    e=x-np.sin(t)
    s=c*e
    thetadot=gamma*s*c*np.sin(x)
    u=-theta*np.sin(x)+np.cos(t)-k*s-eta*np.tanh(s)
    return[u+1.5*np.sin(x),thetadot]
y0=[0.1,0]
sol1=odeint(adapt,y0,t)
sol2=odeint(adapt_slide,y0,t)
u1=-p*sol1[:,0]-sol1[:,1]*np.sin(sol1[:,0])
error2=sol2[:,0]-np.sin(t)
s=c*error2
u2=-sol2[:,1]*np.sin(sol2[:,0])+np.cos(t)-k*s-eta*np.tanh(s)
derror2=u2+sol2[:,1]*np.sin(sol2[:,0])-np.cos(t)
plt.subplot(3,2,1)
plt.plot(t,sol1[:,0])
plt.plot(t,sol1[:,1],'r--')
plt.plot(t,1.5-sol1[:,1],'g--')
plt.subplot(3,2,2)
plt.plot(t,sol2[:,0])
plt.plot(t,sol2[:,1],'r--')
plt.plot(t,1.5-sol2[:,1],'g--')
plt.plot(t,np.sin(t),'b--')
plt.plot(t,error2)
plt.plot(t,derror2)
plt.subplot(3,2,3)
plt.plot(t,u1)
plt.subplot(3,2,4)
plt.plot(t,u2)
#plt.subplot(3,2,5)
#plt.plot(error1,derror1)
plt.subplot(3,2,6)
plt.plot(error2,derror2)
print(derror2)
print(error2)
plt.show()
