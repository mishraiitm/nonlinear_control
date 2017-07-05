import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 

t=np.linspace(0,1000,10000)
y0=[0.1,0.5]
c=10
k=10
eta=1.5
gamma=.1
hi=.5
r=4
#def h(h,eta,s,y):
#    return(0 if h==0 else h*(np.cos(t))-eta*s-h*y-h*y**2)

def f1(Y,t):
    y,h=Y[0],Y[1]
    e=y-np.sin(t)
    s=c*e
    #u=h(h,eta,s,y)
    u=h*(np.cos(t))-eta*np.tanh(10*s)-k*s-h*y-h*y**2

    ydot=y+y**2+(u/hi)
    hdot=gamma*(y+y**2-np.cos(t))
    return[ydot,hdot]

sol=odeint(f1,y0,t,mxstep=500000,atol=1e-6)
e=sol[:,0]-np.sin(t)
s=c*e
u=sol[:,1]*(np.cos(t))-eta*np.tanh(10*s)-k*s-sol[:,1]*sol[:,0]-sol[:,1]*sol[:,0]**2
print(1/sol[:,1])
plt.subplot(3,1,1)
#plt.plot(t,sol[:,0],t,np.sin(t),'r--')
plt.plot(t,sol[:,0]-np.sin(t),'g--')
#plt.ylim([-1,1])
plt.subplot(3,1,2)
plt.plot(t,u)
#plt.ylim([-10,10])
plt.subplot(3,1,3)
plt.plot(t,1/sol[:,1])
plt.show()
