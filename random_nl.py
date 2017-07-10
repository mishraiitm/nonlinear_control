import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
t=np.linspace(0,250,10000)
c=2
k=25
eta=2.5
gamma,gamma1,gamma2=10,9,10
b,a1,a2=2,3,4
def check(x1,u,a1,x2,a2,b):
    if np.sin(x1)<=1e-6 and np.sin(x1)>=0:
        l=1e-6
        return b*(1.0/0.1)*u-a1*x2-a2*x1**2
    elif np.sin(x1)>=-1e-6 and np.sin(x1)<0:
        l=-1e-6
        return b*(1.0/-0.1)*u-a1*x2-a2*x1**2
    else:
        return b*(1.0/np.sin(x1))*u-a1*x2-a2*x1**2


def f(Y,t):
    x1,x2,h,h1,h2=Y[0],Y[1],Y[2],Y[3],Y[4]
    #x"+a1x'+a2x^2=bcosec(x)u
    e=x1-np.sin(t)
    de=x2-np.cos(t)
    s=c*e+de
    u=-k*s-eta*np.tanh(s)+s*h1*x2+np.sin(x1)*h2*x1**2-np.sin(x1)*h*np.sin(t)-h*np.sin(t)*c*de-h*np.cos(x1)*s/2
    x1dot=x2
    #x2dot=b*(1.0/(np.sin(x1)))*u-a1*x2-a2*x1**2
    x2dot=check(x1,u,a1,x2,a2,b)
    hdot=gamma*(s*np.tanh(10*np.sin(x1))*(s*np.cos(x1)/2+np.sin(x1)*np.sin(t)+np.sin(x1)*c*de))
    h1dot=-gamma1*(s*np.tanh(10*np.sin(x1)))*np.sin(x1)*x2
    h2dot=-gamma2*(s*np.tanh(10*np.sin(x1)))*np.sin(x1)*x1**2
    return[x1dot,x2dot,hdot,h1dot,h2dot]

y0=[0.1,0.2,0,0,0]
sol=odeint(f,y0,t,mxstep=500000,atol=1e-6)
x=sol[:,0]
xd=sol[:,1]
h=sol[:,2]
h1=sol[:,3]
h2=sol[:,4]
e,de=x-np.sin(t),xd-np.cos(t)
s=c*e+de
u=-k*s-eta*np.tanh(s)+s*h1*xd+np.sin(x)*h2*x**2-np.sin(x)*h*np.sin(t)-h*np.sin(t)*c*de-h*np.cos(x)*s/2
plt.subplot(2,3,1)
plt.plot(t,x,t,e,'r--')
#plt.plot(t,np.sin(t),'g--')
plt.subplot(2,3,2)
plt.plot(x,xd)
plt.subplot(2,3,3)
plt.plot(t,u)
plt.subplot(2,3,4)
plt.plot(t,h)
plt.subplot(2,3,5)
plt.plot(t,h1)
plt.subplot(2,3,6)
plt.plot(t,h2)
print(x[t])
print(h)
plt.show()
