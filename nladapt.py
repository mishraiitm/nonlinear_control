import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

c=1.5
k=2
gamma=1.2
t=np.linspace(0,20,10000)
def f1(Y,t):
    y,b=Y[0],Y[1]
    s=c*(y-np.sin(t))
    u=(np.cos(t)-y-y**2-k*s)/b
    ydot=y+y**2+2*u
    bdot=gamma*s*c*u
    return[ydot,bdot]

y0=[0.1,0]
sol=odeint(f1,y0,t)
s=c*(sol[:,0]-np.sin(t))
u=(np.cos(t)-sol[:,0]-sol[:,0]**2-k*s)/sol[:,1]
plt.subplot(3,1,1)
plt.plot(t,sol[:,0])
plt.subplot(3,1,2)
plt.plot(t,sol[:,1])
plt.subplot(3,1,3)
plt.plot(t,u)
plt.show()

