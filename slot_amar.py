import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
k=0.3
eta=1.9
t=np.linspace(0,20,1000)
def f(Y,t):
    x1,x2=Y[0],Y[1]
    x1dot=x2
    u=x1**2-x1-x2**3-eta*np.tanh(10*x2)-k*x2
    x2dot=u-x1**2+x2**3
    return[x1dot,x2dot]

y0=[0.1,20]
sol=odeint(f,y0,t)
u=sol[:,0]**2-sol[:,0]-sol[:,1]**3-eta*np.tanh(10*sol[:,1])-k*sol[:,1]
plt.subplot(3,1,1)
plt.plot(t,sol[:,0],'r--',t,sol[:,1])
plt.subplot(3,1,2)
plt.plot(sol[:,0],sol[:,1])
plt.plot([sol[0,0]],[sol[0,1]],'o')
plt.plot([sol[-1,0]],[sol[-1,1]],'s')

plt.subplot(3,1,3)
plt.plot(t,u)
plt.show()
