import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

t=np.linspace(0,800,100000)
def f(Y,t):
    x1,x2=Y[0],Y[1]
    u=-2*x2**3-5*x1**2*np.tanh(10*x1)
    x1dot=x2
    x2dot=u-x1**2+x2**3
    return[x1dot,x2dot]

y0=[0.1,0.2]
sol=odeint(f,y0,t)
u=-2*sol[:,1]**3-5*sol[:,0]**2*np.tanh(10*sol[:,0])
plt.subplot(3,1,1)
plt.plot(t,sol[:,0])
plt.subplot(3,1,2)
plt.plot(sol[:,0],sol[:,1])
plt.plot([sol[0,0]],[sol[0,1]],'o')
plt.plot([sol[-1,0]],[sol[-1,1]],'s')
plt.subplot(3,1,3)
plt.plot(t,u)
plt.show()

