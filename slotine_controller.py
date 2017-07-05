import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


t=np.linspace(0,200,100000)
def f(Y,t):
    x1,x2=Y[0],Y[1]
    x1dot=x2
    x2dot=-5*x1**2*np.tanh(100*x1)-2*x1**2-2*x2**3-x2**3
    return[x1dot,x2dot]


y0=[0.1,0]
sol=odeint(f,y0,t)
plt.subplot(2,1,1)
plt.plot(t,sol[:,0])
plt.plot(t,sol[:,1])
plt.subplot(2,1,2)
plt.plot(sol[:,0],sol[:,1])
plt.show()
