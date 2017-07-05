import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

t=np.linspace(0,400,100000)
def f(Y,t):
    x1,x2=Y[0],Y[1]
    return[x2,-x2**3-x1**5+x1**4*(np.sin(x1))**2]

y0=[[0.1,0.5],[-0.8,0.2],[-0.9,-1.1],[0.9,-1.8]]
for i in y0:
    sol=odeint(f,i,t)
    plt.subplot(2,1,1)
    plt.plot(t,sol[:,0])
    plt.subplot(2,1,2)
    plt.plot(sol[:,0],sol[:,1],'b-')
    plt.plot([sol[0,0]],[sol[0,1]],'o')
    plt.plot([sol[-1,0]],[sol[-1,1]],'s')
plt.show()
