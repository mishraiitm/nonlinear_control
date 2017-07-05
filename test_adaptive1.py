import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

t=np.linspace(0,10,1000)
c=2
eta=0.45
def f1(Y,t):
    x1,x2=Y[0],Y[1]
    e1=x1-np.sin(t)
    e2=x2-np.sin(t)

    x1dot=a*x1**2+b*u1*np.cos(x2)+c*u2
    x2dot=d*x2**2+e*u2*np.cos(x1)+f*u1
    return[x1dot,x2dot]

y0=[0.1,0.4]

