import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

t=np.linspace(0,100,10000)
y0=[0.3,0,0.1,0.3,0.5]
b,a1,a2,gamma,gamma1,gamma2,c,eta,k=2,1,3,7,5,6,3,1.5,15
def f(Y,t):
    x1,x2,h,h1,h2=Y[0],Y[1],Y[2],Y[3],Y[4]
    e=x1-np.sin(t)
    edot=x2-np.cos(t)
    s=c*e+edot
    u=-k*s-eta*np.tanh(s)+h1*x2+h2*x1**2-h*np.sin(t)-h*c*edot
    xdot=x2
    xddot=b*u-a1*x2-a2*x1**2
    hdot=gamma*(c*edot*s+np.sin(t)*s)
    h1dot=-gamma1*s*x2
    h2dot=-gamma2*x1**2*s
    return[xdot,xddot,hdot,h1dot,h2dot]

sol=odeint(f,y0,t)
x=sol[:,0]
xdot=sol[:,1]
h=sol[:,2]
h1=sol[:,3]
h2=sol[:,4]
e,de=x-np.sin(t),xdot-np.cos(t)
s=c*e+de
u=-k*s-eta*np.tanh(s)+h1*xdot+h2*x**2-h*np.sin(t)-h*c*de

xddot=b*u-a1*xdot-a2*x**2
ds=c*de+xddot+np.sin(t)
u=-k*s-eta*np.tanh(s)+h1*xdot+h2*x**2-h*np.sin(t)-h*c*de
plt.subplot(2,3,1)
plt.plot(t,x,label='$x$')
plt.plot(t,e,'r--',label='$error$')
plt.legend(loc='lower left')
plt.subplot(2,3,2)
plt.plot(t,u,'g-',label='$control$')
plt.legend(loc='upper left')
plt.subplot(2,3,3)
plt.plot(x,xdot)
plt.subplot(2,3,4)
plt.plot(t,1/h,label='b=2')
plt.legend(loc='upper left')
plt.subplot(2,3,5)
plt.plot(t,1/h1,label='a=1')
plt.legend(loc='upper left')
plt.subplot(2,3,6)
plt.plot(t,1/h2,label='a=3')
plt.legend(loc='upper right')
plt.savefig('slot_1986_2a.eps')
plt.show()
