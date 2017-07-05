import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

t=np.linspace(0,10,10000)
y0=[0.1,0.4,0.1,0.1]
b=2
r=4*np.sin(3*t)
#r=2.5
gamma=.5
gamma1=.9
c=2
eta=1.05
k=15.5
def f1(Y,t):
	y,h,h1,ym=Y[0],Y[1],Y[2],Y[3]
	r=4*np.sin(3*t)
	e=y-ym
	u=h*r-h*y**2-h1*y
	ydot=y+y**2+b*u
	hdot=(e*y**2-e*r)*gamma*np.tanh(b)
	h1dot=e*y*gamma1*np.tanh(b)
	ymdot=-2*ym+r
	return[ydot,hdot,h1dot,ymdot]

def f2(Y,t):
	y,h=Y[0],Y[1]
	e=y-4*np.sin(3*t)
	s=c*e
	u=h*12*np.cos(3*t)-h*y-h*y**2-eta*h*np.tanh(s)-k*s
	ydot=y+y**2+b*u
	hdot=c*gamma*s*(y+y**2-12*np.cos(3*t))
	return[ydot,hdot]

y1=[0.1,0.4]
sol1=odeint(f1,y0,t)
sol2=odeint(f2,y1,t,mxstep=500000,atol=1e-6)
y,h,h1,ym=sol1[:,0],sol1[:,1],sol1[:,2],sol1[:,3]
y1,h01=sol2[:,0],sol2[:,1]
e1=y1-4*np.sin(3*t)
s=c*e1
u1=h01*12*np.cos(3*t)-h01*y1-h01*y1**2-eta*h01*np.tanh(s)-k*s

e,u=y-ym,h*r-h*y**2-h1*y
plt.figure(1)
plt.subplot(2,2,1)
plt.plot(t,y)
plt.plot(t,ym,'g--')
plt.plot(t,e,'r--')
plt.subplot(2,2,2)
plt.plot(t,h)
plt.plot(t,h-1/b,'r--')
plt.ylabel('$h$')
plt.subplot(2,2,3)
plt.plot(t,h1)
plt.ylabel('$h_{1}$')
plt.subplot(2,2,4)
plt.plot(t,u)

plt.figure(2)
plt.subplot(3,1,1)
plt.plot(t,y1,t,e1,'r--')
plt.ylabel('$y$')
plt.subplot(3,1,2)
plt.plot(t,h01)
plt.ylabel('$parameter$')
plt.subplot(3,1,3)
plt.plot(t,u1)
plt.ylabel('$control$')
plt.show()
