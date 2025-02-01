# Langevin Eqをシミュレートする試み

# 以下のオイラー・丸山法に関する記事参考にした。
# https://qiita.com/chemweb000/items/1a7333bc485fb36cfb5f

import numpy as np
import random as rd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def Langevin_2d(m, zeta, kBT, tmax, dt):

  t, x, y, vx, vy = 0, 0, 0, np.sqrt(kBT/m), np.sqrt(kBT/m)       # 初速は1 um/msになる（長さはum単位ということ）

  t_list, x_list, y_list , vx_list, vy_list = [], [], [], [], []

  while t <= tmax:
    t_list.append(t)
    x_list.append(x)
    y_list.append(y)
    vx_list.append(vx)
    vy_list.append(vy)

    t += dt
    x += vx*dt
    y += vy*dt
    vx += (-zeta*vx*dt + np.sqrt(2*zeta*kBT*dt)*np.random.normal(0,1))/m  # Euler-Maruyama Approx.
    vy += (-zeta*vy*dt + np.sqrt(2*zeta*kBT*dt)*np.random.normal(0,1))/m

  return t_list, x_list, y_list , vx_list, vy_list

tmax = 10   # ms単位
dt = 0.1    # ms単位

m = 1.0     # 10^(-15) kg単位で考えている
zeta = 1.0  # 10^(-12) kg/sに相当
kBT = 1.0   # m = 10^(-15) kg, v = 10^(-3) m/s = 1 um/msの速さに対応する熱エネルギー

t_list, x_list, y_list , vx_list, vy_list = Langevin_2d(m, zeta, kBT, tmax, dt)

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='$x$ position [um]', ylabel='$y$ position [um]', xlim=[-10, 10], ylim=(-10, 10))
ax.grid()

particle, = ax.plot([], [], 'ro', markersize='15', animated=True)

time_template = 'time = %.1f ms'
time_text = ax.text(0.1, 0.9, '', transform=ax.transAxes)

def init():
    time_text.set_text('')
    return particle, time_text

def update(i):
    particle.set_data(x_list[i], y_list[i])
    time_text.set_text(time_template % (i*dt))
    return particle, time_text

frame_int = 1000 * dt       # [ms] interval between frames
fps = 1000/frame_int        # frames per second

ani = FuncAnimation(fig, update, frames=np.arange(0, int(tmax/dt)),
                    init_func=init, blit=True, interval=frame_int, repeat=True)
plt.show()

ani.save('./gif/Langevin_2d.gif', writer='pillow', fps=fps)