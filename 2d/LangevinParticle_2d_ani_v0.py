# Animation of 2d Langevin Particle

# brownianMotion_ani_v2.pyをLangevin方程式のシミュレーション用に移植
# 試しに作ったLangevinEq_2d_ani_v0.pyはlistが先にできるので、そのための修正も必要だった
# 軌跡のアニメーションだけ出すバージョン

import numpy as np
import matplotlib.pyplot as plt
import animatplot as amp
import LangevinFunc_2d_v1 as lvf

try:
    N = int(input('Number of steps (default = 200): '))
except ValueError:
    N = 200         # 200*0.1 = 20 msに相当
try:
    m = int(input('Mass of particle (default = 1.0): '))
except ValueError:
    m = 1.0         # 10^(-15) kg単位で考えている、1.0 umくらいのサイズの粒子に相当
try:
    zeta = int(input('Frictional coef of particle (default = 1.0): '))
except ValueError:
    zeta = 1.0      # 10^(-12) kg/sに相当 
try:
    kBT = int(input('Thermal Energy of particle (default= 1.0): '))
except ValueError:
    kBT = 1.0       # m = 10^(-15) kg, v = 10^(-3) m/s = 1 um/msの速さに対応する熱エネルギー 

t_array, x_array, y_array , vx_array, vy_array, tmax = lvf.Langevin_2d_dt01(m, zeta, kBT, N)

t_array_steps = lvf.array2StepArray(t_array, N)
x_array_steps = lvf.array2StepArray(x_array, N)
y_array_steps = lvf.array2StepArray(y_array, N)

plot_lim = np.sqrt(N)
steps = np.linspace(0, N, N+1)

fig_title = "2-dimensional Langevin Particle ({0} steps, d$t$ = 0.1 ms)".format(N)

fig_text1 = "$m$ = {0:.1f} x 10$^{{-15}}$ kg".format(m)
fig_text2 = "$\zeta$ = {0:.1f} x 10$^{{-8}}$ kg/s".format(zeta)
fig_text3 = "$k_B$$T$ = {0:.1f} x 10$^{{-21}}$ J".format(kBT)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, title=fig_title, xlabel='$X$ [um]', ylabel='$Y$ [um]',
                        xlim=[-plot_lim, plot_lim], ylim=[-plot_lim , plot_lim])
ax.grid(axis='both', color="gray", lw=0.5)

fig.text(0.15, 0.80, fig_text1)
fig.text(0.15, 0.75, fig_text2)
fig.text(0.15, 0.70, fig_text3)

LangevinParticle = amp.blocks.Line(x_array_steps, y_array_steps, ax=ax, ls='-', marker="o", markersize=2, color='blue')

timeline = amp.Timeline(steps, units=' steps', fps=30)

anim = amp.Animation([LangevinParticle], timeline)
anim.controls()

savefile = "./gif/LangevinParticle_2d_{0}steps".format(N)
anim.save_gif(savefile)

plt.show()
plt.close()
