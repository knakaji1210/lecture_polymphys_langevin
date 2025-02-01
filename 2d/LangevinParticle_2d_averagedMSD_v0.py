# Statistics of 2d Langevin Particle
# さらにbrownianMotion_with_DstProfile_stat_v2.pyを参考に原点からの距離の時間変化の平均を求める

# brownianMotion_ani_v2.pyをLangevin方程式のシミュレーション用に移植
# 試しに作ったLangevinEq_2d_ani_v0.pyはlistが先にできるので、そのための修正も必要だった

# Dの計算だけ行えるものとして名前を変えて残す

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
try:
    M = int(input('Number of repeat (default=10000): '))
except ValueError:
    M = 10000

t_list_repeat = []
d2_list_repeat = []

for i in range (M):
    t_array, x_array, y_array , vx_array, vy_array, tmax = lvf.Langevin_2d_dt01(m, zeta, kBT, N)

    d_array = lvf.distFromOrigin(x_array, y_array)
    d2_array = lvf.dist2FromOrigin(x_array, y_array)

    t_list_repeat.append(t_array)
    d2_list_repeat.append(d2_array)

plot_lim = np.sqrt(N)
t = np.linspace(0, N, N+1)

t_mean_list = lvf.calcMean(t_list_repeat, N, M)
d2_mean_list = lvf.calcMean(d2_list_repeat, N, M)

l = len(t_mean_list)

t_mean_list_fit = t_mean_list[int(l*0.2):]
d2_mean_list_fit = d2_mean_list[int(l*0.2):]

param, cov = curve_fit(lvf.linearFit, t_mean_list_fit, d2_mean_list_fit)
slope = param[0]
diffConst = slope / 4.0     # 2次元なので<d^2> = 4Dt
sect = param[1]
err_diffConst = np.sqrt(cov[0][0]) / 4.0
d2_fit_list = [ lvf.linearFit(tim, slope, sect) for tim in t_mean_list ]

fig_text1 = "Number of steps: {}".format(N)
fig_text2 = "Number of repetition: {}".format(M)

fig_text3 = "$m$ = {0:.1f} x 10$^{{-15}}$ kg".format(m)
fig_text4 = "$\zeta$ = {0:.1f} x 10$^{{-8}}$ kg/s".format(zeta)
fig_text5 = "$k_B$$T$ = {0:.1f} x 10$^{{-21}}$ J".format(kBT)

result_text = "$D$ = {0:.3f}±{1:.3f} um$^{{2}}$/ms".format(diffConst, err_diffConst)

fig_title = "<$d^{{2}}$> vs $t$"

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, title=fig_title, xlabel='$t$ [ms]', ylabel='$d^{{2}}$ [um$^{{2}}$]')
ax.grid(axis='both', color="gray", lw=0.5)

# MSD
ax.scatter(t_mean_list, d2_mean_list, marker="o", color="red")
ax.plot(t_mean_list, d2_fit_list, lw=1, color='black')

fig.text(0.15, 0.80, fig_text1)
fig.text(0.15, 0.75, fig_text2)
fig.text(0.15, 0.70, fig_text3)
fig.text(0.15, 0.65, fig_text4)
fig.text(0.15, 0.60, fig_text5)
fig.text(0.15, 0.50, result_text)

savefile = "./png/LangevinParticle_2d_averagedMSD_{0}steps_{1}repetition".format(N, M)
fig.savefig(savefile, dpi=300)

plt.show()
plt.close()
