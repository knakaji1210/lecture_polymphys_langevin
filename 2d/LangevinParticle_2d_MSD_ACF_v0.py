# Statistics of 2d Langevin Particle
# さらにbrownianMotion_with_DstProfile_stat_v2.pyを参考に原点からの距離の時間変化の平均を求める

# brownianMotion_ani_v2.pyをLangevin方程式のシミュレーション用に移植
# 試しに作ったLangevinEq_2d_ani_v0.pyはlistが先にできるので、そのための修正も必要だった

# v1では速度の自己相関関数もやってみた
# v1の自分のやり方は間違えているようなので、以下を参考に作り直した
# その変更はLangevinFunc_v1.pyの中で実現
# https://tech.gijukatsu.com/numpy_autocorrelation/

# LangevinParticle_2d_MSD_ACF_v0.pyと名前を変更

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import animatplot as amp
import LangevinFunc_2d_v1 as lvf

try:
    N = int(input('Number of steps (default = 2000): '))
except ValueError:
    N = 2000         # 200*0.1 = 20 msに相当
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
    M = int(input('Number of repeat (default=5000): '))
except ValueError:
    M = 5000

t_list_repeat = []
d2_list_repeat = []
vACF_list_repeat = []

for i in range (M):
    t_array, x_array, y_array , vx_array, vy_array, tmax = lvf.Langevin_2d_dt001(m, zeta, kBT, N)

    d_array = lvf.distFromOrigin(x_array, y_array)
    d2_array = lvf.dist2FromOrigin(x_array, y_array)
    vACF_array = lvf.velocityACF(vx_array, vy_array)

    t_list_repeat.append(t_array)
    d2_list_repeat.append(d2_array)
    vACF_list_repeat.append(vACF_array)

plot_lim = np.sqrt(N)
t = np.linspace(0, N, N+1)

t_mean_list = lvf.calcMean(t_list_repeat, N, M)
d2_mean_list = lvf.calcMean(d2_list_repeat, N, M)
vACF_mean_list = lvf.calcMean(vACF_list_repeat, N, M)

l = len(t_mean_list)

# 拡散係数Dの計算
t_mean_list_fit = t_mean_list[int(l*0.2):]
d2_mean_list_fit = d2_mean_list[int(l*0.2):]

param, cov = curve_fit(lvf.linearFit, t_mean_list_fit, d2_mean_list_fit)
slope = param[0]
diffConst = slope / 4.0     # 2次元なので<d^2> = 4Dt
sect = param[1]
err_diffConst = np.sqrt(cov[0][0]) / 4.0
d2_fit_list = [ lvf.linearFit(tim, slope, sect) for tim in t_mean_list ]

result_text1 = "$D$ = {0:.3f}±{1:.3f} um$^{{2}}$/ms".format(diffConst, err_diffConst)

# 速さの自己相関関数の計算
# log_t_mean_list = [ np.log10(t) for t in t_mean_list ]
# log_veloSCF_list = [ np.log10(v) for v in veloSCF_list ]
# 後半でACFが上がっちゃうので、そこをフィッティング範囲に含めないために必要なスライス
# 後半のACF上がる問題は、今のACFの計算の仕方からして仕方ないと思われる
t_mean_list_fit2 = t_mean_list[:int(l*0.75)]
vACF_mean_list_fit = vACF_mean_list[:int(l*0.75)]
param, cov = curve_fit(lvf.expFit, t_mean_list_fit2, vACF_mean_list_fit)
ampl = param[0]
tau = param[1]
base = param[2]
err_ampl = np.sqrt(cov[0][0])
err_tau = np.sqrt(cov[1][1])

vACF_fit_list = [ lvf.expFit(tim, ampl, tau, base) for tim in t_mean_list ]

result_text2 = "$A$ = {0:.3f}±{1:.3f} (um/ms)$^2$".format(ampl, err_ampl)
result_text3 = "$\u03C4$ = {0:.3f}±{1:.3f} ms".format(tau, err_tau)

fig_text1 = "Number of steps: {}".format(N)
fig_text2 = "Number of repetition: {}".format(M)

fig_text3 = "$m$ = {0:.1f} x 10$^{{-15}}$ kg".format(m)
fig_text4 = "$\zeta$ = {0:.1f} x 10$^{{-8}}$ kg/s".format(zeta)
fig_text5 = "$k_B$$T$ = {0:.1f} x 10$^{{-21}}$ J".format(kBT)

fig_title1 = "<$d^{{2}}$> vs $t$"
fig_title2 = "<$v$($t$)$v$(0)> vs $t$"

fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(121, title=fig_title1, xlabel='$t$ [ms]', ylabel='$d^{{2}}$ [um$^{{2}}$]')
ax1.grid(axis='both', color="gray", lw=0.5)

# MSD
ax1.scatter(t_mean_list, d2_mean_list, marker="o", color="red")
ax1.plot(t_mean_list, d2_fit_list, lw=1, color='black')

# SCF
ax2 = fig.add_subplot(122, title=fig_title2, xlabel='$t$ [ms]', ylabel='<$v$($t$)$v$(0)>')
ax2.grid(axis='both', color="gray", lw=0.5)
ax2.scatter(t_mean_list_fit2, vACF_mean_list_fit, marker="o", color="green")
ax2.plot(t_mean_list, vACF_fit_list, lw=1, color='black')

fig.text(0.15, 0.80, fig_text1)
fig.text(0.15, 0.75, fig_text2)
fig.text(0.15, 0.70, fig_text3)
fig.text(0.15, 0.65, fig_text4)
fig.text(0.15, 0.60, fig_text5)
fig.text(0.15, 0.50, result_text1)

fig.text(0.60, 0.80, result_text2)
fig.text(0.60, 0.75, result_text3)

savefile = "./png/LangevinParticle_2d_MSD_ACF_N{0}_M{1}".format(N, M)
fig.savefig(savefile, dpi=300)

plt.show()
plt.close()
