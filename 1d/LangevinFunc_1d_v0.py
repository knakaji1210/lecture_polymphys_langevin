# Function to draw animation of 1d Langevin Particle

# brownianFunc_v2.pyをLangevin方程式のシミュレーション用に移植
# 試しに作ったLangevinEq_2d_ani_v0.pyはlistが先にできるので、そのための修正も必要だった

# 速度の自己相関関数について自分のやり方は間違えているようなので、以下を参考に作り直した
# https://tech.gijukatsu.com/numpy_autocorrelation/

# 2dバージョンを元に1dバージョンを作成

import random as rd
import numpy as np
from math import *

def Langevin_1d_dt01(m, zeta, kBT, N):
    dt = 0.1        # ms単位、通常はこちらで
    tmax = N*dt     # 100x0.1 = 10 msに

#   t, x, y, vx, vy = 0, 0, 0, 0, 0     # 初期座標(0,0)、初速(0,0)
    t, x, vx = 0, 0, np.sqrt(kBT/m)     # 初期座標(0,0)、初速1/2kBT程度

    t_list, x_list, vx_list = [], [], []

    while t <= tmax:
        t_list.append(t)
        x_list.append(x)
        vx_list.append(vx)

        t += dt
        x += vx*dt
        vx += (-zeta*vx*dt + np.sqrt(2*zeta*kBT*dt)*np.random.normal(0,1))/m  # Euler-Maruyama Approx.

    t_array = np.asanyarray(t_list, dtype=object)   # animatplotエラー対策でndarrayに変換
    x_array = np.asanyarray(x_list, dtype=object)   # animatplotエラー対策でndarrayに変換
    vx_array = np.asanyarray(vx_list, dtype=object) # animatplotエラー対策でndarrayに変換

    return t_array, x_array, vx_array, tmax

def Langevin_1d_dt001(m, zeta, kBT, N):
    dt = 0.01       # ms単位、速さの自己相関関数のときはこちらで
    tmax = N*dt     # 100x0.1 = 10 msに

#   t, x, y, vx, vy = 0, 0, 0, 0, 0     # 初期座標(0,0)、初速(0,0)
    t, x, vx = 0, 0, np.sqrt(kBT/m)     # 初期座標(0,0)、初速1/2kBT程度

    t_list, x_list, vx_list = [], [], []

    while t <= tmax:
        t_list.append(t)
        x_list.append(x)
        vx_list.append(vx)

        t += dt
        x += vx*dt
        vx += (-zeta*vx*dt + np.sqrt(2*zeta*kBT*dt)*np.random.normal(0,1))/m  # Euler-Maruyama Approx.

    t_array = np.asanyarray(t_list, dtype=object)   # animatplotエラー対策でndarrayに変換
    x_array = np.asanyarray(x_list, dtype=object)   # animatplotエラー対策でndarrayに変換
    vx_array = np.asanyarray(vx_list, dtype=object) # animatplotエラー対策でndarrayに変換

    return t_array, x_array, vx_array, tmax

def array2StepArray(array, N):
    array_steps = [ array[:i] for i in range(N+2) ]
    array_steps = array_steps[1:]

    # numpyのバージョンアップにより、""ndarray from ragged nested sequences"の制限が厳しくなり、
    # animatplotの途中でエラーが出るようになった。そのための修正が以下の１行
    array_steps = np.asanyarray(array_steps, dtype=object)

    return array_steps

# 原点からの距離（使っていない）
def distFromOrigin_1d(x_array):
    d_list = [ np.sqrt(x**2) for x in x_array] # x_arrayと同じものに・・・
    d_array = np.asanyarray(d_list, dtype=object)   # animatplotエラー対策でndarrayに変換

    return d_array

# 原点からの距離の二乗
def dist2FromOrigin_1d(x_array):
    d2_list = [ x**2 for x in x_array]
    d2_array = np.asanyarray(d2_list, dtype=object)  # animatplotエラー対策でndarrayに変換
 
    return d2_array

# 速さの自己相関関数（記事を参考にしたバージョン）
def velocityACF_1d(vx_array):    # auto-correlation function, ACF
    vACF_orig = np.correlate(vx_array, vx_array, "full")                 # np.correlateの元々の計算、結果が左右対称に出てくる
    vACF_slice = vACF_orig[int(vACF_orig.size/2):]                       # 右半分だけにスライス
    vACF_array = vACF_slice / np.arange(len(vx_array), 0, -1)                   # 1つ当たりに規格化              
    return vACF_array

def calcMean(list_repeat, N, M):
    mean_list = []
    for i in range(N):
        rep_list = [ list_repeat[j][i] for j in range(M) ]
        mean = np.mean(rep_list)
        mean_list.append(mean)
    return mean_list

def linearFit(t, a, b):
    return  a*t + b

def expFit(t, a, b, c):
    return  a*np.exp(-t/b) + c