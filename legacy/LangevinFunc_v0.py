# Function to draw animation of 2d Langevin Particle

# brownianFunc_v2.pyをLangevin方程式のシミュレーション用に移植
# 試しに作ったLangevinEq_2d_ani_v0.pyはlistが先にできるので、そのための修正も必要だった

import random as rd
import numpy as np
from math import *

def Langevin_2d_dt01(m, zeta, kBT, N):
    dt = 0.1        # ms単位、通常はこちらで
    tmax = N*dt     # 100x0.1 = 10 msに

#   t, x, y, vx, vy = 0, 0, 0, 0, 0     # 初期座標(0,0)、初速(0,0)
#  （補足）まだやってないが<v(0)v(t)>を求めるとき、v(0)=0は困るので・・・
    t, x, y, vx, vy = 0, 0, 0, np.sqrt(kBT/m), np.sqrt(kBT/m)     # 初期座標(0,0)、初速1/2kBT程度

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
        vy += (-zeta*vy*dt + np.sqrt(2*zeta*kBT*dt)*np.random.normal(0,1))/m  # Euler-Maruyama Approx.

    t_array = np.asanyarray(t_list, dtype=object)   # animatplotエラー対策でndarrayに変換
    x_array = np.asanyarray(x_list, dtype=object)   # animatplotエラー対策でndarrayに変換
    y_array = np.asanyarray(y_list, dtype=object)   # animatplotエラー対策でndarrayに変換
    vx_array = np.asanyarray(vx_list, dtype=object) # animatplotエラー対策でndarrayに変換
    vy_array = np.asanyarray(vy_list, dtype=object) # animatplotエラー対策でndarrayに変換

    return t_array, x_array, y_array , vx_array, vy_array, tmax

def Langevin_2d_dt001(m, zeta, kBT, N):
    dt = 0.01       # ms単位、速さの自己相関関数のときはこちらで
    tmax = N*dt     # 100x0.1 = 10 msに

#   t, x, y, vx, vy = 0, 0, 0, 0, 0     # 初期座標(0,0)、初速(0,0)
#  （補足）まだやってないが<v(0)v(t)>を求めるとき、v(0)=0は困るので・・・
    t, x, y, vx, vy = 0, 0, 0, np.sqrt(kBT/m), np.sqrt(kBT/m)     # 初期座標(0,0)、初速1/2kBT程度

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
        vy += (-zeta*vy*dt + np.sqrt(2*zeta*kBT*dt)*np.random.normal(0,1))/m  # Euler-Maruyama Approx.

    t_array = np.asanyarray(t_list, dtype=object)   # animatplotエラー対策でndarrayに変換
    x_array = np.asanyarray(x_list, dtype=object)   # animatplotエラー対策でndarrayに変換
    y_array = np.asanyarray(y_list, dtype=object)   # animatplotエラー対策でndarrayに変換
    vx_array = np.asanyarray(vx_list, dtype=object) # animatplotエラー対策でndarrayに変換
    vy_array = np.asanyarray(vy_list, dtype=object) # animatplotエラー対策でndarrayに変換

    return t_array, x_array, y_array , vx_array, vy_array, tmax

def array2StepArray(array, N):
    array_steps = [ array[:i] for i in range(N+2) ]
    array_steps = array_steps[1:]

    # numpyのバージョンアップにより、""ndarray from ragged nested sequences"の制限が厳しくなり、
    # animatplotの途中でエラーが出るようになった。そのための修正が以下の１行
    array_steps = np.asanyarray(array_steps, dtype=object)

    return array_steps

# 原点からの距離（使っていない）
def distFromOrigin(x_array, y_array):
    d_list = [ np.sqrt(x**2 + y**2) for x, y in zip(x_array, y_array)]
    d_array = np.asanyarray(d_list, dtype=object)  # animatplotエラー対策でndarrayに変換

    return d_array

# 原点からの距離の二乗
def dist2FromOrigin(x_array, y_array):
    d2_list = [ x**2 + y**2 for x, y in zip(x_array, y_array)]
    d2_array = np.asanyarray(d2_list, dtype=object)  # animatplotエラー対策でndarrayに変換
 
    return d2_array

# 速さの自己相関関数（これは間違えている）
def velocitySCF(vx_array, vy_array):
    velo_list = [ np.sqrt(vx**2 + vy**2) for vx, vy in zip(vx_array, vy_array)]
    v0 = velo_list[0]
    veloSCF_list = [ v0*v for v in velo_list ]
    veloSCF_array = np.asanyarray(veloSCF_list, dtype=object)  # animatplotエラー対策でndarrayに変換
 
    return veloSCF_array


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