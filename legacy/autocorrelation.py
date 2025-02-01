# 自己相関関数の計算
# 以下を参考に作成
# https://tech.gijukatsu.com/numpy_autocorrelation/

import numpy as np
import matplotlib.pyplot as plt

def auto_correlation(x):
    corre_orig = np.correlate(x, x, "full")                 # np.correlateの元々の計算、結果が左右対称に出てくる
    corre_slice = corre_orig[int(corre_orig.size/2):]       # 右半分だけにスライス
    corre = corre_slice / np.arange(len(x), 0, -1)
    return corre_orig, corre_slice, corre

a = np.array([1,2,3])
  
a_ACF_orig, a_ACF_slice, a_ACF = auto_correlation(a)

print(a_ACF_orig)
print(a_ACF_slice)
print(np.arange(len(a), 0, -1))
print(a_ACF)