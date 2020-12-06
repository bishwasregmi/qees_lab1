import numpy as np
import scipy
from scipy.stats import t, norm
import matplotlib.pyplot as plt


f1 = open("measurements/transport_time_120_fastrtps_RT_OFF_LD_OFF/transport_time_256byte.txt", "r")
d1 = f1.read().split()
d1 = list(map(float, d1))
f2 = open("measurements/transport_time_120_connext_RT_ON_LD_OFF/transport_time_256byte.txt", "r")
d2 = f2.read().split()
d2 = list(map(float, d2))
f3 = open("measurements/transport_time_120_opensplice_RT_ON_LD_OFF/transport_time_256byte.txt", "r")
d3 = f3.read().split()
d3 = list(map(float, d3))

alpha = 0.95
dof1= len(d1) - 1
mu1 = np.mean(d1)
std1 = np.std(d1)
sem1 = std1 / np.sqrt(len(d1))

t1 = abs(t.ppf((1 - alpha) / 2, df=dof1)) * sem1
c1 = [mu1 - t1, mu1 + t1]

dof2 = len(d2) - 1
mu2 = np.mean(d2)
std2 = np.std(d2)
sem2 = std2 / np.sqrt(len(d2))

t2 = abs(t.ppf((1 - alpha) / 2, df=dof2)) * sem2
c2 = [mu2 - t2, mu2 + t2]

dof3 = len(d3) - 1
mu3 = np.mean(d3)
std3 = np.std(d3)
sem3 = std3 / np.sqrt(len(d3))

t3 = abs(t.ppf((1 - alpha) / 2, df=dof3)) * sem3
c3 = [mu3 - t3, mu3 + t3]

print(c1)
print(c2)
print(c3)
