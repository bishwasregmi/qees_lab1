# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import scipy
from scipy.stats import t, norm
import matplotlib.pyplot as plt

fileObject = open("measurements/question2_transport_time.txt", "r")
# fileObject = open("transport_time_64Kbyte.txt", "r")
data = fileObject.read().split()
data = list(map(float, data))

alpha_1 = 0.98
alpha_2 = 0.80
dof = len(data) - 1
mu = np.mean(data)
std = np.std(data)
sem = std / np.sqrt(len(data))

t1 = abs(t.ppf((1 - alpha_1) / 2, df=dof))
t2 = abs(t.ppf((1 - alpha_2) / 2, df=dof))

c1 = [mu - t1*sem, mu + t1*sem]
c2 = [mu - t2*sem, mu + t2*sem]
print(f"dof {dof}")
print(f"std {std}")
print(f"sem = {sem}")
print(f"mu {mu}")
print(f"c1 {c1}")
print(f"c2 {c2}")
print(f"t1 {t1}")
print(f"t2 {t2}")
# plt.hist(data, density=True, bins=20, color='y')
plt.axvline(mu, color='k', label="sample_mean")
plt.axvline(c1[0], color='g', label="ConfidenceInterval_"+str(alpha_1))
plt.axvline(c1[1], color='g')
plt.axvline(c2[0], color='r', label="ConfidenceInterval_"+str(alpha_2))
plt.axvline(c2[1], color='r')

xmin, xmax = plt.xlim()
x = np.linspace(0.00085, 0.00105, 1000)
t_plot = t.pdf(x, dof, mu, sem)
plt.plot(x, t_plot)
plt.legend()
plt.show()
