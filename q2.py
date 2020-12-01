# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import scipy
from scipy.stats import t
import matplotlib.pyplot as plt

fileObject = open("question2_transport_time.txt", "r")
data =fileObject.read().split()
data =list(map(float,data))
print(data)

confidence_level = 0.95
degrees_freedom =len(data)-1
sample_mean = np.mean(data)
sample_standard_error = scipy.stats.sem(data)


confidence_interval = t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
tplot = t.pdf(data,degrees_freedom)
print(sample_mean)
print(sample_standard_error)
print(confidence_interval)
plt.plot(data)
plt.plot(tplot)
plt.show()




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
