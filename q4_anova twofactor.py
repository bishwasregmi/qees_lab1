# load packages
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp
from statsmodels.stats.anova import anova_lm
import seaborn as sns
import researchpy as rp
import statsmodels.api as sm
from scipy.stats import t
import scipy.stats as st

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy import stats
from python_scripts.definitions import *
# load data file
# file_object = open("measurements/transport_time_120_annova analysis", "r")
# data = file_object.readlines()
# print(data)

latency = np.array(load_meas_data_q4())
latency = latency.flatten('C')
print(f"latency shape {latency.shape}")

df = pd.DataFrame({'latency' : latency,
                   'data_size' : np.repeat(['256B', '128kB'], 1500),
                   'dds' : np.r_[np.repeat(['fastrtps', 'opensplice', 'connext'],500),
                                 np.repeat(['fastrtps', 'opensplice', 'connext'],500)]})
# sns.boxplot(x = 'dds', y = 'latency', hue= 'data_size', data = df, showfliers=False)
# plt.grid()
# plt.show()
print(f"DataFrame summary:\n{rp.summary_cont(df.groupby(['data_size', 'dds']))['latency']}")
# print(df)

model = ols('latency ~ C(data_size)*C(dds)', df).fit()

print(f"Overall model: F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")

print(f"Model summary:\n {model.summary()}")

res = sm.stats.anova_lm(model, typ= 2)
print(f"Anova table:\n {res}")



# mc = statsmodels.stats.multicomp.MultiComparison(df['latency'],df['dds'])
# mc_results = mc.tukeyhsd()
# print(mc_results)
#
# mc = statsmodels.stats.multicomp.MultiComparison(df['latency'],df['data_size'])
# mc_results = mc.tukeyhsd()
# print(mc_results)


# ------------CALCULATING CONFIDENCE INTERVALS FOR DDSes-----------
latency = load_meas_data_q4()
fastrtps = np.concatenate((latency[0][0], latency[1][0]))
opensplice = np.concatenate((latency[0][1], latency[1][1]))
connext = np.concatenate((latency[0][2], latency[1][2]))

dds_meas = [fastrtps, opensplice, connext]

# the global mean of the measurements
total_mean = np.mean(latency)

# error variance(ssr) taken from the anova table
ssr = res['sum_sq'][3]

# dtandard deviation of the factor dds: dds_std = ssr * sqrt((a-1)/abr)
dds_std = ssr*np.sqrt(((3-1)/(3*2*500)))

# degree of freedom of error: dof = ab(r-1) = 2994
dof = 3*2*(500-1)
#confidence level = 0.95 -> alpha = 0.025
alpha = (1-0.95)/2
# using normal instead of t-distr because dof very big -> z = 1.9599639845400545
z_val = st.norm.ppf(alpha)
# calculating half of the interval
side = dds_std * abs(z_val)

#calculating effect of the levels of the factor dds
fastrtps_effect = np.mean(fastrtps)-total_mean
opensplice_effect =  np.mean(opensplice)-total_mean
connext_effect =  np.mean(connext)-total_mean

# calculating the intervals for each level
ci_fastrtps = [fastrtps_effect - side, fastrtps_effect + side]
ci_opensplice = [opensplice_effect - side, opensplice_effect + side]
ci_connext = [connext_effect - side, connext_effect + side]

print(f"CI fastrtps: {ci_fastrtps}")     # -> [-0.0007447844375547059, 0.0007490710375547059]
print(f"CI opensplice: {ci_opensplice}") # -> [-0.0007709446765547059, 0.0007229107985547059]
print(f"CI connext: {ci_connext}")       # -> [-0.0007250540985547059, 0.0007688013765547059]

print(min([np.linalg(ci_fastrtps), np.linalg(ci_opensplice), np.linalg(ci_connext)]))


